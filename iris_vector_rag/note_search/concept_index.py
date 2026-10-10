"""A postings table of concepts per section, and search by concept id (spec 091).

One row per (section, concept): `section_id` = ``<parent>#<ordinal>``, `parent_id`,
`concept_id`, `polarity` (the section polarity of that concept's mentions), `mentions`
and `section_label`. Search is AND across concepts at the parent level: a parent is
returned when it has a posting for every requested concept (each posting matching the
polarity and section filters when given).

Needs an `IRISVectorStore` over the chunk table, because section texts are read back
from it. The postings table lives in the same schema.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Dict, List, Optional, Tuple

from iris_vector_rag.core.models import Document
from iris_vector_rag.note_search.concepts import ConceptTagger, postings_for_section
from iris_vector_rag.note_search.polarity import (
    AFFIRMED,
    NEGATED,
    UNKNOWN,
    section_polarity,
)

__all__ = [
    "ConceptIndex",
    "ConceptIndexError",
    "ConceptIndexMissingError",
]

DEFAULT_TABLE_NAME = "SectionConcepts"
POLARITIES = (AFFIRMED, NEGATED, UNKNOWN)
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")
_COLUMNS = (
    "section_id",
    "parent_id",
    "concept_id",
    "polarity",
    "mentions",
    "section_label",
)


class ConceptIndexError(RuntimeError):
    """Postings could not be written or read; a write leaves the earlier rows in place."""


class ConceptIndexMissingError(RuntimeError):
    """The postings table does not exist. Names the fix: ``ensure_table()``."""


class ConceptIndex:
    """Concept postings for the sections of parent documents.

    Args:
        vector_store: An `IRISVectorStore` over the chunk-shaped table that holds the
            sections (see `SectionStore`).
        table: ``Schema.Table`` or bare name for the postings table. Default:
            ``SectionConcepts`` in the schema of the vector store's table.

    Raises:
        ValueError: the table name is not a plain (optionally schema-qualified) name.
    """

    def __init__(self, vector_store: Any, table: Optional[str] = None) -> None:
        if table is not None and not _NAME.match(table):
            raise ValueError(f"Not a plain table name: {table!r}")
        self.vector_store = vector_store
        self._table = table

    # -- naming ----------------------------------------------------------------------

    @property
    def table_name(self) -> str:
        """The qualified ``Schema.Table`` of the postings table."""
        if self._table is None:
            self._table = f"{self.vector_store.shape().schema}.{DEFAULT_TABLE_NAME}"
        if "." not in self._table:
            return f"{self.vector_store.shape().schema}.{self._table}"
        return self._table

    def _split(self) -> Tuple[str, str]:
        schema, _, table = self.table_name.partition(".")
        return schema, table

    def _connection(self) -> Any:
        return self.vector_store._get_connection()

    def _exists(self, cursor: Any) -> bool:
        schema, table = self._split()
        cursor.execute(
            "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES "
            "WHERE UPPER(TABLE_SCHEMA) = UPPER(?) AND UPPER(TABLE_NAME) = UPPER(?)",
            [schema, table],
        )
        row = cursor.fetchone()
        return bool(row and int(row[0]) > 0)

    def _missing(self) -> ConceptIndexMissingError:
        return ConceptIndexMissingError(
            f"{self.table_name} does not exist, so concept postings cannot be written "
            f"or searched. Create it once with ConceptIndex(vector_store).ensure_table()."
        )

    # -- schema ----------------------------------------------------------------------

    def ensure_table(self) -> None:
        """Create the postings table and its two indexes if they are absent.

        Idempotent: an existing table is left alone. Creation is not wrapped in a larger
        transaction, so a failure part-way can leave the table without its indexes; run
        `ensure_table()` again after fixing the cause only if the table is absent.

        Raises:
            ConceptIndexError: the database refused the DDL.
        """
        table = self.table_name
        _, bare = self._split()
        connection = self._connection()
        cursor = connection.cursor()
        try:
            if self._exists(cursor):
                return
            cursor.execute(
                f"CREATE TABLE {table} ("
                "section_id VARCHAR(255), "
                "parent_id VARCHAR(255), "
                "concept_id VARCHAR(100), "
                "polarity VARCHAR(10), "
                "mentions INTEGER, "
                "section_label VARCHAR(100))"
            )
            cursor.execute(
                f"CREATE INDEX {bare}ByConcept ON {table} (concept_id, polarity)"
            )
            cursor.execute(f"CREATE INDEX {bare}ByParent ON {table} (parent_id)")
            connection.commit()
        except Exception as exc:
            connection.rollback()
            raise ConceptIndexError(
                f"Could not create the concept postings table {table}: {exc}"
            ) from exc
        finally:
            cursor.close()

    # -- writing ---------------------------------------------------------------------

    def index_parent(
        self,
        parent_id: str,
        sections: Iterable[Any],
        tagger: ConceptTagger,
        detector: Optional[Callable[[str, Tuple[int, int]], str]] = None,
    ) -> int:
        """Replace the postings of ``parent_id`` with those of ``sections``.

        ``sections`` are the `Section` objects `SectionStore.store` returns. Each
        section's postings come from `postings_for_section`. The old rows of the parent
        are deleted and the new ones inserted in one transaction: a failure rolls back
        and leaves the old postings in place.

        Returns:
            The number of postings written.

        Raises:
            ValueError: the parent id is empty or holds ``#``.
            ConceptIndexMissingError: the postings table does not exist.
            ConceptIndexError: the database refused a write (rolled back).
        """
        if not parent_id or "#" in parent_id:
            raise ValueError(
                f"A parent id must be non-empty and hold no '#', got {parent_id!r}."
            )
        rows: List[List[Any]] = []
        for section in sections:
            section_id = f"{parent_id}#{section.ordinal}"
            for posting in postings_for_section(section.text, tagger, detector):
                rows.append(
                    [
                        section_id,
                        parent_id,
                        posting["concept_id"],
                        posting["polarity"],
                        posting["mentions"],
                        section.label,
                    ]
                )

        table = self.table_name
        connection = self._connection()
        cursor = connection.cursor()
        try:
            if not self._exists(cursor):
                raise self._missing()
            # The session connection autocommits, and a rollback cannot undo a statement
            # that already committed, so the replace opens its own transaction.
            cursor.execute("START TRANSACTION")
            cursor.execute(f"DELETE FROM {table} WHERE parent_id = ?", [parent_id])
            for row in rows:
                cursor.execute(
                    f"INSERT INTO {table} (section_id, parent_id, concept_id, polarity, "
                    "mentions, section_label) VALUES (?, ?, ?, ?, ?, ?)",
                    row,
                )
            connection.commit()
        except ConceptIndexMissingError:
            connection.rollback()
            raise
        except Exception as exc:
            connection.rollback()
            raise ConceptIndexError(
                f"Could not write the concept postings of {parent_id!r}: {exc}"
            ) from exc
        finally:
            cursor.close()
        return len(rows)

    # -- reading ---------------------------------------------------------------------

    def search(
        self,
        concept_ids: Sequence[str],
        polarity: Optional[str] = None,
        section: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Document]:
        """Parents that have a posting for every concept in ``concept_ids``.

        Args:
            concept_ids: at least one concept id. Unknown ids match nothing.
            polarity: when given, each posting used must have this polarity.
            section: when given, each posting used must be in a section with this label.
            top_k: most parents returned.

        Returns:
            One Document per parent, best first: score is the sum of ``mentions`` over the
            postings used (ties by parent id ascending). ``id`` is the parent id,
            ``page_content`` the text of the section with the most mentions, and
            ``metadata`` holds ``text_score``, ``polarity`` (affirmed if any used posting
            is affirmed, else negated, else unknown) and ``matched_sections`` (best first:
            ``id``, ``label``, ``concept_id``, ``polarity``, ``mentions``).

        Raises:
            ValueError: no concept ids, a polarity outside affirmed/negated/unknown, or
                ``top_k`` below 1.
            ConceptIndexMissingError: the postings table does not exist.
            ConceptIndexError: the database refused the read, or a section's text is no
                longer in the chunk table.
        """
        if isinstance(concept_ids, str):
            raise ValueError("concept_ids must be a list of concept ids, not a string.")
        wanted = list(dict.fromkeys(concept_ids))
        if not wanted:
            raise ValueError("A concept search needs at least one concept id.")
        if polarity is not None and polarity not in POLARITIES:
            raise ValueError(
                f"polarity must be one of {POLARITIES} or None, got {polarity!r}."
            )
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}.")

        table = self.table_name
        connection = self._connection()
        cursor = connection.cursor()
        try:
            if not self._exists(cursor):
                raise self._missing()
            placeholders = ", ".join("?" for _ in wanted)
            sql = (
                "SELECT section_id, parent_id, concept_id, polarity, mentions, "
                f"section_label FROM {table} WHERE concept_id IN ({placeholders})"
            )
            params: List[Any] = list(wanted)
            if polarity is not None:
                sql += " AND polarity = ?"
                params.append(polarity)
            if section is not None:
                sql += " AND section_label = ?"
                params.append(section)
            try:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
            except Exception as exc:
                raise ConceptIndexError(
                    f"Concept search over {table} failed: {exc}"
                ) from exc
        finally:
            cursor.close()

        ranked = _rank_parents(rows, wanted)[:top_k]
        texts = self._section_texts([best for _, _, best, _ in ranked])
        documents = []
        for score, parent_id, best_section_id, postings in ranked:
            documents.append(
                Document(
                    id=parent_id,
                    page_content=texts[best_section_id],
                    metadata={
                        "text_score": score,
                        "polarity": section_polarity(p["polarity"] for p in postings),
                        "matched_sections": [
                            {
                                "id": p["section_id"],
                                "label": p["label"],
                                "concept_id": p["concept_id"],
                                "polarity": p["polarity"],
                                "mentions": p["mentions"],
                            }
                            for p in postings
                        ],
                    },
                )
            )
        return documents

    def _section_texts(self, section_ids: List[str]) -> Dict[str, str]:
        if not section_ids:
            return {}
        found = {
            str(doc.id): doc.page_content
            for doc in self.vector_store.fetch_documents_by_ids(section_ids)
        }
        missing = [sid for sid in section_ids if sid not in found]
        if missing:
            raise ConceptIndexError(
                f"The postings name sections that are not in the chunk table: {missing}. "
                f"Store the parent again with SectionStore and index it again."
            )
        return found


def _rank_parents(rows: Iterable[Any], wanted: List[str]) -> List[Tuple]:
    """Group postings by parent, keep the parents holding every wanted concept, and
    order them: score (sum of mentions) descending, then parent id ascending.

    Returns (score, parent_id, best_section_id, postings) with postings best first.
    """
    by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for section_id, parent_id, concept_id, pol, mentions, label in rows:
        by_parent.setdefault(str(parent_id), []).append(
            {
                "section_id": str(section_id),
                "concept_id": str(concept_id),
                "polarity": str(pol),
                "mentions": int(mentions or 0),
                "label": str(label or ""),
                "ordinal": int(str(section_id).rsplit("#", 1)[-1]),
            }
        )

    ranked = []
    for parent_id, postings in by_parent.items():
        if not set(wanted) <= {p["concept_id"] for p in postings}:
            continue
        mentions_by_section: Dict[str, int] = {}
        ordinal_of: Dict[str, int] = {}
        for p in postings:
            mentions_by_section[p["section_id"]] = (
                mentions_by_section.get(p["section_id"], 0) + p["mentions"]
            )
            ordinal_of[p["section_id"]] = p["ordinal"]
        best_section = min(
            mentions_by_section,
            key=lambda sid: (-mentions_by_section[sid], ordinal_of[sid]),
        )
        ordered = sorted(
            postings,
            key=lambda p: (-p["mentions"], p["ordinal"], p["concept_id"]),
        )
        score = sum(p["mentions"] for p in postings)
        ranked.append((score, parent_id, best_section, ordered))
    ranked.sort(key=lambda r: (-r[0], r[1]))
    return ranked
