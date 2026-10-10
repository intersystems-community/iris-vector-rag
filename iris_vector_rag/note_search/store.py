"""Store a long document as sections: one chunk-shaped row per section.

A section row is `doc_id = <parent>#<ordinal>`, `document_id = <parent>`,
`chunk_text` = the section text, and `metadata` JSON holding `section` (the label),
`start`, `end`, `ordinal` and `parent_id`. The embedding stays NULL. The table's shape is
read from the database, never assumed (feature 082). The label sits in `metadata` for
now (spec 088, decision 1); `SectionStore._metadata` is the one place that decides so.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, List, Mapping, Optional, Sequence

from iris_vector_rag.note_search.sections import Heading, Section, split_sections
from iris_vector_rag.storage.table_shape import ColumnRole

logger = logging.getLogger(__name__)

#: The chunk-table column that points at the parent document.
PARENT_COLUMN = "document_id"

#: Width of the metadata column in the canonical chunk shape.
METADATA_LIMIT = 2000

#: The opt-in column that holds a section's label, bitmap-indexed (spec 089). A table
#: has it only after `ensure_section_column`; without it the label lives in metadata.
SECTION_COLUMN = "section_label"

#: Width of `SECTION_COLUMN`. A longer label cannot be stored in it.
SECTION_COLUMN_WIDTH = 100

#: The bitmap index on `SECTION_COLUMN`.
SECTION_INDEX = "SectionLabelIdx"

__all__ = [
    "SECTION_COLUMN",
    "SectionStore",
    "SectionStoreError",
    "ensure_section_column",
]


class SectionStoreError(RuntimeError):
    """Sections could not be stored; the rows written before it were kept or restored."""


class SectionStore:
    """Writes the sections of a parent document into a chunk-shaped table.

    Args:
        vector_store: An `IRISVectorStore` over the chunk-shaped table.
    """

    def __init__(self, vector_store: Any) -> None:
        self.vector_store = vector_store

    def store(
        self,
        parent_id: str,
        text: str,
        detector: Callable[[str], Sequence[Heading]],
        label_map: Optional[Mapping[str, str]] = None,
    ) -> List[Section]:
        """Split ``text`` and replace the parent's sections with the result.

        Storing a parent again replaces its rows, in one transaction: a failure leaves
        the earlier rows in place.

        Raises:
            SectionStoreError: the parent id is empty or holds ``#``, the table has no
                parent column or metadata column, a row's metadata is too wide, or the
                database refused a write.
        """
        if not parent_id or "#" in parent_id:
            raise SectionStoreError(
                f"A parent id must be non-empty and hold no '#' (it joins the section "
                f"ordinal), got {parent_id!r}."
            )
        sections = split_sections(text, detector, label_map)

        shape = self.vector_store.shape()
        identifier = shape.column_for(ColumnRole.IDENTIFIER).name
        content = shape.column_for(ColumnRole.TEXT).name
        metadata_column = shape.optional_column_for(ColumnRole.METADATA)
        parent = shape.columns.get(PARENT_COLUMN)
        if metadata_column is None or parent is None:
            raise SectionStoreError(
                f"{shape.qualified_name} is not chunk-shaped: it needs a metadata "
                f"column and a {PARENT_COLUMN!r} column to hold sections."
            )

        label_column = SECTION_COLUMN in shape.columns
        rows = []
        for section in sections:
            metadata = json.dumps(self._metadata(parent_id, section))
            if len(metadata) > METADATA_LIMIT:
                raise SectionStoreError(
                    f"Metadata of section {section.ordinal} of {parent_id!r} is "
                    f"{len(metadata)} characters; the column holds {METADATA_LIMIT}."
                )
            row = [f"{parent_id}#{section.ordinal}", parent_id, section.text, metadata]
            if label_column:
                _check_label_width(section.label, parent_id)
                row.append(section.label)
            rows.append(row)

        table = self.vector_store.table_name
        label_sql = f", {SECTION_COLUMN}" if label_column else ""
        marks = ", ".join("?" * len(rows[0])) if rows else ""
        connection = self.vector_store._get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f"DELETE FROM {table} WHERE {parent.name} = ?", [parent_id])
            for row in rows:
                cursor.execute(
                    f"INSERT INTO {table} ({identifier}, {parent.name}, {content}, "
                    f"{metadata_column.name}{label_sql}) VALUES ({marks})",
                    row,
                )
            connection.commit()
        except Exception as exc:
            connection.rollback()
            raise SectionStoreError(
                f"Could not store the sections of {parent_id!r}: {exc}"
            ) from exc
        finally:
            cursor.close()
        return sections

    @staticmethod
    def _metadata(parent_id: str, section: Section) -> dict:
        """What a section row records about itself. The label lives here, as `section`."""
        return {
            "section": section.label,
            "start": section.start,
            "end": section.end,
            "ordinal": section.ordinal,
            "parent_id": parent_id,
        }


def _check_label_width(label: str, where: str) -> None:
    if len(label) > SECTION_COLUMN_WIDTH:
        raise SectionStoreError(
            f"A section label of {where!r} is {len(label)} characters; {SECTION_COLUMN} "
            f"holds {SECTION_COLUMN_WIDTH}."
        )


def _has_index(cursor: Any, schema: str, table: str) -> bool:
    cursor.execute(
        "SELECT INDEX_NAME FROM INFORMATION_SCHEMA.INDEXES "
        "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?",
        [schema, table],
    )
    wanted = SECTION_INDEX.upper()
    return any(str(row[0]).upper() == wanted for row in cursor.fetchall())


def ensure_section_column(vector_store: Any) -> bool:
    """Add the opt-in bitmap-indexed `section_label` column to a section table.

    Idempotent. If the table lacks the column it is added (VARCHAR(100)), the bitmap
    index `SectionLabelIdx` is created if absent, and every row whose metadata carries a
    `section` label gets that label in the column (rows already holding it are left
    alone). Rows whose metadata does not parse are skipped and counted in the log. A
    label over 100 characters raises `SectionStoreError` before anything is altered.

    Returns True if the table changed, False if it already had the column, the index and
    every label.

    Raises:
        SectionStoreError: a label is too wide, the table has no metadata column, or the
            database refused a statement (the DDL cannot be rolled back).
    """
    shape = vector_store.shape()
    identifier = shape.column_for(ColumnRole.IDENTIFIER).name
    metadata_column = shape.optional_column_for(ColumnRole.METADATA)
    if metadata_column is None:
        raise SectionStoreError(
            f"{shape.qualified_name} has no metadata column; sections are labelled "
            "there, so there is nothing to backfill."
        )
    table = vector_store.table_name
    connection = vector_store._get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(f"SELECT {identifier}, {metadata_column.name} FROM {table}")
        labels: list = []
        skipped = 0
        for row_id, metadata in cursor.fetchall():
            if metadata is None:
                continue
            try:
                parsed = json.loads(metadata)
            except (TypeError, ValueError):
                skipped += 1
                continue
            label = parsed.get("section") if isinstance(parsed, dict) else None
            if isinstance(label, str):
                _check_label_width(label, table)
                labels.append((row_id, label))

        has_column = SECTION_COLUMN in shape.columns
        current: dict = {}
        if has_column:
            cursor.execute(f"SELECT {identifier}, {SECTION_COLUMN} FROM {table}")
            current = {row[0]: row[1] for row in cursor.fetchall()}

        altered = False
        if not has_column:
            cursor.execute(
                f"ALTER TABLE {table} ADD COLUMN {SECTION_COLUMN} "
                f"VARCHAR({SECTION_COLUMN_WIDTH})"
            )
            altered = True
            # The cached shape predates the column; the next read must see it.
            vector_store.refresh_shape()
        if not _has_index(cursor, shape.schema, shape.table):
            cursor.execute(
                f"CREATE BITMAP INDEX {SECTION_INDEX} ON {table} ({SECTION_COLUMN})"
            )
            altered = True

        pending = [(rid, label) for rid, label in labels if current.get(rid) != label]
        for rid, label in pending:
            cursor.execute(
                f"UPDATE {table} SET {SECTION_COLUMN} = ? WHERE {identifier} = ?",
                [label, rid],
            )
        connection.commit()
    except SectionStoreError:
        raise
    except Exception as exc:
        connection.rollback()
        raise SectionStoreError(
            f"Could not add {SECTION_COLUMN} to {table}: {exc}"
        ) from exc
    finally:
        cursor.close()

    if skipped:
        logger.warning(
            "ensure_section_column skipped %d row(s) of %s: metadata is not JSON",
            skipped,
            table,
        )
    return altered or bool(pending)
