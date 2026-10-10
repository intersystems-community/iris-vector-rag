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

__all__ = ["SectionStore", "SectionStoreError"]


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

        rows = []
        for section in sections:
            metadata = json.dumps(self._metadata(parent_id, section))
            if len(metadata) > METADATA_LIMIT:
                raise SectionStoreError(
                    f"Metadata of section {section.ordinal} of {parent_id!r} is "
                    f"{len(metadata)} characters; the column holds {METADATA_LIMIT}."
                )
            rows.append(
                [f"{parent_id}#{section.ordinal}", parent_id, section.text, metadata]
            )

        table = self.vector_store.table_name
        connection = self.vector_store._get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(f"DELETE FROM {table} WHERE {parent.name} = ?", [parent_id])
            for row in rows:
                cursor.execute(
                    f"INSERT INTO {table} ({identifier}, {parent.name}, {content}, "
                    f"{metadata_column.name}) VALUES (?, ?, ?, ?)",
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
