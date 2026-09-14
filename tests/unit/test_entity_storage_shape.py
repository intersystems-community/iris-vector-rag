"""The entity adapter resolves a source document through the shape (feature 082, T052).

`EntityStorageAdapter._resolve_source_document` used to ask
`RAG.SourceDocuments` for `id` and OR in `doc_id`, which is a guess with two ways
to be wrong: on a canonical table `id` does not exist, and on a legacy table the
`OR` made every lookup a full scan of two columns. The shape names the one
identifier column, and says whether the metadata column can be matched with
`LIKE` at all (FR-003, FR-009, FR-012).

Recorded catalog rows, no database: the shapes here are what IRIS reported for the
DDL in `tests/fixtures/shapes/ddl.py`.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from iris_vector_rag.services.storage import EntityStorageAdapter
from iris_vector_rag.storage.catalog import RecordedCatalog
from iris_vector_rag.storage.shape_resolver import TableShapeResolver
from tests.fixtures.shapes.ddl import SHAPE_SCHEMA, SHAPE_TABLES

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "shapes"


class RecordingCursor:
    """Answers `fetchone()` from a script of rows, and keeps every statement."""

    def __init__(self, *rows):
        self._rows = list(rows)
        self.statements: list[tuple[str, object]] = []

    def execute(self, sql, params=None):
        self.statements.append((" ".join(str(sql).split()), params))

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None

    def close(self):
        pass


def adapter_for(shape: str) -> EntityStorageAdapter:
    """An adapter whose document table is one of the recorded shapes."""
    config = {
        "entity_extraction": {
            "storage": {
                "documents_table": f"{SHAPE_SCHEMA}.{SHAPE_TABLES[shape]}",
            }
        }
    }
    return EntityStorageAdapter(
        Mock(),
        config,
        shapes=TableShapeResolver(RecordedCatalog.from_fixture_dir(FIXTURE_DIR)),
    )


def test_the_identifier_column_comes_from_the_shape():
    """One column, named by the shape -- no `id = ? OR doc_id = ?`."""
    adapter = adapter_for("canonical")
    cursor = RecordingCursor(("DOC-1",))

    assert adapter._resolve_source_document(cursor, "DOC-1") == "DOC-1"

    sql, params = cursor.statements[0]
    assert sql == ("SELECT doc_id FROM ShapeTest.ShapeCanonical WHERE doc_id = ?"), sql
    assert params == ["DOC-1"]


def test_a_legacy_table_is_queried_on_its_own_identifier():
    """The same code, a table whose identifier is `id`: no second spelling tried."""
    adapter = adapter_for("legacy_id")
    cursor = RecordingCursor(("DOC-9",))

    assert adapter._resolve_source_document(cursor, "DOC-9") == "DOC-9"

    sql, _ = cursor.statements[0]
    assert sql == "SELECT id FROM ShapeTest.ShapeLegacyId WHERE id = ?", sql
    assert len(cursor.statements) == 1, cursor.statements


def test_the_parent_document_fallback_uses_top_and_like():
    """A miss falls back to the metadata JSON, with IRIS's own row limit."""
    adapter = adapter_for("canonical")
    cursor = RecordingCursor(None, ("DOC-PARENT",))

    assert adapter._resolve_source_document(cursor, "CHUNK-1") == "DOC-PARENT"

    sql, params = cursor.statements[1]
    assert sql == (
        "SELECT TOP 1 doc_id FROM ShapeTest.ShapeCanonical WHERE metadata LIKE ?"
    ), sql
    assert params == ['%"parent_doc_id":"CHUNK-1"%']


def test_a_stream_metadata_column_is_matched_with_substring():
    """`LIKE` on a LONGVARCHAR column matches nothing silently, so it is not used."""
    adapter = adapter_for("legacy_id")
    cursor = RecordingCursor(None, ("DOC-PARENT",))

    assert adapter._resolve_source_document(cursor, "CHUNK-1") == "DOC-PARENT"

    sql, _ = cursor.statements[1]
    assert "SUBSTRING(metadata,1," in sql, sql
    assert "WHERE metadata LIKE" not in sql, sql


def test_an_unknown_document_table_returns_the_identifier_unchanged():
    """Resolution is best-effort: an absent table is not a failed entity write."""
    adapter = adapter_for("canonical")
    adapter.documents_table = f"{SHAPE_SCHEMA}.NoSuchTable"
    cursor = RecordingCursor()

    assert adapter._resolve_source_document(cursor, "DOC-1") == "DOC-1"
    assert cursor.statements == []


def test_a_table_without_a_metadata_column_stops_after_the_identifier():
    """No metadata column, no JSON fallback to build.

    All six recorded shapes have one, so the snapshot is stripped of it here: the
    guard exists because a table is not required to carry metadata at all.
    """
    recorded = RecordedCatalog.from_fixture_dir(FIXTURE_DIR)
    snapshot = recorded.snapshot(SHAPE_SCHEMA, SHAPE_TABLES["canonical"])
    stripped = replace(
        snapshot,
        columns=[
            column
            for column in snapshot.columns
            if str(column["COLUMN_NAME"]).lower() != "metadata"
        ],
    )
    adapter = adapter_for("canonical")
    adapter.shapes = TableShapeResolver(
        RecordedCatalog({(snapshot.schema, snapshot.table): stripped})
    )
    cursor = RecordingCursor(None)

    assert adapter._resolve_source_document(cursor, "DOC-1") == "DOC-1"
    assert len(cursor.statements) == 1, cursor.statements


@pytest.mark.parametrize("shape", sorted(SHAPE_TABLES))
def test_every_shape_is_queried_with_one_statement_on_a_hit(shape):
    """No shape sends a second identifier query to cover for the first."""
    adapter = adapter_for(shape)
    cursor = RecordingCursor(("DOC-1",))

    adapter._resolve_source_document(cursor, "DOC-1")

    assert len(cursor.statements) == 1, cursor.statements
