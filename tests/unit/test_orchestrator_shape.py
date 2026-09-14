"""The setup orchestrator asks the shape which columns to fill (feature 082, T053).

`_generate_missing_document_embeddings` used to read `INFORMATION_SCHEMA.COLUMNS`
itself and then pick a text column from `("content", "text_content", "text",
"document_text")` and an identifier from `doc_id` before `id`. Two problems, both
shipped: the precedence order was its own, so on a table carrying `content` and
`text_content` it embedded the opposite column from the one the store reads, and the
`UPDATE` it wrote afterwards said `WHERE doc_id = ?` regardless of which column it
had just selected (FR-003, FR-009, FR-012, FR-013).

Recorded catalog rows, no database and no embedding model: what is under test is
the SQL, so the cursor and the embedder are both recordings.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from iris_vector_rag.storage.catalog import RecordedCatalog
from iris_vector_rag.storage.shape_resolver import TableShapeResolver
from iris_vector_rag.validation.orchestrator import SetupOrchestrator
from tests.fixtures.shapes.ddl import SHAPE_SCHEMA, SHAPE_TABLES

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "shapes"


class RecordingCursor:
    """Answers `fetchone()`/`fetchall()` from scripts, and keeps every statement."""

    def __init__(self, ones=(), alls=()):
        self._ones = list(ones)
        self._alls = list(alls)
        self.statements: list[tuple[str, object]] = []

    def execute(self, sql, params=None):
        self.statements.append((" ".join(str(sql).split()), params))

    def fetchone(self):
        return self._ones.pop(0) if self._ones else None

    def fetchall(self):
        return self._alls.pop(0) if self._alls else []

    def close(self):
        pass

    @property
    def sql(self) -> list[str]:
        return [sql for sql, _ in self.statements]

    def matching(self, fragment: str) -> list[tuple[str, object]]:
        return [(sql, params) for sql, params in self.statements if fragment in sql]


class RecordingConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


class RecordedEmbedder:
    """One fixed vector per text, so the UPDATE parameters are predictable."""

    def embed_texts(self, texts):
        return [[0.5, 0.25] for _ in texts]


def orchestrator_for(shape: str, cursor: RecordingCursor) -> SetupOrchestrator:
    """An orchestrator whose document table is one of the recorded shapes."""
    config = Mock()
    config.get.return_value = {
        "table_name": f"{SHAPE_SCHEMA}.{SHAPE_TABLES[shape]}",
    }
    connections = Mock()
    connections.get_connection.return_value = RecordingConnection(cursor)

    with (
        patch("iris_vector_rag.validation.orchestrator.EmbeddingManager"),
        patch("iris_vector_rag.validation.orchestrator.PreConditionValidator"),
    ):
        orchestrator = SetupOrchestrator(
            connections,
            config,
            shapes=TableShapeResolver(RecordedCatalog.from_fixture_dir(FIXTURE_DIR)),
        )
    orchestrator.embedding_manager = RecordedEmbedder()
    return orchestrator


def test_no_catalog_read_of_its_own():
    """The columns are asked for once, of the shape -- not of INFORMATION_SCHEMA."""
    cursor = RecordingCursor(alls=[[]])
    orchestrator_for("fixture", cursor)._generate_missing_document_embeddings()

    assert not any("INFORMATION_SCHEMA" in sql for sql in cursor.sql), cursor.sql


def test_the_selected_columns_are_the_shapes_columns():
    """`content` on this table, because that is where the store put the text."""
    cursor = RecordingCursor(alls=[[]])
    orchestrator_for("fixture", cursor)._generate_missing_document_embeddings()

    assert cursor.sql[0] == (
        "SELECT doc_id, content FROM ShapeTest.ShapeFixture WHERE embedding IS NULL"
    ), cursor.sql


def test_a_mixed_table_selects_the_canonical_text_column():
    """`content` and `text_content` both present: the canonical one wins (FR-003).

    The old candidate list tried `content` first, so a mixed table had its legacy
    column embedded while the store read the canonical one.
    """
    cursor = RecordingCursor(alls=[[]])
    orchestrator_for("mixed", cursor)._generate_missing_document_embeddings()

    assert cursor.sql[0] == (
        "SELECT doc_id, text_content FROM ShapeTest.ShapeMixed WHERE embedding IS NULL"
    ), cursor.sql


def test_the_update_writes_back_through_the_same_identifier():
    """Whatever column was selected is the column the UPDATE matches on."""
    cursor = RecordingCursor(
        ones=[(0,)],
        alls=[[("DOC-1", "warfarin monitoring")]],
    )
    orchestrator_for("legacy_serial", cursor)._generate_missing_document_embeddings()

    updates = cursor.matching("UPDATE")
    assert len(updates) == 1, cursor.sql
    sql, params = updates[0]
    assert sql == (
        "UPDATE ShapeTest.ShapeLegacySerial SET embedding = TO_VECTOR(?) WHERE id = ?"
    ), sql
    assert params == ["[0.5,0.25]", "DOC-1"]


def test_a_table_without_an_embedding_column_uses_the_vector_table():
    """No embedding column, so the join to VectorEmbeddings names the identifier."""
    cursor = RecordingCursor(alls=[[]])
    orchestrator_for("legacy_id", cursor)._generate_missing_document_embeddings()

    assert cursor.sql[0] == (
        "SELECT s.id, s.content FROM ShapeTest.ShapeLegacyId s "
        "LEFT JOIN ShapeTest.VectorEmbeddings v ON s.id = v.chunk_id "
        "WHERE v.embedding IS NULL OR v.id IS NULL"
    ), cursor.sql


def test_the_missing_count_is_taken_on_the_shapes_embedding_column():
    """`_ensure_document_embeddings` counts the same column it would fill."""
    cursor = RecordingCursor(ones=[(3,), (0,)])
    orchestrator_for("fixture", cursor)._ensure_document_embeddings()

    assert cursor.sql == [
        "SELECT COUNT(*) FROM ShapeTest.ShapeFixture",
        "SELECT COUNT(*) FROM ShapeTest.ShapeFixture WHERE embedding IS NULL",
    ], cursor.sql


def test_a_missing_document_table_says_so():
    """No table, no shape: the error names the table instead of a column later on."""
    from iris_vector_rag.storage.table_shape import TableNotFound

    cursor = RecordingCursor()
    orchestrator = orchestrator_for("fixture", cursor)
    orchestrator.documents_table = f"{SHAPE_SCHEMA}.NoSuchTable"

    with pytest.raises(TableNotFound, match="NoSuchTable"):
        orchestrator._generate_missing_document_embeddings()

    assert cursor.statements == []
