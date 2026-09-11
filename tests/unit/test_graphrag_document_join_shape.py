"""The GraphRAG entity-to-document join is named by the shape (feature 082, T055).

The join was hand-written text:

    FROM RAG.SourceDocuments sd JOIN RAG.Entities e ON sd.doc_id = e.source_doc_id

which is wrong on any table not keyed by `doc_id`: it matched no rows, the entity
path returned nothing, and the pipeline fell back to plain vector search without
saying why. The identifier, the text column and the metadata column now come from
the shape the vector store reads with, so the retriever and the store cannot
disagree about which table column holds what (FR-012).

`graphrag_merged.py` carries a second copy of the same join. It is migrated to the
same form, but it has no test here because the module cannot be imported at all --
it imports `iris_vector_rag.visualization`, which is not part of the package.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

from iris_vector_rag.pipelines.graphrag import GraphRAGPipeline
from iris_vector_rag.storage.catalog import RecordedCatalog
from iris_vector_rag.storage.shape_resolver import TableShapeResolver
from tests.fixtures.shapes.ddl import SHAPE_SCHEMA, SHAPE_TABLES

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "shapes"


class RecordingCursor:
    def __init__(self, rows=()):
        self._rows = list(rows)
        self.statements: list[tuple[str, object]] = []

    def execute(self, sql, params=None):
        self.statements.append((" ".join(str(sql).split()), params))

    def fetchall(self):
        return list(self._rows)

    def close(self):
        pass

    @property
    def sql(self) -> str:
        return self.statements[0][0]

    @property
    def params(self):
        return self.statements[0][1]


class RecordedStore:
    """The vector store, reduced to the two things the join asks it for.

    `without` drops a column from the recorded snapshot: all six recorded shapes carry
    a metadata column, so a table that has none can only be described by stripping one.
    """

    def __init__(self, shape_name: str, without: str = "") -> None:
        table = SHAPE_TABLES[shape_name]
        self.table_name = f"{SHAPE_SCHEMA}.{table}"
        snapshot = RecordedCatalog.from_fixture_dir(FIXTURE_DIR).snapshot(
            SHAPE_SCHEMA, table
        )
        if without:
            snapshot = replace(
                snapshot,
                columns=[
                    column
                    for column in snapshot.columns
                    if str(column["COLUMN_NAME"]).lower() != without.lower()
                ],
            )
        self._shapes = TableShapeResolver(
            RecordedCatalog({(SHAPE_SCHEMA, table): snapshot})
        )

    def shape(self, cursor=None):
        return self._shapes.resolve(SHAPE_SCHEMA, self.table_name.partition(".")[2])


def pipeline_for(shape_name: str, cursor: RecordingCursor, without: str = ""):
    """A pipeline with a connection and a store, and nothing else running."""
    pipeline = GraphRAGPipeline.__new__(GraphRAGPipeline)
    connection = Mock()
    connection.cursor.return_value = cursor
    pipeline.connection_manager = Mock()
    pipeline.connection_manager.get_connection.return_value = connection
    pipeline.config_manager = Mock()
    pipeline.vector_store = RecordedStore(shape_name, without=without)
    return pipeline


def rows_for(identifier: str) -> list[tuple]:
    return [(identifier, "warfarin monitoring", json.dumps({"source": "test"}))]


def test_the_join_names_the_shapes_identifier():
    """Canonical table: the same `doc_id` the store writes, and its own schema."""
    cursor = RecordingCursor(rows=rows_for("DOC-1"))

    pipeline_for("canonical", cursor)._get_documents_from_entities({"E1"}, top_k=5)

    assert cursor.sql == (
        "SELECT DISTINCT sd.doc_id, sd.text_content, sd.metadata "
        f"FROM {SHAPE_SCHEMA}.{SHAPE_TABLES['canonical']} sd "
        f"JOIN {SHAPE_SCHEMA}.Entities e ON sd.doc_id = e.source_doc_id "
        "WHERE e.entity_id IN (?) "
        "ORDER BY sd.doc_id"
    ), cursor.sql
    assert cursor.params == ["E1"]


def test_a_table_keyed_by_id_joins_on_id():
    """The case the hand-written `sd.doc_id` silently returned nothing for."""
    cursor = RecordingCursor(rows=rows_for("DOC-1"))

    docs = pipeline_for("legacy_id", cursor)._get_documents_from_entities(
        {"E1"}, top_k=5
    )

    assert "SELECT DISTINCT sd.id, sd.content" in cursor.sql, cursor.sql
    assert "ON sd.id = e.source_doc_id" in cursor.sql, cursor.sql
    assert "ORDER BY sd.id" in cursor.sql, cursor.sql
    assert [doc.id for doc in docs] == ["DOC-1"]


def test_a_mixed_table_reads_the_canonical_text_column():
    """`content` and `text_content` both present: the store's column wins (FR-003)."""
    cursor = RecordingCursor(rows=rows_for("DOC-1"))

    pipeline_for("mixed", cursor)._get_documents_from_entities({"E1"}, top_k=5)

    assert "SELECT DISTINCT sd.doc_id, sd.text_content" in cursor.sql, cursor.sql


def test_the_entities_table_lives_in_the_documents_schema():
    """A corpus in its own schema keeps its own entities; `RAG.` was fixed text."""
    cursor = RecordingCursor(rows=rows_for("DOC-1"))

    pipeline_for("legacy_serial", cursor)._get_documents_from_entities({"E1"}, top_k=5)

    assert f"JOIN {SHAPE_SCHEMA}.Entities e" in cursor.sql, cursor.sql
    assert "RAG.Entities" not in cursor.sql, cursor.sql


def test_a_table_without_a_metadata_column_still_joins():
    """No metadata column: the select drops it and the documents carry none."""
    cursor = RecordingCursor(rows=[("DOC-1", "warfarin monitoring")])

    docs = pipeline_for(
        "canonical", cursor, without="metadata"
    )._get_documents_from_entities({"E1"}, top_k=5)

    assert "sd.metadata" not in cursor.sql, cursor.sql
    assert docs[0].metadata == {"retrieval_method": "knowledge_graph"}


def test_the_top_k_limit_still_holds():
    """Unchanged behaviour, asserted because the loop was rewritten around it."""
    cursor = RecordingCursor(rows=rows_for("DOC-1") + rows_for("DOC-2"))

    docs = pipeline_for("canonical", cursor)._get_documents_from_entities(
        {"E1"}, top_k=1
    )

    assert [doc.id for doc in docs] == ["DOC-1"]
