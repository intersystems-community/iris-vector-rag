"""CRAG: table and column names come from the table shape (PL-11), and the embedder
is called one way (PL-12): a list of texts in, a list of vectors out."""

from __future__ import annotations

import re

import pytest

from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.crag import CRAGPipeline
from iris_vector_rag.storage.table_shape import (
    ColumnFact,
    ColumnRole,
    ShapeKind,
    TableShape,
    VectorSpec,
)


def strict_embedder(texts):
    """Rejects the bare-string call the list convention does not allow."""
    assert isinstance(texts, list), f"embedder called with {type(texts).__name__}"
    return [[0.25, 0.5, 0.75, 1.0] for _ in texts]


class RecordingCursor:
    def __init__(self, log, rows):
        self.log, self.rows = log, rows

    def execute(self, sql, params=None):
        self.log.append(sql)

    def fetchall(self):
        return self.rows

    def close(self):
        pass


class RecordingConnection:
    def __init__(self, rows=()):
        self.sql = []
        self.rows = list(rows)

    def cursor(self):
        return RecordingCursor(self.sql, self.rows)


class ConnectionManager:
    def __init__(self, connection):
        self.connection = connection

    def get_connection(self):
        return self.connection


class FakeStore:
    vector_dimension = 4

    def __init__(self):
        self.calls = []

    def similarity_search(self, query_embedding, top_k, filter=None):
        self.calls.append((query_embedding, top_k))
        return [
            (Document(id=f"d{i}", page_content=f"doc {i}", metadata={}), 0.9 - i / 10)
            for i in range(top_k)
        ]


def make_pipeline(connection=None, store=None):
    pipeline = CRAGPipeline.__new__(CRAGPipeline)
    pipeline.embedding_func = strict_embedder
    pipeline.vector_store = store or FakeStore()
    pipeline.connection_manager = ConnectionManager(connection or RecordingConnection())
    return pipeline


def legacy_chunk_shape():
    """A chunk table from an older schema generation: id / content / embedding."""
    columns = {
        "id": ColumnFact("id", "varchar", 255, False, 1, is_key=True),
        "content": ColumnFact("content", "longvarchar", 2147483647, True, 2),
        "embedding": ColumnFact(
            "embedding", "varchar", 132863, True, 3, vector=VectorSpec("FLOAT", 4)
        ),
    }
    return TableShape(
        schema="RAG",
        table="DocumentChunks",
        class_name="RAG.DocumentChunks",
        columns=columns,
        roles={
            ColumnRole.IDENTIFIER: "id",
            ColumnRole.TEXT: "content",
            ColumnRole.EMBEDDING: "embedding",
        },
        kind=ShapeKind.LEGACY_ID,
    )


def test_knowledge_base_expansion_uses_the_store_not_raw_sql():
    connection = RecordingConnection()
    pipeline = make_pipeline(connection)
    docs = pipeline._knowledge_base_expansion("what is x", top_k=2)
    assert connection.sql == []  # no SQL of its own, so no hardcoded names
    assert [d.id for d in docs] == ["d0", "d1"]
    assert all(
        d.metadata["retrieval_method"] == "knowledge_base_expansion" for d in docs
    )
    assert all("similarity_score" in d.metadata for d in docs)


def test_knowledge_base_expansion_does_not_log_sample_rows_with_a_debug_select():
    connection = RecordingConnection()
    make_pipeline(connection)._knowledge_base_expansion("q", top_k=1)
    assert not any("text_content" in sql for sql in connection.sql)


def test_enhance_retrieval_reads_columns_from_the_chunk_shape(monkeypatch):
    connection = RecordingConnection(rows=[("c1", "chunk one", 0.8)])
    pipeline = make_pipeline(connection)
    monkeypatch.setattr(pipeline, "_chunk_shape", lambda conn: legacy_chunk_shape())
    docs = pipeline._enhance_retrieval("q", [], top_k=3)
    (sql,) = connection.sql
    assert re.search(r"FROM\s+RAG\.DocumentChunks", sql)
    assert "content" in sql and "embedding" in sql and "id" in sql
    assert "chunk_text" not in sql and "chunk_embedding" not in sql
    assert [d.page_content for d in docs] == ["chunk one"]


def test_enhance_retrieval_refuses_a_query_of_the_wrong_width(monkeypatch):
    pipeline = make_pipeline(RecordingConnection())
    pipeline.embedding_func = lambda texts: [[0.1] * 8 for _ in texts]
    monkeypatch.setattr(pipeline, "_chunk_shape", lambda conn: legacy_chunk_shape())
    with pytest.raises(ValueError, match="dimension mismatch"):
        pipeline._enhance_retrieval("q", [], top_k=3)


def test_both_corrective_paths_call_the_embedder_with_a_list(monkeypatch):
    pipeline = make_pipeline()
    monkeypatch.setattr(pipeline, "_chunk_shape", lambda conn: legacy_chunk_shape())
    pipeline._enhance_retrieval("q", [], top_k=1)  # strict_embedder asserts the list
    pipeline._knowledge_base_expansion("q", top_k=1)


def test_a_malformed_embedder_result_is_an_error_not_a_guess():
    pipeline = make_pipeline()
    pipeline.embedding_func = lambda texts: [0.1, 0.2]  # flat list: wrong convention
    with pytest.raises(TypeError, match="embedding_func"):
        pipeline._knowledge_base_expansion("q", top_k=1)


def test_enhance_retrieval_binds_the_query_vector(monkeypatch):
    """The chunk search sends the vector as a parameter (TO_VECTOR(?, FLOAT, 4))."""
    executed = []

    class ParamCursor(RecordingCursor):
        def execute(self, sql, params=None):
            executed.append((sql, params))

    class ParamConnection(RecordingConnection):
        def cursor(self):
            return ParamCursor(self.sql, self.rows)

    pipeline = make_pipeline(ParamConnection(rows=[("c1", "chunk one", 0.8)]))
    monkeypatch.setattr(pipeline, "_chunk_shape", lambda conn: legacy_chunk_shape())
    pipeline._enhance_retrieval("q", [], top_k=3)

    ((sql, params),) = executed
    assert "TO_VECTOR(?, FLOAT, 4)" in sql
    assert "'[" not in sql
    assert len(params) == 1 and params[0].startswith("[0.25")
