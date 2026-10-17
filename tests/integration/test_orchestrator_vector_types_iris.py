"""Setup writes embeddings into DOUBLE vector columns as well as FLOAT ones, on real IRIS.

The setup orchestrator used to write `TO_VECTOR(?)`. Untyped, that makes a FLOAT vector,
which a DOUBLE column refuses (SQLCODE -259 on IRIS 2026.1 and 2026.3), so filling in
missing embeddings failed on any table declared `VECTOR(DOUBLE, n)`. The UPDATE now names
the column's type from the table shape, and the chunk insert hands the column a bare `?`,
which IRIS converts to the column's own type.

Works in a schema of its own (`ORCHTV`); nothing here touches `RAG`. No embedding model:
the embedder returns fixed vectors, so a result is a property of the SQL.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.validation.orchestrator import SetupOrchestrator
from tests.fixtures.shapes.ddl import VECTOR_DIMENSION, create_statement

pytestmark = pytest.mark.integration

SCHEMA = "ORCHTV"
DOCS = [("orch-1", "warfarin dosing and monitoring"), ("orch-2", "a note on asthma")]


def _run(connection, sql, params=None):
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params or [])
        rows = cursor.fetchall() if cursor.description else None
        connection.commit()
        return rows
    finally:
        cursor.close()


class FixedEmbedder:
    """The n-th text gets a vector whose n-th component is 1, the rest 0."""

    def embed_texts(self, texts):
        return [
            [1.0 if i == n % VECTOR_DIMENSION else 0.0 for i in range(VECTOR_DIMENSION)]
            for n, _ in enumerate(texts)
        ]


@pytest.fixture
def clean(iris_connection):
    def drop():
        for table in ("Docs", "DocumentChunks"):
            _run(iris_connection, f"DROP TABLE IF EXISTS {SCHEMA}.{table}")

    drop()
    yield iris_connection
    drop()


def _orchestrator(monkeypatch) -> SetupOrchestrator:
    monkeypatch.setenv("RAG_STORAGE__IRIS__TABLE_NAME", f"{SCHEMA}.Docs")
    config = ConfigurationManager()
    with (
        patch("iris_vector_rag.validation.orchestrator.EmbeddingManager"),
        patch("iris_vector_rag.validation.orchestrator.PreConditionValidator"),
    ):
        orchestrator = SetupOrchestrator(ConnectionManager(config), config)
    orchestrator.embedding_manager = FixedEmbedder()
    return orchestrator


def _load_documents(connection, shape: str) -> None:
    _run(connection, create_statement(shape, SCHEMA, "Docs"))
    text_column = "text_content" if shape == "double_vector" else "content"
    for doc_id, text in DOCS:
        _run(
            connection,
            f"INSERT INTO {SCHEMA}.Docs (doc_id, {text_column}) VALUES (?, ?)",
            [doc_id, text],
        )


@pytest.mark.parametrize("shape", ["fixture", "double_vector"])
def test_missing_embeddings_are_filled_on_float_and_double_columns(
    shape, clean, monkeypatch
):
    _load_documents(clean, shape)

    _orchestrator(monkeypatch)._generate_missing_document_embeddings()

    ((missing,),) = _run(
        clean, f"SELECT COUNT(*) FROM {SCHEMA}.Docs WHERE embedding IS NULL"
    )
    assert missing == 0
    element = "DOUBLE" if shape == "double_vector" else "FLOAT"
    probe = (
        "[" + ",".join("1" if i == 1 else "0" for i in range(VECTOR_DIMENSION)) + "]"
    )
    rows = _run(
        clean,
        f"SELECT TOP 1 doc_id FROM {SCHEMA}.Docs ORDER BY VECTOR_DOT_PRODUCT("
        f"embedding, TO_VECTOR(?, {element}, {VECTOR_DIMENSION})) DESC",
        [probe],
    )
    assert rows[0][0] == "orch-2", "the second document holds the second unit vector"


@pytest.mark.parametrize("element", ["FLOAT", "DOUBLE"])
def test_chunks_are_written_into_a_float_or_a_double_chunk_table(
    element, clean, monkeypatch
):
    _load_documents(clean, "fixture")
    _run(
        clean,
        f"CREATE TABLE {SCHEMA}.DocumentChunks (doc_id VARCHAR(255) PRIMARY KEY, "
        "document_id VARCHAR(255), chunk_text LONGVARCHAR, "
        f"chunk_embedding VECTOR({element}, {VECTOR_DIMENSION}), "
        "metadata VARCHAR(2000))",
    )

    count = _orchestrator(monkeypatch)._generate_document_chunks()

    assert count == len(DOCS)
    ((embedded,),) = _run(
        clean,
        f"SELECT COUNT(*) FROM {SCHEMA}.DocumentChunks "
        "WHERE VECTOR_DOT_PRODUCT(chunk_embedding, chunk_embedding) = 1",
    )
    assert embedded == len(DOCS)
