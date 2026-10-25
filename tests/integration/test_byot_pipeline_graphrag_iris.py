"""`where` through GraphRAGPipeline, against a real IRIS (feature 092, FR-021).

A small attached table (``BYOTGraph.Chunk``: two patients, two chunks each) in a schema of
its own, created before each test and dropped after. The query embedding is the test
embedder's, the answer model a stub. The knowledge graph's own tables (``RAG.Entities``)
are not touched: an executor stands in for them and names every patient's chunks, so a
result that holds only one patient's chunks was scoped by the store's SQL, not by the graph.
"""

from __future__ import annotations

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.exceptions import FilterError, RequiredFilterMissing
from iris_vector_rag.pipelines.graphrag import GraphRAGPipeline
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

pytestmark = pytest.mark.integration

SCHEMA, TABLE = "BYOTGraph", "Chunk"
QUALIFIED = f"{SCHEMA}.{TABLE}"
DIM = 384
ROWS = (
    ("P01-1", "P01", "m1", "apixaban stopped before the biopsy"),
    ("P01-2", "P01", "m1", "allergies denied"),
    ("P02-1", "P02", "m1", "warfarin monitoring weekly"),
    ("P02-2", "P02", "m1", "anticoagulation resumed after the procedure"),
)
P02_IDS = ["P02-1", "P02-2"]
ATTACHED = {
    "table_name": QUALIFIED,
    "id_col": "ChunkId",
    "text_col": "Text",
    "embedding_col": "Embedding",
}
SCOPED = {
    **ATTACHED,
    "filter_columns": ["PatientId", "EmbeddingModel"],
    "required_filters": ["PatientId", "EmbeddingModel"],
}
WHERE = {"PatientId": "P02", "EmbeddingModel": "m1"}


class EntityRows:
    """Stands in for ``RAG.Entities``: every entity points at every chunk."""

    def __init__(self):
        self.statements = []

    def execute(self, sql, params=None):
        self.statements.append(sql)
        return [{"source_doc_id": row[0]} for row in ROWS]


def stub_llm(prompt: str) -> str:
    return "stub answer"


def vector_literal(vector) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def table_state(connection):
    cursor = connection.cursor()
    try:
        cursor.execute(
            f"SELECT ChunkId, PatientId, Text FROM {QUALIFIED} ORDER BY ChunkId"
        )
        return [tuple(r) for r in cursor.fetchall()]
    finally:
        cursor.close()


def drop(connection):
    cursor = connection.cursor()
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {QUALIFIED}")
        connection.commit()
    finally:
        cursor.close()


@pytest.fixture
def chunks(iris_connection, use_test_embedder):
    """The table, built with the test embedder's vectors; unchanged by the test."""
    from iris_vector_rag.embeddings.manager import EmbeddingManager

    embedder = EmbeddingManager(ConfigurationManager())
    drop(iris_connection)
    cursor = iris_connection.cursor()
    try:
        cursor.execute(
            f"CREATE TABLE {QUALIFIED} (ChunkId VARCHAR(50) NOT NULL, PatientId VARCHAR(16), "
            f"EmbeddingModel VARCHAR(50), Text VARCHAR(2000), Embedding VECTOR(DOUBLE, {DIM}))"
        )
        for chunk_id, patient, model, text in ROWS:
            cursor.execute(
                f"INSERT INTO {QUALIFIED} (ChunkId, PatientId, EmbeddingModel, Text, Embedding) "
                f"VALUES (?,?,?,?,TO_VECTOR(?, DOUBLE, {DIM}))",
                [
                    chunk_id,
                    patient,
                    model,
                    text,
                    vector_literal(embedder.embed_text(text)),
                ],
            )
        iris_connection.commit()
    finally:
        cursor.close()
    before = table_state(iris_connection)
    yield iris_connection
    after = table_state(iris_connection)
    drop(iris_connection)
    assert after == before, "the pipeline changed an attached table"


def pipeline_for(settings, seeds=()):
    config = ConfigurationManager()
    connections = ConnectionManager(config)
    store = IRISVectorStore(connections, config, **settings)
    pipeline = GraphRAGPipeline(
        connection_manager=connections,
        config_manager=config,
        llm_func=stub_llm,
        vector_store=store,
        executor=EntityRows(),
    )
    pipeline._find_seed_entities = lambda query: list(seeds)
    pipeline._expand_neighborhood = lambda start, depth: set(start)
    return pipeline


def ids(response):
    return sorted(d.id for d in response["retrieved_documents"])


def test_the_vector_fallback_returns_only_the_patient_asked_for(chunks):
    response = pipeline_for(SCOPED).query("anticoagulation", top_k=10, where=WHERE)
    assert response["metadata"]["retrieval_method"] == "vector_fallback"
    assert ids(response) == P02_IDS
    assert response["answer"] == "stub answer"


def test_the_graph_leg_fetches_only_documents_in_scope(chunks):
    pipeline = pipeline_for(SCOPED, seeds=("E1",))
    response = pipeline.query("anticoagulation", top_k=10, where=WHERE)
    # the graph named all four chunks; the store's SQL kept the patient's two
    assert len(pipeline._executor.statements) == 1
    assert "FROM RAG.Entities" in pipeline._executor.statements[0]
    assert response["metadata"]["retrieval_method"] == "knowledge_graph"
    assert ids(response) == P02_IDS


def test_the_graph_leg_would_have_returned_everyone_without_the_scope(chunks):
    """The guard for the test above: unscoped, the same graph returns every chunk."""
    pipeline = pipeline_for(ATTACHED, seeds=("E1",))
    response = pipeline.query("anticoagulation", top_k=10)
    assert ids(response) == sorted(row[0] for row in ROWS)


def test_a_missing_required_filter_is_an_error_not_a_wider_read(chunks):
    pipeline = pipeline_for(SCOPED, seeds=("E1",))
    with pytest.raises(RequiredFilterMissing, match="EmbeddingModel"):
        pipeline.query("anticoagulation", where={"PatientId": "P02"})
    with pytest.raises(RequiredFilterMissing):
        pipeline.query("anticoagulation")


def test_where_on_a_table_that_cannot_scope_it_is_refused(chunks):
    """No filter column is declared, so a `where` cannot be applied: it is refused."""
    pipeline = pipeline_for(ATTACHED, seeds=("E1",))
    with pytest.raises(FilterError, match="PatientId"):
        pipeline.query("anticoagulation", where={"PatientId": "P02"})
    assert pipeline._executor.statements == []  # refused before the graph was read
