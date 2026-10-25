"""`where` through HybridGraphRAGPipeline, against a real IRIS (feature 092, FR-021).

A small attached table (``BYOTHyb.Chunk``: two patients, two chunks each) and a hand-made
``BYOTHyb.Entities`` map (entity -> source chunk) live in a schema of their own, created
before each test and dropped after. An attached table has no entity graph, so the graph
legs are stood in for: a stub engine names every patient's entities for every question,
and the pipeline has to keep only the patient asked for. The query embedding is the test
embedder's, the answer model a stub.

Without an engine the five methods run on the store's vector search and the entity map.
With one, hybrid, rrf, text and vector run the engine's legs and read each entity's chunk
under the predicate. ``iris_vector_graph``'s own tables are never read here.
"""

from __future__ import annotations

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.exceptions import FilterError, RequiredFilterMissing
from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline
from iris_vector_rag.pipelines.hybrid_graphrag_retrieval import HybridRetrievalMethods
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

pytestmark = pytest.mark.integration

SCHEMA = "BYOTHyb"
CHUNKS = f"{SCHEMA}.Chunk"
ENTITIES = f"{SCHEMA}.Entities"
DIM = 384
ROWS = (
    ("P01-1", "P01", "m1", "apixaban stopped before the biopsy"),
    ("P01-2", "P01", "m1", "allergies denied"),
    ("P02-1", "P02", "m1", "warfarin monitoring weekly"),
    ("P02-2", "P02", "m1", "anticoagulation resumed after the procedure"),
)
#: One entity per chunk: "e-<chunk id>".
ENTITY_IDS = [f"e-{row[0]}" for row in ROWS]
P02_IDS = ["P02-1", "P02-2"]
ATTACHED = {
    "table_name": CHUNKS,
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
METHODS = ["hybrid", "rrf", "text", "vector", "kg"]
GRAPH_METHODS = ["hybrid", "rrf", "text", "vector"]


class Engine:
    """The graph engine, naming every patient's entities for every question."""

    def __init__(self, entities):
        self.entities = list(entities)

    def kg_RRF_FUSE(self, **kwargs):
        return [(e, 0.5, 0.4, 0.3) for e in self.entities]

    def kg_TXT(self, **kwargs):
        return [(e, 0.9) for e in self.entities]

    def kg_KNN_VEC(self, **kwargs):
        return [(e, 0.8) for e in self.entities]

    def multi_modal_search(self, **kwargs):
        return [{"entity_id": e, "fusion_score": 0.7} for e in self.entities]


def stub_llm(prompt: str) -> str:
    return "stub answer"


def vector_literal(vector) -> str:
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def table_state(connection):
    cursor = connection.cursor()
    try:
        cursor.execute(
            f"SELECT ChunkId, PatientId, Text FROM {CHUNKS} ORDER BY ChunkId"
        )
        return [tuple(r) for r in cursor.fetchall()]
    finally:
        cursor.close()


def drop(connection):
    cursor = connection.cursor()
    try:
        for table in (CHUNKS, ENTITIES):
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
        connection.commit()
        try:
            cursor.execute(f"DROP SCHEMA {SCHEMA}")
            connection.commit()
        except Exception:  # noqa: BLE001 - an empty schema left behind is harmless
            pass
    finally:
        cursor.close()


@pytest.fixture
def corpus(iris_connection, use_test_embedder):
    """The tables, built with the test embedder's vectors; the chunks unchanged by the test."""
    from iris_vector_rag.embeddings.manager import EmbeddingManager

    embedder = EmbeddingManager(ConfigurationManager())
    drop(iris_connection)
    cursor = iris_connection.cursor()
    try:
        cursor.execute(
            f"CREATE TABLE {CHUNKS} (ChunkId VARCHAR(50) NOT NULL, PatientId VARCHAR(16), "
            f"EmbeddingModel VARCHAR(50), Text VARCHAR(2000), Embedding VECTOR(DOUBLE, {DIM}))"
        )
        cursor.execute(
            f"CREATE TABLE {ENTITIES} (entity_id VARCHAR(100) NOT NULL, "
            "entity_name VARCHAR(200), source_doc_id VARCHAR(50))"
        )
        for chunk_id, patient, model, text in ROWS:
            cursor.execute(
                f"INSERT INTO {CHUNKS} (ChunkId, PatientId, EmbeddingModel, Text, Embedding) "
                f"VALUES (?,?,?,?,TO_VECTOR(?, DOUBLE, {DIM}))",
                [
                    chunk_id,
                    patient,
                    model,
                    text,
                    vector_literal(embedder.embed_text(text)),
                ],
            )
            cursor.execute(
                f"INSERT INTO {ENTITIES} (entity_id, entity_name, source_doc_id) "
                "VALUES (?,?,?)",
                [f"e-{chunk_id}", text.split()[0], chunk_id],
            )
        iris_connection.commit()
    finally:
        cursor.close()
    before = table_state(iris_connection)
    yield iris_connection
    after = table_state(iris_connection)
    drop(iris_connection)
    assert after == before, "the pipeline changed an attached table"


def pipeline_for(settings, engine=None):
    """A pipeline over the attached table; with ``engine``, its graph legs run."""
    config = ConfigurationManager()
    connections = ConnectionManager(config)
    store = IRISVectorStore(connections, config, **settings)
    pipeline = HybridGraphRAGPipeline(
        connection_manager=connections,
        config_manager=config,
        llm_func=stub_llm,
        vector_store=store,
    )
    pipeline.iris_engine = engine
    pipeline.fusion_engine = engine
    pipeline.text_engine = None
    pipeline.vector_optimizer = None
    pipeline.retrieval_methods = (
        HybridRetrievalMethods(engine, engine, None, None, pipeline.embedding_manager)
        if engine is not None
        else None
    )
    # No LLM names the query's entities: the seeds are every patient's.
    pipeline._find_seed_entities = lambda query: list(ENTITY_IDS)
    pipeline._expand_neighborhood = lambda start, depth: set(start)
    return pipeline


def ids(response):
    return sorted(d.id for d in response["retrieved_documents"] if d.id)


def texts(response):
    return sorted(d.page_content for d in response["retrieved_documents"])


P02_TEXTS = sorted(row[3] for row in ROWS if row[1] == "P02")


@pytest.mark.parametrize("method", GRAPH_METHODS)
def test_a_graph_leg_that_names_every_patient_returns_only_the_one_asked_for(
    corpus, method
):
    pipeline = pipeline_for(SCOPED, Engine(ENTITY_IDS))

    response = pipeline.query("anticoagulation", method=method, top_k=10, where=WHERE)

    assert response["metadata"]["retrieval_method"] != "vector_fallback"
    assert texts(response) == P02_TEXTS
    assert response["answer"] == "stub answer"


@pytest.mark.parametrize("method", GRAPH_METHODS)
def test_a_graph_leg_that_names_only_the_other_patient_falls_back_in_scope(
    corpus, method
):
    pipeline = pipeline_for(SCOPED, Engine(["e-P01-1", "e-P01-2"]))

    response = pipeline.query("anticoagulation", method=method, top_k=10, where=WHERE)

    assert response["metadata"]["retrieval_method"] == "vector_fallback"
    assert ids(response) == P02_IDS


@pytest.mark.parametrize("method", METHODS)
def test_every_method_without_the_engine_returns_only_the_patient_asked_for(
    corpus, method
):
    response = pipeline_for(SCOPED).query(
        "anticoagulation", method=method, top_k=10, where=WHERE
    )

    assert response["retrieved_documents"]
    assert set(texts(response)) <= set(P02_TEXTS)
    assert not any(d.id.startswith("P01") for d in response["retrieved_documents"])


def test_the_entity_graph_leg_reads_chunks_through_the_store_under_the_predicate(
    corpus,
):
    response = pipeline_for(SCOPED).query(
        "anticoagulation", method="kg", top_k=10, where=WHERE
    )

    assert response["metadata"]["retrieval_method"] == "knowledge_graph"
    assert ids(response) == P02_IDS


@pytest.mark.parametrize("engine", [None, Engine(ENTITY_IDS)], ids=["plain", "engine"])
@pytest.mark.parametrize("method", METHODS)
def test_a_required_filter_cannot_be_left_out(corpus, method, engine):
    pipeline = pipeline_for(SCOPED, engine)

    with pytest.raises(RequiredFilterMissing, match="PatientId"):
        pipeline.query("anticoagulation", method=method, top_k=10)
    with pytest.raises(RequiredFilterMissing, match="EmbeddingModel"):
        pipeline.query(
            "anticoagulation", method=method, top_k=10, where={"PatientId": "P02"}
        )


@pytest.mark.parametrize("method", METHODS)
def test_a_filter_on_an_undeclared_column_is_refused(corpus, method):
    pipeline = pipeline_for(SCOPED, Engine(ENTITY_IDS))

    with pytest.raises(FilterError, match="Text"):
        pipeline.query(
            "anticoagulation",
            method=method,
            top_k=10,
            where={**WHERE, "Text": "anything"},
        )


def test_a_table_with_no_required_filter_still_queries_without_where(corpus):
    response = pipeline_for(ATTACHED, Engine(ENTITY_IDS)).query(
        "anticoagulation", method="rrf", top_k=10
    )

    assert len(response["retrieved_documents"]) == len(ROWS)
