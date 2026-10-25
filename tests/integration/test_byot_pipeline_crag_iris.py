"""CRAG over an attached table, scoped by ``where`` (feature 092, FR-021), on real IRIS.

The attached table is `BYOTCrag.Chunk`, built here with two patients whose rows sit
side by side in vector space, so a search that ignored ``where`` would return both. For
a store on an attached table CRAG's chunk search reads that table (it is the chunk
table), under the same predicate as the vector searches. Run through each of CRAG's
three outcomes, every retrieved document must be the asked-for patient's. The schema
is created and dropped around each test.
"""

from __future__ import annotations

import random

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.exceptions import RequiredFilterMissing
from iris_vector_rag.pipelines.crag import CRAGPipeline
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("use_test_embedder")]

SCHEMA, TABLE = "BYOTCrag", "Chunk"
FULL = f"{SCHEMA}.{TABLE}"
DIM = 384
QUERY = "apixaban was held before the biopsy"
PATIENTS = ("PA", "PB")
ROWS = 6

SETTINGS = {
    "table_name": FULL,
    "id_col": "ChunkId",
    "text_col": "Text",
    "embedding_col": "Embedding",
    "filter_columns": ["PatientId"],
    "required_filters": ["PatientId"],
}


def _literal(vector):
    return "[" + ",".join(repr(float(x)) for x in vector) + "]"


def _run(connection, sql, params=None):
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params) if params else cursor.execute(sql)
    finally:
        cursor.close()


def _embedder():
    from iris_vector_rag.common.utils import get_embedding_func

    return get_embedding_func()


@pytest.fixture
def attached(iris_connection):
    """The attached table: rows of both patients all within a hair of the query."""
    connection = iris_connection
    _run(connection, f"DROP TABLE IF EXISTS {FULL}")
    _run(
        connection,
        f"CREATE TABLE {FULL} (ChunkId VARCHAR(100) NOT NULL, PatientId VARCHAR(16), "
        f"Text VARCHAR(2000), Embedding VECTOR(DOUBLE, {DIM}))",
    )
    base = _embedder()([QUERY])[0]
    rng = random.Random(7)
    owners = {}
    for patient in PATIENTS:
        for n in range(ROWS):
            chunk_id = f"{patient}#{n}"
            owners[chunk_id] = patient
            vector = [x + rng.uniform(-1e-3, 1e-3) for x in base]
            _run(
                connection,
                f"INSERT INTO {FULL} (ChunkId, PatientId, Text, Embedding) "
                f"VALUES (?, ?, ?, TO_VECTOR(?, DOUBLE, {DIM}))",
                [chunk_id, patient, f"{patient} note {n}: {QUERY}", _literal(vector)],
            )
    connection.commit()
    try:
        yield owners
    finally:
        _run(connection, f"DROP TABLE IF EXISTS {FULL}")
        connection.commit()


def crag(settings=SETTINGS) -> CRAGPipeline:
    config = ConfigurationManager()
    connections = ConnectionManager(config)
    store = IRISVectorStore(connections, config, **settings)
    return CRAGPipeline(
        connection_manager=connections,
        config_manager=config,
        vector_store=store,
        embedding_func=_embedder(),
        llm_func=lambda prompt: "ok",
    )


def owner_of(doc, owners):
    return owners[doc.id.removesuffix("_chunk")]


@pytest.mark.parametrize("status", ["confident", "ambiguous", "disoriented"])
def test_every_retrieved_document_is_the_asked_for_patients(attached, status):
    pipeline = crag()
    pipeline.evaluator.evaluate = lambda query, docs: status

    result = pipeline.query(QUERY, top_k=4, where={"PatientId": "PA"})

    assert result["error"] is None
    assert result["metadata"]["retrieval_status"] == status
    docs = result["retrieved_documents"]
    assert docs, "the scoped search found nothing"
    assert {owner_of(d, attached) for d in docs} == {"PA"}
    assert result["answer"] == "ok"


def test_the_ambiguous_path_runs_its_chunk_search_on_the_attached_table(attached):
    pipeline = crag()
    pipeline.evaluator.evaluate = lambda query, docs: "ambiguous"
    ran = []
    real = pipeline._chunk_search_sql
    pipeline._chunk_search_sql = lambda *a, **k: ran.append(real(*a, **k)) or ran[-1]

    result = pipeline.query(QUERY, top_k=4, where={"PatientId": "PB"})

    ((sql, params),) = ran
    assert f"FROM {FULL}" in sql and '"PatientId" = ?' in sql
    assert params[1:] == ["PB"]
    # The chunk hits repeat the first search's text, so the de-duplication keeps one.
    assert {owner_of(d, attached) for d in result["retrieved_documents"]} == {"PB"}


def test_the_scope_is_what_keeps_the_other_patient_out(attached):
    """Without the scope, the same table does return both: the test above is not vacuous."""
    unscoped = crag({k: v for k, v in SETTINGS.items() if k != "required_filters"})
    result = unscoped.query(QUERY, top_k=10, generate_answer=False)
    assert {owner_of(d, attached) for d in result["retrieved_documents"]} == set(
        PATIENTS
    )


def test_a_query_without_the_required_filter_raises(attached):
    pipeline = crag()
    with pytest.raises(RequiredFilterMissing, match="PatientId"):
        pipeline.query(QUERY, generate_answer=False)
    with pytest.raises(RequiredFilterMissing):
        pipeline.query(QUERY, generate_answer=False, where={})


OWN_ID, OWN_TEXT = (
    "OWN#1",
    "the library's own corpus: apixaban was held before the biopsy",
)


@pytest.fixture
def own_chunk(iris_connection, attached):
    """A row in ivr's own RAG.DocumentChunks that matches the query exactly.

    The table exists once a CRAG pipeline is built (its constructor creates it through
    the library's SchemaManager). The row is removed afterwards.
    """
    pipeline = crag()  # ensures RAG.DocumentChunks
    vector = _literal(_embedder()([QUERY])[0])
    _run(iris_connection, "DELETE FROM RAG.DocumentChunks WHERE doc_id = ?", [OWN_ID])
    _run(
        iris_connection,
        "INSERT INTO RAG.DocumentChunks (doc_id, document_id, chunk_text, "
        f"chunk_embedding) VALUES (?, ?, ?, TO_VECTOR(?, FLOAT, {DIM}))",
        [OWN_ID, OWN_ID, OWN_TEXT, vector],
    )
    iris_connection.commit()
    try:
        yield pipeline
    finally:
        _run(
            iris_connection, "DELETE FROM RAG.DocumentChunks WHERE doc_id = ?", [OWN_ID]
        )
        iris_connection.commit()


def test_the_librarys_own_corpus_is_never_mixed_into_an_attached_query(
    own_chunk, attached
):
    # Positive control: a store on the library's own tables does find the row.
    config = ConfigurationManager()
    plain = CRAGPipeline(
        connection_manager=ConnectionManager(config),
        config_manager=config,
        vector_store=IRISVectorStore(ConnectionManager(config), config),
        embedding_func=_embedder(),
        llm_func=lambda prompt: "ok",
    )
    own = plain._enhance_retrieval(QUERY, [], top_k=5)
    assert any(d.page_content == OWN_TEXT for d in own), "the control row is not found"

    # The attached store's ambiguous path must not see it, on any leg.
    pipeline = own_chunk
    pipeline.evaluator.evaluate = lambda query, docs: "ambiguous"
    result = pipeline.query(QUERY, top_k=10, where={"PatientId": "PA"})
    assert result["error"] is None
    texts = [d.page_content for d in result["retrieved_documents"]]
    assert texts and OWN_TEXT not in texts
    assert OWN_ID not in {
        d.id.removesuffix("_chunk") for d in result["retrieved_documents"]
    }
