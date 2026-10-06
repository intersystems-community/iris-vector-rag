"""Pipelines-fix, RAG and LLM-factory stories against a real IRIS.

Each test works in schemas of its own (`PLFIXA` canonical, `PLFIXB` legacy) so nothing
touches `RAG` tables another test owns. Runs on the ephemeral container `iris_target`
starts. Everything is real: the LLM is OpenAI's (OPENAI_API_KEY from the environment),
the embedder is the real sentence-transformer. The one fault injected is a reranker that
raises, because that failure is what the rerank test is about.
"""

from __future__ import annotations

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.crag import CRAGPipeline
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from tests.support.real_services import EMBEDDING_WIDTH, real_embedder, real_llm

pytestmark = pytest.mark.integration

WIDTH = EMBEDDING_WIDTH
A, B = "PLFIXA", "PLFIXB"
DOCS = {
    "doc1": "alpha insulin regulates glucose",
    "doc2": "beta neural networks learn",
    "doc3": "gamma transformers use attention",
}


_EMBEDDER = real_embedder()


def embed(texts):
    """The real embedder, on a list of texts."""
    assert isinstance(texts, list)
    return _EMBEDDER(texts)


def _vec(text):
    return "[" + ",".join(f"{x:.6f}" for x in embed([text])[0]) + "]"


def _drop(connection):
    cursor = connection.cursor()
    try:
        for schema in (A, B):
            for table in ("DocumentChunks", "SourceDocuments", "SchemaMetadata"):
                cursor.execute(f"DROP TABLE IF EXISTS {schema}.{table}")
        connection.commit()
    finally:
        cursor.close()


def _exec(connection, sql):
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        connection.commit()
    finally:
        cursor.close()


@pytest.fixture
def stack(iris_connection):
    config = ConfigurationManager()
    connections = ConnectionManager(config)
    _drop(iris_connection)
    yield config, connections
    _drop(iris_connection)


def _canonical(stack, connection):
    config, connections = stack
    manager = SchemaManager(connections, config, schema_prefix=A)
    assert manager.ensure_table_schema("SourceDocuments")
    assert manager.ensure_table_schema("DocumentChunks")
    for doc_id, text in DOCS.items():
        _exec(
            connection,
            f"INSERT INTO {A}.SourceDocuments (doc_id, text_content, embedding) "
            f"VALUES ('{doc_id}', '{text}', TO_VECTOR('{_vec(text)}', FLOAT, {WIDTH}))",
        )
        _exec(
            connection,
            f"INSERT INTO {A}.DocumentChunks (doc_id, document_id, chunk_text, chunk_embedding) "
            f"VALUES ('{doc_id}_c', '{doc_id}', '{text}', "
            f"TO_VECTOR('{_vec(text)}', FLOAT, {WIDTH}))",
        )
    return manager


def _legacy(stack, connection):
    """Older generation: `id` / `content` / `embedding` in both tables."""
    config, connections = stack
    for table in ("SourceDocuments", "DocumentChunks"):
        _exec(
            connection,
            f"CREATE TABLE {B}.{table} (id VARCHAR(255) PRIMARY KEY, "
            f"content VARCHAR(4000), embedding VECTOR(FLOAT, {WIDTH}))",
        )
        for doc_id, text in DOCS.items():
            _exec(
                connection,
                f"INSERT INTO {B}.{table} (id, content, embedding) VALUES "
                f"('{doc_id}', '{text}', TO_VECTOR('{_vec(text)}', FLOAT, {WIDTH}))",
            )
    return SchemaManager(connections, config, schema_prefix=B)


def _pipeline(stack, manager, prefix):
    config, connections = stack
    store = IRISVectorStore(connections, config, schema_manager=manager)
    store.table_name = f"{prefix}.SourceDocuments"
    return CRAGPipeline(
        connection_manager=connections,
        config_manager=config,
        vector_store=store,
        embedding_func=embed,
        llm_func=real_llm(),
    )


@pytest.mark.parametrize("prefix", [A, B])
def test_crag_reads_canonical_and_legacy_tables(stack, iris_connection, prefix):
    manager = (_canonical if prefix == A else _legacy)(stack, iris_connection)
    pipeline = _pipeline(stack, manager, prefix)
    query = DOCS["doc1"]

    expanded = pipeline._knowledge_base_expansion(query, top_k=2)
    assert expanded[0].page_content == DOCS["doc1"]
    assert all(
        d.metadata["retrieval_method"] == "knowledge_base_expansion" for d in expanded
    )

    enhanced = pipeline._enhance_retrieval(query, [], top_k=2)
    chunks = [
        d
        for d in enhanced
        if d.metadata.get("retrieval_method") == "chunk_based_enhancement"
    ]
    assert chunks, "the chunk table was not read"
    assert chunks[0].page_content == DOCS["doc1"]

    result = pipeline.query(query, top_k=2)
    assert result["error"] is None, result["error"]
    assert result["metadata"]["retrieval_method"] != "vector_fallback"
    assert result["sources"] is not None
    assert result["contexts"]


def test_rerank_failure_is_loud_against_a_real_store(stack, iris_connection):
    from iris_vector_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline

    config, connections = stack
    manager = SchemaManager(connections, config, schema_prefix=A)
    store = IRISVectorStore(connections, config, schema_manager=manager)
    store.table_name = f"{A}.SourceDocuments"
    store.add_documents(
        [Document(id=k, page_content=v, metadata={}) for k, v in DOCS.items()],
        auto_chunk=False,
    )

    # FAULT INJECTION: the reranker is down; store, embedder and LLM stay real.
    def broken(query, docs):
        raise RuntimeError("reranker is down")

    def make(**kw):
        return BasicRAGRerankingPipeline(
            connections,
            config,
            reranker_func=broken,
            llm_func=real_llm(),
            vector_store=store,
        )

    with pytest.raises(RuntimeError, match="reranker is down"):
        make().query("insulin", top_k=2)

    result = make().query("insulin", top_k=2, allow_rerank_fallback=True)
    assert result["metadata"]["reranked"] is False
    assert "reranker is down" in result["metadata"]["rerank_error"]
    assert isinstance(result["answer"], str) and result["answer"].strip()


def test_create_pipeline_with_embedding_func_and_a_key_reaches_a_real_pipeline(stack):
    """OS-10 end to end: the OpenAI client must not see the pipeline's kwargs.

    The real `ChatOpenAI` rejects an unknown keyword, so a leaked `embedding_func` would
    fail here; the real model then has to answer.
    """
    import iris_vector_rag.common.utils as utils

    utils._llm_instances.clear()

    from iris_vector_rag import create_pipeline

    pipeline = create_pipeline(
        "basic", embedding_func=embed, validate_requirements=False
    )
    answer = pipeline.llm_func("Reply with the single word: ready")
    assert isinstance(answer, str) and answer.strip()
    utils._llm_instances.clear()
