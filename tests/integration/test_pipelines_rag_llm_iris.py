"""Pipelines-fix, RAG and LLM-factory stories against a real IRIS.

Each test works in schemas of its own (`PLFIXA` canonical, `PLFIXB` legacy) so nothing
touches `RAG` tables another test owns. Runs on the ephemeral container `iris_target`
starts. The only fakes are at the boundary: the OpenAI client class, and the embedding
function (a pure function of the text).
"""

from __future__ import annotations

import sys
import types

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.crag import CRAGPipeline
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

pytestmark = pytest.mark.integration

WIDTH = 12
A, B = "PLFIXA", "PLFIXB"
DOCS = {
    "doc1": "alpha insulin regulates glucose",
    "doc2": "beta neural networks learn",
    "doc3": "gamma transformers use attention",
}


def embed(texts):
    """Deterministic 12-wide vectors; the same text gives the same vector."""
    assert isinstance(texts, list)
    out = []
    for text in texts:
        padded = (text * WIDTH)[:WIDTH]
        out.append([float(ord(c) % 7 + 1) for c in padded])
    return out


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
def stack(iris_connection, monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDINGS__PROVIDER", "stub")
    monkeypatch.setenv("RAG_EMBEDDINGS__DIMENSION", str(WIDTH))
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
        llm_func=lambda prompt: "an answer",
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

    def broken(query, docs):
        raise RuntimeError("reranker is down")

    def make(**kw):
        return BasicRAGRerankingPipeline(
            connections,
            config,
            reranker_func=broken,
            llm_func=lambda p: "local answer",
            vector_store=store,
        )

    with pytest.raises(RuntimeError, match="reranker is down"):
        make().query("insulin", top_k=2)

    result = make().query("insulin", top_k=2, allow_rerank_fallback=True)
    assert result["metadata"]["reranked"] is False
    assert "reranker is down" in result["metadata"]["rerank_error"]
    assert result["answer"] == "local answer"  # no OPENAI_API_KEY needed


def test_create_pipeline_with_embedding_func_and_a_key_reaches_a_real_pipeline(
    stack, monkeypatch
):
    """OS-10 end to end: the OpenAI client must not see the pipeline's kwargs."""
    seen = {}

    class StrictChatOpenAI:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            if "embedding_func" in kwargs:
                raise TypeError("unexpected keyword argument 'embedding_func'")

        def invoke(self, prompt):
            return types.SimpleNamespace(content="fake openai answer")

    module = types.ModuleType("langchain_openai")
    module.ChatOpenAI = StrictChatOpenAI
    monkeypatch.setitem(sys.modules, "langchain_openai", module)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    import iris_vector_rag.common.utils as utils

    utils._llm_instances.clear()
    monkeypatch.setattr(utils, "_setup_caching_if_needed", lambda enable=None: False)

    from iris_vector_rag import create_pipeline

    pipeline = create_pipeline(
        "basic", embedding_func=embed, validate_requirements=False
    )
    assert "embedding_func" not in seen
    assert pipeline.llm_func("hello") == "fake openai answer"
    utils._llm_instances.clear()
