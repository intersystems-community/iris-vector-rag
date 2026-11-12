"""query() raises when answer generation fails; it does not return a stand-in answer.

Until 2026-11 most pipelines caught the LLM's exception. Rerank, CRAG, PyLate,
ColBERT-IRIS and HybridGraphRAG answered "Error generating answer", a string a
caller could show as the answer. BasicRAG and MultiQueryRRF returned
`answer=None` with the error in `result["error"]`. The exception now propagates
with its own type (Tom, 2026-11-12: same rule as retrieval). BasicRAG, PyLate
and HybridGraphRAG are covered next to their other unit tests.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
from iris_vector_rag.pipelines.crag import CRAGPipeline
from iris_vector_rag.pipelines.multi_query_rrf import MultiQueryRRFPipeline

DOCS = [
    Document(id="d1", page_content="Diabetes causes thirst.", metadata={}),
    Document(id="d2", page_content="Insulin lowers blood sugar.", metadata={}),
]


def llm_down(*args, **kwargs):
    raise RuntimeError("LLM endpoint returned 401")


@pytest.mark.usefixtures("use_test_embedder")
def test_rerank_raises_the_llm_error():
    store = MagicMock(name="vector_store")
    store.search_by_text.return_value = list(DOCS)
    pipeline = BasicRAGRerankingPipeline(
        connection_manager=Mock(),
        reranker_func=lambda q, docs: [(d, 1.0) for d in docs],
        vector_store=store,
        llm_func=llm_down,
    )

    with pytest.raises(RuntimeError, match="401"):
        pipeline.query("What causes thirst?")


@pytest.mark.usefixtures("use_test_embedder")
def test_crag_raises_the_llm_error():
    pipeline = CRAGPipeline(
        connection_manager=Mock(), vector_store=Mock(), llm_func=llm_down
    )

    with (
        patch.object(pipeline, "_initial_retrieval", return_value=list(DOCS)),
        patch.object(pipeline.evaluator, "evaluate", return_value="confident"),
        patch.object(pipeline, "_apply_corrective_actions", return_value=list(DOCS)),
    ):
        with pytest.raises(RuntimeError, match="401"):
            pipeline.query("What causes thirst?")


@pytest.mark.usefixtures("use_test_embedder")
def test_colbert_iris_raises_the_llm_error():
    from iris_vector_rag.pipelines.colbert_iris import pipeline as colbert

    with patch.object(colbert, "ColBERTSchema"), patch.object(
        colbert, "ColBERTIngestor"
    ), patch.object(colbert, "MaxSimInDB"):
        pipeline = colbert.IRISColBERTPipeline(
            connection_manager=Mock(), vector_store=Mock(), llm_func=llm_down
        )

    pipeline._maxsim.indb_maxsim.return_value = [("d1", 0.9), ("d2", 0.8)]
    with patch.object(pipeline, "_load_model"), patch.object(
        pipeline, "_encode_query", return_value=[[0.1] * 4]
    ), patch.object(pipeline, "_fetch_docs_by_ids", return_value=list(DOCS)):
        with pytest.raises(RuntimeError, match="401"):
            pipeline.query("What causes thirst?")


def test_multi_query_rrf_raises_the_llm_error():
    store = MagicMock(name="vector_store")
    store.similarity_search.return_value = list(DOCS)
    pipeline = MultiQueryRRFPipeline(
        connection_manager=Mock(),
        config_manager=Mock(),
        vector_store=store,
        llm_func=llm_down,
    )
    pipeline.llm = llm_down

    with patch.object(
        pipeline, "generate_query_variations", return_value=["What causes thirst?"]
    ):
        with pytest.raises(RuntimeError, match="401"):
            pipeline.query("What causes thirst?")
