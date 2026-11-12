"""query() raises a retrieval failure; it does not return it in `result["error"]`.

Until 2026-11 BasicRAG (and Rerank, which calls it) caught every retrieval
exception and returned `answer=None` with the error in a field, and CRAG answered
from a made-up "fallback_context" document holding the query text. A caller who
did not check the field read a failure as "nothing matched". The exception now
propagates with its own type, so a dimension mismatch is still a
`DimensionMismatchError` with the store's fix in its message.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from iris_vector_rag.exceptions import DimensionMismatchError
from iris_vector_rag.pipelines.basic import BasicRAGPipeline
from iris_vector_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
from iris_vector_rag.pipelines.crag import CRAGPipeline

MISMATCH = DimensionMismatchError(
    got=768, expected=384, where="RAG.SourceDocuments.embedding (query)"
)


def _store_raising(exc):
    store = MagicMock(name="vector_store")
    store.search_by_text.side_effect = exc
    return store


@pytest.mark.usefixtures("use_test_embedder")
def test_basic_raises_the_stores_dimension_mismatch_unchanged():
    pipeline = BasicRAGPipeline(
        connection_manager=Mock(), vector_store=_store_raising(MISMATCH)
    )

    with pytest.raises(DimensionMismatchError) as exc_info:
        pipeline.query("What are the symptoms of diabetes?", generate_answer=False)

    assert exc_info.value is MISMATCH


@pytest.mark.usefixtures("use_test_embedder")
def test_rerank_raises_the_retrieval_failure_and_never_reranks():
    reranker = Mock(name="reranker")
    pipeline = BasicRAGRerankingPipeline(
        connection_manager=Mock(),
        reranker_func=reranker,
        vector_store=_store_raising(MISMATCH),
    )

    with pytest.raises(DimensionMismatchError):
        pipeline.query("What are the symptoms of diabetes?", generate_answer=False)

    reranker.assert_not_called()


@pytest.mark.usefixtures("use_test_embedder")
def test_crag_raises_when_initial_retrieval_fails():
    llm = Mock(name="llm", return_value="an answer")
    pipeline = CRAGPipeline(
        connection_manager=Mock(), vector_store=Mock(), llm_func=llm
    )

    with patch.object(pipeline, "_initial_retrieval", side_effect=MISMATCH):
        with pytest.raises(DimensionMismatchError):
            pipeline.query("What are the symptoms of diabetes?")

    # No answer was generated from a stand-in context.
    llm.assert_not_called()


@pytest.mark.usefixtures("use_test_embedder")
def test_crag_raises_a_failure_after_retrieval_instead_of_an_error_dict():
    """CRAG's outermost `except Exception` returned an error dict for any failure."""
    pipeline = CRAGPipeline(
        connection_manager=Mock(), vector_store=Mock(), llm_func=lambda p: "a"
    )
    boom = RuntimeError("building the result broke")

    with patch.object(pipeline, "_initial_retrieval", return_value=[]), patch.object(
        pipeline.evaluator, "evaluate", return_value="confident"
    ), patch.object(
        pipeline, "_apply_corrective_actions", return_value=[]
    ), patch.object(
        CRAGPipeline, "_correction_strategy", side_effect=boom
    ):
        with pytest.raises(RuntimeError, match="building the result broke"):
            pipeline.query("What are the symptoms of diabetes?", generate_answer=False)
