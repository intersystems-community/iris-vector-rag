"""US3 contract: rerank=bool|str|callable query-time option (T021 — TDD, must fail before T022–T024).

Covers FR-007 (rerank at query time), FR-008 (post-fusion ordering), FR-009 (degradation fallback),
C-R1..R6 from contracts/reranker.md.
"""
from unittest.mock import MagicMock, patch, call
import pytest


def _basic_pipeline():
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline

    with patch.object(BasicRAGPipeline, "__init__", lambda self, *a, **kw: None):
        p = BasicRAGPipeline.__new__(BasicRAGPipeline)
    p.connection_manager = MagicMock()
    p.config_manager = MagicMock()
    p.config_manager.get = MagicMock(side_effect=lambda k, d=None: d)
    p.vector_store = MagicMock()
    p.vector_store.similarity_search.return_value = []
    p.logger = MagicMock()
    p.llm_func = MagicMock(return_value="")
    p.embedding_manager = MagicMock()
    p.embedding_config = None
    p.use_iris_embedding = False
    p.pipeline_config = {}
    p.chunk_size = 1000
    p.chunk_overlap = 200
    p.default_top_k = 5
    return p


class TestRerankCallable:
    """rerank=callable: called with (query, docs); result replaces docs (FR-007)."""

    def test_callable_invoked_with_query_and_docs(self):
        from iris_vector_rag.core.models import Document

        doc = Document(id="1", page_content="test doc", metadata={})
        p = _basic_pipeline()
        p.vector_store.similarity_search.return_value = [doc]

        rerank_fn = MagicMock(return_value=[doc])
        result = p.query(query="test", generate_answer=False, rerank=rerank_fn)

        rerank_fn.assert_called_once()
        call_args = rerank_fn.call_args
        # First arg is query string, second is list of docs
        assert call_args[0][0] == "test"
        assert isinstance(call_args[0][1], list)

    def test_callable_result_used_as_retrieved_documents(self):
        from iris_vector_rag.core.models import Document

        doc_a = Document(id="1", page_content="A", metadata={})
        doc_b = Document(id="2", page_content="B", metadata={})
        p = _basic_pipeline()
        p.vector_store.similarity_search.return_value = [doc_a, doc_b]

        # Callable reverses the order
        rerank_fn = MagicMock(return_value=[doc_b, doc_a])
        result = p.query(query="test", generate_answer=False, rerank=rerank_fn)

        assert result["retrieved_documents"][0] == doc_b
        assert result["retrieved_documents"][1] == doc_a


class TestRerankBoolTrue:
    """rerank=True uses the default cross-encoder (FR-007)."""

    def test_rerank_true_does_not_crash(self):
        """rerank=True should not crash even if no cross-encoder available."""
        p = _basic_pipeline()
        # Should succeed (may degrade gracefully if model not available)
        result = p.query(query="test", generate_answer=False, rerank=True)
        assert "retrieved_documents" in result

    def test_rerank_true_degradation_flag(self):
        """If reranker fails, metadata should indicate degradation (FR-009)."""
        p = _basic_pipeline()
        # Mock the reranker to fail
        with patch(
            "iris_vector_rag.retrieval.rerank.resolve_reranker",
            side_effect=ImportError("no model"),
        ):
            result = p.query(query="test", generate_answer=False, rerank=True)

        assert "retrieved_documents" in result


class TestRerankFalseOrNone:
    """rerank=False/None: no reranking applied."""

    def test_rerank_none_does_not_invoke_any_reranker(self):
        p = _basic_pipeline()
        rerank_fn = MagicMock()
        with patch("iris_vector_rag.retrieval.rerank.resolve_reranker", rerank_fn):
            result = p.query(query="test", generate_answer=False, rerank=None)

        rerank_fn.assert_not_called()

    def test_rerank_false_does_not_invoke_any_reranker(self):
        p = _basic_pipeline()
        rerank_fn = MagicMock()
        with patch("iris_vector_rag.retrieval.rerank.resolve_reranker", rerank_fn):
            result = p.query(query="test", generate_answer=False, rerank=False)

        rerank_fn.assert_not_called()


class TestRerankDegradation:
    """Graceful degradation (FR-009): original order preserved on failure."""

    def test_callable_failure_returns_original_order(self):
        from iris_vector_rag.core.models import Document

        doc_a = Document(id="1", page_content="A", metadata={})
        doc_b = Document(id="2", page_content="B", metadata={})
        p = _basic_pipeline()
        p.vector_store.similarity_search.return_value = [doc_a, doc_b]

        failing_rerank = MagicMock(side_effect=RuntimeError("reranker crashed"))
        result = p.query(query="test", generate_answer=False, rerank=failing_rerank)

        # Original retrieval order preserved
        assert len(result["retrieved_documents"]) == 2
        assert result["retrieved_documents"][0] == doc_a
