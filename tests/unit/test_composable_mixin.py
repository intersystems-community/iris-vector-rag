"""Unit tests for ComposableQueryMixin delegation seam (T006 — TDD, must fail before T007)."""
from unittest.mock import MagicMock, patch, call

import pytest


class TestComposableQueryMixinNormalize:
    """_normalize_query: produces QueryOptions from pipeline.query() kwargs."""

    def test_normalize_passes_query_through(self):
        from iris_vector_rag.core.composable_query import ComposableQueryMixin

        mixin = ComposableQueryMixin()
        opts = mixin._normalize_query(query="test question")
        assert opts.query == "test question"

    def test_normalize_accepts_query_text_alias(self):
        from iris_vector_rag.core.composable_query import ComposableQueryMixin

        mixin = ComposableQueryMixin()
        opts = mixin._normalize_query(query_text="test alias")
        assert opts.query == "test alias"

    def test_normalize_uses_top_k(self):
        from iris_vector_rag.core.composable_query import ComposableQueryMixin

        mixin = ComposableQueryMixin()
        opts = mixin._normalize_query(query="test", top_k=10)
        assert opts.top_k == 10

    def test_normalize_metadata_filter(self):
        from iris_vector_rag.core.composable_query import ComposableQueryMixin

        mixin = ComposableQueryMixin()
        opts = mixin._normalize_query(query="test", metadata_filter={"source": "pub"})
        assert opts.metadata_filter == {"source": "pub"}


class TestComposableQueryMixinRunRetrieval:
    """_run_retrieval: dispatches to the concrete pipeline's retrieval method."""

    def test_run_retrieval_calls_do_retrieval(self):
        from iris_vector_rag.core.composable_query import ComposableQueryMixin
        from iris_vector_rag.core.query_options import QueryOptions

        mixin = ComposableQueryMixin()
        mixin._do_retrieval = MagicMock(return_value=[])
        opts = QueryOptions(query="test", top_k=5)
        mixin._run_retrieval(opts)
        mixin._do_retrieval.assert_called_once_with(opts)

    def test_run_retrieval_returns_documents(self):
        from iris_vector_rag.core.composable_query import ComposableQueryMixin
        from iris_vector_rag.core.query_options import QueryOptions
        from iris_vector_rag.core.models import Document

        doc = Document(id="1", page_content="test", metadata={})
        mixin = ComposableQueryMixin()
        mixin._do_retrieval = MagicMock(return_value=[doc])
        opts = QueryOptions(query="test")
        result = mixin._run_retrieval(opts)
        assert result == [doc]

    def test_run_retrieval_unsupported_mode_raises(self):
        from iris_vector_rag.core.composable_query import ComposableQueryMixin
        from iris_vector_rag.core.query_options import QueryOptions

        mixin = ComposableQueryMixin()
        mixin.supported_retrieval_modes = ["vector"]
        opts = QueryOptions(query="test", retrieval="text")
        with pytest.raises((ValueError, NotImplementedError)):
            mixin._run_retrieval(opts)


class TestComposableQueryMixinMaybeRerank:
    """_maybe_rerank: reranks when opts.rerank is set, passes through otherwise."""

    def test_no_rerank_when_opts_rerank_none(self):
        from iris_vector_rag.core.composable_query import ComposableQueryMixin
        from iris_vector_rag.core.query_options import QueryOptions
        from iris_vector_rag.core.models import Document

        docs = [Document(id="1", page_content="a", metadata={})]
        mixin = ComposableQueryMixin()
        opts = QueryOptions(query="test", rerank=None)
        result, degraded = mixin._maybe_rerank(docs, opts)
        assert result == docs
        assert degraded is False

    def test_no_rerank_when_opts_rerank_false(self):
        from iris_vector_rag.core.composable_query import ComposableQueryMixin
        from iris_vector_rag.core.query_options import QueryOptions
        from iris_vector_rag.core.models import Document

        docs = [Document(id="1", page_content="a", metadata={})]
        mixin = ComposableQueryMixin()
        opts = QueryOptions(query="test", rerank=False)
        result, degraded = mixin._maybe_rerank(docs, opts)
        assert result == docs
        assert degraded is False

    def test_callable_rerank_used_directly(self):
        from iris_vector_rag.core.composable_query import ComposableQueryMixin
        from iris_vector_rag.core.query_options import QueryOptions
        from iris_vector_rag.core.models import Document

        doc_a = Document(id="1", page_content="a", metadata={})
        doc_b = Document(id="2", page_content="b", metadata={})
        reranked = [doc_b, doc_a]
        rerank_fn = MagicMock(return_value=reranked)

        mixin = ComposableQueryMixin()
        opts = QueryOptions(query="test", rerank=rerank_fn)
        result, degraded = mixin._maybe_rerank([doc_a, doc_b], opts)
        rerank_fn.assert_called_once()
        assert result == reranked
        assert degraded is False

    def test_rerank_failure_falls_back_gracefully(self):
        from iris_vector_rag.core.composable_query import ComposableQueryMixin
        from iris_vector_rag.core.query_options import QueryOptions
        from iris_vector_rag.core.models import Document

        doc = Document(id="1", page_content="a", metadata={})
        failing_rerank = MagicMock(side_effect=RuntimeError("reranker crashed"))

        mixin = ComposableQueryMixin()
        opts = QueryOptions(query="test", rerank=failing_rerank)
        result, degraded = mixin._maybe_rerank([doc], opts)
        assert result == [doc]  # original order preserved
        assert degraded is True


class TestSupportedRetrievalModes:
    """supported_retrieval_modes: default is ['vector']."""

    def test_default_modes(self):
        from iris_vector_rag.core.composable_query import ComposableQueryMixin

        mixin = ComposableQueryMixin()
        assert "vector" in mixin.supported_retrieval_modes
