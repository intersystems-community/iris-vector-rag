"""US3 contract: rerank=bool|str|callable query-time option (T021 — TDD, must fail before T022–T024).

Covers FR-007 (rerank at query time), FR-008 (rerank_score in metadata), FR-009 (degradation fallback),
C-R1..R6 from contracts/reranker.md.
"""

from unittest.mock import MagicMock, patch, call
import pytest

from iris_vector_rag.core.models import Document

STUB_DOCS = [
    Document(id="1", page_content="doc alpha", metadata={}),
    Document(id="2", page_content="doc beta", metadata={}),
    Document(id="3", page_content="doc gamma", metadata={}),
]


def _stub_vector_store():
    vs = MagicMock()
    vs.similarity_search.return_value = list(STUB_DOCS)
    vs.search_by_text.return_value = list(STUB_DOCS)
    return vs


def _basic_pipeline():
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline

    with patch.object(BasicRAGPipeline, "__init__", lambda self, *a, **kw: None):
        p = BasicRAGPipeline.__new__(BasicRAGPipeline)
    p.connection_manager = MagicMock()
    p.config_manager = MagicMock()
    p.config_manager.get = MagicMock(side_effect=lambda k, d=None: d)
    p.vector_store = _stub_vector_store()
    p.logger = MagicMock()
    p.llm_func = None
    p.embedding_manager = MagicMock()
    p.embedding_config = None
    p.use_iris_embedding = False
    p.pipeline_config = {}
    p.chunk_size = 1000
    p.chunk_overlap = 200
    p.default_top_k = 5
    return p


def _crag_pipeline():
    from iris_vector_rag.pipelines.crag import CRAGPipeline

    with patch.object(CRAGPipeline, "__init__", lambda self, *a, **kw: None):
        p = CRAGPipeline.__new__(CRAGPipeline)
    p.connection_manager = MagicMock()
    p.vector_store = _stub_vector_store()
    p.llm_func = None
    p.config_manager = MagicMock()
    p.top_k = 5
    p.correction_threshold = 0.5
    p.max_web_results = 3
    p.embedding_func = MagicMock(return_value=[[0.1] * 768])
    p.evaluator = MagicMock()
    p.evaluator.evaluate_retrieval.return_value = (0.9, "CORRECT")
    return p


def _multi_query_rrf_pipeline():
    from iris_vector_rag.pipelines.multi_query_rrf import MultiQueryRRFPipeline

    with patch.object(MultiQueryRRFPipeline, "__init__", lambda self, *a, **kw: None):
        p = MultiQueryRRFPipeline.__new__(MultiQueryRRFPipeline)
    p.connection_manager = MagicMock()
    p.vector_store = _stub_vector_store()
    p.llm_func = None
    p.config_manager = MagicMock()
    p.rrf_k = 60
    p.num_queries = 1
    p.use_llm_expansion = False
    p.retrieved_k = 10
    return p


def _hybrid_graphrag_pipeline():
    from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

    with patch.object(HybridGraphRAGPipeline, "__init__", lambda self, *a, **kw: None):
        p = HybridGraphRAGPipeline.__new__(HybridGraphRAGPipeline)
    p.connection_manager = MagicMock()
    p.vector_store = _stub_vector_store()
    p.llm_func = None
    p.config_manager = MagicMock()
    p.iris_engine = None
    p.retrieval_methods = None
    p.default_top_k = 5
    p.entity_extraction_enabled = False
    p.use_global_communities = False
    return p


def _run_pipeline_query(name, factory, **kwargs):
    p = factory()
    ctx = {}
    if name == "crag":
        ctx["ir"] = patch.object(
            type(p), "_initial_retrieval", return_value=list(STUB_DOCS), create=True
        )
        ctx["ca"] = patch.object(
            type(p),
            "_apply_corrective_actions",
            return_value=list(STUB_DOCS),
            create=True,
        )
        p.evaluator = MagicMock()
        p.evaluator.evaluate.return_value = "CORRECT"
    if name == "hybrid_graphrag":
        ctx["vkg"] = patch.object(
            type(p), "_validate_knowledge_graph", lambda *a, **kw: None, create=True
        )
        ctx["ehf"] = patch.object(
            type(p),
            "_enhanced_hybrid_fallback",
            return_value=(list(STUB_DOCS), "fallback"),
            create=True,
        )
    patches = list(ctx.values())
    for ptch in patches:
        ptch.start()
    try:
        return p.query(query="test rerank", generate_answer=False, **kwargs)
    finally:
        for ptch in patches:
            ptch.stop()


ALL_PIPELINES = [
    ("basic", _basic_pipeline),
    ("crag", _crag_pipeline),
    ("multi_query_rrf", _multi_query_rrf_pipeline),
    ("hybrid_graphrag", _hybrid_graphrag_pipeline),
]


class TestRerankCallable:
    """rerank=callable: called with (query, docs); result replaces docs (FR-007)."""

    def test_callable_invoked_with_query_and_docs(self):
        from iris_vector_rag.core.models import Document

        doc = Document(id="1", page_content="test doc", metadata={})
        p = _basic_pipeline()
        p.vector_store.similarity_search.return_value = [doc]
        p.vector_store.search_by_text.return_value = [doc]

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
        p.vector_store.search_by_text.return_value = [doc_a, doc_b]

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
        p.vector_store.search_by_text.return_value = [doc_a, doc_b]

        failing_rerank = MagicMock(side_effect=RuntimeError("reranker crashed"))
        result = p.query(query="test", generate_answer=False, rerank=failing_rerank)

        # Original retrieval order preserved
        assert len(result["retrieved_documents"]) == 2
        assert result["retrieved_documents"][0] == doc_a


# ──────────────────────────────────────────────────────────────────────────────
# C-R1 / FR-008: rerank_score in doc.metadata after reranking
# ──────────────────────────────────────────────────────────────────────────────


class TestRerankScoreInMetadata:
    """FR-008: reranked docs must have rerank_score in their metadata."""

    def test_callable_reranker_scores_echoed_to_metadata(self):
        """When rerank callable returns (doc, score) tuples, scores go into metadata."""
        doc_a = Document(id="1", page_content="A", metadata={})
        doc_b = Document(id="2", page_content="B", metadata={})
        p = _basic_pipeline()
        p.vector_store.similarity_search.return_value = [doc_a, doc_b]
        p.vector_store.search_by_text.return_value = [doc_a, doc_b]

        # Callable returns (doc, score) tuples
        def scoring_reranker(query, docs):
            return [(doc_b, 0.9), (doc_a, 0.3)]

        result = p.query(query="test", generate_answer=False, rerank=scoring_reranker)
        docs = result["retrieved_documents"]
        assert docs, "reranked docs must be returned"
        assert "rerank_score" in docs[0].metadata, (
            "FR-008: rerank_score must be set in doc.metadata after reranking. "
            f"Got metadata keys: {list(docs[0].metadata.keys())}"
        )

    def test_resolve_reranker_scores_echoed_to_metadata(self):
        """When rerank=True, resolved reranker scores must appear in metadata."""
        doc_a = Document(id="1", page_content="A", metadata={})
        p = _basic_pipeline()
        p.vector_store.similarity_search.return_value = [doc_a]
        p.vector_store.search_by_text.return_value = [doc_a]

        def mock_reranker(query, docs):
            return [(doc_a, 0.95)]

        with patch(
            "iris_vector_rag.retrieval.rerank.resolve_reranker",
            return_value=mock_reranker,
        ):
            result = p.query(query="test", generate_answer=False, rerank=True)

        docs = result["retrieved_documents"]
        assert docs
        assert "rerank_score" in docs[0].metadata, (
            "FR-008: rerank_score must be set in doc.metadata when rerank=True. "
            f"Got: {list(docs[0].metadata.keys())}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# FR-009: rerank_degraded flag in response metadata
# ──────────────────────────────────────────────────────────────────────────────


class TestRerankDegradedFlag:
    """FR-009: metadata["rerank_degraded"]=True when reranking fails."""

    def test_callable_failure_sets_degraded_flag(self):
        doc_a = Document(id="1", page_content="A", metadata={})
        p = _basic_pipeline()
        p.vector_store.similarity_search.return_value = [doc_a]
        p.vector_store.search_by_text.return_value = [doc_a]

        failing_rerank = MagicMock(side_effect=RuntimeError("boom"))
        result = p.query(query="test", generate_answer=False, rerank=failing_rerank)

        assert result["metadata"].get("rerank_degraded") is True, (
            "FR-009: metadata['rerank_degraded'] must be True when reranker raises. "
            f"Got metadata: {result['metadata']}"
        )

    def test_resolve_failure_sets_degraded_flag(self):
        p = _basic_pipeline()
        with patch(
            "iris_vector_rag.retrieval.rerank.resolve_reranker",
            side_effect=ImportError("no model"),
        ):
            result = p.query(query="test", generate_answer=False, rerank=True)

        assert result["metadata"].get("rerank_degraded") is True, (
            "FR-009: metadata['rerank_degraded'] must be True when resolver fails. "
            f"Got metadata: {result['metadata']}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# C-R2 / FR-007: rerank= accepted on ALL registered pipelines
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,factory", ALL_PIPELINES, ids=[n for n, _ in ALL_PIPELINES]
)
def test_rerank_callable_accepted_on_all_pipelines(name, factory):
    """C-R2 / FR-007: rerank=callable works on every registered pipeline."""
    rerank_fn = MagicMock(return_value=list(STUB_DOCS))
    result = _run_pipeline_query(name, factory, rerank=rerank_fn)
    assert isinstance(result, dict), f"{name}: query must return a dict with rerank="
    # Callable must have been invoked
    rerank_fn.assert_called_once()


@pytest.mark.parametrize(
    "name,factory", ALL_PIPELINES, ids=[n for n, _ in ALL_PIPELINES]
)
def test_rerank_none_passthrough_on_all_pipelines(name, factory):
    """C-R4 / FR-013: rerank=None (default) is identical to no-rerank path."""
    result = _run_pipeline_query(name, factory, rerank=None)
    assert isinstance(result, dict), f"{name}: query with rerank=None must return dict"
