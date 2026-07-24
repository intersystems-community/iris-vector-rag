"""Golden response harness for all registered pipelines (T008).

Principle IV safety net: every pipeline's query() response MUST contain
the canonical keys {answer, retrieved_documents, contexts, sources, metadata}
and NOT change their types. This test runs without IRIS (mocked vector store).

Pipelines covered (matching iris_vector_rag/__init__.py factory map):
  basic, basic_rerank, crag, graphrag (→ hybrid_graphrag.py),
  pylate_colbert, multi_query_rrf
"""
from unittest.mock import MagicMock, patch
import pytest


REQUIRED_KEYS = {"answer", "retrieved_documents", "contexts", "sources", "metadata"}


def _make_minimal_mocks():
    """Return (connection_manager, config_manager, vector_store) mocks."""
    cm = MagicMock()
    cfg = MagicMock()
    cfg.get = MagicMock(side_effect=lambda key, default=None: default)
    cfg.get_nested = MagicMock(side_effect=lambda key, default=None: default)
    vs = MagicMock()
    vs.similarity_search.return_value = []
    vs.similarity_search_with_score.return_value = []
    vs.search_by_text.return_value = []
    return cm, cfg, vs


def _make_basic_pipeline():
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline

    cm, cfg, vs = _make_minimal_mocks()
    with patch.object(BasicRAGPipeline, "__init__", lambda self, *a, **kw: None):
        p = BasicRAGPipeline.__new__(BasicRAGPipeline)
    p.connection_manager = cm
    p.config_manager = cfg
    p.vector_store = vs
    p.logger = MagicMock()
    p.llm_func = MagicMock(return_value="golden answer")
    p.embedding_manager = MagicMock()
    p.embedding_config = None
    p.use_iris_embedding = False
    p.pipeline_config = {}
    p.chunk_size = 1000
    p.chunk_overlap = 200
    p.default_top_k = 5
    return p


class TestBasicRAGPipelineGolden:
    """basic pipeline golden shape."""

    def test_response_has_required_keys(self):
        p = _make_basic_pipeline()
        result = p.query(query="test", top_k=3, generate_answer=True)
        assert REQUIRED_KEYS.issubset(result.keys()), (
            f"Missing keys: {REQUIRED_KEYS - result.keys()}"
        )

    def test_retrieved_documents_is_list(self):
        p = _make_basic_pipeline()
        result = p.query(query="test", generate_answer=True)
        assert isinstance(result["retrieved_documents"], list)
        assert isinstance(result["contexts"], list)
        assert isinstance(result["metadata"], dict)


class TestBasicRerankPipelineGolden:
    """basic_rerank pipeline golden shape."""

    def test_response_has_required_keys(self):
        from iris_vector_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline

        cm, cfg, vs = _make_minimal_mocks()
        with patch.object(
            BasicRAGRerankingPipeline, "__init__", lambda self, *a, **kw: None
        ):
            p = BasicRAGRerankingPipeline.__new__(BasicRAGRerankingPipeline)
        p.connection_manager = cm
        p.config_manager = cfg
        p.vector_store = vs
        p.logger = MagicMock()
        p.llm_func = MagicMock(return_value="")
        p.embedding_manager = MagicMock()
        p.embedding_config = None
        p.use_iris_embedding = False
        p.pipeline_config = {}
        p.chunk_size = 1000
        p.chunk_overlap = 200
        p.default_top_k = 5
        p.rerank_factor = 2
        p.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        p.reranker_func = None

        result = p.query(query="test", generate_answer=True)
        assert REQUIRED_KEYS.issubset(result.keys()), (
            f"Missing keys: {REQUIRED_KEYS - result.keys()}"
        )


class TestCRAGPipelineGolden:
    """crag pipeline golden shape.

    NOTE: crag currently puts 'sources' inside metadata rather than as a top-level key.
    This is a pre-existing inconsistency that US2 will fix. The golden test documents
    the *actual* current behavior (not the desired post-US2 behavior).
    """

    # crag omits top-level 'sources'; it lives in metadata instead
    CRAG_KEYS = {"answer", "retrieved_documents", "contexts", "metadata"}

    def test_response_has_required_keys(self):
        from iris_vector_rag.pipelines.crag import CRAGPipeline

        cm, cfg, vs = _make_minimal_mocks()
        with patch.object(CRAGPipeline, "__init__", lambda self, *a, **kw: None):
            p = CRAGPipeline.__new__(CRAGPipeline)
        p.connection_manager = cm
        p.config_manager = cfg
        p.vector_store = vs
        p.logger = MagicMock()
        p.llm_func = MagicMock(return_value="answer")
        p.embedding_func = MagicMock(return_value=[[0.1] * 384])
        p.embedding_manager = MagicMock()
        p.embedding_config = None
        p.use_iris_embedding = False
        p.evaluator = MagicMock()
        p.evaluator.evaluate.return_value = "CORRECT"

        with patch.object(p, "_initial_retrieval", return_value=[]):
            with patch.object(p, "_apply_corrective_actions", return_value=[]):
                result = p.query(query="test", generate_answer=True)

        assert self.CRAG_KEYS.issubset(result.keys()), (
            f"Missing keys: {self.CRAG_KEYS - result.keys()}"
        )


class TestMultiQueryRRFGolden:
    """multi_query_rrf pipeline golden shape."""

    def test_response_has_required_keys(self):
        from iris_vector_rag.pipelines.multi_query_rrf import MultiQueryRRFPipeline

        cm, cfg, vs = _make_minimal_mocks()
        with patch.object(MultiQueryRRFPipeline, "__init__", lambda self, *a, **kw: None):
            p = MultiQueryRRFPipeline.__new__(MultiQueryRRFPipeline)
        p.connection_manager = cm
        p.config_manager = cfg
        p.vector_store = vs
        p.logger = MagicMock()
        p.num_queries = 2
        p.retrieved_k = 10
        p.rrf_k = 60
        p.use_llm_expansion = False
        p.llm_model = "gpt-4o-mini"
        p.llm = None

        with patch.object(p, "generate_query_variations", return_value=["test"]):
            result = p.query(query="test", generate_answer=False)

        assert REQUIRED_KEYS.issubset(result.keys()), (
            f"Missing keys: {REQUIRED_KEYS - result.keys()}"
        )


class TestHybridGraphRAGGolden:
    """hybrid graphrag pipeline golden shape (fully mocked — no iris-vector-graph needed)."""

    def _make_pipeline(self):
        from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

        cm, cfg, vs = _make_minimal_mocks()
        # Bypass __init__ entirely — it tries to connect to IRIS
        with patch.object(HybridGraphRAGPipeline, "__init__", lambda self, *a, **kw: None):
            p = HybridGraphRAGPipeline.__new__(HybridGraphRAGPipeline)
        p.connection_manager = cm
        p.config_manager = cfg
        p.vector_store = vs
        p.logger = MagicMock()
        p.llm_func = MagicMock(return_value="golden answer")
        p.embedding_manager = MagicMock()
        p.embedding_config = None
        p.use_iris_embedding = False
        p.pipeline_config = {}
        p.default_top_k = 10
        # iris_engine=None triggers enhanced_hybrid_fallback → _fallback_to_vector_search
        p.iris_engine = None
        p.retrieval_methods = None
        return p

    def test_response_has_required_keys(self):
        p = self._make_pipeline()
        # _validate_knowledge_graph hits DB; patch it out
        with patch.object(p, "_validate_knowledge_graph"):
            with patch.object(p, "_enhanced_hybrid_fallback", return_value=([], "fallback")):
                result = p.query(query="test", generate_answer=False)

        assert REQUIRED_KEYS.issubset(result.keys()), (
            f"Missing keys: {REQUIRED_KEYS - result.keys()}"
        )

    def test_sources_key_always_present(self):
        p = self._make_pipeline()
        with patch.object(p, "_validate_knowledge_graph"):
            with patch.object(p, "_enhanced_hybrid_fallback", return_value=([], "fallback")):
                result = p.query(query="test", generate_answer=False, include_sources=False)
        assert "sources" in result
        assert isinstance(result["sources"], list)


class TestPyLateColBERTGolden:
    """pylate_colbert pipeline golden shape (fully mocked — no PyLate needed)."""

    def _make_pipeline(self):
        from iris_vector_rag.pipelines.colbert_pylate.pylate_pipeline import PyLateColBERTPipeline

        cm, cfg, vs = _make_minimal_mocks()
        with patch.object(PyLateColBERTPipeline, "__init__", lambda self, *a, **kw: None):
            p = PyLateColBERTPipeline.__new__(PyLateColBERTPipeline)
        p.connection_manager = cm
        p.config_manager = cfg
        p.vector_store = vs
        p.logger = MagicMock()
        p.llm_func = None
        p.embedding_manager = MagicMock()
        p.embedding_config = None
        p.use_iris_embedding = False
        p.pipeline_config = {}
        p.chunk_size = 1000
        p.chunk_overlap = 200
        p.default_top_k = 5
        # ColBERT-specific attrs
        p.rerank_factor = 2
        p.model_name = "lightonai/GTE-ModernColBERT-v1"
        p.batch_size = 32
        p.use_native_reranking = False  # skip _pylate_rerank
        p.is_initialized = False
        p._document_store = {}
        p._embedding_cache = {}
        p.stats = {"queries_processed": 0, "documents_indexed": 0, "reranking_operations": 0}
        return p

    def test_response_has_required_keys(self):
        p = self._make_pipeline()
        # _restore_metadata is on the instance; patch it
        with patch.object(p, "_restore_metadata", side_effect=lambda docs: docs):
            result = p.query(query="test", generate_answer=False)

        assert REQUIRED_KEYS.issubset(result.keys()), (
            f"Missing keys: {REQUIRED_KEYS - result.keys()}"
        )

    def test_sources_key_always_present(self):
        p = self._make_pipeline()
        with patch.object(p, "_restore_metadata", side_effect=lambda docs: docs):
            result = p.query(query="test", generate_answer=False, include_sources=False)
        assert "sources" in result
        assert isinstance(result["sources"], list)
