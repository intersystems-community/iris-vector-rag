"""US4 contract: retrieval mode selection, weights, prerequisite errors (T025 — TDD, must fail before T027–T030).

Covers FR-010 (mode selection), FR-011 (weights), FR-012 (prereq error not silent),
C-M1..M7 from contracts/retrieval_modes.md.
"""
from unittest.mock import MagicMock, patch
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
    p.llm_func = None
    p.embedding_manager = MagicMock()
    p.embedding_config = None
    p.use_iris_embedding = False
    p.pipeline_config = {}
    p.chunk_size = 1000
    p.chunk_overlap = 200
    p.default_top_k = 5
    return p


class TestRetrievalModeRegistry:
    """RetrievalMode registry: declared modes, prerequisites (FR-010, FR-012)."""

    def test_vector_mode_registered(self):
        from iris_vector_rag.retrieval.modes import get_mode

        mode = get_mode("vector")
        assert mode is not None
        assert mode.name == "vector"

    def test_text_mode_registered(self):
        from iris_vector_rag.retrieval.modes import get_mode

        mode = get_mode("text")
        assert mode is not None
        assert mode.name == "text"

    def test_hybrid_mode_registered(self):
        from iris_vector_rag.retrieval.modes import get_mode

        mode = get_mode("hybrid")
        assert mode.fusion is not None

    def test_rrf_mode_registered(self):
        from iris_vector_rag.retrieval.modes import get_mode

        mode = get_mode("rrf")
        assert mode.fusion == "rrf"

    def test_unknown_mode_raises(self):
        from iris_vector_rag.retrieval.modes import get_mode

        with pytest.raises((ValueError, KeyError)):
            get_mode("turbo_magic")

    def test_text_mode_has_prerequisite(self):
        from iris_vector_rag.retrieval.modes import get_mode

        mode = get_mode("text")
        assert len(mode.requires) > 0  # needs BM25 / iris-vector-graph


class TestPrerequisiteErrors:
    """Missing prerequisite → named error, not silent fallback (FR-012)."""

    def test_text_mode_missing_prereq_raises_named_error(self):
        from iris_vector_rag.retrieval.modes import check_prerequisites

        # Simulate text mode missing iris-vector-graph BM25
        with patch("iris_vector_rag.retrieval.modes._check_bm25_available", return_value=False):
            with pytest.raises(Exception) as exc_info:
                check_prerequisites("text", connection=MagicMock())
            assert "bm25" in str(exc_info.value).lower() or "text" in str(exc_info.value).lower()


class TestWeightValidation:
    """weights without fusion mode → ValueError (from normalize_query_params)."""

    def test_weights_without_fusion_raises(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        with pytest.raises(ValueError, match="weights"):
            normalize_query_params(
                query="test",
                weights={"vector": 0.7, "text": 0.3},
                retrieval="vector",
            )

    def test_weights_with_hybrid_accepted(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        opts = normalize_query_params(
            query="test",
            weights={"vector": 0.6, "text": 0.4},
            retrieval="hybrid",
        )
        assert opts.weights == {"vector": 0.6, "text": 0.4}


class TestRetrievalEngineDispatch:
    """RetrievalEngine dispatches to correct strategy per mode."""

    def test_vector_mode_uses_vector_store(self):
        from iris_vector_rag.retrieval.engine import RetrievalEngine
        from iris_vector_rag.core.query_options import QueryOptions

        vs = MagicMock()
        vs.search_by_text.return_value = []
        engine = RetrievalEngine(vector_store=vs)
        opts = QueryOptions(query="test", retrieval="vector", top_k=5)
        engine.retrieve(opts)
        vs.search_by_text.assert_called_once()

    def test_unsupported_mode_raises(self):
        from iris_vector_rag.retrieval.engine import RetrievalEngine
        from iris_vector_rag.core.query_options import QueryOptions

        vs = MagicMock()
        engine = RetrievalEngine(vector_store=vs)
        opts = QueryOptions(query="test", retrieval="turbo_mode", top_k=5)
        with pytest.raises((ValueError, NotImplementedError)):
            engine.retrieve(opts)
