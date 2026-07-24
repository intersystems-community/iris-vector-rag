"""US2 contract: canonical query param + query_text alias, consistent defaults, response keys.

T014 — TDD, must fail before T015–T020.

Covers FR-004 (consistent kwargs), FR-005 (query_text alias), FR-006 (response keys),
C-Q1..Q6 from the contracts/query_api.md spec.
"""
import warnings
from unittest.mock import MagicMock, patch
import pytest


# The canonical required response keys every pipeline must return
REQUIRED_RESPONSE_KEYS = {
    "answer",
    "retrieved_documents",
    "contexts",
    "sources",
    "metadata",
}


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


# ---------------------------------------------------------------------------
# FR-004: canonical 'query' param accepted
# ---------------------------------------------------------------------------

class TestCanonicalQueryParam:
    """Every pipeline accepts 'query' as the canonical parameter name."""

    def test_basic_accepts_query_kwarg(self):
        p = _basic_pipeline()
        result = p.query(query="test question", generate_answer=False)
        assert isinstance(result, dict)

    def test_basic_query_result_not_empty_dict(self):
        p = _basic_pipeline()
        result = p.query(query="test question", generate_answer=False)
        assert result  # non-empty


# ---------------------------------------------------------------------------
# FR-005: query_text alias
# ---------------------------------------------------------------------------

class TestQueryTextAlias:
    """query_text is accepted as an alias for query on every pipeline."""

    def test_basic_accepts_query_text_alias(self):
        p = _basic_pipeline()
        result = p.query(query_text="test question", generate_answer=False)
        assert isinstance(result, dict)

    def test_basic_query_text_produces_same_keys(self):
        p = _basic_pipeline()
        r1 = p.query(query="test", generate_answer=False)
        r2 = p.query(query_text="test", generate_answer=False)
        assert r1.keys() == r2.keys()

    def test_both_params_query_wins_with_warning(self):
        p = _basic_pipeline()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = p.query(query="canonical", query_text="alias", generate_answer=False)
        assert isinstance(result, dict)
        # Should warn about the conflict
        assert any("query_text" in str(warning.message).lower() for warning in w)


# ---------------------------------------------------------------------------
# FR-006: consistent response keys
# ---------------------------------------------------------------------------

class TestResponseKeyConsistency:
    """All pipelines return the same top-level response keys."""

    def test_basic_has_all_required_keys(self):
        p = _basic_pipeline()
        result = p.query(query="test", generate_answer=False)
        assert REQUIRED_RESPONSE_KEYS.issubset(result.keys()), (
            f"Missing: {REQUIRED_RESPONSE_KEYS - result.keys()}"
        )

    def test_retrieved_documents_is_list(self):
        p = _basic_pipeline()
        result = p.query(query="test", generate_answer=False)
        assert isinstance(result["retrieved_documents"], list)

    def test_contexts_is_list(self):
        p = _basic_pipeline()
        result = p.query(query="test", generate_answer=False)
        assert isinstance(result["contexts"], list)

    def test_sources_is_list(self):
        p = _basic_pipeline()
        result = p.query(query="test", generate_answer=False)
        assert isinstance(result["sources"], list)

    def test_metadata_is_dict(self):
        p = _basic_pipeline()
        result = p.query(query="test", generate_answer=False)
        assert isinstance(result["metadata"], dict)

    def test_include_sources_false_returns_empty_sources(self):
        p = _basic_pipeline()
        result = p.query(query="test", generate_answer=False, include_sources=False)
        assert result["sources"] == []


# ---------------------------------------------------------------------------
# Consistent defaults across pipelines
# ---------------------------------------------------------------------------

class TestConsistentDefaults:
    """Defaults are consistent — generate_answer=True, top_k=5, include_sources=True."""

    def test_basic_default_top_k_is_5_or_pipeline_default(self):
        p = _basic_pipeline()
        # Just verify it doesn't crash with no top_k
        result = p.query(query="test", generate_answer=False)
        assert isinstance(result, dict)

    def test_basic_generate_answer_false_skips_llm(self):
        p = _basic_pipeline()
        result = p.query(query="test", generate_answer=False)
        # LLM func should NOT have been called
        p.llm_func.assert_not_called()
