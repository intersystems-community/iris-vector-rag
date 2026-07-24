"""Unit tests for QueryOptions and normalize_query_params (T004 — TDD, must fail before T005)."""

import warnings

import pytest


class TestNormalizeQueryParams:
    """normalize_query_params: alias resolution, defaults, validation."""

    def test_canonical_query_param(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        opts = normalize_query_params(query="what is IRIS?")
        assert opts.query == "what is IRIS?"

    def test_query_text_alias_accepted(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        opts = normalize_query_params(query_text="what is IRIS?")
        assert opts.query == "what is IRIS?"

    def test_query_wins_over_query_text_when_both_given(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            opts = normalize_query_params(query="canonical", query_text="alias")
        assert opts.query == "canonical"
        assert any("query_text" in str(warning.message).lower() for warning in w)

    def test_missing_query_raises(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        with pytest.raises((ValueError, TypeError)):
            normalize_query_params(top_k=5)

    def test_empty_query_raises(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        with pytest.raises(ValueError):
            normalize_query_params(query="   ")

    def test_default_top_k(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        opts = normalize_query_params(query="test")
        assert opts.top_k == 5

    def test_custom_top_k(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        opts = normalize_query_params(query="test", top_k=10)
        assert opts.top_k == 10

    def test_top_k_bounds(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        with pytest.raises(ValueError):
            normalize_query_params(query="test", top_k=0)
        with pytest.raises(ValueError):
            normalize_query_params(query="test", top_k=101)

    def test_default_generate_answer(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        opts = normalize_query_params(query="test")
        assert opts.generate_answer is True

    def test_default_include_sources(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        opts = normalize_query_params(query="test")
        assert opts.include_sources is True

    def test_default_metadata_filter_none(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        opts = normalize_query_params(query="test")
        assert opts.metadata_filter is None

    def test_default_similarity_threshold(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        opts = normalize_query_params(query="test")
        assert opts.similarity_threshold == 0.0

    def test_similarity_threshold_bounds(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        with pytest.raises(ValueError):
            normalize_query_params(query="test", similarity_threshold=-0.1)
        with pytest.raises(ValueError):
            normalize_query_params(query="test", similarity_threshold=1.1)

    def test_weights_without_fusion_raises(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        with pytest.raises(ValueError, match="weights"):
            normalize_query_params(
                query="test", weights={"vector": 0.7, "text": 0.3}, retrieval="vector"
            )

    def test_weights_with_hybrid_ok(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        opts = normalize_query_params(
            query="test", weights={"vector": 0.7, "text": 0.3}, retrieval="hybrid"
        )
        assert opts.weights == {"vector": 0.7, "text": 0.3}

    def test_weights_with_rrf_ok(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        opts = normalize_query_params(
            query="test", weights={"vector": 0.5, "text": 0.5}, retrieval="rrf"
        )
        assert opts.retrieval == "rrf"

    def test_default_retrieval_none(self):
        """retrieval=None means 'use pipeline default'."""
        from iris_vector_rag.core.query_options import normalize_query_params

        opts = normalize_query_params(query="test")
        assert opts.retrieval is None

    def test_default_rerank_none(self):
        from iris_vector_rag.core.query_options import normalize_query_params

        opts = normalize_query_params(query="test")
        assert opts.rerank is None

    def test_all_defaults_preserve_existing_behavior(self):
        """When retrieval/rerank/weights all unset, opts represents no-change (Principle IV)."""
        from iris_vector_rag.core.query_options import normalize_query_params

        opts = normalize_query_params(query="test", top_k=5)
        assert opts.retrieval is None
        assert opts.rerank is None
        assert opts.weights is None
        assert opts.metadata_filter is None
        assert opts.similarity_threshold == 0.0


class TestQueryOptionsDataclass:
    """QueryOptions dataclass: fields accessible, frozen or mutable as designed."""

    def test_fields_accessible(self):
        from iris_vector_rag.core.query_options import QueryOptions

        opts = QueryOptions(query="test")
        assert opts.query == "test"
        assert opts.top_k == 5
        assert opts.generate_answer is True
        assert opts.include_sources is True
        assert opts.metadata_filter is None
        assert opts.similarity_threshold == 0.0
        assert opts.retrieval is None
        assert opts.weights is None
        assert opts.rerank is None
        assert opts.custom_prompt is None

    def test_custom_prompt_field(self):
        from iris_vector_rag.core.query_options import QueryOptions

        opts = QueryOptions(query="test", custom_prompt="Answer in bullets.")
        assert opts.custom_prompt == "Answer in bullets."
