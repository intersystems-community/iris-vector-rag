"""
Contract tests for Hybrid Fusion query path (FR-010 to FR-012).

Tests validate that HybridGraphRAG's hybrid fusion method executes correctly:
- On an empty graph, falls back to vector search
- Exceptions from hybrid fusion propagate to caller
- Zero results from hybrid fusion fall back to vector search

Contract: HYBRID-001
Requirements: FR-010, FR-011, FR-012
"""

import pytest


class TestHybridFusionContract:
    """Contract tests for hybrid fusion query path."""

    def test_hybrid_fusion_on_empty_graph_answers_from_vector_fallback(
        self, hybrid_on_known_corpus
    ):
        """
        FR-010: Hybrid fusion on empty graph falls back to vector search.

        Given: HybridGraphRAG pipeline with empty graph (no entities/relationships)
        When: Query executed with method="hybrid"
        Then: Query succeeds via vector_fallback

        Note: The graph is deployed (IVG classes installed) but contains no data,
        so hybrid fusion returns zero results and falls back to vector search.
        """
        query = "Which salamander can regrow its legs?"

        # Execute query with hybrid method - graph is empty, will fall back
        result = hybrid_on_known_corpus.query(
            query, method="hybrid", generate_answer=False
        )

        # Verify result structure
        assert result is not None, "Result should not be None"
        assert "contexts" in result, "Result should have contexts"
        assert "metadata" in result, "Result should have metadata"

        # Verify fallback to vector search
        assert (
            len(result["contexts"]) > 0
        ), "Should retrieve documents via vector fallback"

        assert (
            result["metadata"]["retrieval_method"] == "vector_fallback"
        ), f"Expected vector_fallback, got {result['metadata']['retrieval_method']}"

        # Assert the answer: known query gets its known document
        assert (
            result["retrieved_documents"][0].id == "kc-axolotl"
        ), f"Top result should be kc-axolotl, got {result['retrieved_documents'][0].id}"

    def test_hybrid_fusion_zero_results_falls_back(
        self, hybrid_on_known_corpus, mocker
    ):
        """
        FR-011: Hybrid fusion MUST fall back to vector search when returning 0 results.

        Given: HybridGraphRAG pipeline initialized
        And: Hybrid fusion returns 0 results (mocked)
        When: Query executed with method="hybrid"
        Then: System falls back to IRISVectorStore
        """
        query = "Which salamander can regrow its legs?"

        # Mock hybrid fusion to return 0 results
        if hasattr(hybrid_on_known_corpus, "retrieval_methods"):
            mocker.patch.object(
                hybrid_on_known_corpus.retrieval_methods,
                "retrieve_via_hybrid_fusion",
                return_value=([], "hybrid_fusion"),
            )
        else:
            mocker.patch.object(
                hybrid_on_known_corpus,
                "_retrieve_via_hybrid_fusion",
                return_value=([], "hybrid_fusion"),
            )

        # Execute query
        result = hybrid_on_known_corpus.query(
            query, method="hybrid", generate_answer=False
        )

        # Verify fallback
        assert len(result["contexts"]) > 0, "Fallback should retrieve documents"

        assert (
            result["metadata"]["retrieval_method"] == "vector_fallback"
        ), f"Expected vector_fallback, got {result['metadata']['retrieval_method']}"

        # Assert the answer
        assert (
            result["retrieved_documents"][0].id == "kc-axolotl"
        ), f"Top result should be kc-axolotl, got {result['retrieved_documents'][0].id}"

    def test_hybrid_fusion_failure_raises(self, hybrid_on_known_corpus, mocker):
        """
        FR-012: Hybrid fusion exception MUST propagate to caller (no fallback on error).

        Given: HybridGraphRAG pipeline initialized
        And: Hybrid fusion raises exception (mocked)
        When: Query executed with method="hybrid"
        Then: Exception propagates
        """
        query = "Which salamander can regrow its legs?"

        # Mock hybrid fusion to raise exception
        if hasattr(hybrid_on_known_corpus, "retrieval_methods"):
            mocker.patch.object(
                hybrid_on_known_corpus.retrieval_methods,
                "retrieve_via_hybrid_fusion",
                side_effect=RuntimeError("Hybrid fusion failed"),
            )
        else:
            mocker.patch.object(
                hybrid_on_known_corpus,
                "_retrieve_via_hybrid_fusion",
                side_effect=RuntimeError("Hybrid fusion failed"),
            )

        # Execute query - should raise
        with pytest.raises(RuntimeError, match="Hybrid fusion failed"):
            hybrid_on_known_corpus.query(query, method="hybrid", generate_answer=False)
