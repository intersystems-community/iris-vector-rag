"""
Contract tests for HNSW Vector query path (FR-013 to FR-015).

Tests validate that HybridGraphRAG's HNSW vector method executes correctly:
- On an empty graph, falls back to vector search
- Exceptions from HNSW vector propagate to caller
- Zero results from HNSW vector fall back to vector search

Contract: HNSW-001
Requirements: FR-013, FR-014, FR-015
"""

import pytest


class TestHNSWVectorContract:
    """Contract tests for HNSW vector query path."""

    def test_hnsw_vector_on_empty_graph_answers_from_vector_fallback(
        self, hybrid_on_known_corpus
    ):
        """
        FR-013: HNSW vector on empty graph falls back to vector search.

        Given: HybridGraphRAG pipeline with empty graph (no entities/relationships)
        When: Query executed with method="vector"
        Then: Query succeeds via vector_fallback

        Note: The graph is deployed (IVG classes installed) but contains no data,
        so HNSW vector returns zero results and falls back to vector search.
        """
        query = "Which salamander can regrow its legs?"

        # Execute query with vector method - graph is empty, will fall back
        result = hybrid_on_known_corpus.query(
            query, method="vector", generate_answer=False
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

    def test_hnsw_vector_zero_results_falls_back(self, hybrid_on_known_corpus, mocker):
        """
        FR-014: HNSW vector MUST fall back to vector search when returning 0 results.

        Given: HybridGraphRAG pipeline initialized
        And: HNSW vector returns 0 results (mocked)
        When: Query executed with method="vector"
        Then: System falls back to IRISVectorStore
        """
        query = "Which salamander can regrow its legs?"

        # Mock HNSW vector to return 0 results
        if hasattr(hybrid_on_known_corpus, "retrieval_methods"):
            mocker.patch.object(
                hybrid_on_known_corpus.retrieval_methods,
                "retrieve_via_hnsw_vector",
                return_value=([], "hnsw_vector"),
            )
        else:
            mocker.patch.object(
                hybrid_on_known_corpus,
                "_retrieve_via_hnsw_vector",
                return_value=([], "hnsw_vector"),
            )

        # Execute query
        result = hybrid_on_known_corpus.query(
            query, method="vector", generate_answer=False
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

    def test_hnsw_vector_failure_raises(self, hybrid_on_known_corpus, mocker):
        """
        FR-015: HNSW vector exception MUST propagate to caller (no fallback on error).

        Given: HybridGraphRAG pipeline initialized
        And: HNSW vector raises exception (mocked)
        When: Query executed with method="vector"
        Then: Exception propagates
        """
        query = "Which salamander can regrow its legs?"

        # Mock HNSW vector to raise exception
        if hasattr(hybrid_on_known_corpus, "retrieval_methods"):
            mocker.patch.object(
                hybrid_on_known_corpus.retrieval_methods,
                "retrieve_via_hnsw_vector",
                side_effect=RuntimeError("HNSW vector failed"),
            )
        else:
            mocker.patch.object(
                hybrid_on_known_corpus,
                "_retrieve_via_hnsw_vector",
                side_effect=RuntimeError("HNSW vector failed"),
            )

        # Execute query - should raise
        with pytest.raises(RuntimeError, match="HNSW vector failed"):
            hybrid_on_known_corpus.query(query, method="vector", generate_answer=False)
