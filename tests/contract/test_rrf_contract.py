"""
Contract tests for RRF (Reciprocal Rank Fusion) query path (FR-004 to FR-006).

Tests validate that HybridGraphRAG's RRF method executes correctly:
- On an empty graph, falls back to vector search
- Exceptions from RRF propagate to caller
- Zero results from RRF fall back to vector search

Contract: RRF-001
Requirements: FR-004, FR-005, FR-006
"""

import pytest


class TestRRFContract:
    """Contract tests for RRF query path."""

    def test_rrf_on_empty_graph_answers_from_vector_fallback(
        self, hybrid_on_known_corpus
    ):
        """
        FR-004: RRF on empty graph falls back to vector search over RAG.SourceDocuments.

        Given: HybridGraphRAG pipeline with empty graph (no entities/relationships)
        When: Query executed with method="rrf"
        Then: Query succeeds via vector_fallback

        Note: The graph is deployed (IVG classes installed) but contains no data,
        so RRF returns zero results and falls back to vector search.
        """
        query = "Which salamander can regrow its legs?"

        # Execute query with RRF method - graph is empty, will fall back
        result = hybrid_on_known_corpus.query(
            query, method="rrf", generate_answer=False
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

    def test_rrf_zero_results_falls_back(self, hybrid_on_known_corpus, mocker):
        """
        FR-005: RRF MUST fall back to vector search when returning 0 results.

        Given: HybridGraphRAG pipeline initialized
        And: RRF returns 0 results (mocked)
        When: Query executed with method="rrf"
        Then: System falls back to IRISVectorStore
        """
        query = "Which salamander can regrow its legs?"

        # Mock RRF to return 0 results
        if hasattr(hybrid_on_known_corpus, "retrieval_methods"):
            mocker.patch.object(
                hybrid_on_known_corpus.retrieval_methods,
                "retrieve_via_rrf",
                return_value=([], "rrf"),
            )
        else:
            mocker.patch.object(
                hybrid_on_known_corpus, "_retrieve_via_rrf", return_value=([], "rrf")
            )

        # Execute query
        result = hybrid_on_known_corpus.query(
            query, method="rrf", generate_answer=False
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

    def test_rrf_failure_raises(self, hybrid_on_known_corpus, mocker):
        """
        FR-006: RRF exception MUST propagate to caller (no fallback on error).

        Given: HybridGraphRAG pipeline initialized
        And: RRF raises exception (mocked)
        When: Query executed with method="rrf"
        Then: Exception propagates
        """
        query = "Which salamander can regrow its legs?"

        # Mock RRF to raise exception
        if hasattr(hybrid_on_known_corpus, "retrieval_methods"):
            mocker.patch.object(
                hybrid_on_known_corpus.retrieval_methods,
                "retrieve_via_rrf",
                side_effect=RuntimeError("RRF failed"),
            )
        else:
            mocker.patch.object(
                hybrid_on_known_corpus,
                "_retrieve_via_rrf",
                side_effect=RuntimeError("RRF failed"),
            )

        # Execute query - should raise
        with pytest.raises(RuntimeError, match="RRF failed"):
            hybrid_on_known_corpus.query(query, method="rrf", generate_answer=False)
