"""
Contract tests for Text Search query path (FR-007 to FR-009).

Tests validate that HybridGraphRAG's text search method executes correctly:
- On an empty graph, falls back to vector search
- Exceptions from text search propagate to caller
- Zero results from text search fall back to vector search

Contract: TEXT-001
Requirements: FR-007, FR-008, FR-009
"""

import pytest


class TestTextSearchContract:
    """Contract tests for text search query path."""

    def test_text_search_on_empty_graph_answers_from_vector_fallback(
        self, hybrid_on_known_corpus
    ):
        """
        FR-007: Text search on empty graph falls back to vector search.

        Given: HybridGraphRAG pipeline with empty graph (no entities/relationships)
        When: Query executed with method="text"
        Then: Query succeeds via vector_fallback

        Note: The graph is deployed (IVG classes installed) but contains no data,
        so text search returns zero results and falls back to vector search.
        """
        query = "Which salamander can regrow its legs?"

        # Execute query with text method - graph is empty, will fall back
        result = hybrid_on_known_corpus.query(
            query, method="text", generate_answer=False
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

    def test_text_search_zero_results_falls_back(self, hybrid_on_known_corpus, mocker):
        """
        FR-008: Text search MUST fall back to vector search when returning 0 results.

        Given: HybridGraphRAG pipeline initialized
        And: Text search returns 0 results (mocked)
        When: Query executed with method="text"
        Then: System falls back to IRISVectorStore
        """
        query = "Which salamander can regrow its legs?"

        # Mock text search to return 0 results
        if hasattr(hybrid_on_known_corpus, "retrieval_methods"):
            mocker.patch.object(
                hybrid_on_known_corpus.retrieval_methods,
                "retrieve_via_enhanced_text",
                return_value=([], "text"),
            )
        else:
            mocker.patch.object(
                hybrid_on_known_corpus,
                "_retrieve_via_enhanced_text",
                return_value=([], "text"),
            )

        # Execute query
        result = hybrid_on_known_corpus.query(
            query, method="text", generate_answer=False
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

    def test_text_search_failure_raises(self, hybrid_on_known_corpus, mocker):
        """
        FR-009: Text search exception MUST propagate to caller (no fallback on error).

        Given: HybridGraphRAG pipeline initialized
        And: Text search raises exception (mocked)
        When: Query executed with method="text"
        Then: Exception propagates
        """
        query = "Which salamander can regrow its legs?"

        # Mock text search to raise exception
        if hasattr(hybrid_on_known_corpus, "retrieval_methods"):
            mocker.patch.object(
                hybrid_on_known_corpus.retrieval_methods,
                "retrieve_via_enhanced_text",
                side_effect=RuntimeError("Text search failed"),
            )
        else:
            mocker.patch.object(
                hybrid_on_known_corpus,
                "_retrieve_via_enhanced_text",
                side_effect=RuntimeError("Text search failed"),
            )

        # Execute query - should raise
        with pytest.raises(RuntimeError, match="Text search failed"):
            hybrid_on_known_corpus.query(query, method="text", generate_answer=False)
