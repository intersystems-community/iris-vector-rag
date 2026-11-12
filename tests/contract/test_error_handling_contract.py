"""
Contract tests for Error Handling and Edge Cases (FR-023 to FR-025).

Tests validate that HybridGraphRAG handles error conditions correctly:
- Exceptions from graph legs now propagate (no silent fallback)
- Zero-result legs fall back to vector search
- Pipeline remains functional after fallback

Contract: ERROR-001
Requirements: FR-023, FR-024, FR-025
"""

import logging

import pytest


class TestErrorHandlingContract:
    """Contract tests for error handling and edge cases."""

    def test_missing_required_tables_raises(self, hybrid_on_known_corpus, mocker):
        """
        FR-023: A leg that fails due to missing tables MUST raise (not fall back).

        Given: iris_vector_graph expects tables (RDF_EDGES, kg_NodeEmbeddings_optimized)
        And: Tables do not exist (simulated)
        When: Query executed via iris_vector_graph methods
        Then: Exception propagates to caller
        """
        query = "Which salamander can regrow its legs?"

        # Mock to simulate table not found error
        if hasattr(hybrid_on_known_corpus, "retrieval_methods"):
            mocker.patch.object(
                hybrid_on_known_corpus.retrieval_methods,
                "retrieve_via_hnsw_vector",
                side_effect=Exception("Table 'kg_NodeEmbeddings_optimized' not found"),
            )
        else:
            mocker.patch.object(
                hybrid_on_known_corpus,
                "_retrieve_via_hnsw_vector",
                side_effect=Exception("Table 'kg_NodeEmbeddings_optimized' not found"),
            )

        # Execute query - should raise exception
        with pytest.raises(Exception, match="not found"):
            hybrid_on_known_corpus.query(query, method="vector", generate_answer=False)

    def test_iris_vector_graph_connection_failure_raises(
        self, hybrid_on_known_corpus, mocker
    ):
        """
        FR-024: A leg that fails due to connection error MUST raise (not fall back).

        Given: HybridGraphRAG pipeline initialized
        And: iris_vector_graph connection fails (mocked exception)
        When: Query executed
        Then: ConnectionError propagates to caller
        """
        query = "Which salamander can regrow its legs?"

        # Mock to simulate connection failure
        if hasattr(hybrid_on_known_corpus, "retrieval_methods"):
            mocker.patch.object(
                hybrid_on_known_corpus.retrieval_methods,
                "retrieve_via_hybrid_fusion",
                side_effect=ConnectionError("Failed to connect to iris_vector_graph"),
            )
        else:
            mocker.patch.object(
                hybrid_on_known_corpus,
                "_retrieve_via_hybrid_fusion",
                side_effect=ConnectionError("Failed to connect to iris_vector_graph"),
            )

        # Execute query - should raise
        with pytest.raises(ConnectionError, match="Failed to connect"):
            hybrid_on_known_corpus.query(query, method="hybrid", generate_answer=False)

    def test_system_continues_after_fallback(self, hybrid_on_known_corpus, mocker):
        """
        FR-025: System MUST continue functioning after fallback invocation.

        Given: HybridGraphRAG pipeline initialized with known corpus
        And: Multiple queries executed
        When: First query triggers fallback (leg returns zero results)
        And: Second query executed on same pipeline instance
        Then: Both queries succeed, pipeline state remains consistent
        """
        query1 = "Which salamander can regrow its legs?"
        query2 = "What is basalt?"

        # Mock first query to trigger fallback (zero results, not exception)
        if hasattr(hybrid_on_known_corpus, "retrieval_methods"):
            mock_method = mocker.patch.object(
                hybrid_on_known_corpus.retrieval_methods,
                "retrieve_via_rrf",
                return_value=([], "rrf"),
            )
        else:
            mock_method = mocker.patch.object(
                hybrid_on_known_corpus, "_retrieve_via_rrf", return_value=([], "rrf")
            )

        # Execute first query
        result1 = hybrid_on_known_corpus.query(
            query1, method="rrf", generate_answer=False
        )

        # Verify first query used fallback
        assert len(result1["contexts"]) > 0, "First query should succeed via fallback"
        assert (
            result1["metadata"]["retrieval_method"] == "vector_fallback"
        ), "First query should use fallback"
        assert (
            result1["retrieved_documents"][0].id == "kc-axolotl"
        ), f"First query should return kc-axolotl, got {result1['retrieved_documents'][0].id}"

        # Remove mock for second query
        mock_method.stop()

        # Execute second query on same pipeline instance
        result2 = hybrid_on_known_corpus.query(
            query2, method="rrf", generate_answer=False
        )

        # Verify second query succeeded
        assert (
            len(result2["contexts"]) > 0
        ), "Second query should succeed after first query fallback"
        assert (
            result2["retrieved_documents"][0].id == "kc-basalt"
        ), f"Second query should return kc-basalt, got {result2['retrieved_documents'][0].id}"

        # Verify pipeline state is consistent
        assert result2 is not None, "Pipeline should remain functional"

        # Execute third query to verify sustained functionality
        result3 = hybrid_on_known_corpus.query(
            "medieval trade league", method="hybrid", generate_answer=False
        )
        assert (
            len(result3["contexts"]) > 0
        ), "Third query should also succeed - pipeline fully functional"
        assert (
            result3["retrieved_documents"][0].id == "kc-hanseatic"
        ), f"Third query should return kc-hanseatic, got {result3['retrieved_documents'][0].id}"
