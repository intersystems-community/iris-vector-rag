"""
Contract tests for Fallback Mechanism validation (FR-016 to FR-019).

Tests validate that HybridGraphRAG's fallback to IRISVectorStore works correctly
when a graph leg returns zero results, and that exceptions are raised rather than
hidden.

Contract: FALLBACK-001
Requirements: FR-016, FR-017, FR-018, FR-019
"""

import logging

import pytest


class TestFallbackMechanismContract:
    """Contract tests for fallback mechanism validation."""

    def test_fallback_retrieves_documents_successfully(
        self, hybrid_on_known_corpus, mocker
    ):
        """
        FR-016: IRISVectorStore fallback MUST retrieve documents when leg returns [].

        Given: HybridGraphRAG pipeline initialized with known corpus
        And: Primary retrieval method returns zero results (mocked)
        When: Query executed with any method
        Then: Fallback retrieves documents via IRISVectorStore
        """
        query = "Which salamander can regrow its legs?"

        # Mock hybrid fusion to return zero results (not raise)
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

        # Verify fallback succeeded
        assert (
            len(result["retrieved_documents"]) > 0
        ), "Fallback should retrieve documents successfully"

        # Verify documents have expected fields
        for doc in result["retrieved_documents"]:
            assert hasattr(doc, "page_content") or hasattr(
                doc, "content"
            ), "Document should have content field"
            assert hasattr(doc, "metadata"), "Document should have metadata"

        # Verify retrieval method indicates fallback
        assert (
            result["metadata"]["retrieval_method"] == "vector_fallback"
        ), f"Expected vector_fallback, got {result['metadata']['retrieval_method']}"

        # Assert the answer: known query gets its known document
        assert (
            result["retrieved_documents"][0].id == "kc-axolotl"
        ), f"Top result should be kc-axolotl, got {result['retrieved_documents'][0].id}"

    def test_leg_exception_raises(self, hybrid_on_known_corpus, mocker):
        """
        FR-017: A leg that raises MUST fail the query (not fall back).

        Given: HybridGraphRAG pipeline initialized
        And: Primary method raises an exception
        When: Query executed
        Then: Exception propagates (query does not succeed via fallback)
        """
        query = "Which salamander can regrow its legs?"

        # Mock to raise exception
        if hasattr(hybrid_on_known_corpus, "retrieval_methods"):
            mocker.patch.object(
                hybrid_on_known_corpus.retrieval_methods,
                "retrieve_via_rrf",
                side_effect=RuntimeError("Graph leg failed"),
            )
        else:
            mocker.patch.object(
                hybrid_on_known_corpus,
                "_retrieve_via_rrf",
                side_effect=RuntimeError("Graph leg failed"),
            )

        # Execute query - should raise
        with pytest.raises(RuntimeError, match="Graph leg failed"):
            hybrid_on_known_corpus.query(query, method="rrf", generate_answer=False)

    def test_fallback_metadata_indicates_vector_fallback(
        self, hybrid_on_known_corpus, mocker
    ):
        """
        FR-018: Fallback metadata MUST indicate "vector_fallback".

        Given: HybridGraphRAG pipeline initialized
        And: Fallback triggered for any query method (empty graph leg)
        When: Query completes via fallback
        Then: Metadata correctly indicates vector_fallback
        """
        query = "Which salamander can regrow its legs?"

        # Mock text search to return zero results
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

        # Verify metadata
        assert (
            "retrieval_method" in result["metadata"]
        ), "Metadata should contain retrieval_method"

        assert (
            result["metadata"]["retrieval_method"] == "vector_fallback"
        ), f"Metadata should indicate vector_fallback, got {result['metadata']['retrieval_method']}"

        # Verify metadata includes other expected fields
        assert (
            "execution_time" in result["metadata"]
            or "total_time" in result["metadata"]
            or "processing_time" in result["metadata"]
        ), "Metadata should include execution time"

    def test_empty_graph_degrades_to_vector_fallback(
        self, hybrid_on_known_corpus, caplog
    ):
        """
        FR-019: System MUST gracefully degrade when graph has no data.

        Given: HybridGraphRAG pipeline with empty graph (no entities/relationships)
        When: Query executed with any method
        Then: Query succeeds via vector_fallback over RAG.SourceDocuments

        Note: The graph is deployed (IVG classes installed) but contains no data,
        so all legs return zero results and fall back to vector search.
        """
        caplog.set_level(logging.INFO)

        query = "Which salamander can regrow its legs?"

        # Execute query - graph is empty, so this will fall back
        result = hybrid_on_known_corpus.query(
            query, method="hybrid", generate_answer=False
        )

        # Verify query succeeded
        assert result is not None, "Query should succeed despite empty graph"
        assert len(result["contexts"]) > 0, "Should retrieve documents via fallback"

        # Verify fallback was used
        assert (
            result["metadata"]["retrieval_method"] == "vector_fallback"
        ), "Should use vector_fallback when graph is empty"

        # Assert the answer: known query gets its known document
        assert (
            result["retrieved_documents"][0].id == "kc-axolotl"
        ), f"Top result should be kc-axolotl, got {result['retrieved_documents'][0].id}"

        # System should remain functional - try another query
        result2 = hybrid_on_known_corpus.query(
            "What is basalt?", method="hybrid", generate_answer=False
        )
        assert (
            len(result2["contexts"]) > 0
        ), "Pipeline should remain functional after graceful degradation"
        assert (
            result2["retrieved_documents"][0].id == "kc-basalt"
        ), f"Second query should return kc-basalt, got {result2['retrieved_documents'][0].id}"
