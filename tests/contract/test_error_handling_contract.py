"""
Contract tests for Error Handling and Edge Cases (FR-023 to FR-025).

Tests validate that HybridGraphRAG handles error conditions correctly when
required tables are missing, connections fail, or other exceptional scenarios occur.

Contract: ERROR-001
Requirements: FR-023, FR-024, FR-025
"""

import logging
import pytest
from iris_rag import create_pipeline


class TestErrorHandlingContract:
    """Contract tests for error handling and edge cases."""

    @pytest.fixture
    def graphrag_pipeline(self):
        """Create HybridGraphRAG pipeline with validation."""
        return create_pipeline("graphrag", validate_requirements=True)

    @pytest.mark.requires_database
    def test_missing_required_tables_handled(self, graphrag_pipeline, mocker, caplog):
        """
        FR-023: System MUST handle missing required tables gracefully.

        Given: iris_graph_core expects tables (RDF_EDGES, kg_NodeEmbeddings_optimized)
        And: Tables do not exist in database
        When: Query executed via iris_graph_core methods
        Then: System detects missing tables and falls back
        """
        caplog.set_level(logging.ERROR)

        query = "What are the complications of untreated diabetes?"

        # Mock to simulate table not found error
        if hasattr(graphrag_pipeline, 'retrieval_methods'):
            mocker.patch.object(
                graphrag_pipeline.retrieval_methods,
                'retrieve_via_hnsw_vector',
                side_effect=Exception("Table 'kg_NodeEmbeddings_optimized' not found")
            )
        else:
            mocker.patch.object(
                graphrag_pipeline,
                '_retrieve_via_hnsw_vector',
                side_effect=Exception("Table 'kg_NodeEmbeddings_optimized' not found")
            )

        # Execute query - should not raise exception
        result = graphrag_pipeline.query(query, method="vector")

        # Verify fallback succeeded
        assert len(result['contexts']) > 0, \
            "Should fall back to IRISVectorStore when tables missing"

        assert result['metadata']['retrieval_method'] == 'vector_fallback', \
            f"Expected vector_fallback, got {result['metadata']['retrieval_method']}"

        # Verify error was logged
        log_output = caplog.text.lower()
        assert any(keyword in log_output for keyword in ['error', 'table', 'not found']), \
            "Should log error about missing table"

        # Verify no exception propagated to caller
        assert result is not None, "Query should complete without raising exception"

    @pytest.mark.requires_database
    def test_iris_graph_core_connection_failure_handled(self, graphrag_pipeline, mocker, caplog):
        """
        FR-024: System MUST handle iris_graph_core connection failures.

        Given: HybridGraphRAG pipeline initialized
        And: iris_graph_core connection fails (mocked exception)
        When: Query executed
        Then: Connection exception caught, logged, and fallback succeeds
        """
        caplog.set_level(logging.ERROR)

        query = "What are the complications of untreated diabetes?"

        # Mock to simulate connection failure
        if hasattr(graphrag_pipeline, 'retrieval_methods'):
            mocker.patch.object(
                graphrag_pipeline.retrieval_methods,
                'retrieve_via_hybrid_fusion',
                side_effect=ConnectionError("Failed to connect to iris_graph_core")
            )
        else:
            mocker.patch.object(
                graphrag_pipeline,
                '_retrieve_via_hybrid_fusion',
                side_effect=ConnectionError("Failed to connect to iris_graph_core")
            )

        # Execute query - should not raise
        result = graphrag_pipeline.query(query, method="hybrid")

        # Verify fallback succeeded
        assert len(result['contexts']) > 0, \
            "Should fall back after connection failure"

        assert result['metadata']['retrieval_method'] == 'vector_fallback', \
            f"Expected vector_fallback, got {result['metadata']['retrieval_method']}"

        # Verify connection error was logged
        log_output = caplog.text.lower()
        assert any(keyword in log_output for keyword in ['error', 'fail', 'connection']), \
            "Should log error about connection failure"

        # Verify no exception propagated
        assert result is not None, "Query should complete despite connection error"

    @pytest.mark.requires_database
    def test_system_continues_after_fallback(self, graphrag_pipeline, mocker):
        """
        FR-025: System MUST continue functioning after fallback invocation.

        Given: HybridGraphRAG pipeline initialized
        And: Multiple queries executed
        When: First query triggers fallback (iris_graph_core fails)
        And: Second query executed on same pipeline instance
        Then: Both queries succeed, pipeline state remains consistent
        """
        query1 = "What are the complications of untreated diabetes?"
        query2 = "How to prevent diabetes complications?"

        # Mock first query to trigger fallback
        if hasattr(graphrag_pipeline, 'retrieval_methods'):
            mock_method = mocker.patch.object(
                graphrag_pipeline.retrieval_methods,
                'retrieve_via_rrf',
                return_value=([], 'rrf')
            )
        else:
            mock_method = mocker.patch.object(
                graphrag_pipeline,
                '_retrieve_via_rrf',
                return_value=([], 'rrf')
            )

        # Execute first query
        result1 = graphrag_pipeline.query(query1, method="rrf")

        # Verify first query used fallback
        assert len(result1.contexts) > 0, "First query should succeed via fallback"
        assert result1.metadata['retrieval_method'] == 'vector_fallback', \
            "First query should use fallback"

        # Remove mock for second query (or keep it - both should work)
        mock_method.stop()

        # Execute second query on same pipeline instance
        result2 = graphrag_pipeline.query(query2, method="rrf")

        # Verify second query succeeded
        assert len(result2.contexts) > 0, \
            "Second query should succeed after first query fallback"

        # Verify pipeline state is consistent
        assert result2 is not None, "Pipeline should remain functional"

        # Verify no degradation in results
        assert len(result2.contexts) >= len(result1.contexts) * 0.5, \
            "Second query should retrieve similar number of documents (within 50%)"

        # Execute third query to verify sustained functionality
        result3 = graphrag_pipeline.query("diabetes prevention strategies", method="hybrid")
        assert len(result3.contexts) > 0, \
            "Third query should also succeed - pipeline fully functional"
