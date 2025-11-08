"""
Contract tests for Fallback Mechanism validation (FR-016 to FR-019).

Tests validate that HybridGraphRAG's fallback to IRISVectorStore works correctly
across all query methods with appropriate logging and metadata.

Contract: FALLBACK-001
Requirements: FR-016, FR-017, FR-018, FR-019
"""

import logging
import pytest
from iris_rag import create_pipeline


class TestFallbackMechanismContract:
    """Contract tests for fallback mechanism validation."""

    @pytest.fixture
    def graphrag_pipeline(self):
        """Create HybridGraphRAG pipeline with validation."""
        return create_pipeline("graphrag", validate_requirements=True)

    @pytest.mark.requires_database
    def test_fallback_retrieves_documents_successfully(self, graphrag_pipeline, mocker):
        """
        FR-016: IRISVectorStore fallback MUST retrieve documents successfully.

        Given: HybridGraphRAG pipeline initialized
        And: Primary retrieval method unavailable (mocked)
        When: Query executed with any method
        Then: Fallback retrieves documents via IRISVectorStore
        """
        query = "What are the risk factors for diabetes?"

        # Mock hybrid fusion to trigger fallback
        if hasattr(graphrag_pipeline, 'retrieval_methods'):
            mocker.patch.object(
                graphrag_pipeline.retrieval_methods,
                'retrieve_via_hybrid_fusion',
                return_value=([], 'hybrid_fusion')
            )
        else:
            mocker.patch.object(
                graphrag_pipeline,
                '_retrieve_via_hybrid_fusion',
                return_value=([], 'hybrid_fusion')
            )

        # Execute query
        result = graphrag_pipeline.query(query, method="hybrid")

        # Verify fallback succeeded
        assert len(result['contexts']) > 0, \
            "Fallback should retrieve documents successfully"

        # Verify documents have expected fields
        for doc in result['contexts']:
            assert hasattr(doc, 'page_content') or hasattr(doc, 'content'), \
                "Document should have content field"
            assert hasattr(doc, 'metadata'), \
                "Document should have metadata"

        # Verify retrieval method indicates fallback
        assert result['metadata']['retrieval_method'] == 'vector_fallback', \
            f"Expected vector_fallback, got {result['metadata']['retrieval_method']}"

    @pytest.mark.requires_database
    def test_fallback_logs_diagnostic_messages(self, graphrag_pipeline, mocker, caplog):
        """
        FR-017: Fallback MUST log appropriate diagnostic messages.

        Given: HybridGraphRAG pipeline initialized
        And: Primary method fails (mocked exception)
        When: Query executed and fallback triggered
        Then: ERROR and WARNING logs indicate failure and fallback
        """
        caplog.set_level(logging.DEBUG)

        query = "What are the risk factors for diabetes?"

        # Mock to raise exception
        if hasattr(graphrag_pipeline, 'retrieval_methods'):
            mocker.patch.object(
                graphrag_pipeline.retrieval_methods,
                'retrieve_via_rrf',
                side_effect=Exception("Primary method failed")
            )
        else:
            mocker.patch.object(
                graphrag_pipeline,
                '_retrieve_via_rrf',
                side_effect=Exception("Primary method failed")
            )

        # Execute query
        result = graphrag_pipeline.query(query, method="rrf")

        # Verify fallback succeeded
        assert len(result['contexts']) > 0, "Fallback should succeed"

        # Verify diagnostic logging
        log_output = caplog.text.lower()

        # Should have error log about primary failure
        assert any(keyword in log_output for keyword in ['error', 'fail', 'exception']), \
            "Should log error message about primary method failure"

        # Should have warning/info about fallback
        assert any(keyword in log_output for keyword in ['fallback', 'falling back']), \
            "Should log message about fallback activation"

    @pytest.mark.requires_database
    def test_fallback_metadata_indicates_vector_fallback(self, graphrag_pipeline, mocker):
        """
        FR-018: Fallback metadata MUST indicate "vector_fallback".

        Given: HybridGraphRAG pipeline initialized
        And: Fallback triggered for any query method
        When: Query completes via fallback
        Then: Metadata correctly indicates vector_fallback
        """
        query = "What are the risk factors for diabetes?"

        # Mock text search to trigger fallback
        if hasattr(graphrag_pipeline, 'retrieval_methods'):
            mocker.patch.object(
                graphrag_pipeline.retrieval_methods,
                'retrieve_via_enhanced_text',
                return_value=([], 'text')
            )
        else:
            mocker.patch.object(
                graphrag_pipeline,
                '_retrieve_via_enhanced_text',
                return_value=([], 'text')
            )

        # Execute query
        result = graphrag_pipeline.query(query, method="text")

        # Verify metadata
        assert 'retrieval_method' in result['metadata'], \
            "Metadata should contain retrieval_method"

        assert result['metadata']['retrieval_method'] == 'vector_fallback', \
            f"Metadata should indicate vector_fallback, got {result['metadata']['retrieval_method']}"

        # Verify metadata includes other expected fields
        assert 'execution_time' in result['metadata'] or 'total_time' in result['metadata'], \
            "Metadata should include execution time"

        assert 'num_retrieved' in result['metadata'] or len(result['contexts']) > 0, \
            "Metadata should include document count or contexts should have documents"

    @pytest.mark.requires_database
    def test_graceful_degradation_without_iris_vector_graph(self, graphrag_pipeline, mocker, caplog):
        """
        FR-019: System MUST gracefully degrade when iris_vector_graph unavailable.

        Given: iris_vector_graph not installed/available (mocked)
        When: HybridGraphRAG pipeline initializes
        Then: Initialization succeeds and all methods work via fallback
        """
        caplog.set_level(logging.INFO)

        query = "What are the risk factors for diabetes?"

        # Mock all iris_vector_graph methods to simulate unavailability
        if hasattr(graphrag_pipeline, 'retrieval_methods'):
            mocker.patch.object(
                graphrag_pipeline.retrieval_methods,
                'retrieve_via_hybrid_fusion',
                side_effect=AttributeError("iris_vector_graph not available")
            )

        # Execute query - should still work via fallback
        result = graphrag_pipeline.query(query, method="hybrid")

        # Verify query succeeded
        assert result is not None, "Query should succeed despite iris_vector_graph unavailable"
        assert len(result['contexts']) > 0, \
            "Should retrieve documents via fallback"

        # Verify fallback was used
        assert result['metadata']['retrieval_method'] == 'vector_fallback', \
            "Should use vector_fallback when iris_vector_graph unavailable"

        # System should remain functional
        # Try another query to verify state consistency
        result2 = graphrag_pipeline.query("diabetes treatment", method="hybrid")
        assert len(result2.contexts) > 0, \
            "Pipeline should remain functional after graceful degradation"
