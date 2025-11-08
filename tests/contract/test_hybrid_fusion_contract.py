"""
Contract tests for Hybrid Fusion query path (FR-001 to FR-003).

Tests validate that HybridGraphRAG's hybrid fusion method executes correctly
and falls back to vector search when iris_vector_graph fails or returns 0 results.

Contract: HYBRID-001
Requirements: FR-001, FR-002, FR-003
"""

import logging
import pytest
from iris_rag import create_pipeline


class TestHybridFusionContract:
    """Contract tests for hybrid fusion query path."""

    @pytest.fixture
    def graphrag_pipeline(self):
        """Create HybridGraphRAG pipeline with validation."""
        return create_pipeline("graphrag", validate_requirements=True)

    @pytest.mark.requires_database
    def test_hybrid_fusion_executes_successfully(self, graphrag_pipeline):
        """
        FR-001: Hybrid fusion MUST execute multi-modal search when available.

        Given: HybridGraphRAG pipeline initialized with iris_vector_graph
        When: Query executed with method="hybrid"
        Then: Hybrid fusion search executes and returns documents
        """
        query = "What are the symptoms of diabetes?"

        # Execute query with hybrid method
        result = graphrag_pipeline.query(query, method="hybrid")

        # Verify result structure
        assert result is not None, "Result should not be None"
        assert isinstance(result, dict), "Result should be a dictionary"

        # Verify documents retrieved (may be via fallback if iris_vector_graph unavailable)
        assert 'contexts' in result, "Result should contain contexts"
        assert len(result['contexts']) > 0, \
            f"Hybrid fusion (or fallback) should retrieve documents, got {len(result['contexts'])}"

        # Verify metadata contains retrieval_method
        assert 'metadata' in result, "Result should contain metadata"
        assert 'retrieval_method' in result['metadata'], \
            "Metadata should contain retrieval_method key"

        # Retrieval method should be either hybrid_fusion or vector_fallback
        method = result['metadata']['retrieval_method']
        assert method in ['hybrid_fusion', 'vector_fallback', 'hybrid', 'knowledge_graph'], \
            f"Expected hybrid_fusion or fallback method, got {method}"

    @pytest.mark.requires_database
    def test_hybrid_fusion_fallback_on_zero_results(self, graphrag_pipeline, mocker, caplog):
        """
        FR-002: Hybrid fusion MUST fall back to vector search when returning 0 results.

        Given: HybridGraphRAG pipeline initialized
        And: iris_vector_graph hybrid fusion returns 0 results (mocked)
        When: Query executed with method="hybrid"
        Then: System falls back to IRISVectorStore and retrieves documents
        """
        caplog.set_level(logging.WARNING)

        query = "What are the symptoms of diabetes?"

        # Mock retrieval_methods.retrieve_via_hybrid_fusion to return 0 results
        if hasattr(graphrag_pipeline, 'retrieval_methods'):
            mocker.patch.object(
                graphrag_pipeline.retrieval_methods,
                'retrieve_via_hybrid_fusion',
                return_value=([], 'hybrid_fusion')
            )
        else:
            # If retrieval_methods not available, mock _retrieve_via_hybrid_fusion directly
            mocker.patch.object(
                graphrag_pipeline,
                '_retrieve_via_hybrid_fusion',
                return_value=([], 'hybrid_fusion')
            )

        # Execute query
        result = graphrag_pipeline.query(query, method="hybrid")

        # Verify fallback occurred
        assert len(result['contexts']) > 0, \
            "Fallback to vector search should retrieve documents"

        # Verify metadata indicates fallback
        assert result['metadata']['retrieval_method'] == 'vector_fallback', \
            f"Expected vector_fallback, got {result['metadata']['retrieval_method']}"

        # Verify warning log
        log_output = caplog.text
        assert any("fallback" in msg.lower() or "0 results" in msg.lower()
                   for msg in log_output.split('\n')), \
            "Should log warning about fallback"

    @pytest.mark.requires_database
    def test_hybrid_fusion_fallback_on_exception(self, graphrag_pipeline, mocker, caplog):
        """
        FR-003: Hybrid fusion MUST fall back when iris_vector_graph raises exception.

        Given: HybridGraphRAG pipeline initialized
        And: iris_vector_graph hybrid fusion raises exception (mocked)
        When: Query executed with method="hybrid"
        Then: System catches exception, falls back to vector search
        """
        caplog.set_level(logging.ERROR)

        query = "What are the symptoms of diabetes?"

        # Mock retrieval_methods to raise exception
        if hasattr(graphrag_pipeline, 'retrieval_methods'):
            mocker.patch.object(
                graphrag_pipeline.retrieval_methods,
                'retrieve_via_hybrid_fusion',
                side_effect=Exception("iris_vector_graph connection failed")
            )
        else:
            mocker.patch.object(
                graphrag_pipeline,
                '_retrieve_via_hybrid_fusion',
                side_effect=Exception("iris_vector_graph connection failed")
            )

        # Execute query - should not raise exception
        result = graphrag_pipeline.query(query, method="hybrid")

        # Verify fallback succeeded
        assert len(result['contexts']) > 0, \
            "Fallback to vector search should retrieve documents after exception"

        # Verify metadata indicates fallback
        assert result['metadata']['retrieval_method'] == 'vector_fallback', \
            f"Expected vector_fallback after exception, got {result['metadata']['retrieval_method']}"

        # Verify error was logged
        log_output = caplog.text
        assert any("error" in msg.lower() or "fail" in msg.lower()
                   for msg in log_output.split('\n')), \
            "Should log error message about exception"
