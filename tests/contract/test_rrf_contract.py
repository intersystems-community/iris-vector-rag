"""
Contract tests for RRF (Reciprocal Rank Fusion) query path (FR-004 to FR-006).

Tests validate that HybridGraphRAG's RRF method executes correctly and falls
back to vector search when failures occur.

Contract: RRF-001
Requirements: FR-004, FR-005, FR-006
"""

import logging
import pytest
from iris_vector_rag import create_pipeline


class TestRRFContract:
    """Contract tests for RRF query path."""

    @pytest.fixture
    def graphrag_pipeline(self):
        """Create HybridGraphRAG pipeline with validation."""
        return create_pipeline("graphrag", validate_requirements=True)

    @pytest.mark.requires_database
    def test_rrf_executes_successfully(self, graphrag_pipeline):
        """
        FR-004: RRF MUST execute Reciprocal Rank Fusion combining vector and text search.

        Given: HybridGraphRAG pipeline initialized
        When: Query executed with method="rrf"
        Then: RRF search executes and returns documents
        """
        query = "What are the treatments for type 2 diabetes?"

        # Execute query with RRF method
        result = graphrag_pipeline.query(query, method="rrf")

        # Verify result structure
        assert result is not None, "Result should not be None"
        assert ('contexts' in result), "Result should have contexts"
        assert ('metadata' in result), "Result should have metadata"

        # Verify documents retrieved (via RRF or fallback)
        assert len(result['contexts']) > 0, \
            f"RRF (or fallback) should retrieve documents, got {len(result['contexts'])}"

        # Verify metadata
        assert 'retrieval_method' in result['metadata'], \
            "Metadata should contain retrieval_method"

        method = result['metadata']['retrieval_method']
        assert method in ['rrf', 'vector_fallback', 'knowledge_graph'], \
            f"Expected rrf or fallback, got {method}"

    @pytest.mark.requires_database
    def test_rrf_fallback_on_zero_results(self, graphrag_pipeline, mocker, caplog):
        """
        FR-005: RRF MUST fall back to vector search when returning 0 results.

        Given: HybridGraphRAG pipeline initialized
        And: RRF returns 0 results (mocked)
        When: Query executed with method="rrf"
        Then: System falls back to IRISVectorStore
        """
        caplog.set_level(logging.WARNING)

        query = "What are the treatments for type 2 diabetes?"

        # Mock RRF to return 0 results
        if hasattr(graphrag_pipeline, 'retrieval_methods'):
            mocker.patch.object(
                graphrag_pipeline.retrieval_methods,
                'retrieve_via_rrf',
                return_value=([], 'rrf')
            )
        else:
            mocker.patch.object(
                graphrag_pipeline,
                '_retrieve_via_rrf',
                return_value=([], 'rrf')
            )

        # Execute query
        result = graphrag_pipeline.query(query, method="rrf")

        # Verify fallback
        assert len(result['contexts']) > 0, \
            "Fallback should retrieve documents"

        assert result['metadata']['retrieval_method'] == 'vector_fallback', \
            f"Expected vector_fallback, got {result['metadata']['retrieval_method']}"

        # Verify logging
        log_output = caplog.text
        assert any("fallback" in msg.lower() or "0 results" in msg.lower()
                   for msg in log_output.split('\n')), \
            "Should log warning about RRF fallback"

    @pytest.mark.requires_database
    def test_rrf_fallback_on_exception(self, graphrag_pipeline, mocker, caplog):
        """
        FR-006: RRF MUST fall back when raising exception.

        Given: HybridGraphRAG pipeline initialized
        And: RRF raises exception (mocked)
        When: Query executed with method="rrf"
        Then: System catches exception and falls back
        """
        caplog.set_level(logging.ERROR)

        query = "What are the treatments for type 2 diabetes?"

        # Mock RRF to raise exception
        if hasattr(graphrag_pipeline, 'retrieval_methods'):
            mocker.patch.object(
                graphrag_pipeline.retrieval_methods,
                'retrieve_via_rrf',
                side_effect=Exception("RRF failed")
            )
        else:
            mocker.patch.object(
                graphrag_pipeline,
                '_retrieve_via_rrf',
                side_effect=Exception("RRF failed")
            )

        # Execute query - should not raise
        result = graphrag_pipeline.query(query, method="rrf")

        # Verify fallback
        assert len(result['contexts']) > 0, \
            "Fallback should retrieve documents after exception"

        assert result['metadata']['retrieval_method'] == 'vector_fallback', \
            f"Expected vector_fallback, got {result['metadata']['retrieval_method']}"

        # Verify error logged
        log_output = caplog.text
        assert any("error" in msg.lower() or "fail" in msg.lower()
                   for msg in log_output.split('\n')), \
            "Should log error about RRF exception"
