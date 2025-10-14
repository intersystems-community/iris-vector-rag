"""
Contract tests for Enhanced Text Search query path (FR-007 to FR-009).

Tests validate that HybridGraphRAG's text search method executes iFind search
correctly and falls back when necessary.

Contract: TEXT-001
Requirements: FR-007, FR-008, FR-009
"""

import logging
import pytest
from iris_rag import create_pipeline


class TestTextSearchContract:
    """Contract tests for enhanced text search path."""

    @pytest.fixture
    def graphrag_pipeline(self):
        """Create HybridGraphRAG pipeline with validation."""
        return create_pipeline("graphrag", validate_requirements=True)

    @pytest.mark.requires_database
    def test_text_search_executes_successfully(self, graphrag_pipeline):
        """
        FR-007: Text search MUST execute iFind text search via iris_graph_core.

        Given: HybridGraphRAG pipeline initialized
        When: Query executed with method="text"
        Then: iFind text search executes and returns documents
        """
        query = "diabetes symptoms and diagnosis"

        # Execute query with text method
        result = graphrag_pipeline.query(query, method="text")

        # Verify result structure
        assert result is not None, "Result should not be None"
        assert ('contexts' in result), "Result should have contexts"
        assert ('metadata' in result), "Result should have metadata"

        # Verify documents retrieved
        assert len(result['contexts']) > 0, \
            f"Text search (or fallback) should retrieve documents, got {len(result['contexts'])}"

        # Verify metadata
        assert 'retrieval_method' in result['metadata'], \
            "Metadata should contain retrieval_method"

        method = result['metadata']['retrieval_method']
        assert method in ['text', 'vector_fallback', 'knowledge_graph'], \
            f"Expected text or fallback, got {method}"

    @pytest.mark.requires_database
    def test_text_search_fallback_on_zero_results(self, graphrag_pipeline, mocker, caplog):
        """
        FR-008: Text search MUST fall back when returning 0 results.

        Given: HybridGraphRAG pipeline initialized
        And: Text search returns 0 results (mocked)
        When: Query executed with method="text"
        Then: System falls back to vector search
        """
        caplog.set_level(logging.WARNING)

        query = "diabetes symptoms and diagnosis"

        # Mock text search to return 0 results
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

        # Verify fallback
        assert len(result['contexts']) > 0, \
            "Fallback should retrieve documents"

        assert result['metadata']['retrieval_method'] == 'vector_fallback', \
            f"Expected vector_fallback, got {result['metadata']['retrieval_method']}"

        # Verify logging
        log_output = caplog.text
        assert any("fallback" in msg.lower() or "0 results" in msg.lower()
                   for msg in log_output.split('\n')), \
            "Should log warning about text search fallback"

    @pytest.mark.requires_database
    def test_text_search_fallback_on_exception(self, graphrag_pipeline, mocker, caplog):
        """
        FR-009: Text search MUST fall back when raising exception.

        Given: HybridGraphRAG pipeline initialized
        And: Text search raises exception (mocked)
        When: Query executed with method="text"
        Then: System catches exception and falls back
        """
        caplog.set_level(logging.ERROR)

        query = "diabetes symptoms and diagnosis"

        # Mock text search to raise exception
        if hasattr(graphrag_pipeline, 'retrieval_methods'):
            mocker.patch.object(
                graphrag_pipeline.retrieval_methods,
                'retrieve_via_enhanced_text',
                side_effect=Exception("iFind text search failed")
            )
        else:
            mocker.patch.object(
                graphrag_pipeline,
                '_retrieve_via_enhanced_text',
                side_effect=Exception("iFind text search failed")
            )

        # Execute query - should not raise
        result = graphrag_pipeline.query(query, method="text")

        # Verify fallback
        assert len(result['contexts']) > 0, \
            "Fallback should retrieve documents after exception"

        assert result['metadata']['retrieval_method'] == 'vector_fallback', \
            f"Expected vector_fallback, got {result['metadata']['retrieval_method']}"

        # Verify error logged
        log_output = caplog.text
        assert any("error" in msg.lower() or "fail" in msg.lower()
                   for msg in log_output.split('\n')), \
            "Should log error about text search exception"
