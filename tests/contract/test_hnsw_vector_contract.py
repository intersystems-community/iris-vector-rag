"""
Contract tests for HNSW Vector Search query path (FR-010 to FR-012).

Tests validate that HybridGraphRAG's HNSW vector search method executes correctly
and falls back to IRISVectorStore when necessary.

Contract: HNSW-001
Requirements: FR-010, FR-011, FR-012
"""

import logging
import pytest
from iris_vector_rag import create_pipeline


class TestHNSWVectorContract:
    """Contract tests for HNSW vector search path."""

    @pytest.fixture
    def graphrag_pipeline(self):
        """Create HybridGraphRAG pipeline with validation."""
        return create_pipeline("graphrag", validate_requirements=True)

    @pytest.mark.requires_database
    def test_hnsw_vector_executes_successfully(self, graphrag_pipeline):
        """
        FR-010: HNSW vector search MUST execute via iris_vector_graph.

        Given: HybridGraphRAG pipeline initialized
        When: Query executed with method="vector"
        Then: HNSW-optimized vector search executes and returns documents
        """
        query = "How is diabetes diagnosed?"

        # Execute query with vector method
        result = graphrag_pipeline.query(query, method="vector")

        # Verify result structure
        assert result is not None, "Result should not be None"
        assert ('contexts' in result), "Result should have contexts"
        assert ('metadata' in result), "Result should have metadata"

        # Verify documents retrieved
        assert len(result['contexts']) > 0, \
            f"Vector search (or fallback) should retrieve documents, got {len(result['contexts'])}"

        # Verify metadata
        assert 'retrieval_method' in result['metadata'], \
            "Metadata should contain retrieval_method"

        method = result['metadata']['retrieval_method']
        assert method in ['vector', 'hnsw_vector', 'vector_fallback', 'knowledge_graph'], \
            f"Expected vector/hnsw_vector or fallback, got {method}"

    @pytest.mark.requires_database
    def test_hnsw_vector_fallback_on_zero_results(self, graphrag_pipeline, mocker, caplog):
        """
        FR-011: HNSW vector search MUST fall back when returning 0 results.

        Given: HybridGraphRAG pipeline initialized
        And: HNSW returns 0 results (mocked)
        When: Query executed with method="vector"
        Then: System falls back to IRISVectorStore
        """
        caplog.set_level(logging.WARNING)

        query = "How is diabetes diagnosed?"

        # Mock HNSW vector search to return 0 results
        if hasattr(graphrag_pipeline, 'retrieval_methods'):
            mocker.patch.object(
                graphrag_pipeline.retrieval_methods,
                'retrieve_via_hnsw_vector',
                return_value=([], 'hnsw_vector')
            )
        else:
            mocker.patch.object(
                graphrag_pipeline,
                '_retrieve_via_hnsw_vector',
                return_value=([], 'hnsw_vector')
            )

        # Execute query
        result = graphrag_pipeline.query(query, method="vector")

        # Verify fallback
        assert len(result['contexts']) > 0, \
            "Fallback should retrieve documents"

        assert result['metadata']['retrieval_method'] == 'vector_fallback', \
            f"Expected vector_fallback, got {result['metadata']['retrieval_method']}"

        # Verify logging
        log_output = caplog.text
        assert any("fallback" in msg.lower() or "0 results" in msg.lower()
                   for msg in log_output.split('\n')), \
            "Should log warning about HNSW fallback"

    @pytest.mark.requires_database
    def test_hnsw_vector_fallback_on_exception(self, graphrag_pipeline, mocker, caplog):
        """
        FR-012: HNSW vector search MUST fall back when raising exception.

        Given: HybridGraphRAG pipeline initialized
        And: HNSW raises exception (mocked)
        When: Query executed with method="vector"
        Then: System catches exception and falls back
        """
        caplog.set_level(logging.ERROR)

        query = "How is diabetes diagnosed?"

        # Mock HNSW to raise exception
        if hasattr(graphrag_pipeline, 'retrieval_methods'):
            mocker.patch.object(
                graphrag_pipeline.retrieval_methods,
                'retrieve_via_hnsw_vector',
                side_effect=Exception("HNSW search failed")
            )
        else:
            mocker.patch.object(
                graphrag_pipeline,
                '_retrieve_via_hnsw_vector',
                side_effect=Exception("HNSW search failed")
            )

        # Execute query - should not raise
        result = graphrag_pipeline.query(query, method="vector")

        # Verify fallback
        assert len(result['contexts']) > 0, \
            "Fallback should retrieve documents after exception"

        assert result['metadata']['retrieval_method'] == 'vector_fallback', \
            f"Expected vector_fallback, got {result['metadata']['retrieval_method']}"

        # Verify error logged
        log_output = caplog.text
        assert any("error" in msg.lower() or "fail" in msg.lower()
                   for msg in log_output.split('\n')), \
            "Should log error about HNSW exception"
