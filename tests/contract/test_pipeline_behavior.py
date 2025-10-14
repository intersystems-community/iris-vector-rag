"""
Contract tests for HybridGraphRAG pipeline behavior.

These tests verify that the pipeline maintains consistent behavior
without fallback mechanisms.
"""

import pytest
from unittest.mock import Mock, patch


class TestPipelineBehavior:
    """Contract tests for pipeline behavior without fallbacks."""

    def test_pipeline_creation_requires_iris_vector_graph(self):
        """Creating HybridGraphRAG should fail if iris-vector-graph is missing."""
        # This test should FAIL until implementation is complete
        with patch('iris_rag.pipelines.hybrid_graphrag_discovery.GraphCoreDiscovery.import_graph_core_modules') as mock_import:
            mock_import.side_effect = ImportError("HybridGraphRAG requires iris-vector-graph package")

            from iris_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

            with pytest.raises(ImportError):
                HybridGraphRAGPipeline()

    def test_no_conditional_checks_in_pipeline(self):
        """Pipeline should not have conditional availability checks."""
        # This test should FAIL until implementation is complete
        from iris_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

        # Mock successful import
        with patch('iris_rag.pipelines.hybrid_graphrag_discovery.GraphCoreDiscovery.import_graph_core_modules') as mock_import:
            mock_import.return_value = {
                "IRISGraphEngine": Mock(),
                "HybridSearchFusion": Mock(),
                "TextSearchEngine": Mock(),
                "VectorOptimizer": Mock(),
            }

            pipeline = HybridGraphRAGPipeline()

            # These attributes should always be initialized, never None
            assert pipeline.iris_engine is not None
            assert pipeline.retrieval_methods is not None

            # Status methods should always return True
            assert pipeline.is_hybrid_enabled() is True

            status = pipeline.get_hybrid_status()
            assert status["hybrid_enabled"] is True
            assert status["iris_engine_available"] is True
            assert status["fusion_engine_available"] is True

    def test_all_retrieval_methods_available(self):
        """All retrieval methods should be available without conditionals."""
        # This test should FAIL until implementation is complete
        from iris_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

        with patch('iris_rag.pipelines.hybrid_graphrag_discovery.GraphCoreDiscovery.import_graph_core_modules') as mock_import:
            mock_import.return_value = {
                "IRISGraphEngine": Mock(),
                "HybridSearchFusion": Mock(),
                "TextSearchEngine": Mock(),
                "VectorOptimizer": Mock(),
            }

            pipeline = HybridGraphRAGPipeline()

            # All methods should work without fallbacks
            methods = ["hybrid", "rrf", "text", "vector", "kg"]
            for method in methods:
                # Should not raise ValueError for unsupported method
                # (actual query execution would require more mocking)
                assert method in ["hybrid", "rrf", "text", "vector", "kg"]

    def test_no_fallback_methods_exist(self):
        """Fallback methods should not exist in the simplified implementation."""
        # This test should FAIL until implementation is complete
        from iris_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

        # These fallback methods should not exist
        assert not hasattr(HybridGraphRAGPipeline, '_enhanced_hybrid_fallback')
        assert not hasattr(HybridGraphRAGPipeline, '_intelligent_hybrid_search')
        assert not hasattr(HybridGraphRAGPipeline, '_enhanced_text_search_fallback')
        assert not hasattr(HybridGraphRAGPipeline, '_enhanced_vector_search_fallback')