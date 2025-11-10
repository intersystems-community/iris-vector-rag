"""
Unit tests for HybridGraphRAG pipeline.

Tests focus on import behavior and error handling when iris-vector-graph
is not available.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestHybridGraphRAGImportBehavior:
    """Test import behavior and error handling."""

    def test_pipeline_creation_fails_without_iris_vector_graph(self):
        """Pipeline creation should fail with clear error when iris-vector-graph is missing."""
        with patch('iris_rag.pipelines.hybrid_graphrag_discovery.GraphCoreDiscovery.import_graph_core_modules') as mock_import:
            mock_import.side_effect = ImportError(
                "HybridGraphRAG requires iris-vector-graph package. "
                "Install with: pip install rag-templates[hybrid-graphrag]"
            )

            from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

            with pytest.raises(ImportError) as exc_info:
                HybridGraphRAGPipeline()

            assert "iris-vector-graph package" in str(exc_info.value)
            assert "pip install rag-templates[hybrid-graphrag]" in str(exc_info.value)

    def test_create_pipeline_fails_for_graphrag_without_package(self):
        """create_pipeline should fail when requesting graphrag without iris-vector-graph."""
        with patch('iris_rag.pipelines.hybrid_graphrag_discovery.GraphCoreDiscovery.import_graph_core_modules') as mock_import:
            mock_import.side_effect = ImportError(
                "HybridGraphRAG requires iris-vector-graph package. "
                "Install with: pip install rag-templates[hybrid-graphrag]"
            )

            from iris_vector_rag import create_pipeline

            with pytest.raises(ImportError) as exc_info:
                create_pipeline("graphrag")

            assert "iris-vector-graph package" in str(exc_info.value)

    def test_import_error_message_is_actionable(self):
        """Import error should provide clear installation instructions."""
        with patch('builtins.__import__') as mock_import:
            def import_mock(name, *args, **kwargs):
                if 'iris_vector_graph' in name:
                    raise ImportError(f"No module named '{name}'")
                # Return real module for other imports
                return __import__(name, *args, **kwargs)

            mock_import.side_effect = import_mock

            from iris_vector_rag.pipelines.hybrid_graphrag_discovery import GraphCoreDiscovery

            discovery = GraphCoreDiscovery()

            with pytest.raises(ImportError) as exc_info:
                discovery.import_graph_core_modules()

            error_msg = str(exc_info.value)
            assert "HybridGraphRAG requires iris-vector-graph package" in error_msg
            assert "pip install rag-templates[hybrid-graphrag]" in error_msg

    def test_no_graceful_degradation_without_package(self):
        """Pipeline should not degrade gracefully - it should fail fast."""
        with patch('iris_rag.pipelines.hybrid_graphrag_discovery.GraphCoreDiscovery.import_graph_core_modules') as mock_import:
            mock_import.side_effect = ImportError(
                "HybridGraphRAG requires iris-vector-graph package. "
                "Install with: pip install rag-templates[hybrid-graphrag]"
            )

            from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

            # Should not be able to create pipeline at all
            with pytest.raises(ImportError):
                pipeline = HybridGraphRAGPipeline()

            # Should not reach this point
            # No pipeline instance means no graceful degradation

    def test_successful_import_with_package_available(self):
        """When iris-vector-graph is available, import should succeed."""
        # Mock successful import
        mock_modules = {
            "IRISGraphEngine": MagicMock(),
            "HybridSearchFusion": MagicMock(),
            "TextSearchEngine": MagicMock(),
            "VectorOptimizer": MagicMock(),
        }

        with patch('iris_rag.pipelines.hybrid_graphrag_discovery.GraphCoreDiscovery.import_graph_core_modules') as mock_import:
            mock_import.return_value = mock_modules

            from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

            # Should create pipeline successfully
            pipeline = HybridGraphRAGPipeline()

            # Should have all components initialized
            assert hasattr(pipeline, 'iris_engine')
            assert hasattr(pipeline, 'retrieval_methods')

            # Should report as hybrid enabled
            assert pipeline.is_hybrid_enabled() is True