"""
Contract tests for HybridGraphRAG import requirements.

These tests verify that iris-vector-graph is required and proper
error messages are provided when it's missing.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestImportRequirements:
    """Contract tests for import behavior."""

    @patch('builtins.__import__')
    def test_missing_iris_vector_graph_raises_import_error(self, mock_import):
        """When iris-vector-graph is not installed, ImportError must be raised."""
        # Simulate missing package
        def import_mock(name, *args, **kwargs):
            if 'iris_vector_graph' in name:
                raise ImportError("No module named 'iris_vector_graph'")
            return MagicMock()

        mock_import.side_effect = import_mock

        # This test should FAIL until implementation is complete
        from iris_rag.pipelines.hybrid_graphrag_discovery import GraphCoreDiscovery

        discovery = GraphCoreDiscovery()

        with pytest.raises(ImportError) as exc_info:
            discovery.import_graph_core_modules()

        assert "HybridGraphRAG requires iris-vector-graph package" in str(exc_info.value)
        assert "pip install rag-templates[hybrid-graphrag]" in str(exc_info.value)

    def test_successful_import_with_package_installed(self):
        """When iris-vector-graph is installed, imports should succeed."""
        # This test should FAIL until implementation is complete
        from iris_rag.pipelines.hybrid_graphrag_discovery import GraphCoreDiscovery

        discovery = GraphCoreDiscovery()
        modules = discovery.import_graph_core_modules()

        assert "IRISGraphEngine" in modules
        assert "HybridSearchFusion" in modules
        assert "TextSearchEngine" in modules
        assert "VectorOptimizer" in modules

    def test_no_fallback_to_local_paths(self):
        """Discovery should not fall back to local development paths."""
        # This test should FAIL until implementation is complete
        from iris_rag.pipelines.hybrid_graphrag_discovery import GraphCoreDiscovery

        discovery = GraphCoreDiscovery()

        # Should not have methods for path discovery
        assert not hasattr(discovery, 'discover_graph_core_path')
        assert not hasattr(discovery, '_search_sibling_directories')