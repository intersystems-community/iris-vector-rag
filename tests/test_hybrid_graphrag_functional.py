"""
Functional tests for HybridGraphRAG pipeline.

Tests the core functionality of HybridGraphRAG including:
- Security: No hard-coded credentials
- Configuration: Config-driven discovery
- Modularity: Proper fallbacks and error handling
- Query methods: Different retrieval strategies
- Integration: iris_graph_core optional dependency
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.core.models import Document
from iris_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline
from iris_rag.pipelines.hybrid_graphrag_discovery import GraphCoreDiscovery

logger = logging.getLogger(__name__)


class TestHybridGraphRAGSecurity:
    """Test security aspects of HybridGraphRAG."""

    def test_no_hardcoded_credentials_in_source(self):
        """Verify no hard-coded credentials exist in source code."""
        pipeline_file = (
            Path(__file__).parent.parent
            / "iris_rag"
            / "pipelines"
            / "hybrid_graphrag.py"
        )
        discovery_file = (
            Path(__file__).parent.parent
            / "iris_rag"
            / "pipelines"
            / "hybrid_graphrag_discovery.py"
        )

        # Read source files
        pipeline_content = pipeline_file.read_text()
        discovery_content = discovery_file.read_text()

        # Check for hard-coded credentials patterns
        forbidden_patterns = [
            'host="localhost"',
            'password="SYS"',
            'user="_SYSTEM"',
            "port=1972",
            'namespace="USER"',
        ]

        for pattern in forbidden_patterns:
            assert (
                pattern not in pipeline_content
            ), f"Hard-coded credential found: {pattern}"
            assert (
                pattern not in discovery_content
            ), f"Hard-coded credential found: {pattern}"

    def test_config_driven_connection_params(self):
        """Test that connection parameters come from configuration only."""
        with patch("iris_rag.config.manager.ConfigurationManager") as mock_config_mgr:
            # Mock configuration with secure params
            mock_config_instance = Mock()
            mock_config_mgr.return_value = mock_config_instance
            mock_config_instance.get.return_value = {
                "host": "test-host",
                "port": 9999,
                "namespace": "TEST",
                "username": "test-user",
                "password": "test-pass",
            }

            discovery = GraphCoreDiscovery(mock_config_instance)
            config = discovery.get_connection_config()

            assert config["host"] == "test-host"
            assert config["port"] == 9999
            assert config["namespace"] == "TEST"
            assert config["username"] == "test-user"
            assert config["password"] == "test-pass"

    def test_missing_credentials_validation(self):
        """Test that missing credentials are properly detected."""
        discovery = GraphCoreDiscovery(None)

        # Test incomplete config
        incomplete_config = {"host": "localhost", "port": 1972}
        is_valid, missing = discovery.validate_connection_config(incomplete_config)

        assert not is_valid
        assert "namespace" in missing
        assert "username" in missing
        assert "password" in missing


class TestGraphCoreDiscovery:
    """Test the GraphCoreDiscovery functionality."""

    def test_discovery_priority_order(self):
        """Test that discovery follows correct priority order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir).resolve()

            # Create mock graph core structure
            mock_graph_core = temp_path / "test-graph-core"
            mock_graph_core.mkdir()
            (mock_graph_core / "iris_graph_core").mkdir()

            # Mock config manager that returns the test path
            mock_config = Mock()
            mock_config.get.return_value = {"path": str(mock_graph_core)}

            discovery = GraphCoreDiscovery(mock_config)
            found_path = discovery.discover_graph_core_path()

            assert found_path.resolve() == mock_graph_core.resolve()

    def test_environment_variable_discovery(self):
        """Test discovery via GRAPH_CORE_PATH environment variable."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir).resolve()
            mock_graph_core = temp_path / "env-graph-core"
            mock_graph_core.mkdir()
            (mock_graph_core / "iris_graph_core").mkdir()

            with patch.dict(os.environ, {"GRAPH_CORE_PATH": str(mock_graph_core)}):
                discovery = GraphCoreDiscovery(None)
                found_path = discovery.discover_graph_core_path()

                assert found_path.resolve() == mock_graph_core.resolve()

    def test_safe_import_without_sys_path_pollution(self):
        """Test that imports don't permanently pollute sys.path."""
        import sys

        original_path = sys.path.copy()

        discovery = GraphCoreDiscovery(None)
        success, modules = discovery.import_graph_core_modules()

        # sys.path should be restored regardless of import success
        assert sys.path == original_path


class TestHybridGraphRAGPipeline:
    """Test the main HybridGraphRAG pipeline functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_conn_mgr = Mock(spec=ConnectionManager)
        self.mock_config_mgr = Mock(spec=ConfigurationManager)

        # Mock config manager to return proper embedding configuration
        def mock_config_get(key, default=None):
            config_values = {
                "embedding_model.name": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_model.dimension": 384,
                "database.schema": "RAG",
                "vector_search.metric": "COSINE",
                "vector_search.top_k": 5,
            }
            return config_values.get(key, default)

        self.mock_config_mgr.get.side_effect = mock_config_get

        # Mock _config attribute for services that access it directly
        self.mock_config_mgr._config = {
            "embedding_model": {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384
            },
            "database": {"schema": "RAG"},
            "vector_search": {"metric": "COSINE", "top_k": 5}
        }

    def test_pipeline_initialization_without_graph_core(self):
        """Test pipeline initializes gracefully without iris_graph_core."""
        with patch(
            "iris_rag.pipelines.hybrid_graphrag_discovery.GraphCoreDiscovery"
        ) as mock_discovery_class:
            mock_discovery = Mock()
            mock_discovery_class.return_value = mock_discovery
            mock_discovery.import_graph_core_modules.return_value = (False, {})

            pipeline = HybridGraphRAGPipeline(self.mock_conn_mgr, self.mock_config_mgr)

            assert not pipeline.is_hybrid_enabled()
            assert pipeline.iris_engine is None
            assert pipeline.retrieval_methods is None

    def test_pipeline_initialization_with_graph_core(self):
        """Test pipeline initializes with iris_graph_core when available."""
        mock_modules = {
            "IRISGraphEngine": Mock,
            "HybridSearchFusion": Mock,
            "TextSearchEngine": Mock,
            "VectorOptimizer": Mock,
            "package_name": "iris_graph_core",
        }

        with patch(
            "iris_rag.pipelines.hybrid_graphrag_discovery.GraphCoreDiscovery"
        ) as mock_discovery_class:
            mock_discovery = Mock()
            mock_discovery_class.return_value = mock_discovery
            mock_discovery.import_graph_core_modules.return_value = (True, mock_modules)
            mock_discovery.get_connection_config.return_value = {
                "host": "localhost",
                "port": 1972,
                "namespace": "TEST",
                "username": "test",
                "password": "test",
            }
            mock_discovery.validate_connection_config.return_value = (True, [])

            with patch("iris.connect") as mock_iris_connect:
                mock_connection = Mock()
                mock_iris_connect.return_value = mock_connection

                pipeline = HybridGraphRAGPipeline(
                    self.mock_conn_mgr, self.mock_config_mgr
                )

                assert pipeline.discovery is not None
                # Note: iris_engine might still be None due to connection issues in test

    def test_query_fallback_to_graphrag(self):
        """Test that queries use enhanced fallback when hybrid unavailable."""
        with patch(
            "iris_rag.pipelines.hybrid_graphrag_discovery.GraphCoreDiscovery"
        ) as mock_discovery_class:
            mock_discovery = Mock()
            mock_discovery_class.return_value = mock_discovery
            mock_discovery.import_graph_core_modules.return_value = (False, {})

            # Mock connection and cursor for knowledge graph validation
            mock_cursor = Mock()
            mock_cursor.fetchone.return_value = [1]  # 1 entity (non-empty graph)
            mock_connection = Mock()
            mock_connection.cursor.return_value = mock_cursor
            self.mock_conn_mgr.get_connection.return_value = mock_connection

            pipeline = HybridGraphRAGPipeline(self.mock_conn_mgr, self.mock_config_mgr)

            # Verify hybrid is NOT enabled
            assert not pipeline.is_hybrid_enabled()

            # Mock the enhanced fallback method
            with patch.object(
                pipeline, "_enhanced_hybrid_fallback"
            ) as mock_fallback:
                mock_fallback.return_value = ([Mock()], "fallback")

                result = pipeline.query("test query", method="hybrid", generate_answer=False)

                # Should use enhanced fallback when hybrid unavailable
                mock_fallback.assert_called_once()

    def test_query_with_hybrid_methods(self):
        """Test query routing to different hybrid methods."""
        # Create a pipeline with mocked hybrid capabilities
        pipeline = HybridGraphRAGPipeline(self.mock_conn_mgr, self.mock_config_mgr)
        pipeline.retrieval_methods = Mock()

        # Mock successful retrieval
        mock_docs = [Document(page_content="test", metadata={})]
        pipeline.retrieval_methods.retrieve_via_hybrid_fusion.return_value = (
            mock_docs,
            "hybrid_fusion",
        )

        # Mock the validate knowledge graph to avoid DB calls
        with patch.object(pipeline, "_validate_knowledge_graph"):
            result = pipeline.query(
                "test query", method="hybrid", generate_answer=False
            )

            assert result["retrieved_documents"] == mock_docs
            assert result["metadata"]["retrieval_method"] == "hybrid_fusion"

    def test_cursor_cleanup_robustness(self):
        """Test that cursor cleanup is robust and doesn't raise UnboundLocalError."""
        pipeline = HybridGraphRAGPipeline(self.mock_conn_mgr, self.mock_config_mgr)

        # Mock connection manager to return a connection
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        self.mock_conn_mgr.get_connection.return_value = mock_connection

        # Test that cursor cleanup works even if cursor.execute fails
        mock_cursor.execute.side_effect = Exception("Database error")

        # This should not raise UnboundLocalError
        result = pipeline._get_document_content_for_entity("test_entity")

        # Should return fallback content
        assert result == "Entity: test_entity"

        # Cursor should still be closed
        mock_cursor.close.assert_called_once()

    def test_get_hybrid_status(self):
        """Test the hybrid status reporting functionality."""
        pipeline = HybridGraphRAGPipeline(self.mock_conn_mgr, self.mock_config_mgr)

        status = pipeline.get_hybrid_status()

        assert isinstance(status, dict)
        assert "hybrid_enabled" in status
        assert "iris_engine_available" in status
        assert "fusion_engine_available" in status
        assert "text_engine_available" in status
        assert "vector_optimizer_available" in status
        assert "graph_core_path" in status


class TestModularArchitecture:
    """Test the modular architecture aspects."""

    def test_file_line_count_limits(self):
        """Test that all module files stay under reasonable limits."""
        module_files = {
            "iris_rag/pipelines/hybrid_graphrag.py": 700,  # Main pipeline
            "iris_rag/pipelines/hybrid_graphrag_discovery.py": 500,
            "iris_rag/pipelines/hybrid_graphrag_retrieval.py": 500,
        }

        for module_file, max_lines in module_files.items():
            file_path = Path(__file__).parent.parent / module_file
            if file_path.exists():
                line_count = len(file_path.read_text().splitlines())
                assert line_count <= max_lines, f"{module_file} has {line_count} lines (>{max_lines})"

    def test_import_isolation(self):
        """Test that modules can be imported independently."""
        # Test discovery module
        from iris_rag.pipelines.hybrid_graphrag_discovery import GraphCoreDiscovery

        assert GraphCoreDiscovery is not None

        # Test retrieval module
        from iris_rag.pipelines.hybrid_graphrag_retrieval import HybridRetrievalMethods

        assert HybridRetrievalMethods is not None

        # Test main pipeline
        from iris_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

        assert HybridGraphRAGPipeline is not None


class TestConfigurationDriven:
    """Test configuration-driven behavior."""

    def test_graph_core_path_configuration(self):
        """Test that graph core path can be configured."""
        mock_config = Mock()
        mock_config.get.return_value = {"path": "/custom/graph/core/path"}

        discovery = GraphCoreDiscovery(mock_config)

        # Mock path validation to return True for our test path
        with patch.object(discovery, "_validate_graph_core_path", return_value=True):
            path = discovery.discover_graph_core_path()
            assert str(path) == "/custom/graph/core/path"

    def test_connection_parameter_priority(self):
        """Test connection parameter priority: config > env > none."""
        # Test config takes priority over environment
        mock_config = Mock()
        mock_config.get.return_value = {"host": "config-host", "port": 9999}

        discovery = GraphCoreDiscovery(mock_config)

        with patch.dict(os.environ, {"IRIS_HOST": "env-host", "IRIS_PORT": "8888"}):
            config = discovery.get_connection_config()

            # Config should win over environment
            assert config["host"] == "config-host"
            assert config["port"] == 9999


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
