"""
Functional tests for HybridGraphRAG pipeline.

Tests the core functionality of HybridGraphRAG including:
- Security: No hard-coded credentials
- Configuration: Config-driven discovery
- Modularity: Proper fallbacks and error handling
- Query methods: Different retrieval strategies
- Integration: iris_vector_graph optional dependency
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline
from iris_vector_rag.pipelines.hybrid_graphrag_discovery import GraphCoreDiscovery

logger = logging.getLogger(__name__)


class TestHybridGraphRAGSecurity:
    """Test security aspects of HybridGraphRAG."""

    def test_no_hardcoded_credentials_in_source(self):
        """Verify no hard-coded credentials exist in source code."""
        pipeline_file = (
            Path(__file__).parent.parent
            / "iris_vector_rag"
            / "pipelines"
            / "hybrid_graphrag.py"
        )
        discovery_file = (
            Path(__file__).parent.parent
            / "iris_vector_rag"
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

    def test_config_driven_connection_params(self, monkeypatch, tmp_path):
        """Connection parameters come from the one settings resolver, not from code."""
        monkeypatch.chdir(tmp_path)  # no .env here
        for name, value in {
            "IRIS_HOST": "test-host",
            "IRIS_PORT": "9999",
            "IRIS_NAMESPACE": "TEST",
            "IRIS_USERNAME": "test-user",
            "IRIS_PASSWORD": "test-pass",
        }.items():
            monkeypatch.setenv(name, value)
        monkeypatch.delenv("IRIS_CONTAINER_NAME", raising=False)

        config = GraphCoreDiscovery(None).get_connection_config()

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
    """iris_vector_graph is found by import only: no path search, no sys.path edits."""

    def test_discovery_priority_order(self):
        """The installed package is the only source; the import returns its classes."""
        modules = GraphCoreDiscovery(None).import_graph_core_modules()

        for name in (
            "IRISGraphEngine",
            "HybridSearchFusion",
            "TextSearchEngine",
            "VectorOptimizer",
        ):
            assert name in modules, f"{name} missing from the discovered modules"

    def test_environment_variable_discovery(self, tmp_path, monkeypatch):
        """GRAPH_CORE_PATH no longer steers discovery, and nothing is put on sys.path."""
        import sys

        impostor = tmp_path / "env-graph-core"
        (impostor / "iris_vector_graph").mkdir(parents=True)
        monkeypatch.setenv("GRAPH_CORE_PATH", str(impostor))
        original_path = sys.path.copy()

        discovery = GraphCoreDiscovery(None)
        modules = discovery.import_graph_core_modules()

        assert not hasattr(discovery, "discover_graph_core_path")
        assert str(impostor) not in sys.path
        assert sys.path == original_path
        # The real package was imported, not the empty directory named by the variable.
        assert modules["IRISGraphEngine"].__module__.startswith("iris_vector_graph")

    def test_safe_import_without_sys_path_pollution(self, monkeypatch):
        """sys.path is unchanged on success, and a missing package is an ImportError."""
        import sys

        original_path = sys.path.copy()

        modules = GraphCoreDiscovery(None).import_graph_core_modules()
        assert "IRISGraphEngine" in modules
        assert sys.path == original_path

        monkeypatch.setenv("FORCE_IRIS_VECTOR_GRAPH_IMPORT_ERROR", "1")
        with pytest.raises(ImportError, match="iris-vector-graph"):
            GraphCoreDiscovery(None).import_graph_core_modules()
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
                "dimension": 384,
            },
            "database": {"schema": "RAG"},
            "vector_search": {"metric": "COSINE", "top_k": 5},
        }
        self.mock_config_mgr.to_dict.return_value = self.mock_config_mgr._config

    def _fake_modules(self, engine_class=None):
        # Factories, not `Mock` itself: Mock's first positional argument is a spec.
        def make(*args, **kwargs):
            return Mock()

        return {
            "IRISGraphEngine": engine_class or make,
            "HybridSearchFusion": make,
            "TextSearchEngine": make,
            "VectorOptimizer": make,
            "package_name": "iris_vector_graph",
        }

    def _build(self, engine_class=None):
        """Construct the pipeline with iris_vector_graph and IRIS replaced at the
        two seams the constructor uses: the discovery it builds and the session
        connection it opens."""
        database = {
            "host": "localhost",
            "port": 1972,
            "namespace": "TEST",
            "username": "test",
            "password": "test",
        }
        self.mock_config_mgr.get_database_config.return_value = database
        with patch(
            "iris_vector_rag.pipelines.hybrid_graphrag.GraphCoreDiscovery"
        ) as mock_discovery_class, patch(
            "iris_vector_rag.common.iris_connection._session_connection"
        ) as mock_session:
            mock_discovery = Mock(spec=GraphCoreDiscovery)
            mock_discovery_class.return_value = mock_discovery
            mock_discovery.import_graph_core_modules.return_value = self._fake_modules(
                engine_class
            )
            mock_discovery.validate_connection_config.return_value = (True, [])
            mock_session.return_value = Mock()
            return HybridGraphRAGPipeline(self.mock_conn_mgr, self.mock_config_mgr)

    def _degraded_pipeline(self):
        """iris_vector_graph imports but its engine cannot start: no hybrid."""

        def broken_engine(*args, **kwargs):
            raise RuntimeError("engine unavailable")

        return self._build(engine_class=broken_engine)

    def test_pipeline_initialization_without_graph_core(self, monkeypatch):
        """A missing iris_vector_graph is an ImportError that says how to install it.

        The pipeline used to come up half-working; a pipeline that cannot do what
        it is named for now refuses to start.
        """
        monkeypatch.setenv("FORCE_IRIS_VECTOR_GRAPH_IMPORT_ERROR", "1")

        with pytest.raises(ImportError, match="iris-vector-graph"):
            HybridGraphRAGPipeline(self.mock_conn_mgr, self.mock_config_mgr)

    def test_pipeline_initialization_with_graph_core(self):
        """With iris_vector_graph available the engines are built and hybrid is on."""
        pipeline = self._build()

        assert pipeline.discovery is not None
        assert pipeline.is_hybrid_enabled()
        assert pipeline.iris_engine is not None
        assert pipeline.fusion_engine is not None
        assert pipeline.text_engine is not None
        assert pipeline.vector_optimizer is not None
        assert pipeline.retrieval_methods is not None

    def test_query_fallback_to_graphrag(self):
        """Queries use the enhanced fallback when the hybrid engine failed to start."""
        pipeline = self._degraded_pipeline()

        assert not pipeline.is_hybrid_enabled()
        assert pipeline.iris_engine is None
        assert pipeline.retrieval_methods is None

        with patch.object(pipeline, "_validate_knowledge_graph"), patch.object(
            pipeline, "_enhanced_hybrid_fallback"
        ) as mock_fallback:
            mock_fallback.return_value = ([Mock()], "fallback")

            pipeline.query("test query", method="hybrid", generate_answer=False)

            mock_fallback.assert_called_once()

    def test_query_with_hybrid_methods(self):
        """Test query routing to different hybrid methods."""
        pipeline = self._build()
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

            assert [d.page_content for d in result["retrieved_documents"]] == ["test"]
            assert result["metadata"]["retrieval_method"] == "hybrid_fusion"

    def test_cursor_cleanup_robustness(self):
        """The cursor is closed when the query fails, and the failure is not hidden.

        A database error used to become the string "Entity: <id>", which looked like
        document content (PL-10). It now propagates; no row at all is None.
        """
        pipeline = self._degraded_pipeline()

        shape = Mock()
        shape.schema = "RAG"
        shape.qualified_name = "RAG.SourceDocuments"
        shape.column_for.return_value.name = "col"
        pipeline.vector_store = Mock()
        pipeline.vector_store.shape.return_value = shape

        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        self.mock_conn_mgr.get_connection.return_value = mock_connection

        mock_cursor.execute.side_effect = Exception("Database error")
        with pytest.raises(Exception, match="Database error"):
            pipeline._get_document_content_for_entity("test_entity")
        mock_cursor.close.assert_called_once()

        mock_cursor.reset_mock()
        mock_cursor.execute.side_effect = None
        mock_cursor.fetchone.return_value = None
        assert pipeline._get_document_content_for_entity("test_entity") is None
        mock_cursor.close.assert_called_once()

    def test_get_hybrid_status(self):
        """Test the hybrid status reporting functionality."""
        pipeline = self._build()

        status = pipeline.get_hybrid_status()

        assert isinstance(status, dict)
        assert "hybrid_enabled" in status
        assert "iris_engine_available" in status
        assert "fusion_engine_available" in status
        assert "text_engine_available" in status
        assert "vector_optimizer_available" in status
        assert "graph_core_path" in status
        assert status["hybrid_enabled"] is True
        # Found by import, so there is no path to report.
        assert status["graph_core_path"] is None


class TestModularArchitecture:
    """Test the modular architecture aspects."""

    def test_file_line_count_limits(self):
        """Test that all module files stay under reasonable limits."""
        module_files = {
            # These limits were never enforced: the paths named a package that no
            # longer exists, so the loop skipped every file. Set to the size the
            # modules actually are, so growth is caught from here on.
            "iris_vector_rag/pipelines/hybrid_graphrag.py": 1000,  # Main pipeline
            "iris_vector_rag/pipelines/hybrid_graphrag_discovery.py": 500,
            "iris_vector_rag/pipelines/hybrid_graphrag_retrieval.py": 500,
        }

        for module_file, max_lines in module_files.items():
            file_path = Path(__file__).parent.parent / module_file
            if file_path.exists():
                line_count = len(file_path.read_text().splitlines())
                assert (
                    line_count <= max_lines
                ), f"{module_file} has {line_count} lines (>{max_lines})"

    def test_import_isolation(self):
        """Test that modules can be imported independently."""
        # Test discovery module
        from iris_vector_rag.pipelines.hybrid_graphrag_discovery import (
            GraphCoreDiscovery,
        )

        assert GraphCoreDiscovery is not None

        # Test retrieval module
        from iris_vector_rag.pipelines.hybrid_graphrag_retrieval import (
            HybridRetrievalMethods,
        )

        assert HybridRetrievalMethods is not None

        # Test main pipeline
        from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

        assert HybridGraphRAGPipeline is not None


class TestConfigurationDriven:
    """Test configuration-driven behavior."""

    def test_graph_core_path_configuration(self):
        """A graph-core path in configuration is not consulted: import only."""
        mock_config = Mock()
        mock_config.get.return_value = {"path": "/custom/graph/core/path"}

        discovery = GraphCoreDiscovery(mock_config)
        modules = discovery.import_graph_core_modules()

        assert modules["IRISGraphEngine"].__module__.startswith("iris_vector_graph")
        mock_config.get.assert_not_called()

    def test_connection_parameter_priority(self, monkeypatch, tmp_path):
        """Connection parameter priority: the environment beats the config file."""
        monkeypatch.chdir(tmp_path)  # no .env here
        monkeypatch.delenv("IRIS_CONTAINER_NAME", raising=False)
        mock_config = Mock()
        mock_config.get.return_value = {"host": "config-host", "port": 9999}

        discovery = GraphCoreDiscovery(mock_config)

        with patch.dict(os.environ, {"IRIS_HOST": "env-host", "IRIS_PORT": "8888"}):
            config = discovery.get_connection_config()

        assert config["host"] == "env-host"
        assert config["port"] == 8888


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
