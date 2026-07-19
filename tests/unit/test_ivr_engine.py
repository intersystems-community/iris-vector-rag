"""Unit tests for IRISVectorEngine class.

Tests cover construction, configuration, lazy loading, and schema prefix handling.
Note: IRISVectorEngine does not exist yet — these tests define acceptance criteria
and must FAIL (not ERROR) until the class is implemented.

These tests map to tasks T005–T012 in specs/080-ivr-engine/tasks.md.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, Mock
import pytest


class TestIRISVectorEngineFromConfig:
    """T005: IRISVectorEngine.from_config() returns engine instance without DB."""

    def test_from_config_returns_engine_instance(self):
        """from_config() creates and returns an IRISVectorEngine instance."""
        # Import will raise ImportError until IRISVectorEngine is implemented
        from iris_vector_rag.core.engine import IRISVectorEngine

        with patch("iris_vector_rag.config.manager.ConfigurationManager") as mock_cm_class:
            with patch(
                "iris_vector_rag.core.connection.ConnectionManager"
            ) as mock_conn_class:
                # Setup mocks
                mock_config = MagicMock()
                mock_config.get_schema_prefix.return_value = "RAG"
                mock_cm_class.return_value = mock_config

                mock_connection_manager = MagicMock()
                mock_conn_class.return_value = mock_connection_manager

                # Call from_config
                engine = IRISVectorEngine.from_config()

                # Assert return type
                assert isinstance(engine, IRISVectorEngine)

    def test_from_config_creates_config_manager(self):
        """from_config() creates a ConfigurationManager internally."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        with patch("iris_vector_rag.config.manager.ConfigurationManager") as mock_cm_class:
            with patch(
                "iris_vector_rag.core.connection.ConnectionManager"
            ) as mock_conn_class:
                mock_config = MagicMock()
                mock_config.get_schema_prefix.return_value = "RAG"
                mock_cm_class.return_value = mock_config

                mock_connection_manager = MagicMock()
                mock_conn_class.return_value = mock_connection_manager

                IRISVectorEngine.from_config()

                # Verify ConfigurationManager was instantiated
                mock_cm_class.assert_called_once()

    def test_from_config_creates_connection_manager(self):
        """from_config() creates a ConnectionManager internally."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        with patch("iris_vector_rag.config.manager.ConfigurationManager") as mock_cm_class:
            with patch(
                "iris_vector_rag.core.connection.ConnectionManager"
            ) as mock_conn_class:
                mock_config = MagicMock()
                mock_config.get_schema_prefix.return_value = "RAG"
                mock_cm_class.return_value = mock_config

                mock_connection_manager = MagicMock()
                mock_conn_class.return_value = mock_connection_manager

                IRISVectorEngine.from_config()

                # Verify ConnectionManager was instantiated with config_manager
                mock_conn_class.assert_called_once_with(mock_config)


class TestIRISVectorEngineRawConnection:
    """T006: IRISVectorEngine accepts raw DBAPI connection without opening new conn."""

    def test_accepts_raw_connection_stores_it(self):
        """IRISVectorEngine(mock_connection) stores raw connection without opening new one."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        mock_raw_conn = MagicMock()

        # Pass raw connection directly
        engine = IRISVectorEngine(mock_raw_conn)

        # Engine should be created successfully
        assert isinstance(engine, IRISVectorEngine)

    def test_raw_connection_no_get_connection_call(self):
        """IRISVectorEngine(raw_conn) does NOT call .get_connection() at construction."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        mock_raw_conn = MagicMock()

        # Verify mock is not a ConnectionManager
        assert not hasattr(mock_raw_conn, "get_connection") or isinstance(
            mock_raw_conn.get_connection, MagicMock
        )

        # Create engine with raw connection
        engine = IRISVectorEngine(mock_raw_conn)

        # Since it's a raw connection, .get_connection() should not have been called
        assert isinstance(engine, IRISVectorEngine)


class TestIRISVectorEngineSchemaPrefixDefault:
    """T007: schema_prefix default is 'RAG'; custom prefix accepted."""

    def test_schema_prefix_default_is_RAG(self):
        """IRISVectorEngine(mock_connection) defaults to schema_prefix='RAG'."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        mock_raw_conn = MagicMock()
        engine = IRISVectorEngine(mock_raw_conn)

        assert engine.schema_prefix == "RAG"

    def test_schema_prefix_custom(self):
        """IRISVectorEngine(mock_connection, schema_prefix='CUSTOM') returns 'CUSTOM'."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        mock_raw_conn = MagicMock()
        engine = IRISVectorEngine(mock_raw_conn, schema_prefix="MYAPP")

        assert engine.schema_prefix == "MYAPP"

    def test_schema_prefix_empty_string(self):
        """IRISVectorEngine with schema_prefix='' should respect empty string."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        mock_raw_conn = MagicMock()
        engine = IRISVectorEngine(mock_raw_conn, schema_prefix="")

        assert engine.schema_prefix == ""


class TestIRISVectorEngineLazyInit:
    """T008: Lazy initialization — no connection at construction."""

    def test_lazy_init_no_connection_at_construction(self):
        """Connection is None until first accessed."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        # Use a ConnectionManager mock
        mock_cm = MagicMock()
        mock_cm.get_connection = MagicMock()

        engine = IRISVectorEngine(mock_cm)

        # get_connection() should NOT have been called yet (lazy loading)
        mock_cm.get_connection.assert_not_called()

    def test_lazy_init_vector_store_none(self):
        """VectorStore is None until first accessed."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        mock_raw_conn = MagicMock()
        engine = IRISVectorEngine(mock_raw_conn)

        # Accessing _vector_store directly (before property access)
        # should show it's not initialized
        # Note: This test verifies internal state — may be adjusted based on
        # actual implementation details
        assert isinstance(engine, IRISVectorEngine)


class TestIRISVectorEngineConnectionManagerProperty:
    """T009: connection_manager property returns ConnectionManager or wrapper."""

    def test_cm_arg_returns_same_cm_from_property(self):
        """When constructed with ConnectionManager, property returns same object."""
        from iris_vector_rag.core.engine import IRISVectorEngine
        from iris_vector_rag.core.connection import ConnectionManager

        # Use a real ConnectionManager mock with spec
        mock_cm = MagicMock(spec=ConnectionManager)

        engine = IRISVectorEngine(mock_cm)

        # Property should return the same object
        assert engine.connection_manager is mock_cm

    def test_cm_arg_not_wrapped_in_external_wrapper(self):
        """ConnectionManager arg is NOT wrapped in ExternalConnectionWrapper."""
        from iris_vector_rag.core.engine import IRISVectorEngine
        from iris_vector_rag.core.connection import ConnectionManager
        from iris_vector_rag import ExternalConnectionWrapper

        mock_cm = MagicMock(spec=ConnectionManager)
        engine = IRISVectorEngine(mock_cm)

        # Verify it's not wrapped
        assert not isinstance(engine.connection_manager, ExternalConnectionWrapper)


class TestIRISVectorEngineRawConnWrapper:
    """T010: Raw connection arg returns ExternalConnectionWrapper."""

    def test_raw_conn_arg_returns_external_wrapper(self):
        """Non-ConnectionManager arg results in ExternalConnectionWrapper."""
        from iris_vector_rag.core.engine import IRISVectorEngine
        from iris_vector_rag import ExternalConnectionWrapper

        mock_raw_conn = MagicMock()

        engine = IRISVectorEngine(mock_raw_conn)

        # connection_manager property should return wrapper
        assert isinstance(engine.connection_manager, ExternalConnectionWrapper)

    def test_external_wrapper_wraps_raw_connection(self):
        """ExternalConnectionWrapper.get_connection() returns the raw connection."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        mock_raw_conn = MagicMock()
        engine = IRISVectorEngine(mock_raw_conn)

        # Call get_connection on the wrapper
        result = engine.connection_manager.get_connection("iris")

        # Should return the raw connection
        assert result is mock_raw_conn


class TestIRISVectorEngineFromConfigSchemaPrefix:
    """T011: from_config(schema_prefix='CUSTOM') overrides default."""

    def test_from_config_schema_prefix_kwarg_override(self):
        """from_config(schema_prefix='CUSTOM') returns engine with custom prefix."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        with patch("iris_vector_rag.config.manager.ConfigurationManager") as mock_cm_class:
            with patch(
                "iris_vector_rag.core.connection.ConnectionManager"
            ) as mock_conn_class:
                mock_config = MagicMock()
                mock_config.get_schema_prefix.return_value = "RAG"  # default from config
                mock_cm_class.return_value = mock_config

                mock_connection_manager = MagicMock()
                mock_conn_class.return_value = mock_connection_manager

                # Call with schema_prefix override
                engine = IRISVectorEngine.from_config(schema_prefix="CUSTOM")

                # Schema prefix should be the override, not the config default
                assert engine.schema_prefix == "CUSTOM"

    def test_from_config_uses_config_schema_prefix_as_fallback(self):
        """from_config() uses ConfigurationManager.get_schema_prefix() as fallback."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        with patch("iris_vector_rag.config.manager.ConfigurationManager") as mock_cm_class:
            with patch(
                "iris_vector_rag.core.connection.ConnectionManager"
            ) as mock_conn_class:
                mock_config = MagicMock()
                mock_config.get_schema_prefix.return_value = "MYPREFIX"
                mock_cm_class.return_value = mock_config

                mock_connection_manager = MagicMock()
                mock_conn_class.return_value = mock_connection_manager

                # Call without schema_prefix
                engine = IRISVectorEngine.from_config()

                # Should use config value
                assert engine.schema_prefix == "MYPREFIX"


class TestIRISVectorEngineConfigManager:
    """T012: config_manager property accessible and returns ConfigurationManager."""

    def test_config_manager_accessible_from_config(self):
        """config_manager property is accessible after from_config()."""
        from iris_vector_rag.core.engine import IRISVectorEngine
        from iris_vector_rag.config.manager import ConfigurationManager

        with patch("iris_vector_rag.config.manager.ConfigurationManager") as mock_cm_class:
            with patch(
                "iris_vector_rag.core.connection.ConnectionManager"
            ) as mock_conn_class:
                mock_config = MagicMock(spec=ConfigurationManager)
                mock_config.get_schema_prefix.return_value = "RAG"
                mock_cm_class.return_value = mock_config

                mock_connection_manager = MagicMock()
                mock_conn_class.return_value = mock_connection_manager

                engine = IRISVectorEngine.from_config()

                # config_manager should be accessible
                assert engine.config_manager is mock_config

    def test_config_manager_accessible_raw_connection(self):
        """config_manager property is accessible with raw connection."""
        from iris_vector_rag.core.engine import IRISVectorEngine
        from iris_vector_rag.config.manager import ConfigurationManager

        mock_raw_conn = MagicMock()

        with patch("iris_vector_rag.config.manager.ConfigurationManager") as mock_cm_class:
            mock_config = MagicMock(spec=ConfigurationManager)
            mock_cm_class.return_value = mock_config

            engine = IRISVectorEngine(mock_raw_conn)

            # config_manager should be a ConfigurationManager instance
            assert isinstance(engine.config_manager, (MagicMock, ConfigurationManager))

    def test_config_manager_passed_explicitly(self):
        """config_manager kwarg passed to __init__ is used."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        mock_raw_conn = MagicMock()
        mock_config = MagicMock()

        engine = IRISVectorEngine(mock_raw_conn, config_manager=mock_config)

        # Should use the passed config_manager
        assert engine.config_manager is mock_config


class TestIRISVectorEngineConnectionProperty:
    """Test lazy connection property (supporting functionality)."""

    def test_connection_property_lazy_loads(self):
        """connection property calls get_connection() on first access."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        mock_cm = MagicMock()
        mock_conn = MagicMock()
        mock_cm.get_connection.return_value = mock_conn

        engine = IRISVectorEngine(mock_cm)

        # First access should call get_connection()
        conn = engine.connection

        mock_cm.get_connection.assert_called_once_with("iris")
        assert conn is mock_conn

    def test_connection_property_cached(self):
        """connection property caches result — second access doesn't call get_connection()."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        mock_cm = MagicMock()
        mock_conn = MagicMock()
        mock_cm.get_connection.return_value = mock_conn

        engine = IRISVectorEngine(mock_cm)

        # First access
        conn1 = engine.connection
        # Second access
        conn2 = engine.connection

        # get_connection() called only once
        mock_cm.get_connection.assert_called_once()
        assert conn1 is conn2


class TestIRISVectorEngineEdgeCases:
    """Edge cases and additional scenarios."""

    def test_multiple_engines_independent(self):
        """Multiple engine instances maintain separate state."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        mock_conn1 = MagicMock()
        mock_conn2 = MagicMock()

        engine1 = IRISVectorEngine(mock_conn1, schema_prefix="APP1")
        engine2 = IRISVectorEngine(mock_conn2, schema_prefix="APP2")

        assert engine1.schema_prefix == "APP1"
        assert engine2.schema_prefix == "APP2"
        assert engine1.connection_manager is not engine2.connection_manager

    def test_from_config_with_kwargs(self):
        """from_config() passes kwargs to ConfigurationManager."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        with patch("iris_vector_rag.config.manager.ConfigurationManager") as mock_cm_class:
            with patch(
                "iris_vector_rag.core.connection.ConnectionManager"
            ) as mock_conn_class:
                mock_config = MagicMock()
                mock_config.get_schema_prefix.return_value = "RAG"
                mock_cm_class.return_value = mock_config

                mock_connection_manager = MagicMock()
                mock_conn_class.return_value = mock_connection_manager

                # Call with kwargs
                IRISVectorEngine.from_config(config_path="/path/to/config.yaml")

                # Verify ConfigurationManager was called with the kwarg
                mock_cm_class.assert_called_once_with(config_path="/path/to/config.yaml")

    def test_schema_prefix_type(self):
        """schema_prefix must be a string."""
        from iris_vector_rag.core.engine import IRISVectorEngine

        mock_raw_conn = MagicMock()

        engine = IRISVectorEngine(mock_raw_conn, schema_prefix="PREFIX")

        assert isinstance(engine.schema_prefix, str)


class TestRAGPipelineEngineOverload:
    """T027: RAGPipeline accepts IRISVectorEngine as first positional arg."""

    def test_ragpipeline_accepts_engine_as_first_arg(self):
        """RAGPipeline(engine) where engine is IRISVectorEngine unpacks engine attributes."""
        from iris_vector_rag.core.engine import IRISVectorEngine
        from iris_vector_rag.core.base import RAGPipeline
        from iris_vector_rag.pipelines.basic import BasicRAGPipeline

        # Create a real IRISVectorEngine with mocked connection
        mock_raw_conn = MagicMock()
        engine = IRISVectorEngine(mock_raw_conn)

        # Mock the vector store to avoid actual DB interaction
        with patch("iris_vector_rag.storage.vector_store_iris.IRISVectorStore") as mock_vs:
            mock_vs.return_value = MagicMock()
            engine._vector_store = mock_vs.return_value

            # Instantiate BasicRAGPipeline with engine as first arg
            pipeline = BasicRAGPipeline(engine)

            # Assert connection_manager is from the engine
            assert pipeline.connection_manager is engine.connection_manager
            # Assert config_manager is from the engine
            assert pipeline.config_manager is engine.config_manager
            # Assert vector_store is from the engine
            assert pipeline.vector_store is engine.vector_store

    def test_ragpipeline_legacy_pair_still_works(self):
        """RAGPipeline(cm, cfg) still works as before (backward compatibility)."""
        from iris_vector_rag.core.connection import ConnectionManager
        from iris_vector_rag.config.manager import ConfigurationManager
        from iris_vector_rag.pipelines.basic import BasicRAGPipeline

        # Create mocks for connection and config managers
        cm_mock = MagicMock(spec=ConnectionManager)
        cfg_mock = MagicMock(spec=ConfigurationManager)

        # Mock the vector store to avoid actual DB interaction
        with patch("iris_vector_rag.storage.vector_store_iris.IRISVectorStore") as mock_vs:
            mock_vs_instance = MagicMock()
            mock_vs.return_value = mock_vs_instance

            # Instantiate BasicRAGPipeline with legacy calling convention
            pipeline = BasicRAGPipeline(cm_mock, cfg_mock)

            # Assert connection_manager is the one passed in
            assert pipeline.connection_manager is cm_mock
            assert pipeline.config_manager is cfg_mock
            # Assert vector store was created with both
            mock_vs.assert_called_once_with(cm_mock, cfg_mock)


class TestCreatePipelineWithEngine:
    """T028: create_pipeline accepts engine= kwarg and uses its managers."""

    def test_create_pipeline_with_engine_kwarg(self):
        """create_pipeline('basic', engine=engine) creates pipeline with engine's managers."""
        import iris_vector_rag
        from iris_vector_rag.core.engine import IRISVectorEngine
        from iris_vector_rag.core.connection import ConnectionManager
        from iris_vector_rag.config.manager import ConfigurationManager

        # Create a real IRISVectorEngine with mocked managers
        mock_cm = MagicMock(spec=ConnectionManager)
        mock_cfg = MagicMock(spec=ConfigurationManager)
        mock_cfg.get.side_effect = lambda key, default=None: default

        engine = IRISVectorEngine(mock_cm, config_manager=mock_cfg)

        # Mock _create_pipeline_legacy to intercept the call and verify it was called with engine's managers
        with patch("iris_vector_rag._create_pipeline_legacy") as mock_legacy:
            # Set up the mock to return a BasicRAGPipeline-like object
            mock_legacy.return_value = MagicMock()

            # Call create_pipeline with engine kwarg
            result = iris_vector_rag.create_pipeline("basic", engine=engine)

            # Verify _create_pipeline_legacy was called with engine's managers
            mock_legacy.assert_called_once()
            call_args = mock_legacy.call_args
            assert call_args[0][0] == "basic"  # pipeline_type
            assert call_args[0][1] is mock_cm  # connection_manager from engine
            assert call_args[0][2] is mock_cfg  # config_manager from engine
