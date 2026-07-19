"""
Unit tests for pure constructors (no database access on initialization).

Feature: 078-pure-constructors
Tests that constructing RAG pipelines, vector stores, and schema managers
does not trigger any database connections or DDL operations.
"""

import pytest
from unittest.mock import MagicMock, patch, call

from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from iris_vector_rag.core.base import RAGPipeline
from iris_vector_rag.pipelines.basic import BasicRAGPipeline
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.config.manager import ConfigurationManager


class TestSchemaManagerConstructorPure:
    """Test that SchemaManager construction does not access the database."""

    def test_schema_manager_constructor_no_db_call(self):
        """
        Test: SchemaManager.__init__ does not call get_connection().

        Given a mock ConnectionManager with spec=ConnectionManager
        When SchemaManager is constructed
        Then get_connection() is never called
        And _initialized flag is False (not yet initialized)
        """
        # Arrange: Create comprehensive mock for ConfigurationManager
        mock_cm = MagicMock(spec=ConnectionManager)
        mock_cfg = MagicMock(spec=ConfigurationManager)

        # Set up mock config manager to return proper values
        mock_cloud_config = MagicMock()
        mock_cloud_config.vector.vector_dimension = 384
        mock_cfg.get_cloud_config.return_value = mock_cloud_config

        # Make get() return strings for embedding model name
        def mock_get(key, default=None):
            if key == "embedding_model.name":
                return "sentence-transformers/all-MiniLM-L6-v2"
            elif key == "storage:iris":
                return {}
            return default

        mock_cfg.get.side_effect = mock_get

        # Act
        schema_manager = SchemaManager(mock_cm, mock_cfg)

        # Assert
        mock_cm.get_connection.assert_not_called()
        assert hasattr(schema_manager, '_initialized')
        assert schema_manager._initialized is False

    def test_schema_manager_ensure_schema_metadata_table_idempotent(self):
        """
        Test: ensure_schema_metadata_table() is idempotent.

        Given a SchemaManager with _initialized=True
        When ensure_schema_metadata_table() is called
        Then no database connection is made (early return)
        """
        # Arrange: Create comprehensive mock for ConfigurationManager
        mock_cm = MagicMock(spec=ConnectionManager)
        mock_cfg = MagicMock(spec=ConfigurationManager)

        # Set up mock config manager to return proper values
        mock_cloud_config = MagicMock()
        mock_cloud_config.vector.vector_dimension = 384
        mock_cfg.get_cloud_config.return_value = mock_cloud_config

        # Make get() return strings for embedding model name
        def mock_get(key, default=None):
            if key == "embedding_model.name":
                return "sentence-transformers/all-MiniLM-L6-v2"
            elif key == "storage:iris":
                return {}
            return default

        mock_cfg.get.side_effect = mock_get

        schema_manager = SchemaManager(mock_cm, mock_cfg)
        schema_manager._initialized = True

        # Act
        schema_manager.ensure_schema_metadata_table()

        # Assert
        mock_cm.get_connection.assert_not_called()


class TestIRISVectorStoreConstructorPure:
    """Test that IRISVectorStore construction does not access the database."""

    def _create_mock_config_manager(self):
        """Create a properly configured mock ConfigurationManager."""
        mock_cfg = MagicMock(spec=ConfigurationManager)

        # Set up mock config manager to return proper values
        mock_cloud_config = MagicMock()
        mock_cloud_config.vector.vector_dimension = 384
        mock_cfg.get_cloud_config.return_value = mock_cloud_config

        # Make get() return strings for embedding model name
        def mock_get(key, default=None):
            if key == "embedding_model.name":
                return "sentence-transformers/all-MiniLM-L6-v2"
            elif key == "storage:iris":
                return {}
            return default

        mock_cfg.get.side_effect = mock_get
        mock_cfg.to_dict.return_value = {}
        return mock_cfg

    def test_iris_vector_store_constructor_no_db_call(self):
        """
        Test: IRISVectorStore.__init__ does not call get_connection().

        Given mock ConnectionManager and ConfigurationManager
        When IRISVectorStore is constructed
        Then no calls to get_connection() occur
        """
        # Arrange
        mock_cm = MagicMock(spec=ConnectionManager)
        mock_cfg = self._create_mock_config_manager()

        # Act
        vector_store = IRISVectorStore(mock_cm, mock_cfg)

        # Assert
        mock_cm.get_connection.assert_not_called()
        assert vector_store.schema_manager._initialized is False

    def test_iris_vector_store_constructor_with_schema_manager(self):
        """
        Test: IRISVectorStore.__init__ with provided SchemaManager does not call DB.

        Given a mock SchemaManager and mock ConnectionManager
        When IRISVectorStore is constructed with schema_manager parameter
        Then no DB calls occur
        """
        # Arrange
        mock_cm = MagicMock(spec=ConnectionManager)
        mock_cfg = MagicMock(spec=ConfigurationManager)
        mock_cfg.get.return_value = {}
        mock_cfg.to_dict.return_value = {}
        mock_schema = MagicMock(spec=SchemaManager)
        mock_schema.get_vector_dimension.return_value = 384

        # Act
        vector_store = IRISVectorStore(mock_cm, mock_cfg, schema_manager=mock_schema)

        # Assert
        mock_cm.get_connection.assert_not_called()
        assert vector_store.schema_manager is mock_schema


class TestBasicRAGPipelineConstructorPure:
    """Test that BasicRAGPipeline construction does not access the database."""

    def _create_mock_config_manager(self):
        """Create a properly configured mock ConfigurationManager."""
        mock_cfg = MagicMock(spec=ConfigurationManager)

        # Set up mock config manager to return proper values
        mock_cloud_config = MagicMock()
        mock_cloud_config.vector.vector_dimension = 384
        mock_cfg.get_cloud_config.return_value = mock_cloud_config

        # Make get() return strings for embedding model name
        def mock_get(key, default=None):
            if key == "embedding_model.name":
                return "sentence-transformers/all-MiniLM-L6-v2"
            elif key == "storage:iris":
                return {}
            elif key == "storage:chunking":
                return {}
            elif key == "pipelines:basic":
                return {}
            return default

        mock_cfg.get.side_effect = mock_get
        mock_cfg.to_dict.return_value = {}
        return mock_cfg

    def test_basic_rag_pipeline_constructor_no_db_call(self):
        """
        Test: BasicRAGPipeline.__init__ does not call get_connection().

        Given mock ConnectionManager and ConfigurationManager
        When BasicRAGPipeline is constructed
        Then no DB calls occur during initialization
        """
        # Arrange
        mock_cm = MagicMock(spec=ConnectionManager)
        mock_cfg = self._create_mock_config_manager()

        # Act
        pipeline = BasicRAGPipeline(mock_cm, mock_cfg)

        # Assert
        mock_cm.get_connection.assert_not_called()
        assert hasattr(pipeline, '_lazy_init_done')
        assert pipeline._lazy_init_done is False

    def test_basic_rag_pipeline_has_initialize_method(self):
        """
        Test: BasicRAGPipeline has an initialize() method.

        Given a constructed BasicRAGPipeline
        Then it should have initialize() method
        """
        # Arrange
        mock_cm = MagicMock(spec=ConnectionManager)
        mock_cfg = self._create_mock_config_manager()

        # Act
        pipeline = BasicRAGPipeline(mock_cm, mock_cfg)

        # Assert
        assert hasattr(pipeline, 'initialize')
        assert callable(pipeline.initialize)


class TestLazyInitialization:
    """Test lazy initialization on first use."""

    def _create_mock_config_manager(self):
        """Create a properly configured mock ConfigurationManager."""
        mock_cfg = MagicMock(spec=ConfigurationManager)

        # Set up mock config manager to return proper values
        mock_cloud_config = MagicMock()
        mock_cloud_config.vector.vector_dimension = 384
        mock_cfg.get_cloud_config.return_value = mock_cloud_config

        # Make get() return strings for embedding model name
        def mock_get(key, default=None):
            if key == "embedding_model.name":
                return "sentence-transformers/all-MiniLM-L6-v2"
            elif key == "storage:iris":
                return {}
            elif key == "storage:chunking":
                return {}
            elif key == "pipelines:basic":
                return {}
            return default

        mock_cfg.get.side_effect = mock_get
        mock_cfg.to_dict.return_value = {}
        return mock_cfg

    def test_lazy_init_on_first_load_documents(self):
        """
        Test: Lazy initialization happens on first load_documents() call.

        Given a constructed BasicRAGPipeline
        And ensure_schema_metadata_table is mocked to track calls
        When load_documents() is called
        Then ensure_schema_metadata_table() should be called exactly once
        """
        # Arrange
        mock_cm = MagicMock(spec=ConnectionManager)
        mock_cfg = self._create_mock_config_manager()

        pipeline = BasicRAGPipeline(mock_cm, mock_cfg)

        # Mock the schema manager's ensure method
        with patch.object(
            pipeline.vector_store.schema_manager,
            'ensure_schema_metadata_table'
        ) as mock_ensure:
            # Act: Call load_documents (it should trigger lazy init)
            try:
                pipeline.load_documents(documents=[])
            except Exception:
                # We don't care if the call fails for other reasons;
                # we just want to verify ensure was called
                pass

            # Assert
            mock_ensure.assert_called_once()

    def test_lazy_init_idempotent_multiple_calls(self):
        """
        Test: Multiple load_documents() calls trigger ensure_schema_metadata_table() only once.

        Given a constructed BasicRAGPipeline
        When load_documents() is called multiple times
        Then ensure_schema_metadata_table() should be called exactly once
        (subsequent calls skip lazy init due to _lazy_init_done flag)
        """
        # Arrange
        mock_cm = MagicMock(spec=ConnectionManager)
        mock_cfg = self._create_mock_config_manager()

        pipeline = BasicRAGPipeline(mock_cm, mock_cfg)

        with patch.object(
            pipeline.vector_store.schema_manager,
            'ensure_schema_metadata_table'
        ) as mock_ensure:
            # Act: Call load_documents twice
            for _ in range(2):
                try:
                    pipeline.load_documents(documents=[])
                except Exception:
                    pass

            # Assert: ensure_schema_metadata_table called exactly once
            assert mock_ensure.call_count == 1


class TestPyTestCurrentTestGuardRemoved:
    """Test that PYTEST_CURRENT_TEST guard has been removed."""

    def _create_mock_config_manager(self):
        """Create a properly configured mock ConfigurationManager."""
        mock_cfg = MagicMock(spec=ConfigurationManager)

        # Set up mock config manager to return proper values
        mock_cloud_config = MagicMock()
        mock_cloud_config.vector.vector_dimension = 384
        mock_cfg.get_cloud_config.return_value = mock_cloud_config

        # Make get() return strings for embedding model name
        def mock_get(key, default=None):
            if key == "embedding_model.name":
                return "sentence-transformers/all-MiniLM-L6-v2"
            elif key == "storage:iris":
                return {}
            return default

        mock_cfg.get.side_effect = mock_get
        mock_cfg.to_dict.return_value = {}
        return mock_cfg

    def test_no_pytest_current_test_dependency(self):
        """
        Test: IRISVectorStore does not depend on PYTEST_CURRENT_TEST env var.

        Given PYTEST_CURRENT_TEST is unset
        When IRISVectorStore is constructed with mocks
        Then construction succeeds without DB calls (no special test mode)
        """
        # Arrange
        import os
        # Ensure PYTEST_CURRENT_TEST is not set
        original = os.environ.get("PYTEST_CURRENT_TEST")
        if "PYTEST_CURRENT_TEST" in os.environ:
            del os.environ["PYTEST_CURRENT_TEST"]

        try:
            mock_cm = MagicMock(spec=ConnectionManager)
            mock_cfg = self._create_mock_config_manager()

            # Act
            vector_store = IRISVectorStore(mock_cm, mock_cfg)

            # Assert
            mock_cm.get_connection.assert_not_called()
        finally:
            # Restore original value
            if original is not None:
                os.environ["PYTEST_CURRENT_TEST"] = original
