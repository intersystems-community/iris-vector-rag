#!/usr/bin/env python3
"""
Test suite for SchemaManager integration with GraphRAG pipeline.

This test verifies that the SchemaManager correctly handles schema validation,
migration, and integration with the GraphRAG pipeline for Phase 1 implementation.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.core.models import Document

logger = logging.getLogger(__name__)


class TestSchemaManagerIntegration:
    """Test SchemaManager integration with GraphRAG pipeline."""
    
    @pytest.fixture
    def mock_connection_manager(self):
        """Mock connection manager for testing."""
        mock_conn_mgr = Mock(spec=ConnectionManager)
        mock_connection = Mock()
        mock_cursor = Mock()
        
        mock_connection.cursor.return_value = mock_cursor
        mock_conn_mgr.get_connection.return_value = mock_connection
        
        return mock_conn_mgr, mock_connection, mock_cursor
    
    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager with embedding config."""
        mock_config = Mock(spec=ConfigurationManager)
        mock_config.get_embedding_config.return_value = {
            'model': 'all-MiniLM-L6-v2',
            'dimension': 384,
            'provider': 'sentence-transformers'
        }
        mock_config.get.return_value = "FLOAT"  # Default vector data type
        return mock_config
    
    @pytest.fixture
    def schema_manager(self, mock_connection_manager, mock_config_manager):
        """Create SchemaManager instance for testing."""
        conn_mgr, _, _ = mock_connection_manager
        return SchemaManager(conn_mgr, mock_config_manager)
    
    def test_schema_manager_initialization(self, schema_manager, mock_config_manager):
        """Test that SchemaManager initializes correctly."""
        assert schema_manager.config_manager == mock_config_manager
        assert schema_manager.schema_version == "1.0.0"
    
    def test_ensure_schema_metadata_table(self, schema_manager, mock_connection_manager):
        """Test that schema metadata table is created correctly."""
        conn_mgr, mock_connection, mock_cursor = mock_connection_manager
        
        schema_manager.ensure_schema_metadata_table()
        
        # Verify table creation SQL was executed
        mock_cursor.execute.assert_called()
        create_sql_call = mock_cursor.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS RAG.SchemaMetadata" in create_sql_call
        assert "vector_dimension INTEGER" in create_sql_call
        assert "embedding_model VARCHAR(255)" in create_sql_call
        
        mock_connection.commit.assert_called_once()
        mock_cursor.close.assert_called_once()
    
    def test_get_expected_schema_config_document_entities(self, schema_manager, mock_config_manager):
        """Test expected schema config for DocumentEntities table."""
        config = schema_manager._get_expected_schema_config("DocumentEntities")
        
        assert config["schema_version"] == "1.0.0"
        assert config["vector_dimension"] == 384  # all-MiniLM-L6-v2 dimension
        assert config["embedding_model"] == "all-MiniLM-L6-v2"
        assert config["configuration"]["table_type"] == "entity_storage"
        assert config["configuration"]["supports_vector_search"] is True
        assert config["configuration"]["auto_migration"] is True
        assert config["vector_data_type"] == "FLOAT"  # Default value
    
    def test_get_expected_schema_config_different_model(self, schema_manager, mock_config_manager):
        """Test expected schema config with different embedding model."""
        mock_config_manager.get_embedding_config.return_value = {
            'model': 'all-mpnet-base-v2',
            'dimension': 768,
            'provider': 'sentence-transformers'
        }
        
        config = schema_manager._get_expected_schema_config("DocumentEntities")
        
        assert config["vector_dimension"] == 768  # all-mpnet-base-v2 dimension
        assert config["embedding_model"] == "all-mpnet-base-v2"
    
    def test_get_expected_schema_config_custom_vector_data_type(self, schema_manager, mock_config_manager):
        """Test expected schema config with custom vector data type."""
        # Mock custom vector data type configuration
        mock_config_manager.get.return_value = "DOUBLE"
        
        config = schema_manager._get_expected_schema_config("DocumentEntities")
        
        assert config["vector_data_type"] == "DOUBLE"
        assert config["vector_dimension"] == 384  # all-MiniLM-L6-v2 dimension
        assert config["embedding_model"] == "all-MiniLM-L6-v2"
    
    def test_needs_migration_no_current_config(self, schema_manager, mock_connection_manager):
        """Test migration needed when no current config exists."""
        conn_mgr, mock_connection, mock_cursor = mock_connection_manager
        mock_cursor.fetchone.return_value = None  # No existing metadata
        
        needs_migration = schema_manager.needs_migration("DocumentEntities")
        
        assert needs_migration is True
        mock_cursor.close.assert_called()
    
    def test_needs_migration_dimension_mismatch(self, schema_manager, mock_connection_manager):
        """Test migration needed when vector dimension changes."""
        conn_mgr, mock_connection, mock_cursor = mock_connection_manager
        
        # Mock existing metadata with different dimension
        mock_cursor.fetchone.return_value = (
            "1.0.0",  # schema_version
            768,      # vector_dimension (different from expected 384)
            "all-mpnet-base-v2",  # embedding_model
            '{"table_type": "entity_storage"}'  # configuration
        )
        
        needs_migration = schema_manager.needs_migration("DocumentEntities")
        
        assert needs_migration is True
        mock_cursor.close.assert_called()
    
    def test_needs_migration_up_to_date(self, schema_manager, mock_connection_manager):
        """Test no migration needed when schema is up to date."""
        conn_mgr, mock_connection, mock_cursor = mock_connection_manager
        
        # Mock existing metadata matching expected config
        mock_cursor.fetchone.return_value = (
            "1.0.0",  # schema_version
            384,      # vector_dimension (matches expected)
            "all-MiniLM-L6-v2",  # embedding_model
            '{"table_type": "entity_storage"}'  # configuration
        )
        
        needs_migration = schema_manager.needs_migration("DocumentEntities")
        
        assert needs_migration is False
        mock_cursor.close.assert_called()
    
    def test_migrate_document_entities_table(self, schema_manager, mock_connection_manager):
        """Test DocumentEntities table migration."""
        conn_mgr, mock_connection, mock_cursor = mock_connection_manager
        
        # Mock table row count check
        mock_cursor.fetchone.return_value = [0]  # Empty table
        
        success = schema_manager.migrate_table("DocumentEntities")
        
        assert success is True
        
        # Verify table was dropped and recreated
        execute_calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
        
        # Check for DROP TABLE
        assert any("DROP TABLE IF EXISTS RAG.DocumentEntities" in call for call in execute_calls)
        
        # Check for CREATE TABLE with correct dimension
        create_table_call = next((call for call in execute_calls if "CREATE TABLE RAG.DocumentEntities" in call), None)
        assert create_table_call is not None
        assert "VECTOR(FLOAT, 384)" in create_table_call
        
        # Check for index creation
        assert any("CREATE INDEX" in call for call in execute_calls)
        
        mock_connection.commit.assert_called()
        mock_cursor.close.assert_called()
    
    def test_ensure_table_schema_success(self, schema_manager, mock_connection_manager):
        """Test successful table schema validation."""
        conn_mgr, mock_connection, mock_cursor = mock_connection_manager
        
        # Mock that no migration is needed
        mock_cursor.fetchone.return_value = (
            "1.0.0", 384, "all-MiniLM-L6-v2", '{"table_type": "entity_storage"}'
        )
        
        success = schema_manager.ensure_table_schema("DocumentEntities")
        
        assert success is True
    
    def test_ensure_table_schema_with_migration(self, schema_manager, mock_connection_manager):
        """Test table schema validation that requires migration."""
        conn_mgr, mock_connection, mock_cursor = mock_connection_manager
        
        # Mock that migration is needed (no existing metadata)
        mock_cursor.fetchone.return_value = None
        
        success = schema_manager.ensure_table_schema("DocumentEntities")
        
        assert success is True
        
        # Verify migration was performed
        execute_calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
        assert any("DROP TABLE IF EXISTS RAG.DocumentEntities" in call for call in execute_calls)
    
    def test_graphrag_pipeline_schema_integration(self, mock_connection_manager, mock_config_manager):
        """Test that GraphRAG pipeline integrates SchemaManager correctly."""
        conn_mgr, mock_connection, mock_cursor = mock_connection_manager
        
        # Mock successful schema validation
        mock_cursor.fetchone.return_value = (
            "1.0.0", 384, "all-MiniLM-L6-v2", '{"table_type": "entity_storage"}'
        )
        # Ensure config_manager.get("pipelines:graphrag", {}) returns a dict
        mock_config_manager.get.return_value = {"top_k": 5, "max_entities": 10, "relationship_depth": 2}
        
        # Create GraphRAG pipeline
        with patch('iris_rag.pipelines.graphrag.IRISStorage'), \
             patch('iris_rag.pipelines.graphrag.EmbeddingManager'):
            
            pipeline = GraphRAGPipeline(conn_mgr, mock_config_manager)
            
            # Verify SchemaManager was initialized
            assert hasattr(pipeline, 'schema_manager')
            assert isinstance(pipeline.schema_manager, SchemaManager)
    
    @patch('common.vector_format_fix.format_vector_for_iris')
    @patch('common.vector_format_fix.validate_vector_for_iris')
    @patch('common.vector_format_fix.create_iris_vector_string')
    def test_graphrag_store_entities_calls_schema_manager(self, mock_create_vector, mock_validate, mock_format,
                                                         mock_connection_manager, mock_config_manager):
        """Test that _store_entities calls SchemaManager before storing."""
        conn_mgr, mock_connection, mock_cursor = mock_connection_manager

        # Configure mock_config_manager.get to return appropriate values for different keys
        def config_get_side_effect(key, default=None):
            if key == "pipelines:graphrag":
                return {"top_k": 5, "max_entities": 10, "relationship_depth": 2}
            elif key == "storage:iris:vector_data_type":
                return "FLOAT"
            elif key == "embeddings":
                return {
                    "default_model": "all-MiniLM-L6-v2",
                    "models": {
                        "all-MiniLM-L6-v2": {"dimensions": 384, "model_name": "all-MiniLM-L6-v2"}
                    }
                }
            elif key == "embedding_models:all-MiniLM-L6-v2":
                 return {"dimensions": 384, "model_name": "all-MiniLM-L6-v2"}
            return default # Important: respect the default passed to .get() for unhandled keys

        mock_config_manager.get.side_effect = config_get_side_effect
    
        # Mock successful schema validation
        mock_cursor.fetchone.return_value = (
            "1.0.0", 384, "all-MiniLM-L6-v2", '{"table_type": "entity_storage"}'
        )
        
        # Mock vector formatting functions
        mock_format.return_value = [0.1, 0.2, 0.3]
        mock_validate.return_value = True
        mock_create_vector.return_value = "0.1,0.2,0.3"
        
        with patch('iris_rag.pipelines.graphrag.IRISStorage'), \
             patch('iris_rag.pipelines.graphrag.EmbeddingManager'):
            
            pipeline = GraphRAGPipeline(conn_mgr, mock_config_manager)
            
            # Create test entities
            entities = [{
                "entity_id": "test_entity_1",
                "entity_text": "Test Entity",
                "entity_type": "KEYWORD",
                "position": 0,
                "embedding": [0.1, 0.2, 0.3]
            }]
            
            # Call _store_entities
            pipeline._store_entities("test_doc_1", entities)
            
            # Verify schema validation was called
            # The schema manager should have been called to ensure table schema
            assert mock_cursor.execute.called
    
    def test_schema_manager_error_handling(self, schema_manager, mock_connection_manager):
        """Test SchemaManager error handling."""
        conn_mgr, mock_connection, mock_cursor = mock_connection_manager
        
        # Mock database error
        mock_cursor.execute.side_effect = Exception("Database error")
        
        success = schema_manager.ensure_table_schema("DocumentEntities")
        
        assert success is False
    
    def test_get_schema_status(self, schema_manager, mock_connection_manager):
        """Test schema status reporting."""
        conn_mgr, mock_connection, mock_cursor = mock_connection_manager
        
        # Mock current schema metadata
        mock_cursor.fetchone.return_value = (
            "1.0.0", 384, "all-MiniLM-L6-v2", '{"table_type": "entity_storage"}'
        )
        
        status = schema_manager.get_schema_status()
        
        assert "DocumentEntities" in status
        assert status["DocumentEntities"]["status"] == "up_to_date"
        assert status["DocumentEntities"]["needs_migration"] is False
        assert status["DocumentEntities"]["current_config"] is not None
        assert status["DocumentEntities"]["expected_config"] is not None


class TestSchemaManagerConfigurationIntegration:
    """Test SchemaManager integration with different configurations."""
    
    def test_embedding_config_integration(self):
        """Test that SchemaManager correctly uses embedding configuration."""
        # Create real ConfigurationManager with test config
        config_manager = ConfigurationManager()
        
        # Mock connection manager
        mock_conn_mgr = Mock(spec=ConnectionManager)
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_conn_mgr.get_connection.return_value = mock_connection
        
        schema_manager = SchemaManager(mock_conn_mgr, config_manager)
        
        # Test default configuration
        config = schema_manager._get_expected_schema_config("DocumentEntities")
        assert config["embedding_model"] == "all-MiniLM-L6-v2"
        assert config["vector_dimension"] == 384
    
    def test_model_dimension_mapping(self):
        """Test that different models map to correct dimensions."""
        mock_conn_mgr = Mock(spec=ConnectionManager)
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_conn_mgr.get_connection.return_value = mock_connection
        
        # Test different model configurations
        test_cases = [
            ("all-MiniLM-L6-v2", 384),
            ("all-mpnet-base-v2", 768),
            ("text-embedding-ada-002", 1536),
            ("text-embedding-3-small", 1536),
            ("text-embedding-3-large", 3072),
            ("unknown-model", 384)  # Should default to 384
        ]
        
        for model_name, expected_dim in test_cases:
            mock_config = Mock(spec=ConfigurationManager)
            mock_config.get_embedding_config.return_value = {
                'model': model_name,
                'dimension': expected_dim,
                'provider': 'sentence-transformers'
            }
            
            schema_manager = SchemaManager(mock_conn_mgr, mock_config)
            config = schema_manager._get_expected_schema_config("DocumentEntities")
            
            assert config["vector_dimension"] == expected_dim
            assert config["embedding_model"] == model_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])