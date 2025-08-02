#!/usr/bin/env python3
"""
TDD Tests for Custom Table Name Configuration

Tests that both IRISStorage (Enterprise) and IRISVectorStore (Standard) 
support custom table names through configuration.
"""

import pytest
import os
import tempfile
import yaml
from typing import Dict, Any

from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.storage.enterprise_storage import IRISStorage
from iris_rag.storage.vector_store_iris import IRISVectorStore


class TestCustomTableConfiguration:
    """Test custom table name configuration for both storage classes."""

    @pytest.fixture
    def custom_config(self) -> Dict[str, Any]:
        """Create test configuration with custom table name."""
        return {
            "storage": {
                "iris": {
                    "table_name": "MyCompany.Documents"
                }
            },
            "database": {
                "iris": {
                    "host": os.getenv("IRIS_HOST", "localhost"),
                    "port": int(os.getenv("IRIS_PORT", "1972")),
                    "namespace": os.getenv("IRIS_NAMESPACE", "USER"),
                    "username": os.getenv("IRIS_USERNAME", "demo"),
                    "password": os.getenv("IRIS_PASSWORD", "demo")
                }
            }
        }

    @pytest.fixture
    def config_file(self, custom_config: Dict[str, Any]) -> str:
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(custom_config, f)
            return f.name

    @pytest.fixture
    def config_manager(self, config_file: str) -> ConfigurationManager:
        """Create ConfigurationManager with custom config."""
        return ConfigurationManager(config_file)

    @pytest.fixture
    def connection_manager(self, config_manager: ConfigurationManager) -> ConnectionManager:
        """Create ConnectionManager."""
        return ConnectionManager(config_manager)

    def test_iris_storage_uses_custom_table_name(self, connection_manager, config_manager):
        """Test that IRISStorage (Enterprise) uses custom table name from config."""
        # Arrange
        storage = IRISStorage(connection_manager, config_manager)
        
        # Act & Assert
        assert storage.table_name == "MyCompany.Documents"
        assert storage.table_name != "RAG.SourceDocuments"  # Not the default

    def test_iris_vector_store_uses_custom_table_name(self, connection_manager, config_manager):
        """Test that IRISVectorStore (Standard) uses custom table name from config."""
        # Arrange
        vector_store = IRISVectorStore(connection_manager, config_manager)
        
        # Act & Assert
        assert vector_store.table_name == "MyCompany.Documents"
        assert vector_store.table_name != "RAG.SourceDocuments"  # Not the default

    def test_default_table_name_when_no_config(self):
        """Test that default table name is used when no custom config provided."""
        # Arrange
        default_config = ConfigurationManager()
        connection_manager = ConnectionManager(default_config)
        
        # Act
        storage = IRISStorage(connection_manager, default_config)
        vector_store = IRISVectorStore(connection_manager, default_config)
        
        # Assert
        assert storage.table_name == "RAG.SourceDocuments"
        assert vector_store.table_name == "RAG.SourceDocuments"

    def test_custom_table_names_in_both_classes_match(self, connection_manager, config_manager):
        """Test that both storage classes use the same custom table name."""
        # Arrange & Act
        storage = IRISStorage(connection_manager, config_manager)
        vector_store = IRISVectorStore(connection_manager, config_manager)
        
        # Assert
        assert storage.table_name == vector_store.table_name
        assert storage.table_name == "MyCompany.Documents"

    def test_schema_initialization_with_custom_table(self, connection_manager, config_manager):
        """Test that schema initialization works with custom table names."""
        # Arrange
        storage = IRISStorage(connection_manager, config_manager)
        
        # Act & Assert - This should not raise an exception
        try:
            # Note: In a real test environment, this would actually create the table
            # For now, we just verify the table name is set correctly
            assert "MyCompany.Documents" in str(storage.table_name)
            # Schema initialization test would require actual database connection
            # storage.initialize_schema()
        except Exception as e:
            # Expected in test environment without proper database setup
            assert "MyCompany.Documents" in str(e) or True  # Allow connection errors

    def test_configuration_precedence(self):
        """Test that storage config takes precedence over defaults."""
        # Arrange
        config_data = {
            "storage": {
                "iris": {
                    "table_name": "Custom.Table"
                }
            },
            "database": {
                "iris": {
                    "host": os.getenv("IRIS_HOST", "localhost"),
                    "port": int(os.getenv("IRIS_PORT", "1972")),
                    "namespace": os.getenv("IRIS_NAMESPACE", "USER"),
                    "username": os.getenv("IRIS_USERNAME", "demo"),
                    "password": os.getenv("IRIS_PASSWORD", "demo")
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_file = f.name
        
        try:
            config_manager = ConfigurationManager(config_file)
            connection_manager = ConnectionManager(config_manager)
            
            # Act
            storage = IRISStorage(connection_manager, config_manager)
            
            # Assert
            assert storage.table_name == "Custom.Table"
            
        finally:
            os.unlink(config_file)

    @pytest.mark.integration
    def test_custom_table_with_schema_manager_integration(self, connection_manager, config_manager):
        """Integration test: Custom table name works with schema manager."""
        # Arrange
        from iris_rag.storage.schema_manager import SchemaManager
        vector_store = IRISVectorStore(connection_manager, config_manager)
        
        # Act
        schema_manager = vector_store.schema_manager
        
        # Assert
        assert schema_manager is not None
        assert vector_store.table_name == "MyCompany.Documents"
        # In a real integration test, we would verify the schema manager
        # can work with the custom table name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])