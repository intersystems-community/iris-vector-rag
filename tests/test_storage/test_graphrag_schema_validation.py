#!/usr/bin/env python3
"""
Validation tests for GraphRAG schema creation through SchemaManager.

This module tests that the DocumentEntities and EntityRelationships tables
can be created successfully and have the correct structure for GraphRAG functionality.
"""

import pytest
import logging
from typing import Dict, Any

from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.config.manager import ConfigurationManager
from common.iris_connection_manager import get_iris_connection

logger = logging.getLogger(__name__)


class TestGraphRAGSchemaValidation:
    """Test suite for GraphRAG schema validation."""

    @pytest.fixture
    def connection_manager(self):
        """Create a connection manager for testing."""
        return type('ConnectionManager', (), {
            'get_connection': lambda self: get_iris_connection()
        })()

    @pytest.fixture
    def config_manager(self):
        """Create a configuration manager for testing."""
        return ConfigurationManager()

    @pytest.fixture
    def schema_manager(self, connection_manager, config_manager):
        """Create a schema manager for testing."""
        return SchemaManager(connection_manager, config_manager)

    def test_document_entities_table_config(self, schema_manager):
        """Test that DocumentEntities table configuration is properly defined."""
        config = schema_manager.get_table_config("DocumentEntities")
        
        assert config is not None, "DocumentEntities table configuration should exist"
        assert config["embedding_column"] == "embedding", "Should use 'embedding' column for vectors"
        assert config["uses_document_embeddings"] is True, "Should use document embeddings"
        assert config["dimension"] == schema_manager.base_embedding_dimension, "Should use base embedding dimension"
        
        # Verify column mappings
        expected_columns = {
            "entity_id": "entity_id",
            "doc_id": "doc_id", 
            "entity_name": "entity_name",
            "entity_type": "entity_type",
            "embedding": "embedding"
        }
        assert config["columns"] == expected_columns, "Column mappings should match expected structure"

    def test_entity_relationships_table_config(self, schema_manager):
        """Test that EntityRelationships table configuration is properly defined."""
        config = schema_manager.get_table_config("EntityRelationships")
        
        assert config is not None, "EntityRelationships table configuration should exist"
        assert config["embedding_column"] is None, "Should not use embeddings"
        assert config["uses_document_embeddings"] is False, "Should not use document embeddings"
        assert config["dimension"] is None, "Should not have vector dimension"
        
        # Verify column mappings
        expected_columns = {
            "relationship_id": "relationship_id",
            "source_entity_id": "source_entity_id",
            "target_entity_id": "target_entity_id", 
            "relationship_type": "relationship_type",
            "metadata": "metadata"
        }
        assert config["columns"] == expected_columns, "Column mappings should match expected structure"

    def test_document_entities_schema_creation(self, schema_manager):
        """Test that DocumentEntities table can be created successfully."""
        # Ensure the table schema is created
        success = schema_manager.ensure_table_schema("DocumentEntities")
        assert success, "DocumentEntities table schema creation should succeed"
        
        # Verify the table exists and has correct structure
        connection = schema_manager.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Check table exists
            cursor.execute("""
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentEntities'
            """)
            table_exists = cursor.fetchone()[0] > 0
            assert table_exists, "DocumentEntities table should exist after schema creation"
            
            # Check required columns exist
            cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentEntities'
                ORDER BY COLUMN_NAME
            """)
            columns = {row[0]: row[1] for row in cursor.fetchall()}
            
            required_columns = {
                "entity_id": "VARCHAR",
                "doc_id": "VARCHAR", 
                "entity_name": "VARCHAR",
                "entity_type": "VARCHAR",
                "embedding": "VECTOR",
                "created_at": "TIMESTAMP"
            }
            
            for col_name, expected_type in required_columns.items():
                assert col_name in columns, f"Column {col_name} should exist"
                if expected_type != "VECTOR":  # VECTOR type checking is more complex
                    assert expected_type in columns[col_name], f"Column {col_name} should be {expected_type} type"
                    
        finally:
            cursor.close()

    def test_entity_relationships_schema_creation(self, schema_manager):
        """Test that EntityRelationships table can be created successfully."""
        # Ensure the table schema is created
        success = schema_manager.ensure_table_schema("EntityRelationships")
        assert success, "EntityRelationships table schema creation should succeed"
        
        # Verify the table exists and has correct structure
        connection = schema_manager.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Check table exists
            cursor.execute("""
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'EntityRelationships'
            """)
            table_exists = cursor.fetchone()[0] > 0
            assert table_exists, "EntityRelationships table should exist after schema creation"
            
            # Check required columns exist
            cursor.execute("""
                SELECT COLUMN_NAME, DATA_TYPE FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'EntityRelationships'
                ORDER BY COLUMN_NAME
            """)
            columns = {row[0]: row[1] for row in cursor.fetchall()}
            
            required_columns = {
                "relationship_id": "VARCHAR",
                "source_entity_id": "VARCHAR",
                "target_entity_id": "VARCHAR",
                "relationship_type": "VARCHAR", 
                "metadata": "VARCHAR",
                "created_at": "TIMESTAMP"
            }
            
            for col_name, expected_type in required_columns.items():
                assert col_name in columns, f"Column {col_name} should exist"
                assert expected_type in columns[col_name], f"Column {col_name} should be {expected_type} type"
                    
        finally:
            cursor.close()

    def test_foreign_key_relationships(self, schema_manager):
        """Test that foreign key relationships are properly established."""
        # Ensure both tables are created
        schema_manager.ensure_table_schema("SourceDocuments")
        schema_manager.ensure_table_schema("DocumentEntities") 
        schema_manager.ensure_table_schema("EntityRelationships")
        
        connection = schema_manager.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Check foreign key constraints exist
            cursor.execute("""
                SELECT CONSTRAINT_NAME, TABLE_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                FROM INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS rc
                JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu 
                ON rc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
                WHERE rc.CONSTRAINT_SCHEMA = 'RAG'
                AND (TABLE_NAME = 'DocumentEntities' OR TABLE_NAME = 'EntityRelationships')
            """)
            
            constraints = cursor.fetchall()
            constraint_info = [(row[1], row[2], row[3]) for row in constraints]
            
            # DocumentEntities should reference SourceDocuments
            doc_entities_fk = any(
                table == 'DocumentEntities' and ref_table == 'SourceDocuments' and ref_col == 'doc_id'
                for table, ref_table, ref_col in constraint_info
            )
            assert doc_entities_fk, "DocumentEntities should have foreign key to SourceDocuments.doc_id"
            
            # EntityRelationships should reference DocumentEntities twice
            entity_rel_fks = [
                (table, ref_table, ref_col) for table, ref_table, ref_col in constraint_info
                if table == 'EntityRelationships' and ref_table == 'DocumentEntities'
            ]
            assert len(entity_rel_fks) >= 2, "EntityRelationships should have two foreign keys to DocumentEntities"
                    
        finally:
            cursor.close()

    def test_schema_metadata_tracking(self, schema_manager):
        """Test that schema metadata is properly tracked for GraphRAG tables."""
        # Ensure tables are created
        schema_manager.ensure_table_schema("DocumentEntities")
        schema_manager.ensure_table_schema("EntityRelationships")
        
        # Check DocumentEntities metadata
        doc_entities_config = schema_manager.get_current_schema_config("DocumentEntities")
        assert doc_entities_config is not None, "DocumentEntities should have schema metadata"
        assert doc_entities_config["vector_dimension"] == schema_manager.base_embedding_dimension
        assert doc_entities_config["embedding_model"] == schema_manager.base_embedding_model
        
        # Check EntityRelationships metadata  
        entity_rel_config = schema_manager.get_current_schema_config("EntityRelationships")
        assert entity_rel_config is not None, "EntityRelationships should have schema metadata"
        assert entity_rel_config["vector_dimension"] is None, "EntityRelationships should not have vector dimension"

    def test_graphrag_tables_migration_support(self, schema_manager):
        """Test that GraphRAG tables support migration when configuration changes."""
        # Test that migration detection works
        needs_migration_entities = schema_manager.needs_migration("DocumentEntities")
        needs_migration_relationships = schema_manager.needs_migration("EntityRelationships")
        
        # After ensuring schema, migration should not be needed
        schema_manager.ensure_table_schema("DocumentEntities")
        schema_manager.ensure_table_schema("EntityRelationships")
        
        assert not schema_manager.needs_migration("DocumentEntities"), "DocumentEntities should not need migration after creation"
        assert not schema_manager.needs_migration("EntityRelationships"), "EntityRelationships should not need migration after creation"

    def test_vector_dimension_consistency(self, schema_manager):
        """Test that vector dimensions are consistent across GraphRAG tables."""
        # DocumentEntities should use document embeddings
        entities_dimension = schema_manager.get_vector_dimension("DocumentEntities")
        expected_dimension = schema_manager.base_embedding_dimension
        
        assert entities_dimension == expected_dimension, f"DocumentEntities dimension should be {expected_dimension}"
        
        # EntityRelationships should not have vector dimension
        relationships_config = schema_manager.get_table_config("EntityRelationships")
        assert relationships_config["dimension"] is None, "EntityRelationships should not have vector dimension"

    def test_graphrag_schema_status(self, schema_manager):
        """Test that schema status reporting works for GraphRAG tables."""
        status = schema_manager.get_schema_status()
        
        assert "DocumentEntities" in status, "DocumentEntities should be in schema status"
        assert "EntityRelationships" in status, "EntityRelationships should be in schema status"
        
        # Check status structure
        entities_status = status["DocumentEntities"]
        assert "current_config" in entities_status
        assert "expected_config" in entities_status
        assert "needs_migration" in entities_status
        assert "vector_dimension" in entities_status
        
        relationships_status = status["EntityRelationships"] 
        assert "current_config" in relationships_status
        assert "expected_config" in relationships_status
        assert "needs_migration" in relationships_status