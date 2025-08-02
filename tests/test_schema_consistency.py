#!/usr/bin/env python3
"""
Test schema consistency and standardization.
"""

import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.database_schema_manager import get_schema_manager

@pytest.mark.schema
def test_schema_manager_initialization():
    """Test that schema manager initializes correctly."""
    manager = get_schema_manager()
    assert manager is not None
    assert manager.get_schema_name() == "RAG"

@pytest.mark.schema
def test_table_name_consistency():
    """Test that all expected tables have consistent naming."""
    manager = get_schema_manager()
    
    expected_tables = [
        'source_documents',
        'document_chunks', 
        'document_entities',
        'document_token_embeddings',
        'ifind_index'
    ]
    
    for table_key in expected_tables:
        # Should not raise exception
        table_name = manager.get_table_name(table_key)
        assert table_name.startswith("RAG.")
        assert len(table_name) > 4  # More than just "RAG."

@pytest.mark.schema
def test_column_name_standardization():
    """Test column name standardization across tables."""
    manager = get_schema_manager()
    
    # Test document ID standardization across tables that reference documents
    document_tables = ['source_documents', 'document_chunks', 'document_token_embeddings', 'document_entities']
    
    for table_key in document_tables:
        if manager.validate_table_exists(table_key):
            try:
                doc_id_col = manager.get_column_name(table_key, 'doc_id')
                assert doc_id_col in ['doc_id', 'document_id'], f"Table {table_key} has unexpected doc_id column: {doc_id_col}"
            except ValueError:
                # Some tables might not have doc_id (that's ok)
                pass
    
    # Test that primary key columns exist where expected
    tables_with_primary_keys = {
        'source_documents': 'id',
        'document_chunks': 'id',  # maps to chunk_id
        'document_entities': 'entity_id'
    }
    
    for table_key, pk_column in tables_with_primary_keys.items():
        if manager.validate_table_exists(table_key):
            try:
                pk_col = manager.get_column_name(table_key, pk_column)
                assert pk_col is not None, f"Table {table_key} missing primary key column {pk_column}"
            except ValueError as e:
                pytest.fail(f"Table {table_key} missing expected column {pk_column}: {e}")

@pytest.mark.schema 
def test_create_table_sql_generation():
    """Test SQL generation for table creation."""
    manager = get_schema_manager()
    
    sql = manager.build_create_table_sql('source_documents')
    assert "CREATE TABLE IF NOT EXISTS RAG.SourceDocuments" in sql
    assert "doc_id" in sql
    assert "text_content" in sql

@pytest.mark.schema
def test_all_tables_accessible():
    """Test that all configured tables are accessible."""
    manager = get_schema_manager()
    all_tables = manager.get_all_tables()
    
    assert len(all_tables) >= 5  # At least 5 main tables
    for table_key, table_name in all_tables.items():
        assert table_name.startswith("RAG.")
        
        # Should be able to get table info
        info = manager.get_table_info(table_key)
        assert info['key'] == table_key
        assert info['name'] is not None
        assert isinstance(info['columns'], dict)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])