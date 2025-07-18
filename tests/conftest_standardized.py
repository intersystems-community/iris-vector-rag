#!/usr/bin/env python3
"""
Standardized pytest configuration and fixtures for RAG Templates.
Provides consistent database setup, cleanup, and test data management.
"""

import pytest
import logging
import sys
import os
from pathlib import Path
from typing import Generator, Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.database_schema_manager import get_schema_manager
from common.iris_connector import get_iris_connection

logger = logging.getLogger(__name__)

# Test configuration
TEST_CONFIG = {
    'sample_sizes': {
        'small': 10,
        'medium': 100, 
        'large': 1000
    },
    'test_schema': 'TEST_RAG',
    'cleanup_after_tests': True
}

@pytest.fixture(scope="session")
def schema_manager():
    """Provide schema manager for all tests."""
    return get_schema_manager()

@pytest.fixture(scope="session") 
def db_connection():
    """Provide database connection for all tests."""
    try:
        conn = get_iris_connection()
        yield conn
    finally:
        if 'conn' in locals():
            conn.close()

@pytest.fixture(scope="function")
def clean_test_tables(db_connection, schema_manager):
    """Clean test tables before and after each test."""
    cursor = db_connection.cursor()
    
    # Get all table names
    table_names = [
        schema_manager.get_table_name(key, fully_qualified=True)
        for key in schema_manager.get_all_tables().keys()
    ]
    
    def cleanup():
        for table_name in table_names:
            try:
                cursor.execute(f"DELETE FROM {table_name}")
                db_connection.commit()
                logger.debug(f"Cleaned table: {table_name}")
            except Exception as e:
                logger.warning(f"Could not clean {table_name}: {e}")
    
    # Clean before test
    cleanup()
    
    yield
    
    # Clean after test if configured
    if TEST_CONFIG['cleanup_after_tests']:
        cleanup()

@pytest.fixture(scope="function")
def sample_documents(schema_manager) -> List[Dict[str, Any]]:
    """Provide sample test documents."""
    return [
        {
            schema_manager.get_column_name('source_documents', 'id'): f"test_doc_{i}",
            schema_manager.get_column_name('source_documents', 'title'): f"Test Document {i}",
            schema_manager.get_column_name('source_documents', 'content'): f"This is test content for document {i}. " * 10,
            schema_manager.get_column_name('source_documents', 'metadata'): '{"test": true}'
        }
        for i in range(TEST_CONFIG['sample_sizes']['small'])
    ]

@pytest.fixture(scope="function") 
def insert_sample_data(db_connection, schema_manager, sample_documents, clean_test_tables):
    """Insert sample data into test tables."""
    cursor = db_connection.cursor()
    
    # Insert into SourceDocuments
    table_name = schema_manager.get_table_name('source_documents')
    doc_id_col = schema_manager.get_column_name('source_documents', 'id')
    title_col = schema_manager.get_column_name('source_documents', 'title')
    content_col = schema_manager.get_column_name('source_documents', 'content')
    metadata_col = schema_manager.get_column_name('source_documents', 'metadata')
    
    sql = f"""
    INSERT INTO {table_name} 
    ({doc_id_col}, {title_col}, {content_col}, {metadata_col})
    VALUES (?, ?, ?, ?)
    """
    
    for doc in sample_documents:
        cursor.execute(sql, (
            doc[doc_id_col],
            doc[title_col], 
            doc[content_col],
            doc[metadata_col]
        ))
    
    db_connection.commit()
    logger.info(f"Inserted {len(sample_documents)} test documents")
    
    yield sample_documents

@pytest.fixture(params=['small', 'medium', 'large'])
def test_size(request):
    """Parameterized fixture for different test sizes."""
    return TEST_CONFIG['sample_sizes'][request.param]

# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers and settings."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test") 
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "pipeline: mark test as pipeline-specific")
    config.addinivalue_line("markers", "schema: mark test as schema-related")

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add markers based on test file location
        if "test_pipelines" in str(item.fspath):
            item.add_marker(pytest.mark.pipeline)
        if "test_schema" in str(item.fspath):
            item.add_marker(pytest.mark.schema)
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

# Helper functions for tests
def get_test_table_name(schema_manager, table_key: str) -> str:
    """Get test table name with TEST_ prefix."""
    base_name = schema_manager.get_table_name(table_key, fully_qualified=False)
    return f"{TEST_CONFIG['test_schema']}.TEST_{base_name}"

def assert_table_count(db_connection, schema_manager, table_key: str, expected_count: int):
    """Assert that a table has the expected number of rows."""
    cursor = db_connection.cursor()
    table_name = schema_manager.get_table_name(table_key)
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    actual_count = cursor.fetchone()[0]
    assert actual_count == expected_count, f"Expected {expected_count} rows in {table_name}, got {actual_count}"