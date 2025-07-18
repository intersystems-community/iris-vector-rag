# tests/test_index_build.py
# Tests for the index building process.

import pytest
import time
from typing import List, Dict, Any
from unittest.mock import call, patch, MagicMock

# Assuming mock fixtures are available from conftest.py

def test_table_creation(mock_iris_connector):
    """
    Test that all required database tables are created.
    """
    print("\nTest: test_table_creation")
    
    from common.db_init import initialize_database
    
    # Setup mock cursor to capture executed SQL
    mock_cursor = mock_iris_connector.cursor.return_value
    
    # Call the function under test
    initialize_database(mock_iris_connector)
    
    # Get all CREATE TABLE statements
    create_table_calls = [call for call in mock_cursor.execute.call_args_list 
                         if "CREATE TABLE" in call[0][0].upper()]
    
    # Should create at least these tables
    required_tables = ["SOURCEDOCUMENTS", "DOCUMENTTOKENEMBEDDINGS", 
                      "KNOWLEDGEGRAPHNODES", "KNOWLEDGEGRAPHEDGES"]
    
    # Verify each required table was created
    for table in required_tables:
        table_created = any(table in call[0][0].upper() for call in create_table_calls)
        assert table_created, f"Table {table} was not created"
    
    print("Table creation test passed.")

def test_hnsw_index_creation(mock_iris_connector):
    """
    Test that HNSW indexes are created successfully.
    """
    print("\nTest: test_hnsw_index_creation")
    
    from common.db_init import initialize_database
    
    # Setup mock cursor
    mock_cursor = mock_iris_connector.cursor.return_value
    
    # Call the function under test
    initialize_database(mock_iris_connector)
    
    # Verify HNSW index creation statements were executed
    # IRIS syntax for HNSW indexes is "CREATE INDEX ... AS HNSW"
    hnsw_calls = [call for call in mock_cursor.execute.call_args_list 
                 if "AS HNSW" in call[0][0].upper()]
    
    # Should have at least 3 HNSW indexes (for SourceDocuments, DocumentTokenEmbeddings, KnowledgeGraphNodes)
    assert len(hnsw_calls) >= 3, "Not enough HNSW indexes created"
    
    # Verify each table has an index
    required_index_tables = ["SOURCEDOCUMENTS", "DOCUMENTTOKENEMBEDDINGS", "KNOWLEDGEGRAPHNODES"]
    for table in required_index_tables:
        index_created = any(table in call[0][0].upper() for call in hnsw_calls)
        assert index_created, f"HNSW index for {table} was not created"
    
    # Check for proper index parameters
    for index_call in hnsw_calls:
        sql = index_call[0][0].upper()
        assert "COSINE" in sql or "L2" in sql, "Missing distance metric in HNSW index"
        assert "EFCONSTRUCTION" in sql, "Missing efConstruction parameter in HNSW index"
    
    print("HNSW index creation test passed.")

def test_standard_index_creation(mock_iris_connector):
    """
    Test that standard indexes are created on key columns.
    """
    print("\nTest: test_standard_index_creation")
    
    from common.db_init import initialize_database
    
    # Setup mock cursor
    mock_cursor = mock_iris_connector.cursor.return_value
    
    # Call the function under test
    initialize_database(mock_iris_connector)
    
    # Verify standard index creation statements
    index_calls = [call for call in mock_cursor.execute.call_args_list 
                  if "CREATE INDEX" in call[0][0].upper() and "HNSW" not in call[0][0].upper()]
    
    # Check for indexes on important columns like doc_id, node_id, etc.
    important_columns = ["DOC_ID", "NODE_ID", "SOURCE_NODE_ID", "TARGET_NODE_ID", "NODE_TYPE"]
    
    for column in important_columns:
        column_indexed = any(column in call[0][0].upper() for call in index_calls)
        assert column_indexed, f"Column {column} is not indexed"
    
    print("Standard index creation test passed.")

def test_views_creation(mock_iris_connector):
    """
    Test that required views are created.
    """
    print("\nTest: test_views_creation")
    
    from common.db_init import initialize_database
    
    # Setup mock cursor
    mock_cursor = mock_iris_connector.cursor.return_value
    
    # Call the function under test
    initialize_database(mock_iris_connector)
    
    # Verify view creation statements
    view_calls = [call for call in mock_cursor.execute.call_args_list 
                  if "CREATE VIEW" in call[0][0].upper()]
    
    # Check for kg_edges view specifically
    kg_edges_view_created = any("KG_EDGES" in call[0][0].upper() for call in view_calls)
    
    assert kg_edges_view_created, "kg_edges view was not created"
    
    print("Views creation test passed.")

def test_initialize_database_error_handling(mock_iris_connector):
    """
    Test that database initialization handles errors properly.
    """
    print("\nTest: test_initialize_database_error_handling")
    
    from common.db_init import initialize_database
    
    # Setup mock cursor to raise an exception
    mock_cursor = mock_iris_connector.cursor.return_value
    mock_cursor.execute.side_effect = Exception("SQL execution failed")
    
    # Call the function under test and expect it to handle the error
    with pytest.raises(Exception):
        initialize_database(mock_iris_connector)
    
    # The test passes if an exception is raised and not swallowed silently
    print("Initialize database error handling test passed.")

def test_index_build_time_small_sample(mock_iris_connector):
    """
    Test that index building time for a small sample dataset is within a reasonable limit.
    (Note: This is a unit test placeholder; actual performance is benchmarked separately).
    """
    print("\nTest: test_index_build_time_small_sample")
    
    from common.db_init import initialize_database
    
    # Setup to measure time
    start_time = time.time()
    
    # Call the function under test
    initialize_database(mock_iris_connector)
    
    # Measure elapsed time
    elapsed_time = time.time() - start_time
    
    # For unit tests with mocks, this should be very fast
    assert elapsed_time < 1.0, f"Index building took too long: {elapsed_time:.2f} seconds"
    
    print(f"Index build time test passed: {elapsed_time:.4f} seconds")

@patch('builtins.open')
def test_sql_file_reading(mock_open, mock_iris_connector):
    """
    Test that the database initialization reads the SQL file properly.
    """
    print("\nTest: test_sql_file_reading")
    
    from common.db_init import initialize_database
    
    # Setup mock for file reading
    mock_file = MagicMock()
    mock_file.__enter__.return_value = mock_file
    mock_file.read.return_value = """
    -- Test SQL content
    CREATE TABLE TestTable (id INT);
    CREATE HNSW INDEX idx_hnsw_test ON TestTable (col) WITH (m = 16);
    """
    mock_open.return_value = mock_file
    
    # Call the function under test
    initialize_database(mock_iris_connector)
    
    # Verify the file was opened
    mock_open.assert_called_once()
    
    # Verify the file was read
    mock_file.read.assert_called_once()
    
    print("SQL file reading test passed.")
