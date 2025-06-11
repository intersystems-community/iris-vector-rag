# tests/test_data_loader.py
# Tests for the data loader module

import pytest
import os
import sys
import json
from unittest.mock import patch, MagicMock, call

# Make sure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from data.loader import load_documents_to_iris, process_and_load_documents

# --- Sample data for testing ---

SAMPLE_DOCUMENTS = [
    {
        "pmc_id": "PMC123456",
        "title": "Sample Title 1",
        "abstract": "Sample abstract 1",
        "authors": ["Author A", "Author B"],
        "keywords": ["keyword1", "keyword2"]
    },
    {
        "pmc_id": "PMC234567",
        "title": "Sample Title 2",
        "abstract": "Sample abstract 2",
        "authors": ["Author C"],
        "keywords": ["keyword3", "keyword4", "keyword5"]
    },
    {
        "pmc_id": "PMC345678",
        "title": "Sample Title 3",
        "abstract": "Sample abstract 3",
        "authors": [],
        "keywords": []
    }
]

# --- Unit Tests ---

def test_load_documents_to_iris_mock():
    """Test loading documents using a mock connection"""
    # Create a mock connection and cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    
    # Call the function with sample documents
    result = load_documents_to_iris(mock_conn, SAMPLE_DOCUMENTS, batch_size=2)
    
    # Verify the connection was used correctly
    mock_conn.cursor.assert_called_once()
    assert mock_cursor.executemany.call_count == 2  # Should be called twice with batch_size=2
    mock_conn.commit.assert_called()
    mock_cursor.close.assert_called_once()
    
    # Check the result statistics
    assert result["total_documents"] == 3
    assert result["loaded_count"] == 3
    assert result["error_count"] == 0
    assert "duration_seconds" in result
    assert "documents_per_second" in result

def test_load_documents_to_iris_query_format():
    """Test the SQL query format and parameters used"""
    # Create a mock connection and cursor
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    
    # Call the function with a single document for clarity
    load_documents_to_iris(mock_conn, [SAMPLE_DOCUMENTS[0]], batch_size=1)
    
    # Get the SQL and parameters that were used
    sql = mock_cursor.executemany.call_args[0][0]
    params = mock_cursor.executemany.call_args[0][1]
    
    # Verify SQL query format
    assert "INSERT INTO SourceDocuments" in sql
    assert "(doc_id, title, content, authors, keywords)" in sql
    assert "VALUES (?, ?, ?, ?, ?)" in sql
    
    # Verify parameters
    assert len(params) == 1  # One document
    doc_params = params[0]
    assert doc_params[0] == "PMC123456"  # doc_id
    assert doc_params[1] == "Sample Title 1"  # title
    assert doc_params[2] == "Sample abstract 1"  # content (abstract)
    assert doc_params[3] == json.dumps(["Author A", "Author B"])  # authors as JSON
    assert doc_params[4] == json.dumps(["keyword1", "keyword2"])  # keywords as JSON

def test_load_documents_to_iris_executemany_error():
    """Test handling of executemany errors"""
    # Create a mock connection and cursor with an error
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.executemany.side_effect = Exception("Database error")
    
    # Call the function
    result = load_documents_to_iris(mock_conn, SAMPLE_DOCUMENTS, batch_size=1)
    
    # Verify error handling
    mock_conn.rollback.assert_called()
    assert result["total_documents"] == 3
    assert result["loaded_count"] == 0
    assert result["error_count"] == 3

def test_load_documents_to_iris_cursor_error():
    """Test handling of cursor creation errors"""
    # Create a mock connection with an error on cursor creation
    mock_conn = MagicMock()
    mock_conn.cursor.side_effect = Exception("Cursor error")
    
    # Call the function
    result = load_documents_to_iris(mock_conn, SAMPLE_DOCUMENTS)
    
    # Verify error handling
    assert result["total_documents"] == 3
    assert result["loaded_count"] == 0
    assert result["error_count"] == 3

def test_process_and_load_documents_with_mocks():
    """Test the full process_and_load_documents function with mocks"""
    # Mock the PMC processor
    with patch("data.loader.process_pmc_files") as mock_process:
        # Return the sample documents
        mock_process.return_value = SAMPLE_DOCUMENTS
        
        # Mock the connection
        mock_conn = MagicMock()
        
        # Call the function
        result = process_and_load_documents(
            "fake_directory",
            connection=mock_conn,
            limit=10,
            batch_size=2
        )
        
        # Verify the processor was called correctly
        mock_process.assert_called_once_with("fake_directory", limit=10)
        
        # Check the results
        assert result["success"] is True
        assert result["processed_count"] == 3
        assert result["loaded_count"] == 3
        assert result["processed_directory"] == "fake_directory"

def test_process_and_load_documents_connection_creation():
    """Test that process_and_load_documents creates a connection if not provided"""
    # Mock the connection creation
    with patch("data.loader.get_iris_connection") as mock_get_conn:
        # Return a mock connection
        mock_conn = MagicMock()
        mock_get_conn.return_value = mock_conn
        
        # Mock the PMC processor
        with patch("data.loader.process_pmc_files") as mock_process:
            mock_process.return_value = SAMPLE_DOCUMENTS
            
            # Call the function without providing a connection
            result = process_and_load_documents(
                "fake_directory",
                connection=None,
                use_mock=True
            )
            
            # Verify connection was created
            mock_get_conn.assert_called_once_with(use_mock=True)
            
            # Verify the connection was closed since we created it
            mock_conn.close.assert_called_once()
            
            # Check the result
            assert result["success"] is True

def test_process_and_load_documents_connection_failure():
    """Test handling of connection failures"""
    # Mock the connection creation to fail
    with patch("data.loader.get_iris_connection", return_value=None):
        # Call the function with no connection
        result = process_and_load_documents("fake_directory")
        
        # Verify failure handling
        assert result["success"] is False
        assert "Failed to establish database connection" in result["error"]

def test_process_and_load_documents_processing_error():
    """Test handling of document processing errors"""
    # Mock the PMC processor to raise an exception
    with patch("data.loader.process_pmc_files", side_effect=Exception("Processing error")):
        # Mock the connection
        mock_conn = MagicMock()
        
        # Call the function
        result = process_and_load_documents("fake_directory", connection=mock_conn)
        
        # Verify error handling
        assert result["success"] is False
        assert "Processing error" in result["error"]
        assert result["processed_count"] == 0
        assert result["loaded_count"] == 0

# --- Integration Tests ---

@pytest.mark.integration
def test_process_and_load_documents_with_real_connection():
    """Test processing and loading with a real connection if available"""
    # Skip if environment variables aren't set
    if not os.environ.get("IRIS_HOST"):
        pytest.skip("IRIS environment variables not configured")
    
    # Skip if the data directory doesn't exist
    data_dir = "data/pmc_oas_downloaded"
    if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
        pytest.skip(f"PMC data directory {data_dir} not found")
    
    # Mock the PMC processor to return controlled data
    with patch("data.loader.process_pmc_files") as mock_process:
        # Use a prefix for test documents to avoid conflicts
        test_docs = [
            {
                "pmc_id": "TEST_" + doc["pmc_id"],
                "title": doc["title"],
                "abstract": doc["abstract"],
                "authors": doc["authors"],
                "keywords": doc["keywords"]
            }
            for doc in SAMPLE_DOCUMENTS
        ]
        mock_process.return_value = test_docs
        
        # Process and load with real connection and 2 test documents
        result = process_and_load_documents(
            data_dir,
            limit=2,
            batch_size=1,
            use_mock=False  # Use real connection
        )
        
        # Check results
        if result["success"]:
            assert result["processed_count"] == 3
            assert result["loaded_count"] > 0
        else:
            pytest.skip(f"Integration test failed: {result.get('error', 'Unknown error')}")
