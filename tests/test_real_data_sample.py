# tests/test_real_data_sample.py
# Example tests demonstrating how to use the real data fixtures

import pytest
import os
import sys

# Make sure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.utils import Document

# --- Sample Tests Using Real Data Fixtures ---

@pytest.mark.force_real
def test_with_real_data_only(iris_connection, use_real_data):
    """
    This test will only run with real data and will be skipped if real data is not available.
    The force_real marker ensures this.
    """
    assert use_real_data is True
    
    # Test a simple query to make sure we have real data
    cursor = iris_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
    result = cursor.fetchone()
    cursor.close()
    
    # Verify we have data
    assert result is not None
    assert result[0] > 0
    print(f"Found {result[0]} documents in real database")

@pytest.mark.force_mock
def test_with_mock_data_only(iris_connection, use_real_data):
    """
    This test will always use mock data, even if real data is available.
    The force_mock marker ensures this.
    """
    assert use_real_data is False
    
    # Get the mock cursor
    cursor = iris_connection.cursor()
    
    # Insert some mock data
    mock_docs = [
        ("doc1", "Test Title 1", "Test Content 1", "[]", "[]"),
        ("doc2", "Test Title 2", "Test Content 2", "[]", "[]")
    ]
    cursor.executemany(
        "INSERT INTO SourceDocuments (doc_id, title, content, authors, keywords) VALUES (?, ?, ?, ?, ?)",
        mock_docs
    )
    
    # Verify the data was "inserted" into our mock
    cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
    result = cursor.fetchone()
    cursor.close()
    
    assert result is not None
    # Convert to int if it's a string (mock implementation might return strings)
    count = int(result[0]) if isinstance(result[0], str) else result[0]
    assert count >= 2  # At least the 2 we inserted
    print(f"Found {count} documents in mock database")

@pytest.mark.real_data
def test_with_adaptive_behavior(iris_connection, use_real_data):
    """
    This test adapts its behavior based on whether real data is available.
    The real_data marker is just for categorization, the use_real_data flag determines behavior.
    """
    cursor = iris_connection.cursor()
    
    if use_real_data:
        print("Using real data for test")
        # Test with real data - just check that we can query it
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        result = cursor.fetchone()
        assert result is not None
        print(f"Found {result[0]} documents in real database")
    else:
        print("Using mock data for test")
        # Test with mock data - insert some test data first
        mock_docs = [
            ("adaptive_doc1", "Adaptive Title 1", "Adaptive Content 1", "[]", "[]"),
            ("adaptive_doc2", "Adaptive Title 2", "Adaptive Content 2", "[]", "[]")
        ]
        cursor.executemany(
            "INSERT INTO SourceDocuments (doc_id, title, content, authors, keywords) VALUES (?, ?, ?, ?, ?)",
            mock_docs
        )
        
        # Then retrieve and verify
        cursor.execute("SELECT doc_id, title FROM SourceDocuments WHERE doc_id LIKE 'adaptive_%'")
        results = cursor.fetchall()
        assert len(results) == 2
        print(f"Successfully inserted and retrieved mock documents")
    
    cursor.close()

# --- Sample Test Using Multiple Fixtures ---

@pytest.mark.real_data
def test_combined_fixtures(iris_connection, use_real_data, mocker):
    """
    This test demonstrates combining real/mock database with mock embedding function.
    It shows how to mix real and mock components in tests.
    """
    # Create a simple mock embedding function
    mock_embedding = mocker.Mock(return_value=[[0.1, 0.2, 0.3, 0.4, 0.5]])
    
    # Use the database connection (real or mock based on availability)
    cursor = iris_connection.cursor()
    cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
    result = cursor.fetchone()
    
    # Handle potential None result or string values
    if result is None:
        doc_count = 0
    else:
        doc_count = int(result[0]) if isinstance(result[0], str) else result[0]
    
    print(f"Database has {doc_count} documents (using {'real' if use_real_data else 'mock'} data)")
    
    # Use the mock embedding function
    text = "This is a test document"
    embeddings = mock_embedding(text)
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0
    print(f"Generated mock embedding with {len(embeddings[0])} dimensions")
    
    # This is where you would combine the two in a real test
    # For example, retrieving documents and generating embeddings for them
    cursor.close()

# --- Example Integration Test That Can Use Real Data ---

@pytest.mark.integration
def test_document_retrieval_with_embedding(iris_connection, use_real_data, mock_embedding_func):
    """
    An integration test that simulates retrieving documents and generating embeddings.
    This could be part of a real pipeline test.
    """
    # Get a document from the database
    cursor = iris_connection.cursor()
    
    if use_real_data:
        # With real data, we get an actual document
        cursor.execute("SELECT doc_id, content FROM SourceDocuments LIMIT 1")
    else:
        # With mock data, we need to insert one first
        doc_id = "test_retrieval_doc"
        content = "This is a test document for retrieval and embedding generation."
        cursor.execute(
            "INSERT INTO SourceDocuments (doc_id, title, content) VALUES (?, ?, ?)",
            (doc_id, "Test Title", content)
        )
        cursor.execute("SELECT doc_id, content FROM SourceDocuments WHERE doc_id = ?", (doc_id,))
    
    row = cursor.fetchone()
    cursor.close()
    
    if row is None:
        pytest.skip("No documents available in the database")
    
    doc_id, content = row
    
    # Create a Document object
    document = Document(id=doc_id, content=content)
    print(f"Retrieved document: {document.id} (length: {len(document.content)} chars)")
    
    # Generate embedding
    embedding = mock_embedding_func(document.content)
    document.embedding = embedding[0]
    print(f"Generated embedding with {len(document.embedding)} dimensions")
    
    # Verify we have a proper document with embedding
    assert document.id is not None
    assert len(document.content) > 0
    assert document.embedding is not None
    assert len(document.embedding) > 0
