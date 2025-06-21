"""
Focused test for document loading with 1000 documents.
This test uses a simplified approach to document loading.
"""

import pytest
import logging
import os
import sys
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for testing
os.environ["TEST_IRIS"] = "true"
os.environ["TEST_DOCUMENT_COUNT"] = "1000"
os.environ["USE_MOCK_EMBEDDINGS"] = "true"

@pytest.mark.force_testcontainer
def test_minimal_document_loading(iris_testcontainer_connection):
    """Test that we can load documents into the IRIS testcontainer."""
    assert iris_testcontainer_connection is not None
    
    # Test basic connection
    cursor = iris_testcontainer_connection.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    assert result[0] == 1
    logger.info("Basic connection verified")
    
    # Create synthetic test documents
    logger.info("Creating synthetic test documents...")
    documents = []
    for i in range(10):  # Start with a small number for quick test
        doc_id = f"doc_{uuid.uuid4()}"
        title = f"Test Document {i+1}"
        # Create some content with a reasonable length
        content = f"This is test document {i+1}. It contains synthetic content for testing document loading."
        documents.append((doc_id, title, content))
    
    # Insert documents directly into the database
    logger.info(f"Inserting {len(documents)} documents...")
    for doc in documents:
        cursor.execute(
            "INSERT INTO SourceDocuments (doc_id, title, content) VALUES (?, ?, ?)",
            doc
        )
    
    # Verify documents were loaded
    cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
    count = cursor.fetchone()[0]
    assert count >= len(documents), f"Expected at least {len(documents)} documents, found {count}"
    logger.info(f"Successfully loaded and verified {count} documents")
    
    # Fetch one document to verify content
    cursor.execute("SELECT doc_id, title, content FROM SourceDocuments LIMIT 1")
    result = cursor.fetchone()
    assert result is not None, "No document found"
    assert result[0], "Document ID is empty"
    assert result[1], "Document title is empty"
    assert result[2], "Document content is empty"
    logger.info(f"Verified document: {result[0]}, Title: {result[1]}")
    
    # Test loading with 1000 documents (quick synthetic test)
    logger.info("Testing with 1000 synthetic documents...")
    # This is a simplified version of what test_load_pmc_documents does
    
    # Clear the table first
    cursor.execute("DELETE FROM SourceDocuments")
    
    # Insert 1000 simple documents
    for i in range(1000):
        doc_id = f"doc_1000_{i}"
        title = f"Test Document {i}"
        content = f"Content for document {i}. This is a test document for the 1000 document test."
        
        cursor.execute(
            "INSERT INTO SourceDocuments (doc_id, title, content) VALUES (?, ?, ?)",
            (doc_id, title, content)
        )
    
    # Verify all 1000 documents were loaded
    cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
    count = cursor.fetchone()[0]
    assert count == 1000, f"Expected 1000 documents, found {count}"
    logger.info(f"Successfully loaded {count} documents")

    # This proves we can load 1000 documents into the database
    return True
