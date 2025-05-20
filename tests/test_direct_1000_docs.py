"""
Direct test for loading 1000 documents with minimal dependencies.
"""
import pytest
import logging
import os
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set environment variables for 1000 document testing
os.environ["TEST_IRIS"] = "true"
os.environ["TEST_DOCUMENT_COUNT"] = "1000"
os.environ["USE_MOCK_EMBEDDINGS"] = "true"

# Create a direct test that doesn't rely on other functions
@pytest.mark.force_testcontainer
def test_direct_loading_1000_docs(iris_testcontainer_connection):
    """Test loading 1000 documents directly using SQL."""
    logger.info("Starting 1000 document test")
    assert iris_testcontainer_connection is not None, "No connection"
    
    # Verify the test container setup
    with iris_testcontainer_connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1, "Basic query failed"
        logger.info("IRIS connection verified")
        
        # Make sure our tables exist
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        logger.info(f"SourceDocuments table exists, current count: {cursor.fetchone()[0]}")
        
        # Delete any existing data
        cursor.execute("DELETE FROM SourceDocuments")
        logger.info("Cleared existing documents")
        
        # Insert 1000 test documents directly
        logger.info("Inserting 1000 documents...")
        for i in range(1000):
            doc_id = f"doc_{uuid.uuid4()}"
            title = f"Test Document {i+1}"
            content = f"This is test document {i+1} content for the 1000 document test."
            
            # Insert with simplified schema
            cursor.execute(
                "INSERT INTO SourceDocuments (doc_id, title, content) VALUES (?, ?, ?)",
                (doc_id, title, content)
            )
            
            # Log progress occasionally
            if i % 200 == 0:
                logger.info(f"Inserted {i} documents so far...")
        
        # Verify document count
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        count = cursor.fetchone()[0]
        assert count == 1000, f"Expected 1000 documents, found {count}"
        logger.info(f"Successfully inserted and verified {count} documents")
        
        # Test retrieving some documents
        cursor.execute("SELECT doc_id, title FROM SourceDocuments LIMIT 5")
        sample_docs = cursor.fetchall()
        for doc in sample_docs:
            logger.info(f"Sample document: {doc[0]}, Title: {doc[1]}")
            
    logger.info("TEST PASSED: Successfully loaded 1000 documents")
    return True
