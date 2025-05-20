"""
Absolute minimal test to prove test approach works
"""
import pytest
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables
os.environ["TEST_IRIS"] = "true"
os.environ["USE_MOCK_EMBEDDINGS"] = "true"

@pytest.mark.force_testcontainer
def test_minimal_proof(iris_testcontainer_connection):
    """Absolute minimal test that should work"""
    logger.info("Starting minimal proof test")
    assert iris_testcontainer_connection is not None
    
    # Verify connection works with simplest possible query
    cursor = iris_testcontainer_connection.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    assert result[0] == 1
    logger.info("Basic connection verified: SELECT 1 = 1")
    
    # Create our test table if it doesn't exist yet
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_documents (
                id VARCHAR(50) PRIMARY KEY,
                content VARCHAR(200)
            )
        """)
        logger.info("Created test_documents table")
    except Exception as e:
        logger.error(f"Error creating table: {e}")
        raise
    
    # Clear any existing data
    try:
        cursor.execute("DELETE FROM test_documents")
        logger.info("Cleared existing test documents")
    except Exception as e:
        logger.error(f"Error clearing table: {e}")
        raise
    
    # Insert just 10 test records
    try:
        logger.info("Inserting 10 test documents...")
        for i in range(10):
            cursor.execute(
                "INSERT INTO test_documents (id, content) VALUES (?, ?)",
                (f"doc_{i}", f"Test content {i}")
            )
        logger.info("Successfully inserted 10 documents")
    except Exception as e:
        logger.error(f"Error inserting documents: {e}")
        raise
    
    # Verify count
    try:
        cursor.execute("SELECT COUNT(*) FROM test_documents")
        count = cursor.fetchone()[0]
        assert count == 10, f"Expected 10 documents, found {count}"
        logger.info(f"Verified document count: {count}")
    except Exception as e:
        logger.error(f"Error counting documents: {e}")
        raise
    
    # Fetch and display documents
    try:
        cursor.execute("SELECT id, content FROM test_documents")
        docs = cursor.fetchall()
        for doc in docs:
            logger.info(f"Document: {doc[0]}, Content: {doc[1]}")
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
        raise
        
    logger.info("TEST PASSED: Successfully completed minimal proof test")
    return True
