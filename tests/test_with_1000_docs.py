"""
Test module specifically for 1000 document tests.

This module focuses on testing RAG techniques with 1000 documents using 
pytest fixtures properly. It uses the existing fixtures from conftest.py
to handle testcontainer setup and data loading.
"""

import pytest
import logging
import os
import time
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a custom marker for 1000 document tests
pytestmark = [
    pytest.mark.force_testcontainer,  # Always use testcontainer
    pytest.mark.document_count(1000)  # Mark as 1000 document test
]

# Register marker in pytest.ini if not already done
def pytest_configure(config):
    config.addinivalue_line("markers", "document_count(count): mark test to run with specific document count")

# Make sure environment variables are set for all tests in this module
os.environ["TEST_IRIS"] = "true" 
os.environ["TEST_DOCUMENT_COUNT"] = "1000"
os.environ["USE_MOCK_EMBEDDINGS"] = "true"  # Use mock embeddings for faster tests

@pytest.fixture(scope="module", autouse=True)
def setup_test_env():
    """
    Setup environment for 1000 document tests.
    This fixture runs automatically before all tests in this module.
    """
    # Log start of test session
    logger.info("=== Starting 1000 document test session ===")
    logger.info(f"TEST_DOCUMENT_COUNT: {os.environ.get('TEST_DOCUMENT_COUNT')}")
    logger.info(f"USE_MOCK_EMBEDDINGS: {os.environ.get('USE_MOCK_EMBEDDINGS')}")
    
    # Return control - setup complete
    yield
    
    # Log end of test session
    logger.info("=== Completed 1000 document test session ===")

def test_iris_container_connection(iris_testcontainer_connection):
    """
    Test that connection to the IRIS testcontainer works correctly.
    This is a basic connectivity test and SQL execution test.
    """
    # Verify the connection is valid
    assert iris_testcontainer_connection is not None, "Failed to create testcontainer connection"
    
    # Test basic SQL execution
    with iris_testcontainer_connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result is not None, "Failed to execute query"
        assert result[0] == 1, "Unexpected query result"
        logger.info("Basic SQL query successful: SELECT 1 = 1")

def test_basic_document_loading(iris_testcontainer_connection):
    """
    Test that we can load documents directly into the database.
    This tests the table structure and basic data insert operations.
    """
    assert iris_testcontainer_connection is not None
    
    # Create a small batch of test documents
    docs = []
    for i in range(10):
        doc_id = f"test_doc_{uuid.uuid4()}"
        title = f"Test Document {i+1}"
        content = f"This is test document {i+1} content for the 1000 docs test."
        docs.append((doc_id, title, content))
    
    # Insert documents
    logger.info(f"Inserting {len(docs)} test documents...")
    with iris_testcontainer_connection.cursor() as cursor:
        # Clear existing documents to start fresh
        cursor.execute("DELETE FROM SourceDocuments")
        
        # Insert new documents
        for doc in docs:
            cursor.execute(
                "INSERT INTO SourceDocuments (doc_id, title, content) VALUES (?, ?, ?)",
                doc
            )
        
        # Verify count
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        count = cursor.fetchone()[0]
        assert count == len(docs), f"Expected {len(docs)} documents, found {count}"
        logger.info(f"Successfully inserted and verified {count} documents")

def test_large_document_loading(iris_testcontainer_connection):
    """
    Test loading 1000 documents into the database.
    This is the main test for the 1000 document requirement.
    """
    assert iris_testcontainer_connection is not None
    start_time = time.time()
    
    # Clear existing documents
    with iris_testcontainer_connection.cursor() as cursor:
        cursor.execute("DELETE FROM SourceDocuments")
        logger.info("Cleared existing documents")
    
    # Generate and insert 1000 documents in batches
    batch_size = 50
    total_docs = 1000
    batches = total_docs // batch_size
    
    logger.info(f"Inserting {total_docs} documents in {batches} batches of {batch_size}...")
    
    for batch in range(batches):
        batch_docs = []
        for i in range(batch_size):
            doc_idx = batch * batch_size + i
            doc_id = f"doc_1000_{doc_idx}"
            title = f"Test Document {doc_idx}"
            # Create content that's a reasonable size but not too large
            content = f"This is content for document {doc_idx}. It contains information for the 1000 document test."
            batch_docs.append((doc_id, title, content))
        
        # Insert batch
        with iris_testcontainer_connection.cursor() as cursor:
            for doc in batch_docs:
                cursor.execute(
                    "INSERT INTO SourceDocuments (doc_id, title, content) VALUES (?, ?, ?)",
                    doc
                )
        
        # Log progress
        if (batch + 1) % 5 == 0 or batch == 0:
            logger.info(f"Inserted {(batch + 1) * batch_size} documents...")
    
    # Verify final count
    with iris_testcontainer_connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        count = cursor.fetchone()[0]
        assert count == total_docs, f"Expected {total_docs} documents, found {count}"
    
    # Log performance metrics
    end_time = time.time()
    duration = end_time - start_time
    docs_per_second = total_docs / duration
    
    logger.info(f"Successfully loaded {count} documents in {duration:.2f} seconds")
    logger.info(f"Performance: {docs_per_second:.2f} docs/sec")

def test_with_pmc_fixture(iris_with_pmc_data):
    """
    Test using the iris_with_pmc_data fixture which should load 1000 documents.
    This test uses the fixture instead of directly loading documents.
    """
    assert iris_with_pmc_data is not None
    
    # The fixture should have already loaded the data, so just verify count
    with iris_with_pmc_data.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
        count = cursor.fetchone()[0]
        
        # Test may be run after other tests that modify the document count,
        # so we check for a reasonable number, not exactly 1000
        assert count > 0, "No documents found - fixture failed to load data"
        logger.info(f"Found {count} documents loaded by fixture")
        
        # Check a sample document
        cursor.execute("SELECT doc_id, title FROM SourceDocuments LIMIT 1")
        sample = cursor.fetchone()
        logger.info(f"Sample document: {sample[0]}, Title: {sample[1]}")
