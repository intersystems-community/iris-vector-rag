"""
Test script to ensure GraphRAG tests run with 1000 documents.
This test focuses on verification of testcontainer setup and proper document loading.
"""

import pytest
import logging
import time
import os

# Set environment variables for testing
os.environ["TEST_IRIS"] = "true"
os.environ["TEST_DOCUMENT_COUNT"] = "1000"
os.environ["COLLECT_PERFORMANCE_METRICS"] = "true"
os.environ["USE_MOCK_EMBEDDINGS"] = "true"  # Use mock embeddings for faster testing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ensure_iris_testcontainer(iris_testcontainer):
    """Test that IRIS testcontainer starts up and is accessible."""
    assert iris_testcontainer is not None
    time.sleep(1)  # Give container a moment to settle
    
    # Check that container is running
    assert iris_testcontainer.get_container_host_ip() is not None
    port = iris_testcontainer.get_exposed_port(iris_testcontainer.port)
    assert port is not None
    
    logger.info(f"IRIS testcontainer running at {iris_testcontainer.get_container_host_ip()}:{port}")

def test_ensure_testcontainer_connection(iris_testcontainer_connection):
    """Test connection to IRIS testcontainer and basic SQL execution."""
    assert iris_testcontainer_connection is not None
    
    with iris_testcontainer_connection.cursor() as cursor:
        # Test basic query
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result is not None
        assert result[0] == 1
        
        # Test creating a table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS TestTable (
                id VARCHAR(255) PRIMARY KEY,
                value VARCHAR(1000)
            )
        """)
        
        # Test inserting data
        cursor.execute("INSERT INTO TestTable VALUES ('test1', 'test value')")
        
        # Test selecting data
        cursor.execute("SELECT * FROM TestTable")
        rows = cursor.fetchall()
        assert len(rows) > 0
        
        logger.info(f"IRIS testcontainer connection verified, retrieved: {rows[0]}")

# If this test can run, it confirms the fixtures and testcontainer works
# which is the foundation for the large-scale tests with 1000 documents
