"""
Simple test to verify testcontainer setup.

This test imports the necessary fixtures from conftest.py.
"""
import pytest
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make sure the project root is in the path so we can import from tests
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import the testcontainer fixture from conftest
from tests.conftest import iris_testcontainer, iris_testcontainer_connection

# Set environment variables for testing
os.environ["TEST_IRIS"] = "true"
os.environ["TEST_DOCUMENT_COUNT"] = "1000"
os.environ["USE_MOCK_EMBEDDINGS"] = "true"

@pytest.mark.force_testcontainer
def test_simple_testcontainer(iris_testcontainer_connection):
    """Very basic test to verify testcontainer is working."""
    assert iris_testcontainer_connection is not None, "Connection is None"
    
    # Test a simple query
    with iris_testcontainer_connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result is not None, "Query returned None"
        assert result[0] == 1, f"Expected 1, got {result[0]}"
        logger.info(f"Query result: {result[0]}")
        
        # Create a simple table and insert data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS SimpleTestTable (
                id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(100)
            )
        """)
        
        cursor.execute("INSERT INTO SimpleTestTable VALUES ('test1', 'Test Name')")
        
        # Verify data was inserted
        cursor.execute("SELECT * FROM SimpleTestTable")
        rows = cursor.fetchall()
        assert len(rows) > 0, "No rows returned"
        logger.info(f"Retrieved row: {rows[0]}")
