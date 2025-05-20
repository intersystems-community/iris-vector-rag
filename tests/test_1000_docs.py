"""
Minimal test file to verify 1000 document testing with testcontainer
This test must be placed in the tests directory to access fixtures from conftest.py
"""

import pytest
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables directly in the test file
# These will be read by the fixtures in conftest.py
os.environ["TEST_IRIS"] = "true"
os.environ["TEST_DOCUMENT_COUNT"] = "1000"
os.environ["COLLECT_PERFORMANCE_METRICS"] = "true" 
os.environ["USE_MOCK_EMBEDDINGS"] = "true"  # Use mock embeddings for faster tests

@pytest.mark.force_testcontainer
def test_iris_testcontainer_setup(iris_testcontainer_connection):
    """Verify that the IRIS testcontainer works correctly."""
    # This is the same as test_graphrag_with_testcontainer.py::test_iris_testcontainer_setup
    assert iris_testcontainer_connection is not None, "Failed to create testcontainer connection"
    
    # Execute a simple query to verify the connection works
    with iris_testcontainer_connection.cursor() as cursor:
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        assert result is not None, "Failed to execute query"
        assert result[0] == 1, "Unexpected query result"
        logger.info("✅ Basic IRIS query succeeded")
