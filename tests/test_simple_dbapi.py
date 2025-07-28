"""
Test for simple database API connection using IRIS testcontainer.

This test verifies basic database connectivity using the iris_testcontainer_connection fixture
and ensures that simple SQL queries can be executed successfully.
"""

import pytest
import logging

logger = logging.getLogger(__name__)


@pytest.mark.force_testcontainer
def test_simple_dbapi_connection(iris_testcontainer_connection):
    """
    Test basic database connection and simple query execution.
    
    This test:
    1. Uses the iris_testcontainer_connection fixture for isolated testing
    2. Creates a cursor from the connection
    3. Executes a simple SELECT 1 query
    4. Asserts that the result is (1,)
    
    Args:
        iris_testcontainer_connection: Database connection fixture from conftest.py
    """
    logger.info("Testing simple database API connection")
    
    # Ensure we have a valid connection
    assert iris_testcontainer_connection is not None, "Database connection should not be None"
    
    # Create a cursor from the connection
    cursor = iris_testcontainer_connection.cursor()
    
    try:
        # Execute a simple query
        cursor.execute("SELECT 1")
        
        # Fetch the result
        result = cursor.fetchone()
        
        # Assert that the result is (1,)
        assert result == (1,), f"Expected (1,) but got {result}"
        
        logger.info("Simple database query test passed successfully")
        
    finally:
        # Clean up the cursor
        cursor.close()