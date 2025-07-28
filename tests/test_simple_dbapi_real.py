"""
Test file for validating basic IRIS database connection using the real connection fixture.

This test validates our ability to connect to the IRIS container managed by docker-compose
and execute simple queries using the DB-API interface.
"""

import pytest
import logging

logger = logging.getLogger(__name__)


def test_simple_dbapi_connection_real(iris_connection_real):
    """
    Test basic database connectivity using the real IRIS connection fixture.
    
    This test:
    1. Uses the iris_connection_real fixture from tests/conftest.py
    2. Creates a cursor from the provided connection
    3. Executes a simple query: SELECT 1
    4. Asserts that the result is (1,)
    
    This validates our ability to connect to the IRIS container managed by docker-compose.
    """
    # Skip test if no real connection is available
    if iris_connection_real is None:
        pytest.skip("Real IRIS connection not available")
    
    logger.info("Testing simple DB-API connection with real IRIS database")
    
    # Create a cursor from the provided connection
    cursor = iris_connection_real.cursor()
    
    try:
        # Execute a simple query
        cursor.execute("SELECT 1")
        
        # Fetch the result
        result = cursor.fetchone()
        
        # Assert that the result is (1,)
        assert result == (1,), f"Expected (1,) but got {result}"
        
        logger.info("✅ Simple DB-API connection test passed")
        
    finally:
        # Clean up the cursor
        cursor.close()