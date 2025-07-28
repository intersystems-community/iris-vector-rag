"""
Test file for simplified database API validation.

This test validates our new, simplified test architecture by testing
basic database connectivity and query execution.
"""

import pytest
import logging
import iris

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def clean_database():
    """Provide a clean database state for each test using native IRIS DBAPI."""
    # Connect using native IRIS DBAPI
    conn = iris.connect(
        hostname="localhost",
        port=1972,
        namespace="USER",
        username="SuperUser",
        password="SYS"
    )
    
    yield conn
    
    # Cleanup after test
    conn.close()


def test_simplified_dbapi_connection(clean_database):
    """
    Test basic database connectivity using the clean_database fixture.
    
    This test validates that:
    1. The clean_database fixture provides a working connection
    2. We can create a cursor from the connection
    3. We can execute a simple query
    4. The query returns the expected result
    """
    # Create cursor from the provided connection
    cursor = clean_database.cursor()
    
    # Execute a simple query
    cursor.execute("SELECT 1")
    
    # Fetch the result
    result = cursor.fetchone()
    
    # Assert that the result contains the value 1
    # IRIS DBAPI returns DataRow objects, so we access the first column
    assert result[0] == 1, f"Expected 1 but got {result[0]}"
    
    # Clean up cursor
    cursor.close()
    
    logger.info("Database API test completed successfully")