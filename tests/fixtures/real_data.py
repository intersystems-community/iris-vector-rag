"""
Real Data Test Fixtures

This module provides fixtures that help tests determine if real data is available
and control whether to use real or mock resources based on that.
"""

import pytest

from common.iris_connector import get_iris_connection

# --- Constants ---

# Minimum number of documents required to consider that real data is loaded
MIN_REAL_DOCS = 10

# --- Fixtures ---

@pytest.fixture(scope="session")
def real_iris_available() -> bool:
    """
    Check if a real IRIS connection is available.
    
    Returns:
        bool: True if a real IRIS connection can be established, False otherwise
    """
    # Try to connect
    # get_iris_connection will use its own defaults (e.g., localhost for IRIS_HOST)
    # if the environment variables are not explicitly set.
    conn = get_iris_connection(use_mock=False)
    if conn is None:
        return False
    
    # Test the connection with a simple query
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return result is not None and result[0] == 1
    except Exception:
        return False

@pytest.fixture(scope="session")
def real_data_available(real_iris_available: bool) -> bool:
    """
    Check if real data is loaded in the IRIS database.
    
    Args:
        real_iris_available: Result of the real_iris_available fixture
        
    Returns:
        bool: True if real data is available, False otherwise
    """
    if not real_iris_available:
        return False
    
    # Connect to IRIS
    conn = get_iris_connection(use_mock=False)
    if conn is None:
        return False
    
    # Check if the SourceDocuments table exists and has data
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return result is not None and result[0] >= MIN_REAL_DOCS
    except Exception:
        return False

@pytest.fixture
def use_real_data(real_data_available: bool, request) -> bool:
    """
    Determine whether to use real data for the test.
    
    Args:
        real_data_available: Result of the real_data_available fixture
        request: Pytest request object with access to test markers
        
    Returns:
        bool: True if the test should use real data, False otherwise
    """
    # Check for force_mock marker
    if request.node.get_closest_marker("force_mock"):
        return False
    
    # Check for force_real marker
    if request.node.get_closest_marker("force_real"):
        if not real_data_available:
            pytest.skip("Test requires real data but none is available")
        return True
    
    # Default: Use real data if available
    return real_data_available

@pytest.fixture
def iris_connection(use_real_data: bool):
    """
    Provide an IRIS connection (real or mock based on availability).
    
    Args:
        use_real_data: Whether to use real data
        
    Returns:
        An IRIS connection object
    """
    conn = get_iris_connection(use_mock=not use_real_data)
    yield conn
    
    # Close connection after test
    try:
        conn.close()
    except Exception:
        pass

# Example usage in a test:
#
# @pytest.mark.force_real
# def test_with_real_data(iris_connection, use_real_data):
#     assert use_real_data is True
#     # Test with iris_connection which is guaranteed to be real
#
# @pytest.mark.force_mock
# def test_with_mock_data(iris_connection, use_real_data):
#     assert use_real_data is False
#     # Test with iris_connection which is guaranteed to be mock
#
# def test_with_whatever_is_available(iris_connection, use_real_data):
#     # Test with iris_connection which could be real or mock
#     # The use_real_data flag tells which it is
