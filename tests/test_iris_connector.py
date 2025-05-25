# tests/test_iris_connector.py
# Tests for the IRIS database connector

import pytest
import os
import sys
from unittest.mock import patch, MagicMock

# Make sure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.iris_connector import (
    get_real_iris_connection,
    get_mock_iris_connection,
    get_iris_connection,
    IRISConnectionError
)

# --- Unit Tests (Mock-based) ---

def test_get_mock_iris_connection():
    """Test that we can get a mock IRIS connection"""
    mock_conn = get_mock_iris_connection()
    assert mock_conn is not None
    # Verify it's a MockIRISConnector
    from tests.mocks.db import MockIRISConnector
    assert isinstance(mock_conn, MockIRISConnector)

def test_get_iris_connection_with_mock_flag():
    """Test that get_iris_connection returns a mock when use_mock=True"""
    conn = get_iris_connection(use_mock=True)
    assert conn is not None
    # Verify it's a MockIRISConnector
    from tests.mocks.db import MockIRISConnector
    assert isinstance(conn, MockIRISConnector)

def test_get_iris_connection_no_mock_no_real_in_pytest():
    """Test that get_iris_connection falls back to mock in pytest context when real fails"""
    # Mock that get_real_iris_connection raises IRISConnectionError
    with patch('common.iris_connector.get_real_iris_connection', side_effect=IRISConnectionError("Simulated connection error")):
        # Mock that we're in pytest context
        with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "yes"}):
            conn = get_iris_connection(use_mock=False)
            assert conn is not None
            # Verify it fell back to mock
            from tests.mocks.db import MockIRISConnector
            assert isinstance(conn, MockIRISConnector)

def test_get_iris_connection_no_mock_no_real_outside_pytest():
    """Test that get_iris_connection returns None outside pytest context when real fails"""
    # Mock that get_real_iris_connection returns None (failure)
    with patch('common.iris_connector.get_real_iris_connection', return_value=None):
        # Ensure PYTEST_CURRENT_TEST is not in environ
        with patch.dict(os.environ, {}, clear=True):
            conn = get_iris_connection(use_mock=False)
            assert conn is None

def test_get_real_iris_connection_success(monkeypatch):
    """Test successful real IRIS connection"""
    # Create a successful mock connection
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = [1]
    
    # Replace the real_iris_connection function with one that returns our mock
    monkeypatch.setattr('common.iris_connector.get_real_iris_connection', 
                        lambda *args, **kwargs: mock_conn)
    
    # Call the main connection function
    conn = get_iris_connection(use_mock=False)
    
    # Verify we got our mock connection back
    assert conn is mock_conn

def test_get_real_iris_connection_with_config():
    """Test connection can be passed a custom config"""
    # This test is informational only, as the actual connection parameters
    # are tested in the real_iris_connection implementation
    
    # Create a custom config
    custom_config = {
        "hostname": "customhost",
        "port": 1234,
        "namespace": "CUSTOM",
        "username": "testuser",
        "password": "testpass"
    }
    
    # We expect this to raise an IRISConnectionError because "customhost" is not real
    with pytest.raises(IRISConnectionError):
        get_real_iris_connection(custom_config)

def test_get_real_iris_connection_import_error(monkeypatch):
    """Test handling of ImportError for intersystems_iris module"""
    # Setup to make get_real_iris_connection raise IRISConnectionError as if import failed
    monkeypatch.setattr('common.iris_connector.get_real_iris_connection',
                        MagicMock(side_effect=IRISConnectionError("Simulated import error")))
    
    # Call the function, it should fail back to mock in pytest context
    # Ensure PYTEST_CURRENT_TEST is set for this specific test case
    with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "yes"}):
        conn = get_iris_connection(use_mock=False)
    
    # Verify we got a mock connector
    from tests.mocks.db import MockIRISConnector
    assert isinstance(conn, MockIRISConnector)

def test_get_real_iris_connection_connect_error(monkeypatch):
    """Test handling of connection error"""
    # Setup to make the function return None as if connection failed
    monkeypatch.setattr('common.iris_connector.get_real_iris_connection', 
                        lambda *args, **kwargs: None)
    
    # Call the function with different environment setup
    with patch.dict(os.environ, {}, clear=True):  # Clear pytest env var
        conn = get_iris_connection(use_mock=False)
        
        # Verify we got None when not in pytest context
        assert conn is None

# --- Integration Tests (Real IRIS) ---

@pytest.mark.integration
def test_real_iris_connection_integration():
    """Test connection to real IRIS database if available."""
    # Skip if environment variables aren't set
    if not os.environ.get("IRIS_HOST"):
        pytest.skip("IRIS environment variables not configured")
    
    # Try to get a real connection
    conn = get_real_iris_connection()
    
    # Check that we got a real connection
    assert conn is not None
    
    # Simple query to verify connection works
    cursor = conn.cursor()
    cursor.execute("SELECT 1 AS test")
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    assert result[0] == 1
