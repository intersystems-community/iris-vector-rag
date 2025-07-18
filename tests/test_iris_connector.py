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
    IRISConnectionError,
    JDBC_DRIVER_CLASS,  # Added
    JDBC_JAR_PATH       # Added
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

@patch('common.iris_connector.os.path.exists')
@patch('common.iris_connector.jaydebeapi.connect')
@patch('common.iris_connector.os.environ.get')
@patch('common.iris_connector.ConfigurationManager') # Applied last, so first arg to test func
def test_get_real_iris_connection_uses_config_manager_defaults(
        mock_config_manager_class,
        mock_os_environ_get,
        mock_jaydebeapi_connect,
        mock_os_path_exists
    ):
    """
    Test that get_real_iris_connection (no config arg) sources credentials
    exclusively from ConfigurationManager, ignoring os.environ.
    This test is expected to FAIL until get_real_iris_connection is refactored.
    """
    # 1. Configure Mock ConfigurationManager
    mock_cm_instance = MagicMock()
    def cm_get_side_effect(key):
        vals = {
            "database:iris:host": "cm_host",
            "database:iris:port": 5432, # int
            "database:iris:namespace": "CM_NS",
            "database:iris:username": "cm_user",
            "database:iris:password": "cm_pass"
        }
        return vals.get(key)
    mock_cm_instance.get.side_effect = cm_get_side_effect
    mock_config_manager_class.return_value = mock_cm_instance # Ensures ConfigurationManager() returns our mock

    # 2. Configure Mock os.environ.get
    def os_environ_get_side_effect(key, default=None):
        vals = {
            "IRIS_HOST": "env_host",
            "IRIS_PORT": "7777", # string, current code will int() this
            "IRIS_NAMESPACE": "ENV_NS",
            "IRIS_USERNAME": "env_user",
            "IRIS_PASSWORD": "env_pass"
        }
        return vals.get(key, default)
    mock_os_environ_get.side_effect = os_environ_get_side_effect

    # 3. Configure Mock jaydebeapi.connect
    mock_db_connection = MagicMock()
    mock_jaydebeapi_connect.return_value = mock_db_connection
    # Mock the cursor and execute for the connection test within get_real_iris_connection
    mock_cursor = MagicMock()
    mock_db_connection.cursor.return_value = mock_cursor
    mock_cursor.fetchone.return_value = [1] # For "SELECT 1"

    # 4. Configure Mock os.path.exists to pass the JDBC_JAR_PATH check
    mock_os_path_exists.return_value = True

    # 5. Call get_real_iris_connection without config argument
    conn = get_real_iris_connection()

    # 6. Assertion (This is expected to FAIL with the current implementation)
    expected_jdbc_url = "jdbc:IRIS://cm_host:5432/CM_NS"
    expected_credentials = ["cm_user", "cm_pass"]
    
    mock_jaydebeapi_connect.assert_called_once_with(
        JDBC_DRIVER_CLASS,
        expected_jdbc_url,
        expected_credentials,
        JDBC_JAR_PATH
    )

    # Also assert that the returned connection is the one from jaydebeapi
    assert conn == mock_db_connection

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
