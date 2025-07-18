import pytest
import os
from unittest import mock

# Attempt to import ConnectionManager, will fail initially
try:
    from iris_rag.core.connection import ConnectionManager
except ImportError:
    ConnectionManager = None # Placeholder if import fails

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mocks environment variables for IRIS connection."""
    monkeypatch.setenv("IRIS_HOST", "testhost")
    monkeypatch.setenv("IRIS_PORT", "1234")
    monkeypatch.setenv("IRIS_NAMESPACE", "TESTNS")
    monkeypatch.setenv("IRIS_USER", "testuser")
    monkeypatch.setenv("IRIS_PASSWORD", "testpass")

@pytest.fixture
def mock_config_manager():
    """Mocks ConfigurationManager to provide a simple, fixed backend configuration."""
    class MockConfigurationManager:
        def __init__(self, config_path=None):
            # This config is fixed for the fixture. Tests can override .get() if needed.
            self._fixed_iris_config = {
                "host": "fixture_host",
                "port": 11111,
                "namespace": "FIXTURE_NS",
                "username": "fixture_user",
                "password": "fixture_password",
                "driver": "intersystems_iris.dbapi"
            }
        
        def get(self, section_key):
            if section_key == "database:iris":
                return self._fixed_iris_config.copy() # Return a copy
            return None
    return MockConfigurationManager()

def test_import_connection_manager():
    """Tests that ConnectionManager can be imported."""
    assert ConnectionManager is not None, "Failed to import ConnectionManager from iris_rag.core.connection"

# mock_env_vars is removed as ConnectionManager should use the config_manager's values directly.
def test_connection_manager_get_iris_connection(mock_config_manager):
    """Tests getting an IRIS connection using config from mock_config_manager."""
    if ConnectionManager is None:
        pytest.fail("ConnectionManager not imported")

    with mock.patch('iris_rag.core.connection.importlib.import_module') as mock_import_module:
        mock_db_api = mock.MagicMock()
        mock_db_api.connect.return_value = "mock_iris_connection_object"
        mock_import_module.return_value = mock_db_api
        
        conn_manager = ConnectionManager(config_manager=mock_config_manager)
        connection = conn_manager.get_connection("iris")
        
        assert connection == "mock_iris_connection_object"
        mock_import_module.assert_called_once_with("intersystems_iris.dbapi")
        
        # Expected values from the simplified mock_config_manager fixture
        expected_config = mock_config_manager.get("database:iris")
        mock_db_api.connect.assert_called_once_with(
            hostname=expected_config["host"], # Should be "fixture_host"
            port=expected_config["port"],     # Should be 11111
            namespace=expected_config["namespace"], # Should be "FIXTURE_NS"
            username=expected_config["username"], # Should be "fixture_user"
            password=expected_config["password"]  # Should be "fixture_password"
        )

def test_connection_manager_unsupported_backend(mock_config_manager):
    """Tests getting a connection for an unsupported backend."""
    if ConnectionManager is None:
        pytest.fail("ConnectionManager not imported")

    # To test the "Unsupported" path, config_manager must return a config for "unsupported_db"
    # so it passes the "config not found" and "driver not specified" checks.
    # It should then fail because "unsupported_db" is not "iris".
    conn_manager = ConnectionManager(config_manager=mock_config_manager)
    with mock.patch.object(conn_manager.config_manager, 'get') as mock_get_method:
        # Make .get() return a dummy config only for "database:unsupported_db"
        mock_get_method.side_effect = lambda key: {"driver": "dummy_driver"} if key == "database:unsupported_db" else None
        
        with pytest.raises(ValueError, match="Unsupported database backend: unsupported_db"):
            conn_manager.get_connection("unsupported_db")
        mock_get_method.assert_called_once_with("database:unsupported_db")

def test_connection_manager_missing_iris_config(mock_config_manager):
    """Tests ConnectionManager behavior when IRIS config section is missing."""
    if ConnectionManager is None:
        pytest.fail("ConnectionManager not imported")

    conn_manager = ConnectionManager(config_manager=mock_config_manager)
    
    # Use patch.object to temporarily change the behavior of the .get() method
    # on the specific mock_config_manager instance used by conn_manager.
    with mock.patch.object(conn_manager.config_manager, 'get', return_value=None) as mock_get:
        with pytest.raises(ValueError, match="Configuration for backend 'iris' not found."):
            conn_manager.get_connection("iris")
        mock_get.assert_called_once_with("database:iris")


# Renamed and monkeypatch removed. Tests ConnectionManager uses what config_manager provides.
def test_connection_manager_iris_uses_provided_config(mock_config_manager):
    """Tests IRIS connection uses the exact config provided by config_manager."""
    if ConnectionManager is None:
        pytest.fail("ConnectionManager not imported")

    # mock_config_manager will provide its fixed "fixture_host" etc.
    with mock.patch('iris_rag.core.connection.importlib.import_module') as mock_import_module:
        mock_db_api = mock.MagicMock()
        # Use a distinct return value to ensure no test pollution
        connect_return_value = "mock_iris_connection_object_specific_test"
        mock_db_api.connect.return_value = connect_return_value
        mock_import_module.return_value = mock_db_api
        
        conn_manager = ConnectionManager(config_manager=mock_config_manager)
        connection = conn_manager.get_connection("iris")
        
        assert connection == connect_return_value
        mock_import_module.assert_called_once_with("intersystems_iris.dbapi")

        expected_config = mock_config_manager.get("database:iris")
        mock_db_api.connect.assert_called_once_with(
            hostname=expected_config["host"],
            port=expected_config["port"],
            namespace=expected_config["namespace"],
            username=expected_config["username"],
            password=expected_config["password"]
        )