import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add workspace root to sys.path to allow importing check_tables
# This assumes tests/test_scripts/ is two levels down from the workspace root
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Define the mock database configuration values that ConfigurationManager should return
MOCKED_DB_CONFIG_VALUES = {
    "host": "test_host_from_config",
    "port": 1234,
    "namespace": "TEST_NAMESPACE_CONFIG",
    "username": "test_user_from_config",
    "password": "test_password_via_config"  # Unique test password
}

# Define the keys that check_tables.py would (eventually) use to fetch config
CONFIG_KEYS = {
    "host": "database:iris:host",
    "port": "database:iris:port",
    "namespace": "database:iris:namespace",
    "username": "database:iris:username",
    "password": "database:iris:password"
}

@patch('check_tables.iris.connect')  # Patch iris.connect where it's used in check_tables.py
@patch('check_tables.ConfigurationManager')  # Patch ConfigurationManager where it is used
def test_check_tables_uses_configuration_manager_for_db_credentials(
    mock_config_manager_class, mock_iris_connect
):
    """
    Tests that check_tables.py attempts to use ConfigurationManager for DB credentials.
    This test is designed to FAIL initially because check_tables.py currently uses
    hardcoded credentials.
    """
    # Configure the mock ConfigurationManager class to return a mock instance
    mock_config_instance = MagicMock()
    mock_config_manager_class.return_value = mock_config_instance

    # Define the behavior of the mocked ConfigurationManager's get() method
    # This simulates check_tables.py fetching each credential individually.
    def mock_get_side_effect(key_string, default=None):
        if key_string == CONFIG_KEYS["host"]:
            return MOCKED_DB_CONFIG_VALUES["host"]
        elif key_string == CONFIG_KEYS["port"]:
            return MOCKED_DB_CONFIG_VALUES["port"]
        elif key_string == CONFIG_KEYS["namespace"]:
            return MOCKED_DB_CONFIG_VALUES["namespace"]
        elif key_string == CONFIG_KEYS["username"]:
            return MOCKED_DB_CONFIG_VALUES["username"]
        elif key_string == CONFIG_KEYS["password"]:
            return MOCKED_DB_CONFIG_VALUES["password"]
        # Fallback for any other keys, though not expected for this specific test
        return default

    mock_config_instance.get.side_effect = mock_get_side_effect

    # Dynamically import the check_tables module and its main function.
    # This ensures mocks are applied before the script's code (specifically iris.connect)
    # is encountered at import time or execution time.
    import check_tables
    
    # Call the main logic of the script
    # This will internally call the (mocked) iris.connect
    try:
        check_tables.check_tables()
    except Exception as e:
        # The script might fail to connect if mocks aren't perfect,
        # but we are primarily interested in the call to iris.connect.
        # For this test, we allow it to proceed to the assertion.
        # In a real scenario, mock_iris_connect.return_value might need further setup
        # (e.g., a mock connection object with a mock cursor).
        print(f"Note: check_tables.check_tables() raised an exception during test: {e}")
        pass


    # Assertion: iris.connect should have been called with credentials 
    # derived from the (mocked) ConfigurationManager.
    # This assertion WILL FAIL because check_tables.py currently uses hardcoded values:
    # hostname="localhost", port=1972, namespace="USER", username="_SYSTEM", password="SYS"
    mock_iris_connect.assert_called_once_with(
        hostname=MOCKED_DB_CONFIG_VALUES["host"],
        port=MOCKED_DB_CONFIG_VALUES["port"],
        namespace=MOCKED_DB_CONFIG_VALUES["namespace"],
        username=MOCKED_DB_CONFIG_VALUES["username"],
        password=MOCKED_DB_CONFIG_VALUES["password"]
    )