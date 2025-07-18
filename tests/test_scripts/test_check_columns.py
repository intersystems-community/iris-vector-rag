import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add workspace root to sys.path to allow importing check_columns
# This assumes tests/test_scripts/ is two levels down from the workspace root
workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Define the mock database configuration values that ConfigurationManager should return
MOCKED_DB_CONFIG_VALUES = {
    "host": "test_host_from_config_cols",
    "port": 5678,
    "namespace": "TEST_NAMESPACE_CONFIG_COLS",
    "username": "test_user_from_config_cols",
    "password": "test_password_via_config_cols"
}

# Define the keys that check_columns.py would use to fetch config
CONFIG_KEYS = {
    "host": "database:iris:host",
    "port": "database:iris:port",
    "namespace": "database:iris:namespace",
    "username": "database:iris:username",
    "password": "database:iris:password"
}

@patch('check_columns.iris.connect')  # Patch iris.connect where it's used in check_columns.py
@patch('check_columns.ConfigurationManager')  # Patch ConfigurationManager where it would be used
def test_check_columns_uses_configuration_manager(
    mock_config_manager_class, mock_iris_connect
):
    """
    Tests that check_columns.py attempts to use ConfigurationManager for DB credentials.
    This test is designed to FAIL initially because check_columns.py currently uses
    hardcoded credentials.
    """
    # Configure the mock ConfigurationManager class to return a mock instance
    mock_config_instance = MagicMock()
    mock_config_manager_class.return_value = mock_config_instance

    # Define the behavior of the mocked ConfigurationManager's get() method
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
        return default

    mock_config_instance.get.side_effect = mock_get_side_effect

    # Dynamically import the check_columns module and its main function.
    # This ensures mocks are applied before the script's code is encountered.
    import check_columns
    
    # Call the main logic of the script
    try:
        check_columns.check_columns()
    except Exception as e:
        # The script might fail (e.g. if mock_iris_connect doesn't return a usable connection)
        # but we are primarily interested in the call to iris.connect.
        print(f"Note: check_columns.check_columns() raised an exception during test: {e}")
        pass

    # Assertion: iris.connect should have been called with credentials 
    # derived from the (mocked) ConfigurationManager.
    # This assertion WILL FAIL because check_columns.py currently uses hardcoded values:
    # hostname="localhost", port=1972, namespace="USER", username="_SYSTEM", password="SYS"
    mock_iris_connect.assert_called_once_with(
        hostname=MOCKED_DB_CONFIG_VALUES["host"],
        port=MOCKED_DB_CONFIG_VALUES["port"],
        namespace=MOCKED_DB_CONFIG_VALUES["namespace"],
        username=MOCKED_DB_CONFIG_VALUES["username"],
        password=MOCKED_DB_CONFIG_VALUES["password"]
    )