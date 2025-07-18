import pytest
import os
from iris_rag.config.manager import ConfigurationManager
import logging

# --- Test Constants ---
# Define some expected default values or values to set via environment for testing
TEST_HOST = "test_db_host"
TEST_PORT = "1234"
TEST_NAMESPACE = "TEST_CONFIG_NS"
TEST_USERNAME = "test_cfg_user"
TEST_PASSWORD = "test_cfg_password" # pragma: allowlist secret
TEST_DRIVER_PATH = "/path/to/test/driver.dylib"
TEST_EMBEDDING_MODEL = "test-embedding-model"
TEST_LOG_LEVEL = "WARNING"
TEST_LOG_PATH = "logs/iris_rag_config_test.log"
TEST_TABLE_NAME = "test_rag_table"
TEST_TOP_K = "7"

@pytest.fixture(scope="function") # Use function scope to isolate env var changes
def setup_test_environment():
    """Sets up environment variables for a ConfigurationManager test run and cleans up afterwards."""
    original_env = os.environ.copy()

    # Set test environment variables
    os.environ["IRIS_HOST"] = TEST_HOST
    os.environ["IRIS_PORT"] = TEST_PORT
    os.environ["IRIS_NAMESPACE"] = TEST_NAMESPACE
    os.environ["IRIS_USERNAME"] = TEST_USERNAME
    os.environ["IRIS_PASSWORD"] = TEST_PASSWORD
    os.environ["IRIS_DRIVER_PATH"] = TEST_DRIVER_PATH
    os.environ["EMBEDDING_MODEL_NAME"] = TEST_EMBEDDING_MODEL
    os.environ["LOG_LEVEL"] = TEST_LOG_LEVEL
    os.environ["LOG_PATH"] = TEST_LOG_PATH
    os.environ["DEFAULT_TABLE_NAME"] = TEST_TABLE_NAME
    os.environ["DEFAULT_TOP_K"] = TEST_TOP_K
    # Add any other relevant env vars your ConfigurationManager uses

    yield # This is where the test runs

    # Teardown: Restore original environment variables
    os.environ.clear()
    os.environ.update(original_env)

    # Clean up the test log file if it was created
    if os.path.exists(TEST_LOG_PATH):
        try:
            os.remove(TEST_LOG_PATH)
            # Attempt to remove the logs directory if it's empty and was created by this test
            log_dir = os.path.dirname(TEST_LOG_PATH)
            if log_dir == "logs" and os.path.exists(log_dir) and not os.listdir(log_dir):
                os.rmdir(log_dir)
        except OSError:
            pass # Ignore errors during cleanup

@pytest.fixture
def config_manager(setup_test_environment):
    """Fixture for ConfigurationManager, using the test environment."""
    return ConfigurationManager()

# --- Test Cases ---

def test_config_manager_instantiation(config_manager):
    """Tests if ConfigurationManager instantiates correctly."""
    assert config_manager is not None, "ConfigurationManager failed to instantiate."

def test_database_config_loading(config_manager):
    """Tests if database configurations are loaded correctly from environment variables."""
    db_config = config_manager.get_database_config()
    assert db_config.get("host") == TEST_HOST
    assert db_config.get("port") == TEST_PORT
    assert db_config.get("namespace") == TEST_NAMESPACE
    assert db_config.get("username") == TEST_USERNAME
    assert db_config.get("password") == TEST_PASSWORD
    assert db_config.get("driver_path") == TEST_DRIVER_PATH

def test_embedding_config_loading(config_manager):
    """Tests if embedding configurations are loaded correctly."""
    embedding_config = config_manager.get_embedding_config()
    assert embedding_config.get("model_name") == TEST_EMBEDDING_MODEL

def test_logging_config_loading(config_manager):
    """Tests if logging configurations are loaded and applied correctly."""
    log_config = config_manager.get_logging_config()
    assert log_config.get("level") == TEST_LOG_LEVEL.upper() # ConfigurationManager might uppercase it
    assert log_config.get("path") == TEST_LOG_PATH

    # Test if the logger was actually configured (basic check)
    logger = logging.getLogger("iris_rag") # Assuming this is your root logger name
    assert logging.getLevelName(logger.getEffectiveLevel()) == TEST_LOG_LEVEL.upper()

    # Check if file handler was added (more involved, might need to inspect logger.handlers)
    # For simplicity, we'll check if the log file is created after a log message
    # This is implicitly tested by ConfigurationManager's __init__ if it sets up logging.
    # If ConfigurationManager.setup_logging() is called explicitly, test that.
    # Here, we assume ConfigurationManager's constructor calls setup_logging.
    assert os.path.exists(os.path.dirname(TEST_LOG_PATH)), "Log directory was not created."
    # A simple log to ensure the file handler is working
    logger.warning("Test log message for config test.")
    assert os.path.exists(TEST_LOG_PATH), "Log file was not created after logging."


def test_general_config_loading(config_manager):
    """Tests loading of other general configurations."""
    assert config_manager.get_default_table_name() == TEST_TABLE_NAME
    assert config_manager.get_default_top_k() == int(TEST_TOP_K)

def test_missing_env_vars_defaults(setup_test_environment): # Uses fixture to ensure clean env
    """Tests default values when optional environment variables are missing."""
    # Remove some optional env vars to test defaults
    # Example: if LOG_PATH had a default
    if "LOG_PATH" in os.environ:
        del os.environ["LOG_PATH"]

    # Re-initialize ConfigurationManager with some vars missing
    cfg_manager_with_defaults = ConfigurationManager()

    # Assert that default values are used
    # Example:
    # assert cfg_manager_with_defaults.get_logging_config().get("path") == "logs/iris_rag.log" # Default path
    # This depends on your ConfigurationManager's actual default logic.
    # For this test suite, most critical vars are expected to be set.
    # We'll test one common default: LOG_LEVEL often defaults to INFO.
    if "LOG_LEVEL" in os.environ:
        del os.environ["LOG_LEVEL"]
    cfg_manager_log_default = ConfigurationManager()
    assert cfg_manager_log_default.get_logging_config().get("level") == "INFO" # Common default

    # Test for a required variable missing (should ideally raise error or have a safe default)
    # For critical ones like DB host, it might be better to fail early.
    # This depends on ConfigurationManager's design.
    if "IRIS_HOST" in os.environ:
        del os.environ["IRIS_HOST"]
    with pytest.raises(ValueError, match="Missing required configuration: IRIS_HOST"):
         ConfigurationManager() # Expecting an error if IRIS_HOST is critical and missing without default


# To run these tests:
# PYTHONPATH=. pytest tests/test_e2e_iris_rag_config_system.py