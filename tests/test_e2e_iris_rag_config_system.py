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

    # Set test environment variables using the RAG_ prefix that ConfigurationManager expects
    os.environ["RAG_DATABASE__IRIS__HOST"] = TEST_HOST
    os.environ["RAG_DATABASE__IRIS__PORT"] = TEST_PORT
    os.environ["RAG_DATABASE__IRIS__NAMESPACE"] = TEST_NAMESPACE
    os.environ["RAG_DATABASE__IRIS__USERNAME"] = TEST_USERNAME
    os.environ["RAG_DATABASE__IRIS__PASSWORD"] = TEST_PASSWORD
    os.environ["RAG_DATABASE__IRIS__DRIVER_PATH"] = TEST_DRIVER_PATH
    os.environ["RAG_EMBEDDINGS__MODEL_NAME"] = TEST_EMBEDDING_MODEL
    os.environ["RAG_LOGGING__LEVEL"] = TEST_LOG_LEVEL
    os.environ["RAG_LOGGING__PATH"] = TEST_LOG_PATH
    os.environ["RAG_DEFAULT_TABLE_NAME"] = TEST_TABLE_NAME
    os.environ["RAG_PIPELINES__BASIC__DEFAULT_TOP_K"] = TEST_TOP_K
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
    # The database config is nested under 'iris' key
    iris_config = db_config.get("iris", {})
    assert iris_config.get("host") == TEST_HOST
    assert iris_config.get("port") == int(TEST_PORT)  # Port should be converted to int
    assert iris_config.get("namespace") == TEST_NAMESPACE
    assert iris_config.get("username") == TEST_USERNAME
    assert iris_config.get("password") == TEST_PASSWORD
    assert iris_config.get("driver_path") == TEST_DRIVER_PATH

def test_embedding_config_loading(config_manager):
    """Tests if embedding configurations are loaded correctly."""
    embedding_config = config_manager.get_embedding_config()
    # The embedding config uses 'model' key, not 'model_name'
    assert embedding_config.get("model_name") == TEST_EMBEDDING_MODEL

def test_general_config_loading(config_manager):
    """Tests loading of other general configurations."""
    assert config_manager.get_default_table_name() == TEST_TABLE_NAME
    # Test the actual config path for top_k
    top_k_value = config_manager.get("pipelines:basic:default_top_k", 5)
    assert top_k_value == int(TEST_TOP_K)

def test_missing_env_vars_defaults(setup_test_environment): # Uses fixture to ensure clean env
    """Tests default values when optional environment variables are missing."""
    # Remove some optional env vars to test defaults
    # Example: if LOG_PATH had a default
    if "RAG_LOGGING__PATH" in os.environ:
        del os.environ["RAG_LOGGING__PATH"]

    # Re-initialize ConfigurationManager with some vars missing
    cfg_manager_with_defaults = ConfigurationManager()

    # Assert that default values are used
    # Example:
    # assert cfg_manager_with_defaults.get_logging_config().get("path") == "logs/iris_rag.log" # Default path
    # This depends on your ConfigurationManager's actual default logic.
    # For this test suite, most critical vars are expected to be set.
    # We'll test one common default: LOG_LEVEL often defaults to INFO.
    if "RAG_LOGGING__LEVEL" in os.environ:
        del os.environ["RAG_LOGGING__LEVEL"]
    cfg_manager_log_default = ConfigurationManager()
    assert cfg_manager_log_default.get_logging_config().get("level") == "INFO" # Common default

    # Test for a required variable missing (should ideally raise error or have a safe default)
    # For critical ones like DB host, it might be better to fail early.
    # This depends on ConfigurationManager's design.
    # Since the current ConfigurationManager doesn't validate required fields,
    # we'll test that it gracefully handles missing values by returning defaults
    if "RAG_DATABASE__IRIS__HOST" in os.environ:
        del os.environ["RAG_DATABASE__IRIS__HOST"]
    cfg_manager_missing_host = ConfigurationManager()
    # Should use default from config file (localhost)
    db_config = cfg_manager_missing_host.get_database_config()
    iris_config = db_config.get("iris", {})
    assert iris_config.get("host") == "localhost"  # Default from config file


# To run these tests:
# PYTHONPATH=. pytest tests/test_e2e_iris_rag_config_system.py