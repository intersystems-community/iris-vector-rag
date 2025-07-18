import pytest
import os
import yaml
from unittest import mock

# Attempt to import ConfigurationManager, will fail initially
try:
    from iris_rag.config.manager import ConfigurationManager, ConfigValidationError
except ImportError:
    ConfigurationManager = None
    ConfigValidationError = None if ConfigurationManager is None else Exception # Placeholder

@pytest.fixture
def mock_config_file(tmp_path):
    """Creates a temporary YAML config file for testing."""
    config_content = {
        "database": {
            "iris": {
                "host": "file_host",
                "port": 1972,
                "namespace": "FILE_NS",
                "user": "file_user",
                "password": "file_pass"
            },
            "common": {
                "timeout": 30
            }
        },
        "llm": {
            "provider": "openai",
            "api_key": "file_llm_key"
        },
        "embedding": {
            "model_name": "text-embedding-ada-002"
        }
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_content, f)
    return str(config_file)

def test_import_configuration_manager():
    """Tests that ConfigurationManager can be imported."""
    assert ConfigurationManager is not None, "Failed to import ConfigurationManager"
    assert ConfigValidationError is not None, "Failed to import ConfigValidationError"

def test_load_from_yaml_file(mock_config_file):
    """Tests loading configuration from a YAML file."""
    if ConfigurationManager is None: pytest.fail("CM not imported")
    
    cm = ConfigurationManager(config_path=mock_config_file)
    
    assert cm.get("database:iris:host") == "file_host"
    assert cm.get("database:iris:port") == 1972
    assert cm.get("llm:provider") == "openai"
    assert cm.get("embedding:model_name") == "text-embedding-ada-002"
    assert cm.get("database:common:timeout") == 30

def test_env_variable_override(mock_config_file, monkeypatch):
    """Tests that environment variables override YAML file settings."""
    if ConfigurationManager is None: pytest.fail("CM not imported")

    monkeypatch.setenv("RAG_DATABASE__IRIS__HOST", "env_host")
    monkeypatch.setenv("RAG_DATABASE__IRIS__PORT", "5432") # Env vars are strings
    monkeypatch.setenv("RAG_LLM__API_KEY", "env_llm_key")
    # RAG_DATABASE__COMMON__TIMEOUT is not set, should use file value

    cm = ConfigurationManager(config_path=mock_config_file)

    assert cm.get("database:iris:host") == "env_host"
    assert cm.get("database:iris:port") == 5432 # Expect type conversion
    assert cm.get("llm:api_key") == "env_llm_key"
    assert cm.get("database:common:timeout") == 30 # From file
    assert cm.get("database:iris:namespace") == "FILE_NS" # From file

def test_get_nested_setting(mock_config_file):
    """Tests getting a deeply nested setting."""
    if ConfigurationManager is None: pytest.fail("CM not imported")
    cm = ConfigurationManager(config_path=mock_config_file)
    assert cm.get("database:iris") == {
        "host": "file_host",
        "port": 1972,
        "namespace": "FILE_NS",
        "user": "file_user",
        "password": "file_pass"
    }

def test_get_non_existent_setting(mock_config_file):
    """Tests getting a non-existent setting returns None or default."""
    if ConfigurationManager is None: pytest.fail("CM not imported")
    cm = ConfigurationManager(config_path=mock_config_file)
    assert cm.get("non:existent:key") is None
    assert cm.get("non:existent:key", "default_val") == "default_val"

def test_missing_config_file():
    """Tests behavior when the config file is missing."""
    if ConfigurationManager is None: pytest.fail("CM not imported")
    
    # Should not raise error if file is optional or defaults are used
    # Or raise a specific error if config_path is given but not found
    with pytest.raises(FileNotFoundError): # Assuming it raises FileNotFoundError
        ConfigurationManager(config_path="non_existent_config.yaml")

def test_config_validation_error_required_key():
    """Tests that a ConfigValidationError is raised for missing required keys (hypothetical)."""
    if ConfigurationManager is None or ConfigValidationError is None:
        pytest.fail("CM or CVE not imported")

    # This test depends on ConfigurationManager implementing validation logic.
    # For now, it will fail as the basic CM won&#x27;t have this.
    # We&#x27;d need to mock a config file missing a required key.
    
    # Example: Assume 'database:iris:host' is required by a schema (not yet implemented)
    bad_config_content = {"database": {"iris": {"port": 1234}}} # Missing host
    bad_config_file = "bad_config.yaml" # Not actually creating file for this initial test
    
    # This part of the test will likely fail until validation is built
    with mock.patch('builtins.open', mock.mock_open(read_data=yaml.dump(bad_config_content))):
        with mock.patch('os.path.exists', return_value=True):
            with pytest.raises(ConfigValidationError, match="Missing required config: database:iris:host"):
                ConfigurationManager(config_path=bad_config_file)

# More tests to be added:
# - Different data types (int, bool, list) from env vars.
# - Default values for get method.
# - Schema validation if implemented.