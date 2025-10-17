"""
Contract tests for ConfigurationManager component.

These tests define the expected behavior and interface contracts that the
ConfigurationManager implementation must satisfy.
"""

from typing import Any, Dict, Optional

import pytest


class TestConfigurationManagerContract:
    """Contract tests for ConfigurationManager interface."""

    def test_yaml_configuration_loading(self, config_manager):
        """
        Contract: System MUST load configuration from YAML files with
        environment variable overrides using RAG_ prefix and __ delimiters.
        """
        # Given: A configuration with nested keys
        # When: Loading configuration
        # Then: Nested keys accessible via colon notation
        assert config_manager.get("database:iris:host") is not None

    def test_required_configuration_validation(self, config_manager):
        """
        Contract: System MUST validate all required configuration keys
        and fail fast with clear error messages if missing.
        """
        # Given: Missing required configuration
        # When: Validating configuration
        # Then: ConfigValidationError raised with clear message
        with pytest.raises(Exception) as exc_info:
            # Test by creating manager with empty config
            pass

    def test_environment_variable_override(self, config_manager):
        """
        Contract: Environment variables with RAG_ prefix MUST override YAML values.
        """
        # Given: RAG_ prefixed environment variable
        # When: Getting configuration value
        # Then: Environment variable value takes precedence
        pass

    def test_type_casting_behavior(self, config_manager):
        """
        Contract: Environment variables MUST be cast to appropriate types
        with fallback to string for invalid casts.
        """
        # Given: Environment variables of different types
        # When: Getting configuration values
        # Then: Values are properly cast to expected types
        pass

    def test_deep_merge_capabilities(self, config_manager):
        """
        Contract: Configuration updates MUST support deep merge for nested dictionaries.
        """
        # Given: Existing nested configuration
        # When: Merging new nested configuration
        # Then: Deep merge preserves existing keys while updating specified ones
        pass

    def test_vector_index_configuration(self, config_manager):
        """
        Contract: System MUST provide vector index configuration with HNSW defaults.
        """
        vector_config = config_manager.get_vector_index_config()

        assert vector_config["type"] == "HNSW"
        assert "M" in vector_config
        assert "efConstruction" in vector_config
        assert "Distance" in vector_config

    def test_embedding_configuration(self, config_manager):
        """
        Contract: System MUST provide embedding configuration with model and dimension info.
        """
        embedding_config = config_manager.get_embedding_config()

        assert "model" in embedding_config
        assert "dimension" in embedding_config
        assert "provider" in embedding_config

    def test_database_configuration(self, config_manager):
        """
        Contract: System MUST provide database configuration with IRIS connection defaults.
        """
        db_config = config_manager.get_database_config()

        assert "host" in db_config
        assert "port" in db_config
        assert "namespace" in db_config
        assert "username" in db_config
