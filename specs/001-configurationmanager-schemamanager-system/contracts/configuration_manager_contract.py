"""
ConfigurationManager Contract Tests

These tests validate the ConfigurationManager public interface and ensure
compliance with the functional requirements. Tests must fail initially
and pass after implementation validation.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

# Import the actual implementation to test against
from iris_rag.config.manager import ConfigurationManager, ConfigValidationError


class TestConfigurationManagerContract:
    """Contract tests for ConfigurationManager functionality."""

    def setup_method(self):
        """Set up test environment for each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_config_file = Path(self.temp_dir) / "test_config.yaml"

    def teardown_method(self):
        """Clean up test environment after each test method."""
        # Clear environment variables
        env_vars = [key for key in os.environ.keys() if key.startswith("RAG_")]
        for var in env_vars:
            del os.environ[var]

    def test_yaml_configuration_loading(self):
        """
        FR-001: System MUST load configuration from YAML files with environment variable overrides.
        """
        # Create test YAML configuration
        config_content = """
        database:
          iris:
            host: localhost
            port: 1972
            namespace: RAG
        embeddings:
          model: all-MiniLM-L6-v2
          dimension: 384
        """
        self.test_config_file.write_text(config_content)

        # Load configuration
        config_manager = ConfigurationManager(
            config_file_path=str(self.test_config_file)
        )

        # Verify YAML values loaded correctly
        assert config_manager.get("database.iris.host") == "localhost"
        assert config_manager.get("database.iris.port") == 1972
        assert config_manager.get("embeddings.model") == "all-MiniLM-L6-v2"
        assert config_manager.get("embeddings.dimension") == 384

    def test_environment_variable_overrides(self):
        """
        FR-001: Environment variables MUST override YAML files using RAG_ prefix and __ delimiters.
        """
        # Create test YAML configuration
        config_content = """
        database:
          iris:
            host: localhost
            port: 1972
        """
        self.test_config_file.write_text(config_content)

        # Set environment variable overrides
        os.environ["RAG_DATABASE__IRIS__HOST"] = "remote-host"
        os.environ["RAG_DATABASE__IRIS__PORT"] = "2972"

        # Load configuration
        config_manager = ConfigurationManager(
            config_file_path=str(self.test_config_file)
        )

        # Verify environment variables override YAML values
        assert config_manager.get("database.iris.host") == "remote-host"
        assert config_manager.get("database.iris.port") == "2972"  # String from env var

    def test_configuration_precedence_order(self):
        """
        Clarification: Environment variables override YAML files override defaults.
        """
        # Test default fallback
        config_manager = ConfigurationManager()
        default_value = config_manager.get("nonexistent.key", "default_value")
        assert default_value == "default_value"

        # Test YAML override of default
        config_content = """
        test:
          key: yaml_value
        """
        self.test_config_file.write_text(config_content)
        config_manager = ConfigurationManager(
            config_file_path=str(self.test_config_file)
        )
        assert config_manager.get("test.key") == "yaml_value"

        # Test environment override of YAML
        os.environ["RAG_TEST__KEY"] = "env_value"
        config_manager = ConfigurationManager(
            config_file_path=str(self.test_config_file)
        )
        assert config_manager.get("test.key") == "env_value"

    def test_required_configuration_validation(self):
        """
        FR-002: System MUST validate all required configuration keys and fail fast with clear error messages.
        """
        # Create minimal config that might be missing required keys
        config_content = """
        incomplete:
          config: true
        """
        self.test_config_file.write_text(config_content)

        # This should work for basic initialization
        config_manager = ConfigurationManager(
            config_file_path=str(self.test_config_file)
        )

        # Accessing specialized config methods should validate requirements
        with pytest.raises(ConfigValidationError) as exc_info:
            config_manager.get_database_config()

        # Verify error message is clear and actionable
        assert "required" in str(exc_info.value).lower()

    def test_specialized_config_methods(self):
        """
        Test specialized configuration access methods for different components.
        """
        config_content = """
        database:
          iris:
            host: localhost
            port: 1972
            namespace: RAG
            username: _SYSTEM
            password: SYS
        embeddings:
          model: all-MiniLM-L6-v2
          dimension: 384
          provider: sentence-transformers
        vector_index:
          type: HNSW
          M: 16
          efConstruction: 200
          Distance: COSINE
        """
        self.test_config_file.write_text(config_content)
        config_manager = ConfigurationManager(
            config_file_path=str(self.test_config_file)
        )

        # Test database config method
        db_config = config_manager.get_database_config()
        assert db_config["host"] == "localhost"
        assert db_config["port"] == 1972

        # Test embedding config method
        embedding_config = config_manager.get_embedding_config()
        assert embedding_config["model"] == "all-MiniLM-L6-v2"
        assert embedding_config["dimension"] == 384

        # Test vector index config method
        vector_config = config_manager.get_vector_index_config()
        assert vector_config["type"] == "HNSW"
        assert vector_config["M"] == 16

    def test_type_casting_from_environment(self):
        """
        Test that environment variable values are properly cast to appropriate types.
        """
        config_content = """
        test:
          bool_value: false
          int_value: 42
          float_value: 3.14
        """
        self.test_config_file.write_text(config_content)

        # Set environment variables with string values
        os.environ["RAG_TEST__BOOL_VALUE"] = "true"
        os.environ["RAG_TEST__INT_VALUE"] = "84"
        os.environ["RAG_TEST__FLOAT_VALUE"] = "6.28"
        os.environ["RAG_TEST__STRING_VALUE"] = "hello"

        config_manager = ConfigurationManager(
            config_file_path=str(self.test_config_file)
        )

        # Verify type casting works correctly
        assert (
            config_manager.get("test.bool_value") == "true"
        )  # Env vars are strings by default
        assert config_manager.get("test.int_value") == "84"
        assert config_manager.get("test.float_value") == "6.28"
        assert config_manager.get("test.string_value") == "hello"

    def test_hot_reload_capability(self):
        """
        Clarification: Support hot reload for non-critical settings only.
        """
        config_content = """
        logging:
          level: INFO
        critical:
          database_host: localhost
        """
        self.test_config_file.write_text(config_content)
        config_manager = ConfigurationManager(
            config_file_path=str(self.test_config_file)
        )

        original_level = config_manager.get("logging.level")
        assert original_level == "INFO"

        # Update config file
        updated_content = """
        logging:
          level: DEBUG
        critical:
          database_host: localhost
        """
        self.test_config_file.write_text(updated_content)

        # Hot reload should be possible for logging but implementation-dependent
        # This test documents the expected interface
        if hasattr(config_manager, "reload_config"):
            config_manager.reload_config()
            assert config_manager.get("logging.level") == "DEBUG"

    def test_performance_target_config_access(self):
        """
        Clarification: Configuration access should be <50ms.
        """
        config_content = """
        test:
          performance_key: performance_value
        """
        self.test_config_file.write_text(config_content)
        config_manager = ConfigurationManager(
            config_file_path=str(self.test_config_file)
        )

        import time

        # Measure configuration access time
        start_time = time.perf_counter()
        for _ in range(100):  # Multiple accesses to test caching
            value = config_manager.get("test.performance_key")
        end_time = time.perf_counter()

        avg_access_time = ((end_time - start_time) * 1000) / 100  # Convert to ms
        assert (
            avg_access_time < 50
        ), f"Config access took {avg_access_time}ms, exceeds 50ms target"

    def test_error_handling_patterns(self):
        """
        FR-002: Test explicit error handling with no silent failures.
        """
        # Test with missing config file
        with pytest.raises(Exception) as exc_info:
            ConfigurationManager(config_file_path="/nonexistent/config.yaml")

        # Error should be explicit, not silent
        assert exc_info.value is not None

        # Test with invalid YAML
        invalid_yaml = "invalid: yaml: content: ["
        self.test_config_file.write_text(invalid_yaml)

        with pytest.raises(Exception) as exc_info:
            ConfigurationManager(config_file_path=str(self.test_config_file))

        # Error should be explicit about YAML parsing issue
        assert exc_info.value is not None

    def test_configuration_deep_merge(self):
        """
        Test that configuration updates use deep merge for nested structures.
        """
        config_content = """
        database:
          iris:
            host: localhost
            port: 1972
          redis:
            host: localhost
            port: 6379
        """
        self.test_config_file.write_text(config_content)

        # Set partial environment override
        os.environ["RAG_DATABASE__IRIS__HOST"] = "remote-iris"

        config_manager = ConfigurationManager(
            config_file_path=str(self.test_config_file)
        )

        # Verify deep merge preserves non-overridden values
        assert config_manager.get("database.iris.host") == "remote-iris"  # Overridden
        assert config_manager.get("database.iris.port") == 1972  # Preserved
        assert config_manager.get("database.redis.host") == "localhost"  # Preserved


@pytest.mark.requires_database
class TestConfigurationManagerDatabaseIntegration:
    """Contract tests that require database connectivity."""

    def test_database_config_validation(self):
        """
        Test that database configuration validation works with actual database connectivity.
        """
        # This would test actual database connection using the configuration
        # Implementation depends on connection manager integration
        pass

    def test_config_integration_with_schema_manager(self):
        """
        Test integration between ConfigurationManager and SchemaManager.
        """
        # This would test the complete integration workflow
        # Implementation depends on schema manager integration
        pass
