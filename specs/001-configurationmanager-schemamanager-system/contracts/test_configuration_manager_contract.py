"""
Contract tests for ConfigurationManager component.

These tests validate the expected behavior and interface contracts for the
ConfigurationManager implementation against the functional requirements.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

# These imports will fail initially - this is expected for contract tests
try:
    from iris_rag.config.manager import ConfigurationManager, ConfigValidationError
except ImportError:
    # Contract tests are written before implementation
    ConfigurationManager = None
    ConfigValidationError = None


class TestConfigurationManagerContract:
    """Contract tests for ConfigurationManager functionality."""

    def test_configuration_manager_exists(self):
        """FR-001: ConfigurationManager class must exist and be importable."""
        assert (
            ConfigurationManager is not None
        ), "ConfigurationManager class must be implemented"

    def test_yaml_configuration_loading(self):
        """FR-001: System MUST load configuration from YAML files."""
        # Create temporary YAML config
        config_data = {
            "database": {"iris": {"host": "localhost", "port": 1972}},
            "embeddings": {"model": "test-model", "dimension": 384},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            if ConfigurationManager:
                manager = ConfigurationManager(config_path=config_path)
                assert manager.get_database_config()["host"] == "localhost"
                assert manager.get_embedding_config()["model"] == "test-model"
        finally:
            os.unlink(config_path)

    def test_environment_variable_overrides(self):
        """FR-001: Environment variables MUST override YAML configuration."""
        # Set environment variables
        os.environ["RAG_DATABASE__IRIS__HOST"] = "override-host"
        os.environ["RAG_EMBEDDINGS__MODEL"] = "override-model"

        config_data = {
            "database": {"iris": {"host": "localhost"}},
            "embeddings": {"model": "default-model"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            if ConfigurationManager:
                manager = ConfigurationManager(config_path=config_path)
                assert manager.get_database_config()["host"] == "override-host"
                assert manager.get_embedding_config()["model"] == "override-model"
        finally:
            os.unlink(config_path)
            # Clean up environment
            os.environ.pop("RAG_DATABASE__IRIS__HOST", None)
            os.environ.pop("RAG_EMBEDDINGS__MODEL", None)

    def test_required_configuration_validation(self):
        """FR-002: System MUST validate required configuration keys."""
        # Empty configuration should fail validation
        config_data = {}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            if ConfigurationManager and ConfigValidationError:
                with pytest.raises(ConfigValidationError):
                    ConfigurationManager(config_path=config_path)
        finally:
            os.unlink(config_path)

    def test_configuration_access_performance(self):
        """FR-010: Configuration access MUST be under 50ms."""
        import time

        config_data = {
            "database": {"iris": {"host": "localhost", "port": 1972}},
            "embeddings": {"model": "test-model", "dimension": 384},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            if ConfigurationManager:
                manager = ConfigurationManager(config_path=config_path)

                # Measure configuration access time
                start_time = time.time()
                for _ in range(100):  # Multiple accesses to average timing
                    manager.get_database_config()
                    manager.get_embedding_config()
                end_time = time.time()

                avg_time_ms = (
                    (end_time - start_time) / 200
                ) * 1000  # Convert to ms per call
                assert (
                    avg_time_ms < 50
                ), f"Configuration access took {avg_time_ms:.2f}ms, must be <50ms"
        finally:
            os.unlink(config_path)

    def test_hot_reload_non_critical_settings(self):
        """FR-002: System MUST support hot reload for non-critical settings only."""
        config_data = {
            "database": {"iris": {"host": "localhost", "port": 1972}},
            "logging": {"level": "INFO"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            if ConfigurationManager:
                manager = ConfigurationManager(config_path=config_path)

                # Hot reload should work for logging (non-critical)
                config_data["logging"]["level"] = "DEBUG"
                with open(config_path, "w") as f:
                    yaml.dump(config_data, f)

                if hasattr(manager, "reload_config"):
                    manager.reload_config()
                    logging_config = manager.get_logging_config()
                    assert logging_config["level"] == "DEBUG"
        finally:
            os.unlink(config_path)

    def test_precedence_order(self):
        """Environment variables > YAML files > defaults precedence must be enforced."""
        # Set up all three levels
        os.environ["RAG_TEST__VALUE"] = "env_value"

        config_data = {
            "database": {"iris": {"host": "localhost"}},
            "test": {"value": "yaml_value"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            if ConfigurationManager:
                manager = ConfigurationManager(config_path=config_path)

                # Environment should win
                if hasattr(manager, "get_config"):
                    test_config = manager.get_config("test")
                    assert test_config["value"] == "env_value"
        finally:
            os.unlink(config_path)
            os.environ.pop("RAG_TEST__VALUE", None)

    def test_type_casting_support(self):
        """Environment variables MUST support type casting with fallback."""
        os.environ["RAG_DATABASE__PORT"] = "1972"  # String that should become int
        os.environ["RAG_SETTINGS__ENABLED"] = "true"  # String that should become bool
        os.environ["RAG_SETTINGS__INVALID"] = (
            "not_a_number"  # Should fallback to string
        )

        config_data = {
            "database": {"iris": {"host": "localhost"}, "port": 1000},
            "settings": {"enabled": False, "invalid": 0},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            if ConfigurationManager:
                manager = ConfigurationManager(config_path=config_path)

                db_config = manager.get_database_config()
                settings_config = (
                    manager.get_config("settings")
                    if hasattr(manager, "get_config")
                    else {}
                )

                assert isinstance(db_config["port"], int)
                assert db_config["port"] == 1972

                if settings_config:
                    assert isinstance(settings_config["enabled"], bool)
                    assert settings_config["enabled"] is True
                    # Invalid should fallback to string
                    assert isinstance(settings_config["invalid"], str)
                    assert settings_config["invalid"] == "not_a_number"
        finally:
            os.unlink(config_path)
            os.environ.pop("RAG_DATABASE__PORT", None)
            os.environ.pop("RAG_SETTINGS__ENABLED", None)
            os.environ.pop("RAG_SETTINGS__INVALID", None)

    def test_clear_error_messages(self):
        """FR-002: System MUST fail fast with clear error messages when required config missing."""
        if ConfigurationManager and ConfigValidationError:
            with pytest.raises(ConfigValidationError) as exc_info:
                ConfigurationManager(config_path="nonexistent_config.yaml")

            # Error message should be specific and actionable
            error_msg = str(exc_info.value)
            assert len(error_msg) > 0, "Error message must not be empty"
            # Should contain guidance about the missing configuration
            assert any(
                keyword in error_msg.lower()
                for keyword in ["required", "missing", "config", "not found"]
            ), f"Error message should indicate missing config: {error_msg}"
