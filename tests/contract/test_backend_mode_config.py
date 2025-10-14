"""
Contract tests for backend mode configuration management.

These tests define the expected API contract for:
- Loading backend configuration from environment, config file, and defaults (FR-001, FR-002)
- Validating configuration against detected IRIS edition (FR-008)
- Logging mode at session start (FR-012)
- Error handling for invalid configurations (FR-009)

Contract: specs/035-make-2-modes/contracts/backend_config_contract.yaml
Status: Tests MUST FAIL until Phase 3.3 implementation
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# These imports will fail until implementation - expected behavior for TDD
from iris_rag.testing.backend_manager import (
    BackendConfiguration,
    ConfigurationError,
    EditionMismatchError,
    IrisDevtoolsMissingError,
    load_configuration,
    log_session_start,
    validate_configuration,
)
from iris_rag.testing.validators import IRISEdition


@pytest.mark.contract
@pytest.mark.requires_backend_mode
class TestBackendConfigurationLoading:
    """Contract tests for load_configuration() operation."""

    def test_load_from_environment_variable(self, monkeypatch):
        """
        Environment variable takes precedence over config file.

        Given: IRIS_BACKEND_MODE=enterprise AND config file has backend_mode=community
        When: load_configuration() called
        Then: Returns BackendConfiguration with mode=ENTERPRISE and source=ENVIRONMENT

        Requirement: FR-002
        """
        monkeypatch.setenv("IRIS_BACKEND_MODE", "enterprise")

        config = load_configuration()

        assert config.mode.value == "enterprise"
        assert config.source.value == "environment"
        assert config.max_connections == 999  # enterprise = unlimited
        assert config.execution_strategy.value == "parallel"

    def test_load_from_config_file(self, monkeypatch, tmp_path):
        """
        Config file used when no environment variable set.

        Given: No IRIS_BACKEND_MODE env var AND config file has backend_mode=community
        When: load_configuration() called
        Then: Returns BackendConfiguration with mode=COMMUNITY and source=CONFIG_FILE

        Requirement: FR-002
        """
        # Ensure no env var
        monkeypatch.delenv("IRIS_BACKEND_MODE", raising=False)

        # Point to test config file
        config_file = tmp_path / "backend_modes.yaml"
        config_file.write_text("backend_mode: community\n")

        with patch("iris_rag.testing.backend_manager.DEFAULT_CONFIG_PATH", config_file):
            config = load_configuration()

        assert config.mode.value == "community"
        assert config.source.value == "config_file"
        assert config.max_connections == 1  # community = single connection
        assert config.execution_strategy.value == "sequential"

    def test_load_default(self, monkeypatch, tmp_path):
        """
        Default to COMMUNITY when no config provided.

        Given: No IRIS_BACKEND_MODE env var AND no config file
        When: load_configuration() called
        Then: Returns BackendConfiguration with mode=COMMUNITY and source=DEFAULT

        Requirement: FR-002
        """
        # Ensure no env var
        monkeypatch.delenv("IRIS_BACKEND_MODE", raising=False)

        # Point to nonexistent config file
        nonexistent_config = tmp_path / "nonexistent.yaml"

        with patch("iris_rag.testing.backend_manager.DEFAULT_CONFIG_PATH", nonexistent_config):
            config = load_configuration()

        assert config.mode.value == "community"
        assert config.source.value == "default"
        assert config.max_connections == 1
        assert config.execution_strategy.value == "sequential"

    def test_invalid_mode_value(self, monkeypatch):
        """
        Invalid mode string raises clear error.

        Given: IRIS_BACKEND_MODE=invalid_value
        When: load_configuration() called
        Then: Raises ConfigurationError with valid values listed

        Requirement: FR-009
        """
        monkeypatch.setenv("IRIS_BACKEND_MODE", "invalid_value")

        with pytest.raises(ConfigurationError) as exc_info:
            load_configuration()

        error_msg = str(exc_info.value)
        assert "invalid_value" in error_msg.lower()
        assert "community" in error_msg.lower()
        assert "enterprise" in error_msg.lower()


@pytest.mark.contract
@pytest.mark.requires_backend_mode
class TestBackendConfigurationValidation:
    """Contract tests for validate_configuration() operation."""

    def test_validate_matching_edition(self):
        """
        Validation passes when mode matches detected edition.

        Given: BackendConfiguration with mode=COMMUNITY AND IRISEdition.COMMUNITY detected
        When: validate_configuration(config, edition) called
        Then: Returns normally without error

        Requirement: FR-008
        """
        config = BackendConfiguration(
            mode=BackendMode.COMMUNITY,
            source=ConfigSource.ENVIRONMENT,
            iris_devtools_path=Path("../iris-devtools"),
        )
        detected_edition = IRISEdition.COMMUNITY

        # Should not raise any exception
        validate_configuration(config, detected_edition)

    def test_validate_mismatched_edition(self):
        """
        Validation fails when mode doesn't match detected edition.

        Given: BackendConfiguration with mode=ENTERPRISE AND IRISEdition.COMMUNITY detected
        When: validate_configuration(config, edition) called
        Then: Raises EditionMismatchError with actionable message

        Requirement: FR-008
        """
        config = BackendConfiguration(
            mode=BackendMode.ENTERPRISE,
            source=ConfigSource.ENVIRONMENT,
            iris_devtools_path=Path("../iris-devtools"),
        )
        detected_edition = IRISEdition.COMMUNITY

        with pytest.raises(EditionMismatchError) as exc_info:
            validate_configuration(config, detected_edition)

        error_msg = str(exc_info.value)
        assert "enterprise" in error_msg.lower()
        assert "community" in error_msg.lower()
        assert "IRIS_BACKEND_MODE" in error_msg

    def test_validate_missing_iris_devtools(self):
        """
        Validation fails when iris-devtools not found.

        Given: BackendConfiguration with iris_devtools_path=/nonexistent
        When: validate_configuration(config, edition) called
        Then: Raises IrisDevtoolsMissingError with installation instructions

        Requirement: FR-007
        """
        config = BackendConfiguration(
            mode=BackendMode.COMMUNITY,
            source=ConfigSource.ENVIRONMENT,
            iris_devtools_path=Path("/nonexistent/iris-devtools"),
        )
        detected_edition = IRISEdition.COMMUNITY

        with pytest.raises(IrisDevtoolsMissingError) as exc_info:
            validate_configuration(config, detected_edition)

        error_msg = str(exc_info.value)
        assert "/nonexistent/iris-devtools" in error_msg
        assert "iris-devtools" in error_msg.lower()


@pytest.mark.contract
@pytest.mark.requires_backend_mode
class TestBackendModeLogging:
    """Contract tests for log_session_start() operation."""

    def test_log_mode_at_session_start(self, caplog):
        """
        Backend mode logged at test session start.

        Given: BackendConfiguration with mode=COMMUNITY
        When: log_session_start(config) called
        Then: Logs "Backend mode: community (source: environment)" at INFO level

        Requirement: FR-012
        """
        config = BackendConfiguration(
            mode=BackendMode.COMMUNITY,
            source=ConfigSource.ENVIRONMENT,
            iris_devtools_path=Path("../iris-devtools"),
        )

        with caplog.at_level("INFO"):
            log_session_start(config)

        # Check log output
        assert len(caplog.records) >= 1
        log_message = caplog.records[0].message.lower()
        assert "backend mode" in log_message
        assert "community" in log_message
        assert "environment" in log_message


# Missing enum imports - will be added when implementation exists
from iris_rag.testing.backend_manager import BackendMode, ConfigSource
