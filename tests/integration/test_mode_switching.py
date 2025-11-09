"""
Integration tests for backend mode switching and immutability.

These tests verify:
- Mode can be switched via environment variable (FR-002)
- Mode is immutable during session (NFR-001)
- Configuration precedence works correctly

Status: Tests MUST FAIL until Phase 3.3 implementation
"""

import os
from pathlib import Path

import pytest

# These imports will fail until implementation - expected behavior for TDD
from iris_vector_rag.testing.backend_manager import (
    BackendConfiguration,
    BackendMode,
    ConfigSource,
    load_configuration,
)


@pytest.mark.integration
@pytest.mark.requires_backend_mode
class TestBackendModeSwitching:
    """Integration tests for mode switching behavior."""

    def test_mode_switch_via_environment_variable(self, monkeypatch):
        """
        Mode changes when environment variable changes.

        Given: No environment variable set
        When: IRIS_BACKEND_MODE set to 'enterprise'
        Then: load_configuration() returns ENTERPRISE mode

        Requirement: FR-002
        """
        # First load without env var (should be COMMUNITY default)
        monkeypatch.delenv("IRIS_BACKEND_MODE", raising=False)
        config1 = load_configuration()
        assert config1.mode == BackendMode.COMMUNITY
        assert config1.source == ConfigSource.DEFAULT or config1.source == ConfigSource.CONFIG_FILE

        # Set env var to enterprise
        monkeypatch.setenv("IRIS_BACKEND_MODE", "enterprise")
        config2 = load_configuration()
        assert config2.mode == BackendMode.ENTERPRISE
        assert config2.source == ConfigSource.ENVIRONMENT

    def test_mode_immutable_during_session(self):
        """
        BackendConfiguration is immutable after creation.

        Given: BackendConfiguration instance
        When: Attempt to modify mode attribute
        Then: Raises AttributeError (frozen dataclass)

        Requirement: NFR-001
        """
        config = BackendConfiguration(
            mode=BackendMode.COMMUNITY,
            source=ConfigSource.DEFAULT,
        )

        # Attempt to modify frozen dataclass
        with pytest.raises(AttributeError):
            config.mode = BackendMode.ENTERPRISE

        with pytest.raises(AttributeError):
            config.source = ConfigSource.ENVIRONMENT

    def test_environment_overrides_config_file(self, monkeypatch, tmp_path):
        """
        Environment variable takes precedence over config file.

        Given: Config file with mode=community AND env var with mode=enterprise
        When: load_configuration() called
        Then: Returns ENTERPRISE mode from environment

        Requirement: FR-002
        """
        # Create config file with community mode
        config_file = tmp_path / "backend_modes.yaml"
        config_file.write_text("backend_mode: community\n")

        # Set environment to enterprise
        monkeypatch.setenv("IRIS_BACKEND_MODE", "enterprise")

        # Mock config file path
        from iris_vector_rag.testing import backend_manager
        with monkeypatch.context() as m:
            m.setattr(backend_manager, "DEFAULT_CONFIG_PATH", config_file)
            config = load_configuration()

        # Environment should win
        assert config.mode == BackendMode.ENTERPRISE
        assert config.source == ConfigSource.ENVIRONMENT

    def test_config_file_used_when_no_env_var(self, monkeypatch, tmp_path):
        """
        Config file used when environment variable not set.

        Given: Config file with mode=enterprise AND no env var
        When: load_configuration() called
        Then: Returns ENTERPRISE mode from config file

        Requirement: FR-002
        """
        # Create config file with enterprise mode
        config_file = tmp_path / "backend_modes.yaml"
        config_file.write_text("backend_mode: enterprise\n")

        # Ensure no env var
        monkeypatch.delenv("IRIS_BACKEND_MODE", raising=False)

        # Mock config file path
        from iris_vector_rag.testing import backend_manager
        with monkeypatch.context() as m:
            m.setattr(backend_manager, "DEFAULT_CONFIG_PATH", config_file)
            config = load_configuration()

        # Config file should be used
        assert config.mode == BackendMode.ENTERPRISE
        assert config.source == ConfigSource.CONFIG_FILE


@pytest.mark.integration
@pytest.mark.requires_backend_mode
class TestConfigurationPrecedence:
    """Integration tests for configuration source precedence."""

    def test_default_when_no_config_sources(self, monkeypatch, tmp_path):
        """
        Default to COMMUNITY when no config sources available.

        Given: No env var AND no config file
        When: load_configuration() called
        Then: Returns COMMUNITY mode with DEFAULT source

        Requirement: FR-002
        """
        # Ensure no env var
        monkeypatch.delenv("IRIS_BACKEND_MODE", raising=False)

        # Point to nonexistent config file
        nonexistent_config = tmp_path / "nonexistent.yaml"

        from iris_vector_rag.testing import backend_manager
        with monkeypatch.context() as m:
            m.setattr(backend_manager, "DEFAULT_CONFIG_PATH", nonexistent_config)
            config = load_configuration()

        assert config.mode == BackendMode.COMMUNITY
        assert config.source == ConfigSource.DEFAULT

    def test_case_insensitive_environment_variable(self, monkeypatch):
        """
        Environment variable is case-insensitive.

        Given: IRIS_BACKEND_MODE=ENTERPRISE (uppercase)
        When: load_configuration() called
        Then: Correctly parses as ENTERPRISE mode

        Requirement: FR-009
        """
        monkeypatch.setenv("IRIS_BACKEND_MODE", "ENTERPRISE")

        config = load_configuration()

        assert config.mode == BackendMode.ENTERPRISE

    def test_case_insensitive_config_file(self, monkeypatch, tmp_path):
        """
        Config file mode is case-insensitive.

        Given: Config file with backend_mode: Community (mixed case)
        When: load_configuration() called
        Then: Correctly parses as COMMUNITY mode

        Requirement: FR-009
        """
        config_file = tmp_path / "backend_modes.yaml"
        config_file.write_text("backend_mode: Community\n")

        monkeypatch.delenv("IRIS_BACKEND_MODE", raising=False)

        from iris_vector_rag.testing import backend_manager
        with monkeypatch.context() as m:
            m.setattr(backend_manager, "DEFAULT_CONFIG_PATH", config_file)
            config = load_configuration()

        assert config.mode == BackendMode.COMMUNITY


@pytest.mark.integration
@pytest.mark.requires_backend_mode
class TestConfigurationImmutability:
    """Integration tests for configuration immutability (NFR-001)."""

    def test_configuration_object_immutable(self):
        """
        BackendConfiguration is fully immutable.

        Given: BackendConfiguration instance
        When: Attempt to modify any attribute
        Then: Raises AttributeError

        Requirement: NFR-001
        """
        config = BackendConfiguration(
            mode=BackendMode.ENTERPRISE,
            source=ConfigSource.ENVIRONMENT,
            iris_devtools_path=Path("../iris-devtools"),
        )

        # Cannot modify mode
        with pytest.raises(AttributeError):
            config.mode = BackendMode.COMMUNITY

        # Cannot modify source
        with pytest.raises(AttributeError):
            config.source = ConfigSource.CONFIG_FILE

        # Cannot modify path
        with pytest.raises(AttributeError):
            config.iris_devtools_path = Path("/other/path")

    def test_configuration_properties_immutable(self):
        """
        Derived properties cannot be modified.

        Given: BackendConfiguration instance
        When: Attempt to modify max_connections or execution_strategy
        Then: Raises AttributeError

        Requirement: NFR-001
        """
        config = BackendConfiguration(
            mode=BackendMode.COMMUNITY,
            source=ConfigSource.DEFAULT,
        )

        # Cannot modify computed properties
        with pytest.raises(AttributeError):
            config.max_connections = 999

        with pytest.raises(AttributeError):
            config.execution_strategy = "parallel"
