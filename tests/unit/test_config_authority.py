"""Unit tests for config single authority (AUD-005).

Tests verify that ConfigurationManager is the single source of truth for
IRIS connection parameters, with proper precedence (env > YAML > defaults).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Dict, Any

import pytest
import yaml

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.common.iris_connection import get_iris_connection


class TestDefaultPortIs1972:
    """Test that default IRIS port is 1972 everywhere."""

    def test_default_config_yaml_port_is_1972(self):
        """default_config.yaml must have port 1972 (not 1974)."""
        default_yaml_path = (
            Path(__file__).parent.parent.parent
            / "iris_vector_rag/config/default_config.yaml"
        )
        with open(default_yaml_path) as f:
            config = yaml.safe_load(f)
        assert (
            config["database"]["iris"]["port"] == 1972
        ), "default_config.yaml port should be 1972"

    def test_config_manager_default_port_is_1972(self):
        """ConfigurationManager.get_database_config() default port is 1972."""
        env_clean = {k: v for k, v in os.environ.items() if not k.startswith("IRIS_")}
        with patch.dict(os.environ, env_clean, clear=True):
            # Set IRIS_HOST to avoid validation error
            with patch.dict(os.environ, {"IRIS_HOST": "localhost"}):
                cm = ConfigurationManager()
                db_config = cm.get_database_config()
                assert db_config["port"] == 1972, "Default database port should be 1972"
                assert isinstance(
                    db_config["port"], int
                ), "Port must be an integer, not string"

    def test_config_manager_no_override_returns_1972(self):
        """With no env vars and empty YAML, port defaults to 1972."""
        # Create temp YAML with host but no iris port override
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"database": {"iris": {"host": "localhost"}}}, f)
            temp_yaml = f.name

        try:
            # Remove IRIS_* vars from environment for this test
            env_clean = {
                k: v for k, v in os.environ.items() if not k.startswith("IRIS_")
            }
            with patch.dict(os.environ, env_clean, clear=True):
                cm = ConfigurationManager(config_path=temp_yaml)
                db_config = cm.get_database_config()
                assert (
                    db_config["port"] == 1972
                ), "Port should default to 1972 with empty YAML"
        finally:
            Path(temp_yaml).unlink()


class TestYAMLDrivesConnection:
    """Test that YAML config drives actual connection parameters."""

    def test_yaml_port_drives_config_manager(self):
        """YAML port 51972 is returned by ConfigurationManager."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "database": {
                        "iris": {
                            "port": 51972,
                            "host": "iris.example.com",
                            "namespace": "CUSTOM",
                        }
                    }
                },
                f,
            )
            temp_yaml = f.name

        try:
            # Remove IRIS_* vars from environment
            env_clean = {
                k: v for k, v in os.environ.items() if not k.startswith("IRIS_")
            }
            with patch.dict(os.environ, env_clean, clear=True):
                cm = ConfigurationManager(config_path=temp_yaml)
                db_config = cm.get_database_config()
                assert db_config["port"] == 51972, "YAML port should be 51972"
                assert (
                    db_config["host"] == "iris.example.com"
                ), "YAML host should be iris.example.com"
                assert (
                    db_config["namespace"] == "CUSTOM"
                ), "YAML namespace should be CUSTOM"
        finally:
            Path(temp_yaml).unlink()

    def test_connection_manager_uses_yaml_port(self):
        """ConnectionManager passes YAML port to get_iris_connection."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "database": {
                        "iris": {
                            "port": 54321,
                            "host": "test-host",
                            "namespace": "TEST",
                            "username": "testuser",
                            "password": "testpass",
                        }
                    }
                },
                f,
            )
            temp_yaml = f.name

        try:
            env_clean = {
                k: v for k, v in os.environ.items() if not k.startswith("IRIS_")
            }
            with patch.dict(os.environ, env_clean, clear=True):
                cm = ConfigurationManager(config_path=temp_yaml)
                conn_mgr = ConnectionManager(config_manager=cm)

                # Mock get_iris_connection to capture the params
                with patch(
                    "iris_vector_rag.common.iris_connection.get_iris_connection"
                ) as mock_get:
                    mock_conn = MagicMock()
                    mock_get.return_value = mock_conn

                    conn_mgr.get_connection()

                    # Verify that get_iris_connection was called with YAML params
                    mock_get.assert_called_once()
                    call_kwargs = mock_get.call_args.kwargs
                    assert (
                        call_kwargs.get("port") == 54321
                    ), f"Expected port 54321, got {call_kwargs.get('port')}"
                    assert (
                        call_kwargs.get("host") == "test-host"
                    ), f"Expected host 'test-host', got {call_kwargs.get('host')}"
                    assert call_kwargs.get("namespace") == "TEST"
                    assert call_kwargs.get("username") == "testuser"
                    assert call_kwargs.get("password") == "testpass"
        finally:
            Path(temp_yaml).unlink()


class TestEnvOverridesYAML:
    """Test that env vars override YAML configuration."""

    def test_env_port_overrides_yaml(self):
        """IRIS_PORT env var overrides YAML port."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "database": {
                        "iris": {
                            "port": 1972,
                            "host": "localhost",
                        }
                    }
                },
                f,
            )
            temp_yaml = f.name

        try:
            with patch.dict(os.environ, {"IRIS_PORT": "51972"}, clear=False):
                cm = ConfigurationManager(config_path=temp_yaml)
                db_config = cm.get_database_config()
                assert (
                    db_config["port"] == 51972
                ), "IRIS_PORT env var should override YAML port"
        finally:
            Path(temp_yaml).unlink()

    def test_env_host_overrides_yaml(self):
        """IRIS_HOST env var overrides YAML host."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "database": {
                        "iris": {
                            "host": "yaml-host",
                        }
                    }
                },
                f,
            )
            temp_yaml = f.name

        try:
            env_clean = {
                k: v
                for k, v in os.environ.items()
                if k
                not in (
                    "IRIS_HOST",
                    "IRIS_PORT",
                    "IRIS_NAMESPACE",
                    "IRIS_USERNAME",
                    "IRIS_PASSWORD",
                    "IRIS_DRIVER_PATH",
                )
            }
            with patch.dict(os.environ, env_clean, clear=True):
                with patch.dict(os.environ, {"IRIS_HOST": "env-host"}):
                    cm = ConfigurationManager(config_path=temp_yaml)
                    db_config = cm.get_database_config()
                    assert (
                        db_config["host"] == "env-host"
                    ), "IRIS_HOST env var should override YAML host"
        finally:
            Path(temp_yaml).unlink()

    def test_env_all_params_override_yaml(self):
        """All IRIS_* env vars override corresponding YAML values."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "database": {
                        "iris": {
                            "host": "yaml-host",
                            "port": 1972,
                            "namespace": "YAML_NS",
                            "username": "yaml_user",
                            "password": "yaml_pass",
                        }
                    }
                },
                f,
            )
            temp_yaml = f.name

        try:
            env_overrides = {
                "IRIS_HOST": "env-host",
                "IRIS_PORT": "54321",
                "IRIS_NAMESPACE": "ENV_NS",
                "IRIS_USERNAME": "env_user",
                "IRIS_PASSWORD": "env_pass",
            }
            # Clear all other IRIS_* vars
            env_clean = {
                k: v for k, v in os.environ.items() if not k.startswith("IRIS_")
            }
            with patch.dict(os.environ, env_clean, clear=True):
                with patch.dict(os.environ, env_overrides):
                    cm = ConfigurationManager(config_path=temp_yaml)
                    db_config = cm.get_database_config()
                    assert db_config["host"] == "env-host"
                    assert db_config["port"] == 54321
                    assert db_config["namespace"] == "ENV_NS"
                    assert db_config["username"] == "env_user"
                    assert db_config["password"] == "env_pass"
        finally:
            Path(temp_yaml).unlink()


class TestIRISUsernameFallback:
    """Test that both IRIS_USERNAME and IRIS_USER work (FR-005)."""

    def test_iris_username_takes_precedence(self):
        """IRIS_USERNAME takes precedence over IRIS_USER."""
        env_clean = {
            k: v
            for k, v in os.environ.items()
            if k not in ("IRIS_USER", "IRIS_USERNAME")
        }
        with patch.dict(os.environ, env_clean, clear=True):
            with patch.dict(
                os.environ,
                {
                    "IRIS_USER": "alice",
                    "IRIS_USERNAME": "bob",
                },
            ):
                cm = ConfigurationManager()
                db_config = cm.get_database_config()
                assert (
                    db_config["username"] == "bob"
                ), "IRIS_USERNAME should take precedence over IRIS_USER"

    def test_iris_user_fallback_when_username_not_set(self):
        """IRIS_USER is used when IRIS_USERNAME is not set."""
        env_clean = {
            k: v
            for k, v in os.environ.items()
            if k not in ("IRIS_USER", "IRIS_USERNAME")
        }
        with patch.dict(os.environ, env_clean, clear=True):
            with patch.dict(os.environ, {"IRIS_USER": "alice"}):
                cm = ConfigurationManager()
                db_config = cm.get_database_config()
                assert (
                    db_config["username"] == "alice"
                ), "IRIS_USER should be used when IRIS_USERNAME is not set"

    def test_default_username_when_neither_set(self):
        """Default username is _SYSTEM when neither IRIS_USERNAME nor IRIS_USER set."""
        env_clean = {
            k: v
            for k, v in os.environ.items()
            if k not in ("IRIS_USER", "IRIS_USERNAME")
        }
        with patch.dict(os.environ, env_clean, clear=True):
            cm = ConfigurationManager()
            db_config = cm.get_database_config()
            assert (
                db_config["username"] == "_SYSTEM"
            ), "Default username should be _SYSTEM"


class TestInvalidPortHandling:
    """Test that invalid IRIS_PORT raises clear error (Edge case)."""

    def test_invalid_port_raises_error(self):
        """Invalid IRIS_PORT string raises ValueError, not silent fallback."""
        env_clean = {k: v for k, v in os.environ.items() if not k.startswith("IRIS_")}
        with patch.dict(os.environ, env_clean, clear=True):
            with patch.dict(os.environ, {"IRIS_PORT": "invalid_string"}):
                cm = ConfigurationManager()
                with pytest.raises(ValueError) as exc_info:
                    cm.get_database_config()
                assert "IRIS_PORT" in str(
                    exc_info.value
                ), f"Error message should mention IRIS_PORT: {exc_info.value}"

    def test_out_of_range_port_raises_error(self):
        """Port outside valid range raises error."""
        env_clean = {k: v for k, v in os.environ.items() if not k.startswith("IRIS_")}
        with patch.dict(os.environ, env_clean, clear=True):
            with patch.dict(os.environ, {"IRIS_PORT": "99999"}):
                cm = ConfigurationManager()
                with pytest.raises(ValueError) as exc_info:
                    cm.get_database_config()
                assert "port" in str(exc_info.value).lower()


class TestPortTypeConsistency:
    """Test that port is always an integer, never a string."""

    def test_yaml_port_string_converted_to_int(self):
        """YAML port "1972" (string) is converted to int."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "database": {
                        "iris": {
                            "host": "localhost",
                            "port": "1972",  # String in YAML
                        }
                    }
                },
                f,
            )
            temp_yaml = f.name

        try:
            env_clean = {
                k: v for k, v in os.environ.items() if not k.startswith("IRIS_")
            }
            with patch.dict(os.environ, env_clean, clear=True):
                cm = ConfigurationManager(config_path=temp_yaml)
                db_config = cm.get_database_config()
                assert isinstance(
                    db_config["port"], int
                ), "Port must be int, even if YAML has string"
                assert db_config["port"] == 1972
        finally:
            Path(temp_yaml).unlink()

    def test_env_port_converted_to_int(self):
        """IRIS_PORT env var (string) is converted to int."""
        env_clean = {k: v for k, v in os.environ.items() if not k.startswith("IRIS_")}
        with patch.dict(os.environ, env_clean, clear=True):
            with patch.dict(os.environ, {"IRIS_PORT": "51972"}):
                cm = ConfigurationManager()
                db_config = cm.get_database_config()
                assert isinstance(
                    db_config["port"], int
                ), "Port must be int from env var"
                assert db_config["port"] == 51972


class TestConnectionManagerPassesConfigToIrisConnection:
    """Test that ConnectionManager actually passes resolved params to get_iris_connection."""

    def test_connection_manager_passes_config_params(self):
        """ConnectionManager.get_connection() passes ConfigManager params to get_iris_connection."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "database": {
                        "iris": {
                            "host": "config-host",
                            "port": 54321,
                            "namespace": "CONFIG_NS",
                            "username": "config_user",
                            "password": "config_pass",
                        }
                    }
                },
                f,
            )
            temp_yaml = f.name

        try:
            env_clean = {
                k: v for k, v in os.environ.items() if not k.startswith("IRIS_")
            }
            with patch.dict(os.environ, env_clean, clear=True):
                cm = ConfigurationManager(config_path=temp_yaml)
                conn_mgr = ConnectionManager(config_manager=cm)

                with patch(
                    "iris_vector_rag.common.iris_connection.get_iris_connection"
                ) as mock_get:
                    mock_conn = MagicMock()
                    mock_get.return_value = mock_conn

                    conn_mgr.get_connection()

                    # Verify all params were passed
                    call_kwargs = mock_get.call_args.kwargs
                    assert call_kwargs.get("host") == "config-host"
                    assert call_kwargs.get("port") == 54321
                    assert call_kwargs.get("namespace") == "CONFIG_NS"
                    assert call_kwargs.get("username") == "config_user"
                    assert call_kwargs.get("password") == "config_pass"
        finally:
            Path(temp_yaml).unlink()

    def test_connection_manager_respects_env_override(self):
        """ConnectionManager uses env-overridden config when env vars present."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(
                {
                    "database": {
                        "iris": {
                            "host": "localhost",
                            "port": 1972,
                        }
                    }
                },
                f,
            )
            temp_yaml = f.name

        try:
            env_clean = {
                k: v for k, v in os.environ.items() if not k.startswith("IRIS_")
            }
            with patch.dict(os.environ, env_clean, clear=True):
                with patch.dict(os.environ, {"IRIS_PORT": "55555"}):
                    cm = ConfigurationManager(config_path=temp_yaml)
                    conn_mgr = ConnectionManager(config_manager=cm)

                    with patch(
                        "iris_vector_rag.common.iris_connection.get_iris_connection"
                    ) as mock_get:
                        mock_conn = MagicMock()
                        mock_get.return_value = mock_conn

                        conn_mgr.get_connection()

                        # Env override should win
                        call_kwargs = mock_get.call_args.kwargs
                        assert (
                            call_kwargs.get("port") == 55555
                        ), "Env var IRIS_PORT should override YAML"
        finally:
            Path(temp_yaml).unlink()


class TestGetIrisConnectionExplicitParams:
    """Test that get_iris_connection accepts explicit params (FR-003)."""

    def test_get_iris_connection_signature_accepts_explicit_params(self):
        """get_iris_connection function signature accepts explicit params."""
        import inspect
        from iris_vector_rag.common.iris_connection import get_iris_connection

        sig = inspect.signature(get_iris_connection)
        param_names = list(sig.parameters.keys())

        # Should accept host, port, namespace, username, password
        assert "host" in param_names
        assert "port" in param_names
        assert "namespace" in param_names
        assert "username" in param_names
        assert "password" in param_names


class TestPrecedenceHierarchy:
    """Test full precedence: env > YAML > defaults."""

    def test_full_precedence_hierarchy(self):
        """Verify complete precedence: env > YAML > defaults."""
        scenarios = [
            # (env_port, yaml_port, expected_port, description)
            (None, None, 1972, "No env, no YAML → default 1972"),
            (None, 51972, 51972, "No env, YAML 51972 → YAML wins"),
            (55555, 51972, 55555, "Env 55555, YAML 51972 → env wins"),
            (55555, None, 55555, "Env 55555, no YAML → env wins"),
        ]

        for env_port, yaml_port, expected_port, description in scenarios:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                yaml_config = {"database": {"iris": {"host": "localhost"}}}
                if yaml_port is not None:
                    yaml_config["database"]["iris"]["port"] = yaml_port
                yaml.dump(yaml_config, f)
                temp_yaml = f.name

            try:
                env_clean = {
                    k: v for k, v in os.environ.items() if not k.startswith("IRIS_")
                }
                with patch.dict(os.environ, env_clean, clear=True):
                    if env_port is not None:
                        with patch.dict(os.environ, {"IRIS_PORT": str(env_port)}):
                            cm = ConfigurationManager(config_path=temp_yaml)
                            db_config = cm.get_database_config()
                    else:
                        cm = ConfigurationManager(config_path=temp_yaml)
                        db_config = cm.get_database_config()

                    assert (
                        db_config["port"] == expected_port
                    ), f"Failed: {description}. Got {db_config['port']}, expected {expected_port}"
            finally:
                Path(temp_yaml).unlink()
