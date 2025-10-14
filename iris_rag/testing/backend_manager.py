"""
Backend mode configuration management.

Handles loading and validating backend configuration from
environment variables and configuration files.

Feature: 035-make-2-modes
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from iris_rag.config.backend_modes import (
    BackendMode,
    ConfigSource,
    ConfigurationError,
    ExecutionStrategy,
)
from iris_rag.testing.validators import (
    EditionMismatchError,
    IRISEdition,
)


# Default configuration file path
DEFAULT_CONFIG_PATH = Path(".specify/config/backend_modes.yaml")

from iris_rag.testing.exceptions import IrisDevtoolsMissingError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendConfiguration:
    """
    Immutable backend mode configuration.

    Holds backend mode settings and derived constraints for test execution.
    """

    mode: BackendMode
    source: ConfigSource
    iris_devtools_path: Path = Path("../iris-devtools")

    @property
    def max_connections(self) -> int:
        """
        Maximum concurrent database connections.

        Returns:
            1 for COMMUNITY mode, 999 for ENTERPRISE mode
        """
        return 1 if self.mode == BackendMode.COMMUNITY else 999

    @property
    def execution_strategy(self) -> ExecutionStrategy:
        """
        Test execution strategy.

        Returns:
            SEQUENTIAL for COMMUNITY mode, PARALLEL for ENTERPRISE mode
        """
        return (
            ExecutionStrategy.SEQUENTIAL
            if self.mode == BackendMode.COMMUNITY
            else ExecutionStrategy.PARALLEL
        )


def load_configuration(
    config_path: Optional[Path] = None,
) -> BackendConfiguration:
    """
    Load backend configuration with precedence: env var > config file > default.

    Configuration precedence order:
    1. IRIS_BACKEND_MODE environment variable
    2. Config file (.specify/config/backend_modes.yaml)
    3. Default (COMMUNITY mode)

    Args:
        config_path: Optional path to config file (defaults to DEFAULT_CONFIG_PATH)

    Returns:
        Loaded BackendConfiguration instance

    Raises:
        ConfigurationError: If configuration is invalid

    Examples:
        >>> os.environ['IRIS_BACKEND_MODE'] = 'enterprise'
        >>> config = load_configuration()
        >>> config.mode
        <BackendMode.ENTERPRISE: 'enterprise'>
        >>> config.source
        <ConfigSource.ENVIRONMENT: 'environment'>
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH

    # 1. Check environment variable (highest precedence)
    env_mode = os.environ.get("IRIS_BACKEND_MODE")
    if env_mode:
        mode = BackendMode.from_string(env_mode)
        return BackendConfiguration(
            mode=mode,
            source=ConfigSource.ENVIRONMENT,
        )

    # 2. Check config file
    if config_path.exists():
        try:
            with open(config_path) as f:
                config_data = yaml.safe_load(f)

            if config_data and "backend_mode" in config_data:
                mode = BackendMode.from_string(config_data["backend_mode"])

                # Optional: custom iris-devtools path
                iris_devtools_path = Path(
                    config_data.get("iris_devtools_path", "../iris-devtools")
                )

                return BackendConfiguration(
                    mode=mode,
                    source=ConfigSource.CONFIG_FILE,
                    iris_devtools_path=iris_devtools_path,
                )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to parse config file {config_path}: {e}"
            ) from e

    # 3. Default to COMMUNITY mode
    return BackendConfiguration(
        mode=BackendMode.COMMUNITY,
        source=ConfigSource.DEFAULT,
    )


def validate_configuration(
    config: BackendConfiguration,
    detected_edition: IRISEdition,
) -> None:
    """
    Validate backend configuration against detected IRIS edition.

    Ensures configured mode matches actual database edition.

    Args:
        config: Backend configuration to validate
        detected_edition: Edition detected from IRIS connection

    Raises:
        EditionMismatchError: If mode doesn't match detected edition
        IrisDevtoolsMissingError: If iris-devtools not found

    Examples:
        >>> config = BackendConfiguration(mode=BackendMode.COMMUNITY, source=ConfigSource.DEFAULT)
        >>> validate_configuration(config, IRISEdition.COMMUNITY)  # OK
        >>> validate_configuration(config, IRISEdition.ENTERPRISE)  # Raises EditionMismatchError
    """
    # Check iris-devtools exists
    if not config.iris_devtools_path.exists():
        raise IrisDevtoolsMissingError(
            f"iris-devtools not found at {config.iris_devtools_path}\n"
            "Required development dependency.\n"
            "Clone from: git clone <iris-devtools-repo> ../iris-devtools"
        )

    # Check mode matches detected edition
    expected_edition = (
        IRISEdition.COMMUNITY
        if config.mode == BackendMode.COMMUNITY
        else IRISEdition.ENTERPRISE
    )

    if detected_edition != expected_edition:
        raise EditionMismatchError(
            f"Backend mode '{config.mode.value}' does not match "
            f"detected IRIS edition '{detected_edition.value}'.\n"
            f"Fix: Set IRIS_BACKEND_MODE={detected_edition.value} "
            f"or update config file"
        )


def log_session_start(config: BackendConfiguration) -> None:
    """
    Log backend mode configuration at test session start.

    Args:
        config: Active backend configuration

    Examples:
        >>> config = BackendConfiguration(mode=BackendMode.COMMUNITY, source=ConfigSource.ENVIRONMENT)
        >>> log_session_start(config)
        # Logs: "Backend mode: community (source: environment)"
    """
    logger.info(
        f"Backend mode: {config.mode.value} (source: {config.source.value})"
    )
    logger.info(f"Max connections: {config.max_connections}")
    logger.info(f"Execution strategy: {config.execution_strategy.value}")
