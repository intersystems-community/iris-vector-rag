"""
Backend mode configuration enums and utilities.

Provides enums and helper classes for configuring IRIS backend modes
(Community vs Enterprise) for test execution.

Feature: 035-make-2-modes
"""

from enum import Enum

from iris_vector_rag.testing.exceptions import ConfigurationError


class BackendMode(Enum):
    """
    IRIS backend mode for test execution.

    - COMMUNITY: Community edition with single connection limit
    - ENTERPRISE: Enterprise edition with unlimited connections
    """

    COMMUNITY = "community"
    ENTERPRISE = "enterprise"

    @classmethod
    def from_string(cls, value: str) -> "BackendMode":
        """
        Parse backend mode from string (case-insensitive).

        Args:
            value: Mode string ("community" or "enterprise")

        Returns:
            Parsed BackendMode enum value

        Raises:
            ConfigurationError: If value is not a valid mode

        Examples:
            >>> BackendMode.from_string("community")
            <BackendMode.COMMUNITY: 'community'>
            >>> BackendMode.from_string("ENTERPRISE")
            <BackendMode.ENTERPRISE: 'enterprise'>
            >>> BackendMode.from_string("invalid")
            ConfigurationError: Invalid backend mode: invalid
        """
        try:
            return cls(value.lower())
        except ValueError:
            raise ConfigurationError(
                f"Invalid backend mode: {value}\n"
                "Valid values: community, enterprise"
            )


class ConfigSource(Enum):
    """
    Source of backend mode configuration.

    Tracks where configuration was loaded from for debugging and logging.
    """

    ENVIRONMENT = "environment"  # From IRIS_BACKEND_MODE env var
    CONFIG_FILE = "config_file"  # From .specify/config/backend_modes.yaml
    DEFAULT = "default"  # Hardcoded default (COMMUNITY)


class ExecutionStrategy(Enum):
    """
    Test execution strategy based on backend mode.

    - SEQUENTIAL: Tests run one at a time (community mode)
    - PARALLEL: Tests run concurrently (enterprise mode)
    """

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
