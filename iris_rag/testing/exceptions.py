"""
Exception hierarchy for backend mode testing infrastructure.

Provides structured error classes with actionable error messages
for backend mode configuration, edition detection, connection pooling,
and iris-devtools integration.

Feature: 035-make-2-modes
"""


class BackendModeError(Exception):
    """Base exception for all backend mode-related errors."""
    pass


# Configuration Errors
class ConfigurationError(BackendModeError):
    """Raised when backend mode configuration is invalid."""
    pass


# Edition Detection Errors
class EditionDetectionError(BackendModeError):
    """Raised when IRIS edition cannot be detected."""
    pass


class EditionMismatchError(BackendModeError):
    """Raised when configured backend mode doesn't match detected IRIS edition."""
    pass


# iris-devtools Errors
class IrisDevtoolsError(BackendModeError):
    """Base exception for iris-devtools related errors."""
    pass


class IrisDevtoolsMissingError(IrisDevtoolsError):
    """Raised when iris-devtools is not found at expected path."""
    pass


class IrisDevtoolsImportError(IrisDevtoolsError):
    """Raised when iris-devtools import fails."""
    pass


# Connection Pool Errors
class ConnectionPoolError(BackendModeError):
    """Base exception for connection pool errors."""
    pass


class ConnectionPoolTimeout(ConnectionPoolError):
    """Raised when connection pool acquisition times out."""
    pass


class ConnectionLimitExceeded(ConnectionPoolError):
    """Raised when connection limit is exceeded."""
    pass


# Error message templates with actionable guidance
ERROR_MESSAGES = {
    "invalid_backend_mode": (
        "Invalid backend mode: {value}\n"
        "Valid values: community, enterprise\n"
        "Fix: Set IRIS_BACKEND_MODE to 'community' or 'enterprise'"
    ),
    "edition_mismatch": (
        "Backend mode '{mode}' does not match detected IRIS edition '{edition}'.\n"
        "Fix: Set IRIS_BACKEND_MODE={edition} or update config file"
    ),
    "iris_devtools_missing": (
        "iris-devtools not found at {path}\n"
        "Required development dependency.\n"
        "Fix: Clone iris-devtools to ../iris-devtools\n"
        "     git clone <iris-devtools-repo> ../iris-devtools"
    ),
    "connection_pool_timeout": (
        "Connection pool timeout after {timeout}s\n"
        "Mode: {mode} (max {max_connections} connections)\n"
        "Possible cause: Test parallelism exceeds connection limit\n"
        "Fix: Reduce test parallelism or switch to enterprise mode"
    ),
    "edition_detection_failed": (
        "Failed to detect IRIS edition: {error}\n"
        "Verify IRIS connection is active and accessible.\n"
        "Fix: Check database connection settings"
    ),
}
