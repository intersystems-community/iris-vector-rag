"""
Testing infrastructure for configurable IRIS backend modes.

Provides backend mode configuration, iris-devtools integration,
edition detection, and connection pooling for pytest-based testing.
"""

# Import for re-export
from iris_vector_rag.config.backend_modes import (
    BackendMode,
    ConfigSource,
    ExecutionStrategy,
)
from iris_vector_rag.testing.backend_manager import (
    BackendConfiguration,
    load_configuration,
    log_session_start,
    validate_configuration,
)
from iris_vector_rag.testing.connection_pool import ConnectionPool
from iris_vector_rag.testing.exceptions import (
    BackendModeError,
    ConfigurationError,
    ConnectionLimitExceeded,
    ConnectionPoolError,
    ConnectionPoolTimeout,
    EditionDetectionError,
    EditionMismatchError,
    IrisDevtoolsError,
    IrisDevtoolsImportError,
    IrisDevtoolsMissingError,
)
from iris_vector_rag.testing.iris_devtools_bridge import IrisDevToolsBridge
from iris_vector_rag.testing.validators import IRISEdition, detect_iris_edition

__all__ = [
    # Enums
    "BackendMode",
    "IRISEdition",
    "ConfigSource",
    "ExecutionStrategy",
    # Classes
    "BackendConfiguration",
    "ConnectionPool",
    "IrisDevToolsBridge",
    # Functions
    "load_configuration",
    "validate_configuration",
    "log_session_start",
    "detect_iris_edition",
    # Exceptions
    "BackendModeError",
    "ConfigurationError",
    "EditionDetectionError",
    "EditionMismatchError",
    "IrisDevtoolsError",
    "IrisDevtoolsMissingError",
    "IrisDevtoolsImportError",
    "ConnectionPoolError",
    "ConnectionPoolTimeout",
    "ConnectionLimitExceeded",
]
