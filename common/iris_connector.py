"""
Unified IRIS Database Connector
This module acts as a compatibility layer, ensuring that all parts of the application
use the centralized connection logic from the IRISConnectionManager.
"""

from .iris_connection_manager import (
    get_iris_connection,
    get_iris_dbapi_connection,
    get_iris_jdbc_connection,
    test_connection,
    IRISConnectionManager,
)

class IRISConnectionError(Exception):
    """Custom exception for IRIS connection errors."""
    pass

__all__ = [
    "get_iris_connection",
    "get_iris_dbapi_connection",
    "get_iris_jdbc_connection",
    "test_connection",
    "IRISConnectionManager",
    "IRISConnectionError",
]