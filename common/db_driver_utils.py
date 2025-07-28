"""
Database driver utility functions.

This module provides low-level utilities for database driver detection
that can be used by other modules without creating circular dependencies.
"""

from enum import Enum
from typing import Any


class DriverType(Enum):
    """Enumeration of supported IRIS database drivers."""
    DBAPI = "dbapi"
    JDBC = "jdbc"


def get_driver_type(connection: Any = None) -> DriverType:
    """
    Determine the database driver type from a connection object.
    
    Args:
        connection: Database connection object (optional, for future use)
        
    Returns:
        DriverType enum value
    """
    if connection is not None:
        connection_type = str(type(connection))
        if 'jaydebeapi' in connection_type or 'JDBC' in connection_type:
            return DriverType.JDBC
        else:
            return DriverType.DBAPI
    
    # Default fallback - this maintains backward compatibility
    # In practice, this should be called with a connection object
    return DriverType.DBAPI


def get_driver_capabilities(driver_type: DriverType) -> dict:
    """
    Get the capabilities of a specific driver type.
    
    Args:
        driver_type: The driver type to get capabilities for
        
    Returns:
        Dictionary of capability flags
    """
    capabilities = {
        DriverType.DBAPI: {
            "vector_operations": True,
            "top_clause": True,
            "to_vector_function": True,
            "vector_cosine": True,
            "auto_parameterization": False,
            "supports_vector_operations": True  # Legacy compatibility
        },
        DriverType.JDBC: {
            "vector_operations": False,  # Due to auto-parameterization bug
            "top_clause": False,         # Gets parameterized incorrectly
            "to_vector_function": False, # Gets parameterized incorrectly
            "vector_cosine": False,      # Requires workarounds
            "auto_parameterization": True,
            "supports_vector_operations": False  # Legacy compatibility
        }
    }
    
    return capabilities.get(driver_type, capabilities[DriverType.DBAPI])