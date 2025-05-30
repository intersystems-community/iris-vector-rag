"""
JDBC-based IRIS connector - Drop-in replacement for iris_connector.py
This module provides JDBC connection as a solution for vector parameter binding issues
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jdbc_exploration.iris_jdbc_connector import (
    get_iris_jdbc_connection as _get_jdbc_connection,
    IRISJDBCConnection
)

def get_iris_connection():
    """
    Drop-in replacement for get_iris_connection() that uses JDBC
    
    This function returns a JDBC connection that supports parameter binding
    with vector functions, solving the ODBC parameter binding issues.
    
    Returns:
        IRISJDBCConnection: A JDBC connection object with cursor() method
    """
    # Override namespace to USER for compatibility
    return _get_jdbc_connection(
        namespace=os.getenv('IRIS_NAMESPACE', 'USER'),
        username=os.getenv('IRIS_USERNAME', '_SYSTEM'),
        password=os.getenv('IRIS_PASSWORD', 'SYS')
    )

# For compatibility with existing code that might import the class
get_real_iris_connection = get_iris_connection
get_mock_iris_connection = get_iris_connection  # No mock for JDBC yet
get_testcontainer_connection = get_iris_connection  # No testcontainer for JDBC yet

# Re-export the connection class
__all__ = ['get_iris_connection', 'IRISJDBCConnection']