"""
Centralized connection manager for RAG pipelines
Handles JDBC, ODBC, and future dbapi connections with a unified interface
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages database connections for RAG pipelines"""

    def __init__(self, connection_type: str = "odbc"):
        """
        Initialize connection manager

        Args:
            connection_type: Either "jdbc", "odbc", or "dbapi" (default: "odbc" for stability)
        """
        self.connection_type = connection_type.lower()
        self._connection = None
        self._cursor = None

        if self.connection_type not in ["jdbc", "odbc", "dbapi"]:
            raise ValueError(
                f"Invalid connection type: {connection_type}. Must be 'jdbc', 'odbc', or 'dbapi'"
            )

    def connect(self):
        """Establish database connection based on configured type"""
        if self._connection:
            return self._connection

        if self.connection_type == "jdbc":
            try:
                from common.iris_connection_manager import get_iris_jdbc_connection

                self._connection = get_iris_jdbc_connection()
                logger.info("Established JDBC connection")
            except Exception as e:
                logger.warning(f"JDBC connection failed: {e}, falling back to ODBC")
                self.connection_type = "odbc"
                return self.connect()

        elif self.connection_type == "dbapi":
            try:
                from common.iris_dbapi_connector import get_iris_dbapi_connection

                self._connection = get_iris_dbapi_connection()
                if not self._connection:  # If get_iris_dbapi_connection returns None
                    raise ConnectionError("DBAPI connection returned None")
                logger.info("Established DBAPI connection")
            except Exception as e:
                logger.warning(f"DBAPI connection failed: {e}, falling back to ODBC")
                self.connection_type = "odbc"  # Fallback
                # Clear any partial connection state before retrying
                if self._connection and hasattr(self._connection, "close"):
                    try:
                        self._connection.close()
                    except Exception as close_err:
                        logger.error(
                            f"Error closing partial DBAPI connection during fallback: {close_err}"
                        )
                self._connection = None
                return self.connect()  # Retry with ODBC

        else:  # odbc (or fallback from dbapi/jdbc)
            from common.iris_connector import get_iris_connection

            self._connection = get_iris_connection()
            logger.info("Established ODBC connection")

        return self._connection

    def execute(self, query: str, params: Optional[List[Any]] = None) -> List[Any]:
        """
        Execute a query with optional parameters

        Args:
            query: SQL query to execute
            params: Optional list of parameters for parameterized queries

        Returns:
            List of result rows
        """
        if not self._connection:
            self.connect()

        if self.connection_type == "jdbc":
            # JDBC connector has execute method
            return self._connection.execute(query, params or [])
        else:  # Covers ODBC and DBAPI
            # ODBC and DBAPI use a similar cursor-based approach
            cursor = self._connection.cursor()
            try:
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)

                # For SELECT queries, fetchall. For others (INSERT, UPDATE, DELETE),
                # fetchall might not be appropriate or available, or might return empty.
                # Standard DBAPI cursor.description is None for non-SELECT.
                if cursor.description:
                    return cursor.fetchall()
                else:
                    # For non-SELECT, perhaps return rowcount or an empty list
                    # For now, returning empty list to maintain consistency with existing behavior
                    # if fetchall() was called on a non-SELECT for ODBC.
                    return []
            finally:
                cursor.close()

    def execute_many(self, query: str, params_list: List[List[Any]]) -> None:
        """
        Execute a query multiple times with different parameters

        Args:
            query: SQL query to execute
            params_list: List of parameter lists
        """
        if not self._connection:
            self.connect()

        if self.connection_type == "jdbc":
            # Execute in a loop for JDBC
            for params in params_list:
                self._connection.execute(query, params)
        else:  # Covers ODBC and DBAPI
            # ODBC and DBAPI support executemany
            cursor = self._connection.cursor()
            try:
                cursor.executemany(query, params_list)
                self._connection.commit()  # Ensure commit for DBAPI as well
            finally:
                cursor.close()

    @contextmanager
    def cursor(self):
        """Get a cursor for direct database operations"""
        if not self._connection:
            self.connect()

        if self.connection_type == "jdbc":
            # For JDBC, we'll create a wrapper that mimics cursor behavior
            class JDBCCursorWrapper:
                def __init__(self, connection):
                    self.connection = connection
                    self._results = None

                def execute(self, query, params=None):
                    self._results = self.connection.execute(query, params or [])

                def fetchall(self):
                    return self._results or []

                def fetchone(self):
                    if self._results and len(self._results) > 0:
                        return self._results[0]
                    return None

                def close(self):
                    pass  # No-op for JDBC

            yield JDBCCursorWrapper(self._connection)
        else:  # Covers ODBC and DBAPI
            # ODBC and DBAPI native cursor
            cursor = self._connection.cursor()
            try:
                yield cursor
            finally:
                cursor.close()

    def close(self):
        """Close the database connection"""
        if self._connection:
            if self.connection_type == "jdbc":
                # JDBC connection might not have close method
                if hasattr(self._connection, "close"):
                    self._connection.close()
            else:  # Covers ODBC and DBAPI
                if hasattr(
                    self._connection, "close"
                ):  # Good practice to check for DBAPI too
                    self._connection.close()
            self._connection = None
            logger.info(f"Closed {self.connection_type.upper()} connection")

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False


# Global connection manager instance (can be configured at startup)
_global_connection_manager = None


def get_connection_manager(connection_type: Optional[str] = None) -> ConnectionManager:
    """
    Get or create a global connection manager

    Args:
        connection_type: Override connection type (default uses environment or "odbc" for stability)

    Returns:
        ConnectionManager instance
    """
    global _global_connection_manager

    if connection_type:
        # Create new manager with specified type
        return ConnectionManager(connection_type)

    if _global_connection_manager is None:
        # Create default manager based on environment
        # Default to ODBC for stability until JDBC issues are resolved
        default_type = os.environ.get("RAG_CONNECTION_TYPE", "odbc")
        _global_connection_manager = ConnectionManager(default_type)

    return _global_connection_manager


def set_global_connection_type(connection_type: str):
    """
    Set the global connection type for all pipelines

    Args:
        connection_type: Either "jdbc", "odbc", or "dbapi"
    """
    global _global_connection_manager
    _global_connection_manager = ConnectionManager(connection_type)
    logger.info(f"Set global connection type to: {connection_type}")
