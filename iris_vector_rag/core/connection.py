import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Attempt to import ConfigurationManager, will be created later
try:
    from iris_vector_rag.config.manager import ConfigurationManager
except ImportError:
    logger.error(
        "ConfigurationManager not found. Ensure iris_rag package is installed correctly."
    )
    raise ImportError(
        "ConfigurationManager not available. Please check your installation."
    )


class ConnectionManager:
    """
    Manages database connections for different backends.
    """

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initializes the ConnectionManager.
        """
        self._connections = {}  # Initialize as instance variable
        self._connection_times = {}  # Track connection creation time
        self._connection_use_counts = {}  # Track number of times connection used
        if config_manager is None:
            self.config_manager = ConfigurationManager()
        else:
            self.config_manager = config_manager

    def _is_connection_healthy(self, connection) -> bool:
        """Check if a connection is still healthy."""
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except Exception:
            return False

    def invalidate_connection(self, backend_name: str = "iris"):
        """Invalidate a cached connection."""
        if backend_name in self._connections:
            try:
                self._connections[backend_name].close()
            except:
                pass
            del self._connections[backend_name]

    def get_connection(self, backend_name: str = "iris"):
        """
        Retrieves or creates a database connection for the specified backend.
        """
        if backend_name != "iris":
            raise ValueError(f"Unsupported database backend: {backend_name}")

        from iris_vector_rag.common.iris_connection import get_iris_connection
        return get_iris_connection()

    def _create_dbapi_connection(self):
        """Create a native IRIS DBAPI connection."""
        from iris_vector_rag.common.iris_connection import get_iris_connection
        return get_iris_connection()

    def close_connection(self, backend_name: str):
        """Closes a specific database connection."""
        if backend_name in self._connections:
            connection = self._connections.pop(backend_name)
            try:
                connection.close()
            except Exception as e:
                logger.error(f"Error closing connection for {backend_name}: {e}")

    def close_all_connections(self):
        """Closes all active database connections."""
        for backend_name in list(self._connections.keys()):
            self.close_connection(backend_name)
