import os
import importlib
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Attempt to import ConfigurationManager, will be created later
try:
    from iris_rag.config.manager import ConfigurationManager
except ImportError:
    # Placeholder if ConfigurationManager doesn't exist yet
    # This allows ConnectionManager to be defined, though tests requiring
    # actual config loading will fail until ConfigurationManager is implemented.
    class ConfigurationManager:
        def __init__(self, config_path=None):
            # This is a placeholder, real implementation will load from file/env
            pass
        def get(self, section_key):
            # Placeholder: always return None. Tests should mock this.
            return None

class ConnectionManager:
    """
    Manages database connections for different backends.

    This class is responsible for establishing and providing database
    connections based on configurations. It supports multiple database
    backends, with an initial focus on InterSystems IRIS.
    """

    def __init__(self, config_manager: ConfigurationManager = None):
        """
        Initializes the ConnectionManager.

        Args:
            config_manager: An instance of ConfigurationManager to load
                            database connection settings. If None, a default
                            ConfigurationManager will be instantiated.
        """
        self._connections = {} # Initialize as instance variable
        if config_manager is None:
            # This will eventually load from a default path or environment
            self.config_manager = ConfigurationManager()
        else:
            self.config_manager = config_manager

    def get_connection(self, backend_name: str = "iris"):
        """
        Retrieves or creates a database connection for the specified backend.

        Connections are cached to avoid redundant establishments.

        Args:
            backend_name: The name of the database backend (e.g., 'iris').
                         Defaults to 'iris' for backward compatibility.

        Returns:
            A database connection object.

        Raises:
            ValueError: If the backend is unsupported or configuration is missing.
            ImportError: If the required database driver cannot be imported.
        """
        if backend_name in self._connections:
            return self._connections[backend_name]

        # Check for supported backend types
        if backend_name != "iris":
            # This can be expanded if more backends are officially supported
            raise ValueError(f"Unsupported database backend: {backend_name}")

        # For IRIS backend, check configuration to determine connection type
        try:
            # Get storage configuration to determine connection type
            storage_config = self.config_manager.get("storage:backends:iris")
            connection_type = storage_config.get("connection_type", "dbapi") if storage_config else "dbapi"
            
            logger.info(f"Establishing connection for backend '{backend_name}' using {connection_type.upper()}")
            
            if connection_type == "dbapi":
                # Use native IRIS DBAPI (preferred)
                connection = self._create_dbapi_connection()
            else:
                # Fallback to JDBC
                from common.iris_connector import get_iris_connection
                connection = get_iris_connection()
            
            self._connections[backend_name] = connection
            return connection
        except Exception as e:
            # Catching a broad exception here as connection creation can raise various errors
            raise ConnectionError(f"Failed to connect to IRIS backend '{backend_name}': {e}")
    
    def _create_dbapi_connection(self):
        """Create a native IRIS DBAPI connection."""
        try:
            import intersystems_iris.dbapi._DBAPI as dbapi
            
            # Get database configuration
            db_config = self.config_manager.get("database")
            if not db_config:
                # Fallback to environment variables
                db_config = {
                    "db_host": os.getenv("IRIS_HOST", "localhost"),
                    "db_port": int(os.getenv("IRIS_PORT", "1972")),
                    "db_namespace": os.getenv("IRIS_NAMESPACE", "USER"),
                    "db_user": os.getenv("IRIS_USERNAME", "_SYSTEM"),
                    "db_password": os.getenv("IRIS_PASSWORD", "SYS")
                }
            
            # Create DBAPI connection
            connection = dbapi.connect(
                hostname=db_config.get("db_host", "localhost"),
                port=db_config.get("db_port", 1972),
                namespace=db_config.get("db_namespace", "USER"),
                username=db_config.get("db_user", "_SYSTEM"),
                password=db_config.get("db_password", "SYS")
            )
            
            logger.info("✅ Successfully connected to IRIS using native DBAPI")
            return connection
            
        except ImportError as e:
            logger.error(f"Native IRIS DBAPI not available: {e}")
            raise ImportError("intersystems_iris package not installed. Install with: pip install intersystems-iris")
        except Exception as e:
            logger.error(f"Failed to create DBAPI connection: {e}")
            raise ConnectionError(f"DBAPI connection failed: {e}")

    def close_connection(self, backend_name: str):
        """Closes a specific database connection."""
        if backend_name in self._connections:
            connection = self._connections.pop(backend_name)
            try:
                connection.close()
            except Exception as e:
                # Log error or handle as appropriate
                logger.error(f"Error closing connection for {backend_name}: {e}")

    def close_all_connections(self):
        """Closes all active database connections."""
        for backend_name in list(self._connections.keys()):
            self.close_connection(backend_name)