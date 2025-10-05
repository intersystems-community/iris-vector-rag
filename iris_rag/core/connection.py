import logging
import os
import time

logger = logging.getLogger(__name__)

# Attempt to import ConfigurationManager, will be created later
try:
    from iris_rag.config.manager import ConfigurationManager
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
        self._connections = {}  # Initialize as instance variable
        self._connection_times = {}  # Track connection creation time
        if config_manager is None:
            # This will eventually load from a default path or environment
            self.config_manager = ConfigurationManager()
        else:
            self.config_manager = config_manager

    def _is_connection_healthy(self, connection) -> bool:
        """
        Check if a connection is still healthy and responsive.

        Args:
            connection: The database connection to check

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except Exception as e:
            logger.warning(f"Connection health check failed: {e}")
            return False

    def get_connection(self, backend_name: str = "iris"):
        """
        Retrieves or creates a database connection for the specified backend.

        Connections are cached but health-checked before reuse. Stale connections
        are automatically refreshed to prevent timeouts during long-running operations.

        Args:
            backend_name: The name of the database backend (e.g., 'iris').
                         Defaults to 'iris' for backward compatibility.

        Returns:
            A database connection object.

        Raises:
            ValueError: If the backend is unsupported or configuration is missing.
            ImportError: If the required database driver cannot be imported.
        """
        # Check if we have a cached connection
        if backend_name in self._connections:
            connection = self._connections[backend_name]
            connection_age = time.time() - self._connection_times.get(backend_name, 0)

            # Refresh connection if older than 20 minutes or unhealthy
            if connection_age > 1200 or not self._is_connection_healthy(connection):
                logger.info(f"Refreshing stale connection for {backend_name} (age: {connection_age:.0f}s)")
                try:
                    connection.close()
                except:
                    pass
                del self._connections[backend_name]
                del self._connection_times[backend_name]
            else:
                return self._connections[backend_name]

        # Get database configuration
        config_key = f"database:{backend_name}"
        db_config = self.config_manager.get(config_key)

        if not db_config:
            raise ValueError(f"Configuration for backend '{backend_name}' not found.")

        # Check for supported backend types
        if backend_name != "iris":
            # This can be expanded if more backends are officially supported
            raise ValueError(f"Unsupported database backend: {backend_name}")

        # For IRIS backend, use the proven database utility
        try:
            logger.info(
                f"Establishing connection for backend '{backend_name}' using DBAPI"
            )

            # Use the existing database utility instead of direct DBAPI imports
            from common.iris_dbapi_connector import get_iris_dbapi_connection

            # Create connection using the proven utility function
            connection = get_iris_dbapi_connection()

            if connection is None:
                raise ConnectionError("IRIS connection utility returned None")

            self._connections[backend_name] = connection
            self._connection_times[backend_name] = time.time()
            return connection
        except ImportError as e:
            logger.error(f"Failed to import database utility: {e}")
            raise ImportError(f"Database utility not available: {e}")
        except Exception as e:
            # Catching a broad exception here as connection creation can raise various errors
            raise ConnectionError(
                f"Failed to connect to IRIS backend '{backend_name}': {e}"
            )

    def _create_dbapi_connection(self):
        """Create a native IRIS DBAPI connection."""
        try:
            # Import the correct IRIS DBAPI module that has connect()
            pass

            # Get database configuration
            db_config = self.config_manager.get("database")
            if not db_config:
                # Fallback to environment variables
                db_config = {
                    "db_host": os.getenv("IRIS_HOST", "localhost"),
                    "db_port": int(os.getenv("IRIS_PORT", "1974")),
                    "db_namespace": os.getenv("IRIS_NAMESPACE", "USER"),
                    "db_user": os.getenv("IRIS_USERNAME", "_SYSTEM"),
                    "db_password": os.getenv("IRIS_PASSWORD", "SYS"),
                }

            # Use our utility connector instead of direct iris.connect
            from common.iris_connection_manager import get_iris_connection

            connection_config = {
                "hostname": db_config.get("db_host", "localhost"),
                "port": db_config.get("db_port", 1974),
                "namespace": db_config.get("db_namespace", "USER"),
                "username": db_config.get("db_user", "_SYSTEM"),
                "password": db_config.get("db_password", "SYS"),
            }
            connection = get_iris_connection(connection_config)

            logger.info("✅ Successfully connected to IRIS using native DBAPI")
            return connection

        except ImportError as e:
            logger.error(f"Native IRIS DBAPI not available: {e}")
            raise ImportError(
                "iris package not installed. Install with: pip install intersystems-irispython"
            )
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
