import logging
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

    Since 0.15 this is a thin adapter over `iris_vector_rag.session`: the session owns
    the connection, its health cadence and its connect back-off, so `get_connection`
    no longer opens or probes anything itself. Callers that only need to run SQL should
    use the session directly (`session_for(config_manager).execute(...)`).
    """

    def __init__(self, config_manager: Optional[ConfigurationManager] = None):
        """
        Initializes the ConnectionManager.
        """
        if config_manager is None:
            self.config_manager = ConfigurationManager()
        else:
            self.config_manager = config_manager

    @property
    def session(self):
        """The session for the configured identity. Acquiring it does not connect."""
        from iris_vector_rag.session import session_for

        return session_for(self.config_manager)

    def get_connection(self, backend_name: str = "iris"):
        """
        Retrieves a database connection for the specified backend.

        Uses ConfigurationManager as the single authority for connection params
        (AUD-005), resolved into the session's identity.
        """
        if backend_name != "iris":
            raise ValueError(f"Unsupported database backend: {backend_name}")

        return self.session.connection_for_adapter()

    def create_connection(self, backend_name: str = "iris"):
        """Create a database connection (alias for get_connection)."""
        return self.get_connection(backend_name)

    def invalidate_connection(self, backend_name: str = "iris"):
        """Discard this thread's connection so the next call opens a fresh one."""
        if backend_name != "iris":
            return
        self.session.close()

    def close_connection(self, backend_name: str):
        """Closes a specific database connection."""
        self.invalidate_connection(backend_name)

    def close_all_connections(self):
        """Closes every connection the session opened, in any thread."""
        self.session.close_all()
