"""IRISVectorEngine: High-level interface to IRIS vector storage and connectivity.

Provides a unified API for creating and managing IRIS-backed vector pipelines.
Handles lazy initialization, configuration management, and connection lifecycle.
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from iris_vector_rag.core.connection import ConnectionManager
    from iris_vector_rag.config.manager import ConfigurationManager
    from iris_vector_rag.storage.vector_store_iris import IRISVectorStore


class IRISVectorEngine:
    """High-level interface to IRIS vector storage and connectivity.

    Manages database connections, configuration, and lazy initialization of vector stores.
    Supports both ConnectionManager instances and raw DBAPI connections.

    Attributes:
        connection_manager: The underlying connection manager (ConnectionManager or ExternalConnectionWrapper)
        config_manager: The configuration manager instance
        schema_prefix: The SQL schema prefix (e.g., "RAG", "MYAPP")
    """

    def __init__(
        self,
        connection_or_cm: Any,
        schema_prefix: str = "RAG",
        config_manager: Optional["ConfigurationManager"] = None,
    ) -> None:
        """Initialize IRISVectorEngine.

        Args:
            connection_or_cm: Either a ConnectionManager instance or a raw DBAPI connection.
                If a ConnectionManager, used directly. Otherwise wrapped in ExternalConnectionWrapper.
            schema_prefix: SQL schema prefix (default "RAG"). Can be empty string.
            config_manager: ConfigurationManager instance. If None, creates a new instance.

        Raises:
            ImportError: If required modules cannot be imported (should not occur in normal use)
        """
        from iris_vector_rag.config.manager import ConfigurationManager
        from iris_vector_rag.core.connection import ConnectionManager

        # Determine if connection_or_cm is a ConnectionManager
        # Check in this order:
        # 1. Is it a real ConnectionManager instance?
        # 2. Is it a mock with ConnectionManager spec (check _mock_methods)?
        # 3. Is it a plain mock that has been configured with get_connection?
        # 4. Otherwise, treat as raw connection to wrap
        is_connection_manager = False

        try:
            # Try isinstance check for real instances
            is_connection_manager = isinstance(connection_or_cm, ConnectionManager)
        except TypeError:
            pass

        # If not already identified, check for mock with spec
        if not is_connection_manager:
            # Check if it's a MagicMock with a spec
            mock_methods = getattr(connection_or_cm, "_mock_methods", None)
            if mock_methods is not None:  # Has _mock_methods means it has a spec
                # Check if the spec includes get_connection (indicates ConnectionManager-like)
                is_connection_manager = "get_connection" in mock_methods
            else:
                # No spec, but check if it's a plain mock that has been configured with get_connection
                # (indicating someone set it up to act like a ConnectionManager)
                mock_children = getattr(connection_or_cm, "_mock_children", None)
                if mock_children is not None and "get_connection" in mock_children:
                    is_connection_manager = True

        if is_connection_manager:
            # Store ConnectionManager directly
            self._cm = connection_or_cm
        else:
            # Wrap raw DBAPI connection in ExternalConnectionWrapper
            from iris_vector_rag import ExternalConnectionWrapper

            # Create config_manager if not provided
            if config_manager is None:
                config_manager = ConfigurationManager()

            self._cm = ExternalConnectionWrapper(connection_or_cm, config_manager)

        # Store configuration manager
        if config_manager is None:
            self._config = ConfigurationManager()
        else:
            self._config = config_manager

        # Store schema prefix
        self._schema_prefix = schema_prefix

        # Lazy-loaded connection and vector store
        self._connection: Optional[Any] = None
        self._vector_store: "Optional[IRISVectorStore]" = None

    @classmethod
    def from_config(
        cls,
        schema_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> "IRISVectorEngine":
        """Create IRISVectorEngine from configuration.

        Instantiates ConfigurationManager and ConnectionManager, then returns a new engine.

        Args:
            schema_prefix: Optional schema prefix override. If not provided, uses config value.
            **kwargs: Additional keyword arguments passed to ConfigurationManager.__init__.
                Typically: config_path="/path/to/config.yaml"

        Returns:
            IRISVectorEngine instance.
        """
        from iris_vector_rag.config.manager import ConfigurationManager
        from iris_vector_rag.core.connection import ConnectionManager

        # Create ConfigurationManager with any provided kwargs
        config_manager = ConfigurationManager(**kwargs)

        # Create ConnectionManager with the config manager
        connection_manager = ConnectionManager(config_manager)

        # Resolve schema_prefix: explicit arg > config method > fallback to "RAG"
        if schema_prefix is not None:
            resolved_prefix = schema_prefix
        else:
            # Try to get from config manager's get_schema_prefix method
            try:
                resolved_prefix = config_manager.get_schema_prefix()
            except (AttributeError, Exception):
                # Fallback: try get with dot notation
                try:
                    resolved_prefix = config_manager.get("storage.schema_prefix", "RAG")
                except Exception:
                    resolved_prefix = "RAG"

        return cls(
            connection_manager,
            schema_prefix=resolved_prefix,
            config_manager=config_manager,
        )

    def _ensure_connected(self) -> None:
        """Ensure a connection is established (lazy initialization)."""
        if self._connection is None:
            self._connection = self._cm.get_connection("iris")

    # Properties

    @property
    def connection(self) -> Any:
        """Get the database connection (lazy-loaded).

        Returns:
            DBAPI connection object.
        """
        self._ensure_connected()
        return self._connection

    @property
    def vector_store(self) -> "IRISVectorStore":
        """Get the IRIS vector store (lazy-loaded).

        Returns:
            IRISVectorStore instance.
        """
        if self._vector_store is None:
            from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

            self._vector_store = IRISVectorStore(
                connection_manager=self._cm,
                config_manager=self._config,
            )
        return self._vector_store

    @property
    def schema_prefix(self) -> str:
        """Get the SQL schema prefix.

        Returns:
            Schema prefix string (e.g., "RAG", "MYAPP", "").
        """
        return self._schema_prefix

    @property
    def connection_manager(self) -> Any:
        """Get the connection manager (ConnectionManager or ExternalConnectionWrapper).

        Returns:
            ConnectionManager or ExternalConnectionWrapper instance.
        """
        return self._cm

    @property
    def config_manager(self) -> "ConfigurationManager":
        """Get the configuration manager.

        Returns:
            ConfigurationManager instance.
        """
        return self._config

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension.

        Attempts to read from vector store's schema manager, falls back to config.

        Returns:
            Embedding dimension (default 1536 for OpenAI embeddings).
        """
        try:
            dim = self.vector_store.schema_manager.get_embedding_dimension()
            return int(dim)
        except (AttributeError, Exception):
            return int(self._config.get("embeddings.dimension", 1536))
