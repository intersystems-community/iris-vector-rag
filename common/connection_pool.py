"""
IRIS Database Connection Pool Manager.

Provides connection pooling for concurrent REST API requests to prevent
connection pool exhaustion and ensure efficient resource management.

Based on research.md findings:
- 20 base connections, 10 overflow (adaptive under load)
- 1-hour connection recycle to prevent stale connections
- Pre-ping validation before each use
- Thread-safe pool management
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

import intersystems_iris.dbapi._DBAPI as iris_dbapi
from intersystems_iris.dbapi._DBAPI import Connection

logger = logging.getLogger(__name__)


class IRISConnectionPool:
    """
    Thread-safe connection pool for IRIS database.

    This class implements connection pooling to support concurrent RAG API requests
    without exhausting the IRIS connection limit.

    Features:
    - Configurable pool size with overflow capacity
    - Connection recycling to prevent stale connections
    - Pre-ping validation for connection health
    - Thread-safe acquire/release operations
    """

    def __init__(
        self,
        host: str,
        port: int,
        namespace: str,
        username: str,
        password: str,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
        pool_timeout: int = 30,
    ):
        """
        Initialize IRIS connection pool.

        Args:
            host: IRIS server hostname
            port: IRIS server port
            namespace: IRIS namespace
            username: Database username
            password: Database password
            pool_size: Number of base connections (default: 20)
            max_overflow: Additional connections under load (default: 10)
            pool_recycle: Seconds before recycling connection (default: 3600)
            pool_pre_ping: Validate connection before use (default: True)
            pool_timeout: Seconds to wait for connection (default: 30)
        """
        self.host = host
        self.port = port
        self.namespace = namespace
        self.username = username
        self.password = password
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping
        self.pool_timeout = pool_timeout

        # Connection tracking
        self._available_connections: list[Connection] = []
        self._in_use_connections: set[Connection] = set()
        self._connection_timestamps: Dict[Connection, float] = {}

        # Thread safety
        import threading

        self._lock = threading.Lock()

        # Metrics
        self._total_connections_created = 0
        self._total_connections_recycled = 0
        self._total_acquisitions = 0

        logger.info(
            f"Initialized IRIS connection pool: "
            f"pool_size={pool_size}, max_overflow={max_overflow}, "
            f"recycle={pool_recycle}s, pre_ping={pool_pre_ping}"
        )

    def _create_connection(self) -> Connection:
        """
        Create a new IRIS database connection.

        Returns:
            IRIS connection object

        Raises:
            Exception: If connection creation fails
        """
        try:
            connection = iris_dbapi.connect(
                hostname=self.host,
                port=self.port,
                namespace=self.namespace,
                username=self.username,
                password=self.password,
            )

            import time

            self._connection_timestamps[connection] = time.time()
            self._total_connections_created += 1

            logger.debug(
                f"Created new IRIS connection "
                f"(total created: {self._total_connections_created})"
            )

            return connection

        except Exception as e:
            logger.error(f"Failed to create IRIS connection: {e}")
            raise

    def _validate_connection(self, connection: Connection) -> bool:
        """
        Validate that a connection is still alive.

        Args:
            connection: Connection to validate

        Returns:
            True if connection is valid, False otherwise
        """
        if not self.pool_pre_ping:
            return True

        try:
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True

        except Exception as e:
            logger.warning(f"Connection validation failed: {e}")
            return False

    def _should_recycle_connection(self, connection: Connection) -> bool:
        """
        Check if connection should be recycled based on age.

        Args:
            connection: Connection to check

        Returns:
            True if connection should be recycled, False otherwise
        """
        if connection not in self._connection_timestamps:
            return False

        import time

        age = time.time() - self._connection_timestamps[connection]
        return age >= self.pool_recycle

    def _recycle_connection(self, connection: Connection) -> None:
        """
        Close and remove a connection from the pool.

        Args:
            connection: Connection to recycle
        """
        try:
            connection.close()
            self._total_connections_recycled += 1

            if connection in self._connection_timestamps:
                del self._connection_timestamps[connection]

            logger.debug(
                f"Recycled IRIS connection "
                f"(total recycled: {self._total_connections_recycled})"
            )

        except Exception as e:
            logger.warning(f"Error recycling connection: {e}")

    def acquire(self) -> Connection:
        """
        Acquire a connection from the pool.

        This method will:
        1. Try to get an available connection from the pool
        2. Validate the connection if pre_ping is enabled
        3. Recycle connections that are too old
        4. Create new connections if pool is not full

        Returns:
            Active IRIS connection

        Raises:
            Exception: If unable to acquire connection within timeout
        """
        import time

        start_time = time.time()

        while (time.time() - start_time) < self.pool_timeout:
            with self._lock:
                # Try to get existing connection
                if self._available_connections:
                    connection = self._available_connections.pop()

                    # Check if connection should be recycled
                    if self._should_recycle_connection(connection):
                        self._recycle_connection(connection)
                        continue

                    # Validate connection health
                    if not self._validate_connection(connection):
                        self._recycle_connection(connection)
                        continue

                    # Connection is good - mark as in use
                    self._in_use_connections.add(connection)
                    self._total_acquisitions += 1

                    logger.debug(
                        f"Acquired connection from pool "
                        f"(available: {len(self._available_connections)}, "
                        f"in_use: {len(self._in_use_connections)})"
                    )

                    return connection

                # No available connections - check if we can create new one
                total_connections = (
                    len(self._available_connections) + len(self._in_use_connections)
                )

                if total_connections < (self.pool_size + self.max_overflow):
                    # Create new connection
                    connection = self._create_connection()
                    self._in_use_connections.add(connection)
                    self._total_acquisitions += 1

                    logger.debug(
                        f"Created new connection "
                        f"(total: {total_connections + 1}/{self.pool_size + self.max_overflow})"
                    )

                    return connection

            # Pool is full - wait briefly and retry
            time.sleep(0.1)

        # Timeout exceeded
        raise Exception(
            f"Timeout acquiring connection from pool after {self.pool_timeout}s "
            f"(pool_size: {self.pool_size}, max_overflow: {self.max_overflow}, "
            f"in_use: {len(self._in_use_connections)})"
        )

    def release(self, connection: Connection) -> None:
        """
        Release a connection back to the pool.

        Args:
            connection: Connection to release
        """
        with self._lock:
            if connection in self._in_use_connections:
                self._in_use_connections.remove(connection)

                # Check if we should recycle instead of returning to pool
                if self._should_recycle_connection(connection):
                    self._recycle_connection(connection)
                else:
                    self._available_connections.append(connection)

                logger.debug(
                    f"Released connection to pool "
                    f"(available: {len(self._available_connections)}, "
                    f"in_use: {len(self._in_use_connections)})"
                )

    @contextmanager
    def get_connection(self) -> Generator[Connection, None, None]:
        """
        Context manager for acquiring and releasing connections.

        Usage:
            >>> pool = IRISConnectionPool(...)
            >>> with pool.get_connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM table")

        Yields:
            IRIS connection
        """
        connection = self.acquire()
        try:
            yield connection
        finally:
            self.release(connection)

    def close_all(self) -> None:
        """
        Close all connections in the pool.

        This should be called on application shutdown.
        """
        with self._lock:
            # Close available connections
            for connection in self._available_connections:
                self._recycle_connection(connection)
            self._available_connections.clear()

            # Close in-use connections (should be empty at shutdown)
            for connection in self._in_use_connections:
                self._recycle_connection(connection)
            self._in_use_connections.clear()

            logger.info(
                f"Closed all connections in pool "
                f"(total created: {self._total_connections_created}, "
                f"recycled: {self._total_connections_recycled}, "
                f"acquisitions: {self._total_acquisitions})"
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.

        Returns:
            Dictionary with pool metrics
        """
        with self._lock:
            return {
                "pool_size": self.pool_size,
                "max_overflow": self.max_overflow,
                "available_connections": len(self._available_connections),
                "in_use_connections": len(self._in_use_connections),
                "total_connections": len(self._available_connections)
                + len(self._in_use_connections),
                "total_created": self._total_connections_created,
                "total_recycled": self._total_connections_recycled,
                "total_acquisitions": self._total_acquisitions,
            }


# Global connection pool instance (initialized by FastAPI app)
_global_pool: Optional[IRISConnectionPool] = None


def get_pool() -> IRISConnectionPool:
    """
    Get the global connection pool instance.

    Returns:
        Global IRISConnectionPool instance

    Raises:
        RuntimeError: If pool has not been initialized
    """
    if _global_pool is None:
        raise RuntimeError(
            "Connection pool not initialized. "
            "Call initialize_pool() during application startup."
        )
    return _global_pool


def initialize_pool(
    host: Optional[str] = None,
    port: Optional[int] = None,
    namespace: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    **kwargs: Any,
) -> IRISConnectionPool:
    """
    Initialize the global connection pool.

    This should be called during FastAPI application startup.

    Args:
        host: IRIS server hostname (default: from IRIS_HOST env)
        port: IRIS server port (default: from IRIS_PORT env)
        namespace: IRIS namespace (default: from IRIS_NAMESPACE env)
        username: Database username (default: from IRIS_USERNAME env)
        password: Database password (default: from IRIS_PASSWORD env)
        **kwargs: Additional IRISConnectionPool arguments

    Returns:
        Initialized global connection pool
    """
    global _global_pool

    # Get connection parameters from environment if not provided
    host = host or os.getenv("IRIS_HOST", "localhost")
    port = port or int(os.getenv("IRIS_PORT", "1972"))
    namespace = namespace or os.getenv("IRIS_NAMESPACE", "USER")
    username = username or os.getenv("IRIS_USERNAME", "_SYSTEM")
    password = password or os.getenv("IRIS_PASSWORD", "SYS")

    _global_pool = IRISConnectionPool(
        host=host,
        port=port,
        namespace=namespace,
        username=username,
        password=password,
        **kwargs,
    )

    logger.info(
        f"Initialized global connection pool: {host}:{port}/{namespace} "
        f"(pool_size={_global_pool.pool_size}, "
        f"max_overflow={_global_pool.max_overflow})"
    )

    return _global_pool


def close_pool() -> None:
    """
    Close the global connection pool.

    This should be called during FastAPI application shutdown.
    """
    global _global_pool

    if _global_pool is not None:
        _global_pool.close_all()
        _global_pool = None
        logger.info("Closed global connection pool")
