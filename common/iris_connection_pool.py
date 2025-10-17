"""
IRIS Connection Pool for high-performance production workloads.

Implements connection pooling to eliminate the connection overhead observed
in production (60s per 100-ticket batch from connection churn).

Features:
- Thread-safe connection pool with min/max sizing
- Automatic connection validation and refresh
- Graceful degradation on pool exhaustion
- Connection health checks
- Pool statistics for monitoring
"""

import logging
import os
import threading
import time
from queue import Queue, Empty, Full
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class IRISConnectionPool:
    """
    Thread-safe connection pool for InterSystems IRIS.

    Maintains a pool of reusable database connections to eliminate
    connection overhead in high-throughput scenarios.

    Example:
        pool = IRISConnectionPool(min_size=5, max_size=20)

        with pool.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM RAG.Entities LIMIT 10")
            results = cursor.fetchall()
            cursor.close()
    """

    def __init__(
        self,
        min_size: int = 5,
        max_size: int = 20,
        connection_timeout: float = 30.0,
        validation_interval: int = 60,
        host: str = None,
        port: int = None,
        namespace: str = None,
        username: str = None,
        password: str = None
    ):
        """
        Initialize connection pool.

        Args:
            min_size: Minimum connections to maintain
            max_size: Maximum connections allowed
            connection_timeout: Timeout in seconds when acquiring connection
            validation_interval: Seconds between connection health checks
            host: IRIS host (defaults to IRIS_HOST env var)
            port: IRIS port (defaults to IRIS_PORT env var)
            namespace: IRIS namespace (defaults to IRIS_NAMESPACE env var)
            username: IRIS username (defaults to IRIS_USERNAME env var)
            password: IRIS password (defaults to IRIS_PASSWORD env var)
        """
        self.min_size = min_size
        self.max_size = max_size
        self.connection_timeout = connection_timeout
        self.validation_interval = validation_interval

        # Connection parameters
        self.host = host or os.environ.get("IRIS_HOST", "localhost")
        self.port = port or int(os.environ.get("IRIS_PORT", 1972))
        self.namespace = namespace or os.environ.get("IRIS_NAMESPACE", "USER")
        self.username = username or os.environ.get("IRIS_USERNAME", "_SYSTEM")
        self.password = password or os.environ.get("IRIS_PASSWORD", "SYS")

        # Pool state
        self._pool = Queue(maxsize=max_size)
        self._active_connections = 0
        self._lock = threading.Lock()
        self._closed = False

        # Statistics
        self._stats = {
            "created": 0,
            "destroyed": 0,
            "hits": 0,
            "misses": 0,
            "timeouts": 0,
            "validation_failures": 0
        }

        # Initialize pool with minimum connections
        self._initialize_pool()

        logger.info(
            f"IRIS Connection Pool initialized: min={min_size}, max={max_size}, "
            f"host={self.host}:{self.port}/{self.namespace}"
        )

    def _initialize_pool(self):
        """Create minimum number of connections on startup."""
        for i in range(self.min_size):
            try:
                conn = self._create_connection()
                if conn:
                    self._pool.put(conn, block=False)
                    self._stats["created"] += 1
                    logger.debug(f"Created initial connection {i+1}/{self.min_size}")
            except Full:
                logger.warning(f"Pool full during initialization at {i+1} connections")
                break
            except Exception as e:
                logger.error(f"Failed to create initial connection {i+1}: {e}")

    def _create_connection(self):
        """
        Create a new IRIS database connection.

        Returns:
            Connection object or None if creation fails
        """
        try:
            # Import here to avoid circular dependencies
            from .iris_dbapi_connector import get_iris_dbapi_connection

            conn = get_iris_dbapi_connection()

            if conn is None:
                logger.error("Failed to create IRIS connection")
                return None

            with self._lock:
                self._active_connections += 1

            logger.debug(
                f"Created new connection (active: {self._active_connections}/{self.max_size})"
            )
            return conn

        except Exception as e:
            logger.error(f"Exception creating IRIS connection: {e}")
            return None

    def _validate_connection(self, conn) -> bool:
        """
        Check if connection is still valid.

        Args:
            conn: Connection to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            return result is not None
        except Exception as e:
            logger.debug(f"Connection validation failed: {e}")
            self._stats["validation_failures"] += 1
            return False

    def _destroy_connection(self, conn):
        """
        Close and destroy a connection.

        Args:
            conn: Connection to destroy
        """
        try:
            conn.close()
            with self._lock:
                self._active_connections -= 1
            self._stats["destroyed"] += 1
            logger.debug(f"Destroyed connection (active: {self._active_connections})")
        except Exception as e:
            logger.warning(f"Error destroying connection: {e}")

    @contextmanager
    def get_connection(self, timeout: float = None):
        """
        Get a connection from the pool (context manager).

        Usage:
            with pool.get_connection() as conn:
                # Use connection
                pass

        Args:
            timeout: Override default connection timeout

        Yields:
            Database connection

        Raises:
            TimeoutError: If connection not available within timeout
            RuntimeError: If pool is closed
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")

        timeout = timeout or self.connection_timeout
        conn = None
        acquired_from_pool = False

        try:
            # Try to get connection from pool
            try:
                conn = self._pool.get(block=True, timeout=timeout)
                acquired_from_pool = True
                self._stats["hits"] += 1
                logger.debug("Acquired connection from pool")

                # Validate connection
                if not self._validate_connection(conn):
                    logger.warning("Pooled connection invalid, creating new one")
                    self._destroy_connection(conn)
                    conn = self._create_connection()
                    acquired_from_pool = False
                    self._stats["misses"] += 1

            except Empty:
                # Pool empty, try to create new connection if under limit
                with self._lock:
                    if self._active_connections < self.max_size:
                        conn = self._create_connection()
                        self._stats["misses"] += 1
                        logger.debug("Created new connection (pool empty)")
                    else:
                        self._stats["timeouts"] += 1
                        raise TimeoutError(
                            f"Connection pool exhausted (max={self.max_size}) "
                            f"and timeout ({timeout}s) reached"
                        )

            if conn is None:
                raise RuntimeError("Failed to acquire database connection")

            yield conn

        finally:
            # Return connection to pool if it's still valid
            if conn is not None:
                if acquired_from_pool or self._pool.qsize() < self.max_size:
                    try:
                        # Re-validate before returning to pool
                        if self._validate_connection(conn):
                            self._pool.put(conn, block=False)
                            logger.debug("Returned connection to pool")
                        else:
                            logger.warning("Connection invalid, destroying instead of returning to pool")
                            self._destroy_connection(conn)
                    except Full:
                        # Pool full, destroy the connection
                        logger.debug("Pool full, destroying connection")
                        self._destroy_connection(conn)
                else:
                    # Destroy excess connections
                    self._destroy_connection(conn)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        return {
            **self._stats,
            "pool_size": self._pool.qsize(),
            "active_connections": self._active_connections,
            "min_size": self.min_size,
            "max_size": self.max_size
        }

    def close(self):
        """
        Close all connections and shut down the pool.
        """
        if self._closed:
            return

        self._closed = True
        logger.info("Closing connection pool...")

        # Close all pooled connections
        while not self._pool.empty():
            try:
                conn = self._pool.get(block=False)
                self._destroy_connection(conn)
            except Empty:
                break

        logger.info(
            f"Connection pool closed. Stats: {self.get_statistics()}"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


# Global singleton pool (lazy-initialized)
_global_pool: Optional[IRISConnectionPool] = None
_pool_lock = threading.Lock()


def get_global_pool(
    min_size: int = 10,
    max_size: int = 50,
    **kwargs
) -> IRISConnectionPool:
    """
    Get or create the global connection pool singleton.

    Args:
        min_size: Minimum pool size (only used on first creation)
        max_size: Maximum pool size (only used on first creation)
        **kwargs: Additional arguments passed to IRISConnectionPool

    Returns:
        Global IRISConnectionPool instance
    """
    global _global_pool

    if _global_pool is None:
        with _pool_lock:
            if _global_pool is None:
                _global_pool = IRISConnectionPool(
                    min_size=min_size,
                    max_size=max_size,
                    **kwargs
                )
                logger.info("Created global IRIS connection pool")

    return _global_pool


def close_global_pool():
    """Close the global connection pool if it exists."""
    global _global_pool

    if _global_pool is not None:
        with _pool_lock:
            if _global_pool is not None:
                _global_pool.close()
                _global_pool = None
                logger.info("Closed global IRIS connection pool")
