"""
Mode-aware database connection pooling.

Manages database connections with backend mode-specific limits
using thread-safe semaphore-based pooling.

Feature: 035-make-2-modes

DEPRECATED: This module is deprecated as of Feature 051 (Simplify IRIS Connection).
Use iris_vector_rag.common.IRISConnectionPool() for production connection pooling.
See specs/051-simplify-iris-connection/quickstart.md for migration guide.
"""

import threading
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Generator

from iris_vector_rag.testing.backend_manager import BackendMode
from iris_vector_rag.testing.exceptions import ConnectionPoolTimeout


class ConnectionPool:
    """
    Thread-safe connection pool with mode-aware limits.

    Community mode: Single connection (Semaphore(1))
    Enterprise mode: Many connections (Semaphore(999))

    DEPRECATED: Use IRISConnectionPool from iris_vector_rag.common instead.
    """

    def __init__(self, mode: BackendMode):
        """
        Initialize connection pool for specified backend mode.

        Args:
            mode: Backend mode (COMMUNITY or ENTERPRISE)

        DEPRECATED: Use IRISConnectionPool for production pooling:
            from iris_vector_rag.common import IRISConnectionPool
            pool = IRISConnectionPool(max_connections=20)
        """
        warnings.warn(
            "ConnectionPool (testing module) is deprecated as of Feature 051. "
            "Use IRISConnectionPool from iris_vector_rag.common for production pooling. "
            "See specs/051-simplify-iris-connection/quickstart.md for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.mode = mode
        self._max_connections = 1 if mode == BackendMode.COMMUNITY else 999
        self._semaphore = threading.Semaphore(self._max_connections)
        self._active_connections: Dict[int, Any] = {}
        self._lock = threading.Lock()

    @property
    def max_connections(self) -> int:
        """
        Maximum concurrent connections allowed.

        Returns:
            1 for COMMUNITY, 999 for ENTERPRISE
        """
        return self._max_connections

    @property
    def active_connections(self) -> int:
        """
        Current number of active connections.

        Returns:
            Count of connections currently acquired
        """
        with self._lock:
            return len(self._active_connections)

    @property
    def available_connections(self) -> int:
        """
        Number of available connection slots.

        Returns:
            max_connections - active_connections
        """
        return self._max_connections - self.active_connections

    @contextmanager
    def acquire(self, timeout: float = 30.0) -> Generator[Any, None, None]:
        """
        Acquire connection from pool (context manager).

        Blocks if connection limit reached until timeout.

        Args:
            timeout: Maximum seconds to wait for connection slot

        Yields:
            Mock connection object (actual connection would come from iris_config)

        Raises:
            ConnectionPoolTimeout: If timeout exceeded waiting for slot

        Examples:
            >>> pool = ConnectionPool(mode=BackendMode.COMMUNITY)
            >>> with pool.acquire(timeout=5.0) as conn:
            ...     # Use connection
            ...     pass
        """
        # Acquire semaphore slot
        acquired = self._semaphore.acquire(timeout=timeout)

        if not acquired:
            raise ConnectionPoolTimeout(
                f"Connection pool timeout after {timeout}s\n"
                f"Mode: {self.mode.value} (max {self._max_connections} connections)\n"
                "Possible cause: Test parallelism exceeds connection limit"
            )

        # Create mock connection (in real implementation, would get from iris_config)
        conn = MockConnection()

        try:
            # Track active connection
            with self._lock:
                self._active_connections[id(conn)] = conn

            yield conn

        finally:
            # Release connection
            with self._lock:
                self._active_connections.pop(id(conn), None)

            # Release semaphore slot
            self._semaphore.release()


class MockConnection:
    """Mock connection for testing purposes."""

    def __init__(self):
        self._closed = False

    def close(self):
        """Close connection."""
        self._closed = True

    def cursor(self):
        """Return mock cursor."""
        return MockCursor()


class MockCursor:
    """Mock cursor for testing purposes."""

    def execute(self, query: str):
        """Execute mock query."""
        pass

    def fetchone(self):
        """Fetch one mock result."""
        return (1,)
