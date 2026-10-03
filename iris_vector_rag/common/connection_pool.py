"""
Connection pool for the HTTP API, now a thin layer over one session.

The pool this replaces kept its own free list and called `iris.dbapi.connect` itself,
so none of the connections it handed out had the socket precheck, the bounded
handshake, and `pool_pre_ping` meant a `SELECT 1`
before *every* request. The session already gives each thread its own connection,
probes it on a cadence rather than per use, and replaces it when it dies -- which is
what the free list, the pre-ping and the recycle timer were each approximating.

What is left here is the API's own concern: a cap on how many requests may hold a
connection at once, and the statistics the health route reports.

Deprecated in 0.15, removed in 0.16: use `session_for(...)` directly. The API keeps
using it until then (FR-017).

Feature: 083-iris-session (FR-017, FR-018)
"""

import logging
import threading
import warnings
from contextlib import contextmanager
from typing import Any, Dict, Generator

from iris_vector_rag.session import SessionIdentity, session_for

logger = logging.getLogger(__name__)


class IRISConnectionPool:
    """Bounded concurrent access to one session's per-thread connections."""

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
        Args:
            host, port, namespace, username, password: the instance to reach.
            pool_size: base number of concurrent connection holders.
            max_overflow: additional holders permitted under load.
            pool_recycle: seconds a connection is trusted; becomes the session's
                health interval, which is checked on a cadence rather than per use.
            pool_pre_ping: kept for compatibility. The session's health deadline
                replaces the per-request probe this used to mean.
            pool_timeout: seconds `acquire()` waits for a free slot.
        """
        warnings.warn(
            "IRISConnectionPool is deprecated and is removed in 0.16. Use "
            "iris_vector_rag.session.session_for(...), which already gives each "
            "thread its own health-checked connection.",
            DeprecationWarning,
            stacklevel=2,
        )
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

        self._identity = SessionIdentity(
            host=host,
            port=int(port),
            namespace=namespace,
            username=username,
            password=password,
        )
        # One slot per permitted concurrent holder. The connections themselves are
        # the session's, one per thread; this only bounds how many are in flight.
        self._slots = threading.BoundedSemaphore(pool_size + max_overflow)
        self._lock = threading.Lock()
        self._in_use = 0
        self._total_acquisitions = 0

        logger.info(
            "IRIS connection pool over a session for %s: capacity=%d, health=%ds",
            self._identity.label,
            pool_size + max_overflow,
            pool_recycle,
        )

    @property
    def session(self):
        """The session every connection in this pool comes from."""
        return session_for(
            identity=self._identity, health_interval=float(self.pool_recycle)
        )

    def acquire(self) -> Any:
        """A connection for this thread, once a slot is free."""
        if not self._slots.acquire(timeout=self.pool_timeout):
            raise TimeoutError(
                f"Timeout acquiring connection from pool after {self.pool_timeout}s "
                f"(pool_size: {self.pool_size}, max_overflow: {self.max_overflow}, "
                f"in_use: {self._in_use})"
            )
        try:
            connection = self.session.connection_for_adapter()
        except BaseException:
            self._slots.release()
            raise
        with self._lock:
            self._in_use += 1
            self._total_acquisitions += 1
        return connection

    def release(self, connection: Any) -> None:
        """Give the slot back, and the connection too if the pool is over capacity.

        A connection normally stays with its thread -- that is what makes the second
        request on a worker thread free. But a server that starts a thread per
        request would then open one connection per request, and a licence counts
        connections: Community Edition refuses the sixth. So once the session holds
        more than this pool advertises, the finishing request gives its connection
        back rather than leaving it parked on a thread that may never return.
        """
        with self._lock:
            if self._in_use == 0:
                return
            self._in_use -= 1
        try:
            self._slots.release()
        except ValueError:  # pragma: no cover - released more often than acquired
            pass
        session = self.session
        if session.open_connection_count > self.pool_size + self.max_overflow:
            session.close()

    @contextmanager
    def get_connection(self) -> Generator[Any, None, None]:
        """
        Usage:
            >>> with pool.get_connection() as conn:
            ...     cursor = conn.cursor()
        """
        connection = self.acquire()
        try:
            yield connection
        finally:
            self.release(connection)

    def close_all(self) -> None:
        """Close every connection the session opened. For application shutdown."""
        self.session.close_all()
        with self._lock:
            self._in_use = 0
        logger.info(
            "Closed all connections for %s (acquisitions: %d)",
            self._identity.label,
            self._total_acquisitions,
        )

    def get_stats(self) -> Dict[str, Any]:
        """Pool metrics, as the health route reports them."""
        with self._lock:
            in_use = self._in_use
            acquisitions = self._total_acquisitions
        return {
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "in_use_connections": in_use,
            "open_connections": self.session.open_connection_count,
            "capacity": self.pool_size + self.max_overflow,
            "total_acquisitions": acquisitions,
            "driver": self.session.driver_name,
            # Not `session.is_connected`: that is the asking thread's own state, and
            # the thread that reads the health route is never one that served a
            # request. For a pool, connected means some thread holds a connection
            # right now -- which is false for an idle pool, because a request thread
            # gives its connection back when it ends.
            "connected": self.session.open_connection_count > 0,
            "identity": self._identity.label,
        }

    def __enter__(self) -> "IRISConnectionPool":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close_all()
        return False


def _session_pool(**parameters: Any) -> IRISConnectionPool:
    """The same pool, without the warning, for the HTTP API that still needs it.

    The deprecation is aimed at callers who can move to `session_for()`. The API
    adapter cannot until 0.16 (FR-017), and a warning it raises against itself is
    one no user can act on. Called once at application startup, so borrowing the
    global filter for the length of the constructor costs nothing.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return IRISConnectionPool(**parameters)
