"""One session owns one instance-and-namespace relationship.

Identity, driver choice, connection lifetime, health cadence and connect back-off
used to be spread across eleven entry points and three module globals. They are one
object here, because they are one fact: whether this thread can talk to IRIS now.

Feature: 083-iris-session
"""

from __future__ import annotations

import logging
import os
import threading
import weakref
from time import monotonic
from typing import Any, Callable, List, Optional, Sequence

from iris_vector_rag.session.errors import (
    CallerConnectionBroken,
    ConnectionLost,
    InBackoff,
    PasswordExpired,
    SessionError,
    looks_like_lost_connection,
    wrap_connect_failure,
)
from iris_vector_rag.session.identity import SessionIdentity
from iris_vector_rag.session.results import Rows, read_cursor

logger = logging.getLogger(__name__)

DEFAULT_HEALTH_INTERVAL = 30.0


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class _ThreadMarker:
    """Nothing but a lifetime: it lives exactly as long as its thread's state.

    A request thread that finishes without closing anything cannot use its connection
    again, but a licence still counts it. Storing this object in the thread's own
    state means Python drops it when the thread ends, which is the signal to close
    what that thread was holding.
    """

    __slots__ = ("__weakref__",)


class _BorrowedDriver:
    """Stands in for a driver when the caller brought their own connection."""

    name = "borrowed"

    def available(self) -> bool:
        return True

    def connect(self, identity: SessionIdentity) -> Any:
        raise CallerConnectionBroken(
            identity,
            "the connection was supplied by the caller; the session does not open another",
        )

    def ping(self, connection: Any) -> bool:
        return True

    def is_closed(self, connection: Any) -> bool:
        # The session cannot replace what it did not open, so it must not decide the
        # caller's connection is finished.
        return False

    def close(self, connection: Any) -> None:
        pass

    def password_expired(self, error: BaseException) -> bool:
        return False

    def recover_password(self, identity: SessionIdentity) -> bool:
        return False


class IRISSession:
    """A caller's relationship with one IRIS instance and namespace."""

    def __init__(
        self,
        identity: Optional[SessionIdentity] = None,
        driver: Optional[Any] = None,
        health_interval: Optional[float] = None,
        connect_timeout: Optional[float] = None,
        backoff_seconds: Optional[float] = None,
        connection: Optional[Any] = None,
        owns_connection: bool = True,
    ) -> None:
        self.identity = identity or SessionIdentity.from_env()
        self._health_interval = health_interval
        self._connect_timeout = connect_timeout
        self._backoff_seconds = backoff_seconds

        self._owns = owns_connection
        self._borrowed_connection = connection
        self._local = threading.local()
        self._connect_lock = threading.RLock()
        self._tracked: List[Any] = []
        self._backoff_until = 0.0
        self._password_recovered = False

        if driver is not None:
            self.driver = driver
        elif not owns_connection:
            self.driver = _BorrowedDriver()
        else:
            from iris_vector_rag.session.drivers import select_driver

            self.driver = select_driver(
                self.identity, connect_timeout=self.connect_timeout
            )

    # -- construction -------------------------------------------------------

    @classmethod
    def from_connection(
        cls,
        connection: Any,
        identity: Optional[SessionIdentity] = None,
    ) -> "IRISSession":
        """Wrap a connection the caller owns. The session never closes it."""
        return cls(
            identity=identity or SessionIdentity.from_env(),
            connection=connection,
            owns_connection=False,
        )

    # -- what a caller can ask without connecting ---------------------------

    @property
    def driver_name(self) -> str:
        return getattr(self.driver, "name", type(self.driver).__name__)

    @property
    def health_interval(self) -> float:
        if self._health_interval is not None:
            return float(self._health_interval)
        return _env_float("IRIS_HEALTH_INTERVAL", DEFAULT_HEALTH_INTERVAL)

    @property
    def connect_timeout(self) -> float:
        if self._connect_timeout is not None:
            return float(self._connect_timeout)
        return _env_float("IRIS_CONNECT_TIMEOUT", 10.0)

    @property
    def backoff_seconds(self) -> float:
        if self._backoff_seconds is not None:
            return float(self._backoff_seconds)
        return _env_float("IRIS_CONNECT_BACKOFF", 30.0)

    @property
    def owns_connection(self) -> bool:
        return self._owns

    @property
    def in_backoff(self) -> bool:
        return monotonic() < self._backoff_until

    @property
    def open_connection_count(self) -> int:
        """How many connections this session holds, across every thread.

        A licence counts connections, not callers: IRIS Community Edition refuses
        the sixth. The HTTP API's pool reads this to decide whether a finished
        request's connection may stay with its thread (FR-017).
        """
        if not self._owns:
            return 1 if self._borrowed_connection is not None else 0
        with self._connect_lock:
            return len(self._tracked)

    @property
    def is_connected(self) -> bool:
        """Recorded state. Asking this never costs a round trip."""
        if not self._owns:
            return self._borrowed_connection is not None
        return getattr(self._local, "connection", None) is not None

    # -- the adapter escape hatch -------------------------------------------

    def connection_for_adapter(self) -> Any:
        """The driver connection itself.

        For adapters that must hold the driver object -- the HTTP API's pool, the
        deprecated entry points, `iris-vector-graph`. Library code uses `execute`.
        """
        return self._thread_connection()

    # -- statements ---------------------------------------------------------

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> Rows:
        statement = _clean(sql)
        return self._attempt(
            lambda connection: self._run(connection, statement, params)
        )

    def execute_many(self, sql: str, seq_of_params: Sequence[Sequence[Any]]) -> int:
        batch = [tuple(params) for params in seq_of_params]
        if not batch:
            return 0
        statement = _clean(sql)
        return self._attempt(
            lambda connection: self._run_many(connection, statement, batch)
        )

    def one(self, sql: str, params: Optional[Sequence[Any]] = None) -> Optional[tuple]:
        """The single row, or None. No rows is an answer."""
        return self.execute(sql, params).one()

    def _run(self, connection: Any, sql: str, params: Optional[Sequence[Any]]) -> Rows:
        cursor = connection.cursor()
        try:
            if params is None:
                cursor.execute(sql)
            else:
                cursor.execute(sql, params)
            return read_cursor(cursor)
        finally:
            _close_cursor(cursor)

    def _run_many(self, connection: Any, sql: str, batch: List[tuple]) -> int:
        cursor = connection.cursor()
        try:
            cursor.executemany(sql, batch)
            reported = getattr(cursor, "rowcount", None)
            if isinstance(reported, int) and reported >= 0:
                return reported
            return len(batch)
        finally:
            _close_cursor(cursor)

    def _attempt(self, action: Callable[[Any], Any]) -> Any:
        """Run an action, replacing a dead connection at most once."""
        replaced = False
        while True:
            try:
                connection = self._thread_connection()
            except SessionError as exc:
                if replaced:
                    raise ConnectionLost(
                        self.identity,
                        f"the connection failed and could not be replaced: {exc}",
                    ) from exc
                raise
            try:
                return action(connection)
            except Exception as exc:
                if not looks_like_lost_connection(exc):
                    # IRIS rejecting a statement will reject it again; a reconnect
                    # would buy the same failure at the price of a handshake.
                    raise
                if not self._owns:
                    raise CallerConnectionBroken(
                        self.identity,
                        "the connection supplied by the caller has failed; "
                        "the session does not replace what it does not own",
                    ) from exc
                if replaced:
                    raise ConnectionLost(
                        self.identity,
                        f"the connection failed again after one replacement: {exc}",
                    ) from exc
                replaced = True
                logger.info(
                    "replacing the connection to %s after %s",
                    self.identity.label,
                    type(exc).__name__,
                )
                self._discard(close=True)

    # -- connection lifetime ------------------------------------------------

    def _thread_connection(self) -> Any:
        if not self._owns:
            if self._borrowed_connection is None:
                raise CallerConnectionBroken(
                    self.identity, "the caller-supplied connection is gone"
                )
            return self._borrowed_connection

        state = self._local
        connection = getattr(state, "connection", None)
        if connection is not None and getattr(state, "pid", None) != os.getpid():
            # A forked child must not use, or close, what the parent opened.
            connection = None
            state.connection = None

        if connection is not None and self._is_closed(connection):
            # Ten years of callers end with `conn.close()`, and the connection they
            # were handed is the session's. Noticing that costs nothing, so the
            # session replaces it rather than letting the next statement fail.
            logger.debug(
                "the connection for %s was closed by its caller; opening another",
                self.identity.label,
            )
            self._discard(close=False)
            connection = None

        if connection is None:
            return self._open()

        if monotonic() >= getattr(state, "healthy_until", 0.0):
            if self.driver.ping(connection):
                state.healthy_until = monotonic() + self.health_interval
            else:
                logger.info(
                    "health probe failed for %s; reconnecting", self.identity.label
                )
                self._discard(close=True)
                return self._open()
        return connection

    def _is_closed(self, connection: Any) -> bool:
        """Has this handle been closed under us? Only drivers that can say are asked."""
        ask = getattr(self.driver, "is_closed", None)
        if ask is None:
            return False
        try:
            return bool(ask(connection))
        except Exception:
            return False

    def _open(self) -> Any:
        now = monotonic()
        if now < self._backoff_until:
            raise InBackoff(
                self.identity,
                f"a connect attempt failed within the last "
                f"{self.backoff_seconds:.0f}s; refusing to retry for another "
                f"{self._backoff_until - now:.0f}s",
            )
        with self._connect_lock:
            try:
                connection = self.driver.connect(self.identity)
            except Exception as exc:
                connection = self._after_connect_failure(exc)
        self._record(connection)
        return connection

    def _after_connect_failure(self, error: Exception) -> Any:
        """An expired password is recoverable once; everything else backs off."""
        if self.driver.password_expired(error):
            if self._password_recovered:
                raise PasswordExpired(
                    self.identity,
                    "the password is still expired after one recovery attempt",
                ) from error
            self._password_recovered = True
            if not self.driver.recover_password(self.identity):
                raise PasswordExpired(
                    self.identity,
                    "the password is expired and could not be cleared",
                ) from error
            logger.info("cleared expired password state for %s", self.identity.label)
            try:
                return self.driver.connect(self.identity)
            except Exception as retry_error:
                if self.driver.password_expired(retry_error):
                    raise PasswordExpired(
                        self.identity,
                        "recovery ran but the password is still expired",
                    ) from retry_error
                raise self._back_off(retry_error) from retry_error
        raise self._back_off(error) from error

    def _back_off(self, error: Exception) -> SessionError:
        failure = wrap_connect_failure(self.identity, error)
        if failure.opens_backoff:
            self._backoff_until = monotonic() + self.backoff_seconds
            logger.warning(
                "connect to %s failed (%s); refusing further attempts for %.0fs",
                self.identity.label,
                failure.__class__.__name__,
                self.backoff_seconds,
            )
        return failure

    def _record(self, connection: Any) -> None:
        state = self._local
        state.connection = connection
        state.pid = os.getpid()
        state.healthy_until = monotonic() + self.health_interval
        with self._connect_lock:
            self._tracked.append(connection)
        marker = _ThreadMarker()
        state.marker = marker
        state.reaper = weakref.finalize(marker, self._reap, connection)

    def _reap(self, connection: Any) -> None:
        """Close a connection whose thread has ended. Never raises: nothing is waiting."""
        with self._connect_lock:
            if connection not in self._tracked:
                return  # already closed by close(), close_all() or a replacement
            self._tracked.remove(connection)
        try:
            self.driver.close(connection)
        except Exception as exc:  # pragma: no cover - teardown of a dead thread
            logger.debug("closing a finished thread's connection failed: %s", exc)

    def _discard(self, close: bool) -> None:
        state = self._local
        connection = getattr(state, "connection", None)
        reaper = getattr(state, "reaper", None)
        if reaper is not None:
            # Before the marker goes: dropping it would otherwise run the reaper, and
            # this thread is closing the connection itself.
            reaper.detach()
        state.connection = None
        state.healthy_until = 0.0
        state.marker = None
        state.reaper = None
        if connection is None:
            return
        with self._connect_lock:
            if connection in self._tracked:
                self._tracked.remove(connection)
        if close:
            self.driver.close(connection)

    # -- teardown -----------------------------------------------------------

    def close(self) -> None:
        """Close this thread's connection, if the session owns it."""
        if not self._owns:
            return
        self._discard(close=True)

    def close_all(self) -> None:
        """Close every connection this session opened, in any thread.

        Only correct at teardown -- the registry's `atexit` and test reset. A
        running thread's connection is its own business.
        """
        if not self._owns:
            return
        with self._connect_lock:
            connections, self._tracked = list(self._tracked), []
        for connection in connections:
            self.driver.close(connection)
        self._local = threading.local()

    def __enter__(self) -> "IRISSession":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"IRISSession({self.identity.label}, driver={self.driver_name}, "
            f"connected={self.is_connected})"
        )


def _clean(sql: str) -> str:
    """IRIS rejects a trailing semicolon, and callers keep writing one."""
    statement = sql.strip()
    if statement.endswith(";"):
        statement = statement[:-1].rstrip()
    return statement


def _close_cursor(cursor: Any) -> None:
    try:
        cursor.close()
    except Exception:
        pass
