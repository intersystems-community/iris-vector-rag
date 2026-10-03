"""A driver that reaches nothing, so a unit test can use a real session.

The point is not to mock the session away -- the health cadence, the reconnect and
the cursor discipline under test are the ones production runs. Only the socket is
missing.

Feature: 083-iris-session (FR-007, SC-008)
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple


class FakeCursor:
    """Records what it was asked to run and hands back programmed rows."""

    def __init__(self, connection: "FakeConnection") -> None:
        self._connection = connection
        self._driver = connection.driver
        self.description: Optional[List[Tuple[str, ...]]] = None
        self.rowcount = -1
        self.closed = False
        self._rows: Tuple[Tuple[Any, ...], ...] = ()

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> None:
        self._connection.note_statement(sql, params)
        if sql.strip().upper().startswith(("SELECT", "WITH")):
            self.description = [(column,) for column in self._driver.columns]
            self._rows = self._driver.rows
            self.rowcount = len(self._rows)
        else:
            self.description = None
            self._rows = ()
            self.rowcount = 1

    def executemany(self, sql: str, seq_of_params: Sequence[Sequence[Any]]) -> None:
        batch = list(seq_of_params)
        self._connection.note_statement(sql, batch)
        self.description = None
        self._rows = ()
        self.rowcount = len(batch)

    def fetchall(self) -> List[Tuple[Any, ...]]:
        rows, self._rows = list(self._rows), ()
        return rows

    def fetchone(self) -> Optional[Tuple[Any, ...]]:
        if not self._rows:
            return None
        first, self._rows = self._rows[0], self._rows[1:]
        return first

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self._connection.cursor_closed()


class FakeConnection:
    """A connection that can be dropped on command."""

    def __init__(self, driver: "FakeDriver") -> None:
        self.driver = driver
        self.closed = False
        self.dropped = False
        self.commits = 0
        self.rollbacks = 0
        self.open_cursors = 0
        self.statements_run = 0

    def cursor(self) -> FakeCursor:
        if self.closed or self.dropped:
            raise ConnectionResetError("fake connection is closed")
        self.open_cursors += 1
        self.driver.cursors_opened += 1
        return FakeCursor(self)

    def cursor_closed(self) -> None:
        self.open_cursors -= 1
        self.driver.cursors_closed += 1

    def note_statement(self, sql: str, params: Any) -> None:
        if self.closed or self.dropped:
            raise ConnectionResetError("fake connection is closed")
        drop_after = self.driver.drop_after
        if drop_after is not None and self.statements_run >= drop_after:
            self.dropped = True
            raise ConnectionResetError("fake connection reset by peer")
        self.statements_run += 1
        self.driver.statements.append(sql)
        self.driver.parameters.append(params)
        if sql in self.driver.fail_statements:
            raise ValueError(f"fake statement failure: {sql}")

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True

    def isClosed(self) -> bool:  # noqa: N802 - the name the IRIS DBAPI driver uses
        return self.closed


class FakeDriver:
    """A ``Driver`` that connects to nothing and records everything."""

    name = "fake"

    def __init__(
        self,
        rows: Sequence[Sequence[Any]] = (),
        columns: Sequence[str] = ("value",),
        is_available: bool = True,
        connect_failures: int = 0,
        connect_error: Optional[BaseException] = None,
        expired_password: bool = False,
        drop_after: Optional[int] = None,
        fail_statements: Sequence[str] = (),
    ) -> None:
        self.rows: Tuple[Tuple[Any, ...], ...] = tuple(tuple(row) for row in rows)
        self.columns: Tuple[str, ...] = tuple(columns)
        self.is_available = is_available
        self.connect_failures = connect_failures
        self.connect_error = connect_error
        self.expired_password = expired_password
        self.drop_after = drop_after
        self.fail_statements = set(fail_statements)

        self.connects = 0
        self.pings = 0
        self.closes = 0
        self.cursors_opened = 0
        self.cursors_closed = 0
        self.statements: List[str] = []
        self.parameters: List[Any] = []
        self.connections: List[FakeConnection] = []

    # -- the Driver protocol ------------------------------------------------

    def available(self) -> bool:
        return self.is_available

    def connect(self, identity: Any) -> FakeConnection:
        self.connects += 1
        if self.expired_password:
            raise RuntimeError("Password change required for user")
        if self.connect_failures > 0:
            self.connect_failures -= 1
            raise self.connect_error or RuntimeError("fake connect failure")
        connection = FakeConnection(self)
        self.connections.append(connection)
        return connection

    def ping(self, connection: FakeConnection) -> bool:
        self.pings += 1
        return not (connection.closed or connection.dropped)

    def is_closed(self, connection: FakeConnection) -> bool:
        # A dropped connection is not a closed one: the peer's reset is only visible
        # to a probe, which is the distinction the session's two paths rely on.
        return connection.closed

    def close(self, connection: FakeConnection) -> None:
        self.closes += 1
        connection.close()

    def password_expired(self, error: BaseException) -> bool:
        return "password change required" in str(error).lower()

    # -- convenience for tests ---------------------------------------------

    @property
    def open_cursors(self) -> int:
        return self.cursors_opened - self.cursors_closed


def fake_identity(**overrides: Any):
    from iris_vector_rag.session import SessionIdentity

    fields = {
        "host": "fake-host",
        "port": 51972,
        "namespace": "USER",
        "username": "_SYSTEM",
        "password": "SYS",
    }
    fields.update(overrides)
    return SessionIdentity(**fields)


def fake_session(**kwargs: Any):
    """A real session over a fake driver, in one line.

    Keyword arguments other than ``identity``, ``health_interval`` and
    ``backoff_seconds`` are passed to :class:`FakeDriver`.
    """
    from iris_vector_rag.session import IRISSession

    identity = kwargs.pop("identity", None) or fake_identity()
    session_kwargs = {
        name: kwargs.pop(name)
        for name in ("health_interval", "backoff_seconds", "connect_timeout")
        if name in kwargs
    }
    driver = kwargs.pop("driver", None) or FakeDriver(**kwargs)
    return IRISSession(identity=identity, driver=driver, **session_kwargs)
