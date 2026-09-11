# Contract: IRIS Session (083)

This feature has no HTTP surface of its own. Its contract is a Python one: the
session, the driver protocol, the error taxonomy, and the deprecation surface.

## The one entry point

```python
from iris_vector_rag.session import session_for, IRISSession, SessionIdentity
```

### `session_for(...) -> IRISSession`

```python
def session_for(
    config_manager=None,
    *,
    identity: SessionIdentity | None = None,
    driver=None,
    health_interval: float | None = None,
) -> IRISSession
```

- Returns **the** session for the resolved identity in this process; a second call
  with the same identity returns the same object (FR-002, US1 scenario 4).
- `config_manager` resolves the identity through `get_database_config()`; `identity`
  overrides it; neither means the environment.
- `driver` substitutes driver selection entirely — nothing real is imported and no
  network call happens (FR-007, US4 scenario 4). A session created with an explicit
  driver is still registered, so two calls with the same identity still share.
- Raises `ValidationError` for an unusable identity, `NoUsableDriver` when no driver
  is available. It does **not** connect: acquiring a session is free.

### `IRISSession.from_connection(connection, *, identity=None) -> IRISSession`

Wraps a connection the caller already holds. Not registered, never closed by the
session, never replaced (FR-010, US1 scenario 3). `identity` defaults to a
`SessionIdentity` describing what the caller said, or the environment's.

### Instance surface

| Member | Type | Contract |
| --- | --- | --- |
| `identity` | `SessionIdentity` | answers without connecting (US1 scenario 5) |
| `driver_name` | `str` | `"embedded"` / `"network"` / whatever a substituted driver reports |
| `is_connected` | `bool` | recorded state only, no probe (US2 scenario 4) |
| `execute(sql, params=None)` | `Rows` | see below |
| `execute_many(sql, seq_of_params)` | `int` | affected rows; one cursor, released either way |
| `one(sql, params=None)` | `tuple \| None` | `None` for no rows (US3 scenario 3) |
| `connection_for_adapter()` | driver connection | the single documented accessor (FR-013) |
| `close()` | `None` | closes this thread's connection if the session owns it; idempotent |
| `__enter__` / `__exit__` | | `close()` on exit |

`execute` contract:

1. Values in the returned `Rows` are readable after the cursor is gone (FR-011).
2. The cursor is released on success and on failure (FR-012).
3. A connection-level failure on an owned connection is retried exactly once after
   replacing the connection (FR-009); a second failure raises `ConnectionLost`.
4. No health probe is issued if the thread's health deadline has not passed (FR-008).
5. SQL is passed through unchanged apart from stripping a trailing `;`, which IRIS
   rejects.

## The driver protocol

```python
@runtime_checkable
class Driver(Protocol):
    name: str
    def available(self) -> bool: ...
    def connect(self, identity: SessionIdentity) -> Any: ...
    def ping(self, connection: Any) -> bool: ...
    def close(self, connection: Any) -> None: ...
    def password_expired(self, error: BaseException) -> bool: ...
    def recover_password(self, identity: SessionIdentity) -> bool: ...
```

Obligations on an implementation:

- `available()` must not open a connection and must not raise.
- `close()` and `ping()` must not raise; `ping()` returns `False` instead.
- `connect()` raises on failure; the session decides whether that failure means
  back-off, password recovery or unreachable.
- `select_driver(identity, prefer_embedded=True)` returns the first available driver,
  logs the choice at INFO with the reason, and raises `NoUsableDriver` listing every
  driver it tried and why each was unusable (US4 scenario 3).

## Error taxonomy (FR-019)

Every error carries `identity` and renders `host:port/namespace` in its message.
All of them subclass `SessionError`, which subclasses the builtin `ConnectionError`
so that existing `except ConnectionError` call sites keep working.

| Error | Raised when | `opens_backoff` |
| --- | --- | --- |
| `IRISUnreachable` | the socket precheck fails — port not accepting connections | `False` |
| `HandshakeTimeout` | TCP connected but the driver handshake exceeded `connect_timeout` | `True` |
| `ConnectFailed` | the driver refused the connect for a reason of its own | `True` |
| `InBackoff` | an attempt was made inside the back-off window opened by a previous failure | `False` |
| `NoUsableDriver` | neither the embedded nor the network driver is available | `False` |
| `PasswordExpired` | the account password is expired and recovery did not succeed | `False` |
| `ConnectionLost` | an owned connection failed and its replacement failed too | `False` |
| `CallerConnectionBroken` | a caller-supplied connection failed; the session replaces nothing | `False` |

Each error class carries a class-level `opens_backoff` flag, and the session opens the
back-off window only for a failure whose flag is `True`: one that spent the whole
connect budget. `IRISUnreachable` costs a two-second socket check and answers a
question that changes the moment IRIS finishes starting, so refusing to ask it again
would keep a caller waiting longer than the truth.

## Deprecation surface

Deprecated in 0.15 — unchanged signature and behaviour, one `DeprecationWarning`
each naming the session as the replacement, removed in 0.16 (FR-015):

| Name | Replacement |
| --- | --- |
| `common.iris_connection.get_iris_connection` | `session_for().connection_for_adapter()` |
| `common.iris_dbapi_connector.get_iris_dbapi_connection` | same |
| `common.iris_connection_manager.IRISConnectionManager` | `session_for()` |
| `common.iris_connection_manager.get_iris_dbapi_connection` | same |
| `common.connection_manager.LegacyConnectionManager` | `session_for()` |
| `common.connection_pool.IRISConnectionPool` | `session_for()` (the API adapter keeps it for 0.15) |
| `core.connection.ConnectionManager` | `session_for()` |

Removed in this feature — re-exports with no callers anywhere (FR-016):

| Name | Why now |
| --- | --- |
| `common.iris_connection.IRISConnectionPool` + `_PooledConnection` | no caller in the package or its tests |
| `common.connection_pool.get_pool` / `initialize_pool` / `close_pool` | module singletons with no caller |
| `common.iris_connection_manager.get_iris_connection_old` | ignores its own arguments, no caller |
| `common.iris_connector` (whole module) | third function named `get_iris_connection`; its JDBC branch imports a name that does not exist |

Renamed so that no two library classes share a name (SC-001): the ODBC/JDBC-era
`common.connection_manager.ConnectionManager` becomes `LegacyConnectionManager`.
`core.connection.ConnectionManager` keeps its name — it is the one storage and the
pipelines pass around. `api.websocket.connection.ConnectionManager` is a WebSocket
registry, unrelated to databases, and is renamed `WebSocketConnectionManager`.

## Quality gate (FR-018, SC-004)

`scripts/gates/antipatterns.py` gains a detector: a call to `iris.dbapi.connect`,
`iris.connect` or `iris.createConnection` anywhere under `iris_vector_rag/` other
than `iris_vector_rag/session/drivers.py` fails the gate. Existing offenders sit in
`antipatterns-baseline.txt` only until the task that removes them.
