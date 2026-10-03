# IRIS Session

One object owns one instance-and-namespace relationship: the identity, the driver
choice, the connection, its health cadence and the connect back-off. Everything that
talks to IRIS goes through it.

```python
from iris_vector_rag.session import session_for

session = session_for(config_manager)          # does not connect
rows = session.execute("SELECT TOP 5 doc_id FROM RAG.SourceDocuments")
for doc_id, in rows:
    print(doc_id)
```

Acquiring a session costs nothing. The first statement opens the connection.

## Why

Before 0.15 there were eleven ways to acquire a connection, and each had its own
subset of the protections: one had the two-second socket precheck and the bounded
handshake; the others had none of them. A caller
that took the wrong one could hang forever on a container that was still starting.

The session is the one way in, so a protection added once applies everywhere.

## Migrating

Everything in the left column still works in 0.15 and warns once, naming its
replacement. All of it is gone in 0.16.

| Before                                                                 | Now                                                          |
| ---------------------------------------------------------------------- | ------------------------------------------------------------ |
| `common.iris_connection.get_iris_connection()`                         | `session_for().connection_for_adapter()`                     |
| `common.iris_dbapi_connector.get_iris_dbapi_connection()`              | same                                                         |
| `common.iris_connection_manager.IRISConnectionManager()`               | `session_for()`                                              |
| `common.connection_manager.ConnectionManager()`                        | `session_for()` (the class is now `LegacyConnectionManager`) |
| `common.connection_pool.IRISConnectionPool()`                          | `session_for()`                                              |
| `cursor = conn.cursor(); ...; cursor.close()`                          | `session.execute(...)`                                       |
| `IRISConnectionPool.get_pool()` / `initialize_pool()` / `close_pool()` | gone in 0.15 — hold a session                                |
| `common.iris_connector` (whole module)                                 | gone in 0.15 — `session_for()`                               |

Three things to know while migrating:

1. **The connection the old names hand you is the session's**, not yours. You may still
   close it — the session notices and opens another — but you no longer have to, and a
   thread that ends releases it anyway.
2. **Rows outlive their cursor.** `session.execute()` returns plain tuples, so the
   `DataRow is inaccessible` failure that follows `cursor.close()` cannot happen.
3. **A wrong password is a failure, not a second session.** The identity is host, port,
   namespace and user; the password is not part of it and never appears in a log, a
   `repr` or an error message.

## Getting a session

| Call                                      | Use                                            |
| ----------------------------------------- | ---------------------------------------------- |
| `session_for(config_manager)`             | Normal use: identity from config, then env.    |
| `session_for()`                           | Identity from the environment alone.           |
| `session_for(identity=...)`               | An explicit `SessionIdentity`.                 |
| `session_for(driver=...)`                 | A substituted driver (tests, other transports) |
| `IRISSession.from_connection(connection)` | A connection the caller owns and will close.   |

One session exists per identity per process: two callers asking for the same
instance, port, namespace and user get the same object, so they share its health
cadence and its back-off window. The password is not part of the identity — a wrong
password is a failure, not a second session.

## Running statements

| Method                      | Returns                                           |
| --------------------------- | ------------------------------------------------- |
| `execute(sql, params=None)` | `Rows` — plain tuples, safe after any close.      |
| `one(sql, params=None)`     | One tuple, or `None`. Two rows is a `ValueError`. |
| `execute_many(sql, seq)`    | Number of rows affected.                          |
| `connection_for_adapter()`  | The driver connection, for code that needs it.    |

`Rows` is iterable, sized, indexable and knows its `columns`; `rows.dicts()` zips
them. Values are copied out of the cursor before it closes, because this driver's row
objects die with their cursor.

A trailing semicolon is stripped: IRIS rejects it, and callers keep writing it.

## Health, reconnects and failures

- **Health is a deadline, not a probe per call.** After a connect or a successful
  probe the connection is trusted for `IRIS_HEALTH_INTERVAL` seconds. A hundred
  statements inside that window cost zero probes. `is_connected` reads recorded
  state and never dials.
- **A dead connection is replaced once per operation.** A statement IRIS rejected is
  never retried: it would fail identically and cost a handshake.
- **A connection you closed is replaced, not mourned.** `conn.close()` on what
  `connection_for_adapter()` (or a deprecated entry point) handed you is safe: asking a
  handle whether it is closed is a local flag read, so the session notices and opens
  another before the next statement. It costs a reconnect, not an error.
- **A caller's own connection is never probed, never replaced, never closed.** Its
  failures are reported as `CallerConnectionBroken`.

Every failure is a `SessionError`, which is a builtin `ConnectionError`, and names the
instance:

| Error                    | Means                                               | Opens back-off |
| ------------------------ | --------------------------------------------------- | -------------- |
| `IRISUnreachable`        | The port is not accepting connections (2 s check).  | no             |
| `HandshakeTimeout`       | TCP connected, the driver handshake never finished. | yes            |
| `ConnectFailed`          | The driver refused for a reason of its own.         | yes            |
| `InBackoff`              | A previous attempt burned the budget; fail now.     | no             |
| `NoUsableDriver`         | Neither driver can be used in this process.         | no             |
| `PasswordExpired`        | Expired password; the library reports it, not fixes | no             |
| `ConnectionLost`         | An owned connection died and its replacement too.   | no             |
| `CallerConnectionBroken` | A borrowed connection failed.                       | no             |

Only failures that cost a whole connect budget open the back-off window. The socket
precheck answers a question that changes the moment IRIS starts, so refusing to ask
it again would make the caller wait longer than the truth.

## Tuning

| Variable                      | Default     | Meaning                                                 |
| ----------------------------- | ----------- | ------------------------------------------------------- |
| `IRIS_HEALTH_INTERVAL`        | `30`        | Seconds a connection is trusted between probes.         |
| `IRIS_CONNECT_TIMEOUT`        | `10`        | Seconds the driver handshake may take.                  |
| `IRIS_CONNECT_BACKOFF`        | `30`        | Seconds before retrying after a budget-burning failure. |
| `IRIS_HOST`                   | `localhost` | Instance host, when no config manager is given.         |
| `IRIS_PORT`                   | detected    | SuperServer port.                                       |
| `IRIS_NAMESPACE`              | `USER`      | Namespace.                                              |
| `IRIS_USER` / `IRIS_USERNAME` | `_SYSTEM`   | Account.                                                |
| `IRIS_PASSWORD`               | `SYS`       | Password. Never appears in a log, repr or error.        |

Per-session overrides: `session_for(health_interval=...)`, and
`IRISSession(connect_timeout=..., backoff_seconds=...)`.

## Drivers

A driver is anything with seven operations: `name`, `available()`, `connect(identity)`,
`ping(connection)`, `close(connection)` and `password_expired(error)` — plus an optional eighth, `is_closed(connection)`, for a
driver that can say locally whether a handle is closed. A driver without it makes the
session fall back to probing.

- `EmbeddedDriver` — IRIS reached from inside the instance.
- `NetworkDriver` — the SuperServer port, with the socket precheck, the bounded
  handshake and expired-password detection. On a "password change required" failure the
  session raises `PasswordExpired` with a hint; it never runs `docker exec`.

`select_driver` prefers embedded, logs which it chose and why, and names every driver
it tried when none can be used. `iris_vector_rag/session/drivers.py` is the only
module allowed to call a driver's own connect; the antipatterns gate enforces that.

## Testing against a session

```python
from tests.support.fake_driver import fake_session

def test_something():
    session = fake_session(columns=("doc_id",), rows=[("d1",)])
    assert session.one("SELECT doc_id FROM RAG.SourceDocuments") == ("d1",)
    assert session.driver.connects == 1
```

`FakeDriver` counts connects, pings, closes, cursors and statements, and can be told
to fail a connect, expire a password or drop a connection after _n_ statements. It
touches no network, so unit tests need no container.

## Threads and forks

Connections are per thread: two threads never share one, and each gets its own server
job. A connection is tagged with the pid that opened it, so a forked child opens its
own rather than corrupting the parent's. Cleanup runs at `atexit`, never in
`__del__`.

**A thread that ends gives its connection back.** IRIS Community Edition refuses the
sixth concurrent connection, and a request thread that has exited cannot use the one it
was holding — so the session closes it when the thread dies. A thread-per-request
server sitting idle therefore holds nothing:

```python
session.open_connection_count      # 0 when no thread is in a statement
```

## Per-thread and process cleanup

| Call                         | Closes                                      |
| ---------------------------- | ------------------------------------------- |
| `session.close()`            | This thread's connection.                   |
| `session.close_all()`        | Every connection the session opened.        |
| `close_all_sessions()`       | Every registered session. Runs at `atexit`. |
| `reset_sessions_for_tests()` | Same, for tests that own the process.       |

A thread that just ends needs none of these.

## The pool

`IRISConnectionPool` still exists for the HTTP API, and in 0.15 it is a pool over one
session. The session already gives each thread its own connection, so what the pool
adds is a capacity bound: `release()` closes a connection once more are held than the
pool advertises.

| Stat               | Means                                                   |
| ------------------ | ------------------------------------------------------- |
| `open_connections` | Connections held right now, across every thread.        |
| `connected`        | `open_connections > 0` — some thread holds one **now**. |
| `pool_size`        | The advertised capacity.                                |

`connected` is `False` for an idle pool, and `True` inside a live `get_connection()`.
That is a pool's question, not a thread's: the thread that reads a health route is
never the thread that served a request.

The pool is deprecated with everything else in this table — a new caller should hold a
session.
