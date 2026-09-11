# Research: IRIS Session (083)

The spec says eleven acquisition paths, two classes sharing a name, protections on one
path only, a probe per storage call, and a retrieval component that gets no connection.
This is where each of those claims is in the tree, and what replaces it.

## The eleven acquisition paths

| # | Path | Hardened? | Callers |
| --- | --- | --- | --- |
| 1 | `common/iris_connection.py::get_iris_connection` | yes — socket precheck, bounded handshake, back-off, password recovery, per-identity cache | storage via `core.ConnectionManager`, 3 pipelines, tests, examples |
| 2 | `common/iris_connection.py::IRISConnectionPool` | inherits 1 | nothing in the package |
| 3 | `common/connection_pool.py::IRISConnectionPool` | **no** — bare `import iris; iris.dbapi.connect(...)` | `api/main.py`, api middleware and services |
| 4 | `common/connection_pool.py::get_pool` / `initialize_pool` / `close_pool` | as 3 | nothing in the package |
| 5 | `common/connection_manager.py::ConnectionManager` (+ `get_connection_manager`, `set_global_connection_type`) | no — ODBC/JDBC/dbapi flavours, already `DeprecationWarning` | nothing in the package |
| 6 | `core/connection.py::ConnectionManager` | delegates to 1 | storage, schema manager, engine, pipelines, `retrieval/modes.py` |
| 7 | `common/iris_connection_manager.py::IRISConnectionManager` | delegates to 1, already `DeprecationWarning` | nothing in the package |
| 8 | `common/iris_connection_manager.py::get_iris_dbapi_connection` | delegates to 1 | `tests/e2e/test_crag_pipeline_e2e.py` and other tests import `get_iris_connection` from this module |
| 9 | `common/iris_connection_manager.py::get_iris_connection_old` | delegates to 1, ignores its arguments | nothing |
| 10 | `common/iris_connector.py::get_iris_connection` — a second function of the same name | no | nothing in the package |
| 11 | `common/iris_dbapi_connector.py::get_iris_dbapi_connection` / `get_iris_dbapi_module` | delegates to 1 | `common/iris_sql_utils.py`, `common/llm_cache_manager.py` |

Two collisions, not one:

- `IRISConnectionPool` in `common/iris_connection.py` (1) and in `common/connection_pool.py` (3).
- `ConnectionManager` in `core/connection.py` (6), `common/connection_manager.py` (5) and
  `api/websocket/connection.py` — the last is a WebSocket registry with nothing to do
  with databases, and it is the one the API's `app_state.connection_manager` holds.

## Decision 1: the session is a class with an identity, not a function with a cache

`get_iris_connection` is the only hardened path, but its state lives in three module
globals (`_connection_cache`, `_handshake_backoff`, `_edition_cache`) keyed by
`(host, port, namespace, user)`. That key is already the session identity; it is just
not addressable, so nothing can be asked "are you connected?" without connecting, and
tests cannot substitute it without patching module globals.

**Decision**: `IRISSession`, in a new `iris_vector_rag/session/`, owns the identity, the
driver, the per-thread connection, the health deadline and the back-off. `for_config()`
returns the one session per identity per process; `from_connection()` wraps a
caller-supplied connection and never closes it.

**Rejected**: keeping the function and adding accessors. The state is per identity, the
function is not, and every accessor would be a second way to reach the same fact.

## Decision 2: the driver is a protocol with six operations

Selection is spread over import-time `try/except` blocks in `iris_connection.py`,
`connection_pool.py`, `iris_dbapi_connector.py` and `iris_connector.py`, so which driver
a process uses depends on which module imported first, and nothing logs the choice.

**Decision**: a `Driver` protocol — `name`, `available()`, `connect(identity)`,
`ping(connection)`, `close(connection)`, `password_expired(error)`,
`recover_password(identity)`. Two implementations: `EmbeddedDriver` (`iris.runtime`
state `embedded-kernel` / `embedded-local`, or `IRISINSTALLDIR`) and `NetworkDriver`
(`iris.dbapi.connect(..., sharedmemory=False)`, falling back to `createConnection` /
`connect`). `select_driver()` prefers embedded, logs the choice and the reason, and
raises `NoUsableDriver` naming both attempts when neither is available.

The bounded handshake stays where it is today — a daemon thread joined with a timeout —
because the driver's connect has no socket timeout of its own. It moves into
`NetworkDriver` so the embedded path is not paying for a thread it does not need.

## Decision 3: health has a deadline, and operations do not probe

Today `core.ConnectionManager.get_connection()` calls `get_iris_connection()` for every
storage operation, which opens a throwaway socket (2 s timeout) *and* runs
`SELECT 1` on the cached connection. Two round trips before the real query, on an
edition where the connection count is 1.

**Decision**: the session records `_healthy_until = monotonic() + health_interval`.
`connection()` returns the thread's connection without any probe while that deadline
holds; past it, one `ping` refreshes the deadline. Default interval **30 s**, from
`IRIS_HEALTH_INTERVAL`. `is_connected` reads recorded state and never probes.

Reconnect is driven by failure, not by probing: `execute` catches a connection-level
error, replaces the connection once, and retries the statement once — but only for a
session that owns its connection (FR-010).

**Rejected**: a background health thread. It would probe an idle session forever and
adds a thread per identity for a fact only the next operation needs.

## Decision 4: rows are copied out of the cursor

The driver's row objects stop being readable when their cursor closes — the failure
this project has already been bitten by (see the memory note on `DataRow` lifetime).

**Decision**: `session.execute(sql, params)` returns `Rows`: a list of plain tuples plus
`columns`, materialised inside the `with cursor` block. `one()` returns `Optional[tuple]`
— `None` for no rows, not an exception. `execute_many` for batches. Every path closes
its cursor in `finally`; no method hands a live cursor to a caller.

`connection_for_adapter()` is the single documented escape hatch (FR-013) for code that
must hold the driver object — today `iris-vector-graph` and the API's pool.

## Decision 5: threads get their own connection, and a fork gets nothing

`_connection_cache` is a plain dict shared by every thread, guarded only while
connecting; two threads then run statements on one connection.

**Decision**: connections live in `threading.local()`. The session records the pid it
was built in; a connection created in another pid is discarded rather than reused, which
is the fork edge case. Registry cleanup at interpreter exit closes connections through
`atexit`, not through `__del__`.

## Decision 6: deprecate the eight, delete the three, keep the API working

**Decision**:

- Deprecated with one warning each, unchanged behaviour, removed in 0.16: paths 1, 3, 5,
  6, 7, 8, 11 (the names with callers, inside or outside this repo).
- Deleted now (FR-016 — re-exports with no callers anywhere): paths 2, 4, 9, 10 —
  `common/iris_connection.py::IRISConnectionPool`, the `connection_pool` module
  singletons, `get_iris_connection_old`, and `common/iris_connector.py` in its entirety.
- `core.ConnectionManager` keeps its signature and its `get_connection()` return type
  (a driver connection) by delegating to a session; it is the name storage and the
  pipelines pass around today, and changing it would change every constructor in the
  package.
- The API keeps `IRISConnectionPool`, which becomes a thin pool over sessions. While
  there: `app_state.connection_pool.dispose()` is called on shutdown and no such method
  exists on that class — the API raises `AttributeError` from its lifespan `finally`
  today. FR-017 covers it.

## Decision 7: the gate check for direct driver connects

**Decision**: extend `scripts/gates/antipatterns.py` with a detector for
`iris.dbapi.connect`, `iris.connect` and `iris.createConnection` outside
`iris_vector_rag/session/drivers.py`, with the current offenders in the baseline file
until their tasks remove them. SC-004 is then enforced rather than asserted.

## Decision 8: the two defects the spec names in passing

- `pipelines/basic.py:589` builds `RetrievalEngine(connection=getattr(self.connection_manager, "connection", None))`.
  `ConnectionManager` has no `connection` attribute, so every non-vector retrieval mode
  runs with `connection=None`; `check_prerequisites` then constructs a *second*
  `ConnectionManager()` of its own to answer whether the knowledge graph exists.
  The engine takes the session instead.
- `core/engine.py:211-223` `embedding_dimension` falls back to `1536` for a project
  whose default model is 384-dimensional (constitution principle 5). It reads the
  configured model's dimension, and raises rather than guessing when there is none.

## Open questions

None. The health interval default (30 s) and the back-off window (existing
`IRIS_CONNECT_BACKOFF`, 30 s) are tuning decisions recorded here per the spec's
assumptions.
