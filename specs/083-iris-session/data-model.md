# Data Model: IRIS Session (083)

Five entities. Four of them are immutable value objects; only the session has state,
and all of its state is about one thing — whether this thread can talk to IRIS right
now.

## SessionIdentity

Which instance and namespace, and as whom.

| Field | Type | Notes |
| --- | --- | --- |
| `host` | `str` | non-empty after strip |
| `port` | `int` | 1–65535 |
| `namespace` | `str` | matches `^[A-Za-z0-9_]+$` |
| `username` | `str` | non-empty |
| `password` | `str` | excluded from equality, hashing and `repr` |

Constructors:

- `SessionIdentity.from_env()` — `IRIS_HOST`, `IRIS_PORT` (falling back to
  `auto_detect_iris_port()`, then 1972), `IRIS_NAMESPACE`, `IRIS_USER`/`IRIS_USERNAME`,
  `IRIS_PASSWORD`.
- `SessionIdentity.from_config(config_manager)` — `get_database_config()`, with any
  key it leaves unset filled from the environment.

Invariants:

- **The key is the first four fields.** `key` is `(host, port, namespace, username)`.
  Two identities differing only in password are the same session; a session that
  cannot connect with the password it holds is a failure, not a second session.
- **The password never reaches a log line, a `repr`, or an error message.**
  `__repr__` renders `SessionIdentity(host=…, port=…, namespace=…, username=…)`.
- **`label` is `host:port/namespace`**, and every error in this feature carries it
  (FR-019).
- Validation happens in `__post_init__`, so an invalid identity cannot exist. It
  raises the project's existing `ValidationError`, which is what the current
  connection path raises for the same inputs.

## Driver

The substitutable means of reaching IRIS. A protocol, not a base class: the fake
driver in `iris_vector_rag/testing/fake_driver.py` satisfies it without importing
anything from `session/drivers.py`.

| Operation | Signature | Contract |
| --- | --- | --- |
| `name` | `str` | `"embedded"`, `"network"`, `"fake"` — appears in the selection log line and in `NoUsableDriver` |
| `available()` | `() -> bool` | true if this driver can be used in this process; must not connect |
| `connect(identity)` | `(SessionIdentity) -> Any` | a live driver connection, or raises |
| `ping(connection)` | `(Any) -> bool` | cheap liveness check; false rather than raising |
| `close(connection)` | `(Any) -> None` | never raises |
| `password_expired(error)` | `(BaseException) -> bool` | is this failure the expired-default-password state? |
| `recover_password(identity)` | `(SessionIdentity) -> bool` | clear that state; true if it was attempted successfully |

Implementations:

- `EmbeddedDriver` — usable when `iris.runtime.get().state` is `embedded-kernel` or
  `embedded-local`, or when `IRISINSTALLDIR`/`ISC_PACKAGE_INSTALLDIR` is set and the
  runtime is `unavailable` (it then configures `mode="embedded"`). Connects with
  `iris.dbapi.connect(namespace=...)`. `recover_password` returns false: an embedded
  process is already inside the instance.
- `NetworkDriver` — usable when the `iris` module exposes `dbapi.connect`,
  `createConnection` or `connect`. Connects with a 2 s socket precheck, then the
  driver handshake on a daemon thread joined for `connect_timeout`, forcing
  `sharedmemory=False`. `recover_password` runs
  `Do ##class(Security.Users).UnExpireUserPasswords("*")` in the container publishing
  the identity's port.

Selection: `select_driver(identity, prefer_embedded=True)` returns the first
available driver and logs `name` plus the reason. If none is available it raises
`NoUsableDriver` naming every driver tried.

## Connection

One thread's live driver handle inside a session. Not a class — the driver's own
object — but it has an owner and a lifetime:

| Property | Values | Meaning |
| --- | --- | --- |
| owner | `session` \| `caller` | a caller-supplied connection is never closed or replaced (FR-010) |
| scope | thread | stored in `threading.local()`; no two threads share one (FR-003) |
| birth pid | `int` | a connection whose pid is not `os.getpid()` is discarded, not reused (fork edge case) |

## Rows (statement result)

Values copied out of the driver's cursor while it is still open, so they outlive it
(FR-011).

| Field | Type | Notes |
| --- | --- | --- |
| `columns` | `tuple[str, ...]` | from `cursor.description`; empty for non-`SELECT` |
| `rows` | `tuple[tuple, ...]` | every value a plain Python object, never a driver row |
| `rowcount` | `int` | driver `rowcount` for statements that report one, else `len(rows)` |

Behaviour: iterable, `len()`, indexable, falsy when empty. `one()` returns the single
row or `None` — never raises for "no rows" (US3 scenario 3) — and raises
`ValueError` if more than one row came back. `dicts()` zips columns with rows.

Invariant: **no `Rows` value is a driver object.** The copy happens inside the
`try` that owns the cursor, and the cursor closes in the `finally`.

## Session

| Field | Type | Notes |
| --- | --- | --- |
| `identity` | `SessionIdentity` | immutable |
| `driver` | `Driver` | chosen once at construction |
| `health_interval` | `float` | seconds; `IRIS_HEALTH_INTERVAL`, default 30 |
| `connect_timeout` | `float` | `IRIS_CONNECT_TIMEOUT`, default 10 |
| `backoff_seconds` | `float` | `IRIS_CONNECT_BACKOFF`, default 30 |
| `_owns_connection` | `bool` | false for `from_connection` |
| `_backoff_until` | `float` | monotonic deadline; per session, therefore per identity |
| `_password_recovered` | `bool` | recovery is attempted once per session (FR-005) |
| `_local.connection` | driver object \| `None` | per thread |
| `_local.healthy_until` | `float` | per thread; monotonic deadline |
| `_pid` | `int` | the pid the session was built in |

### States

```text
                 ┌──────────────┐
      close()    │ disconnected │◄── initial, and after close()
   ┌────────────►└──────┬───────┘
   │                    │ connection() / execute()
   │                    ▼
   │             ┌──────────────┐  handshake exceeds connect_timeout,
   │             │  connecting  ├──────────────────────────┐
   │             └──────┬───────┘  or connect fails        │
   │                    │ success                          ▼
   │             ┌──────▼───────┐                   ┌─────────────┐
   └─────────────┤  connected   │                   │ in-backoff  │
                 └──────┬───────┘                   └──────┬──────┘
                        │ ping fails past deadline,        │ any attempt inside
                        │ or a statement raises a          │ the window raises
                        │ connection-level error           │ InBackoff, without
                        ▼                                  │ touching the network
                 (one replace, then retry once;             │
                  second failure raises ConnectionLost) ◄───┘ window elapses
```

Transitions in words:

- **disconnected → connecting**: first `connection()` or `execute()` on this thread.
- **connecting → connected**: records `healthy_until = monotonic() + health_interval`.
- **connecting → in-backoff**: the handshake exceeded `connect_timeout`, or the
  connect raised. `_backoff_until = monotonic() + backoff_seconds`. An expired
  password is not this transition: it is recovered once and the connect retried.
- **in-backoff → connecting**: only after the window elapses. Callers inside the
  window get `InBackoff` immediately (FR-004), so a caller waits at most the window
  rather than a fresh `connect_timeout` each time.
- **connected → connected (probe)**: `connection()` past `healthy_until` pings once.
  A successful ping pushes the deadline out; nothing else probes (FR-008).
- **connected → connecting (replace)**: a statement raised a connection-level error
  and the session owns the connection. The replacement is attempted once per
  operation (FR-009); if the retry fails, `ConnectionLost`.
- **connected → failed (not owned)**: the same error on a caller-supplied connection
  raises `CallerConnectionBroken` and replaces nothing (FR-010).
- **any → disconnected**: `close()` closes this thread's connection and clears its
  deadline. Registry `atexit` does this for every session without a finaliser.

### Invariants

1. One session per `identity.key` per process (FR-002), held by the registry.
2. A thread never observes a half-built session: construction chooses the driver and
   nothing else; the first connect happens under a lock keyed by the session.
3. `is_connected` reads `_local.connection is not None` and the recorded deadline. It
   never probes (US2 scenario 4).
4. At most one probe per thread per `health_interval` (SC-002).
5. At most one reconnect per operation.
6. `connection_for_adapter()` is the only public way to the driver object (FR-013).
7. A session created in one process is not usable in a forked child; the child sees
   the pid mismatch and connects afresh.
