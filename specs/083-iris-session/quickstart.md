# Quickstart: IRIS Session (083)

## From configuration

```python
from iris_vector_rag.session import session_for

session = session_for()                      # environment
session = session_for(config_manager)        # ConfigurationManager

rows = session.execute(
    "SELECT TOP 5 doc_id, chunk_text FROM RAG.DocumentChunks WHERE doc_id = ?",
    ("doc-1",),
)
for doc_id, text in rows:                    # readable after the cursor is gone
    print(doc_id, text[:40])

print(session.identity.label)                # localhost:51972/USER — no connection opened
print(session.driver_name)                   # "network"
print(session.is_connected)                  # recorded state, no probe
```

`session_for()` does not connect. The first `execute()` connects, and every later
call inside the health interval (30 s by default) issues no probe.

One row, or none:

```python
row = session.one("SELECT COUNT(*) FROM RAG.SourceDocuments")
count = row[0] if row else 0
```

A batch:

```python
session.execute_many(
    "INSERT INTO RAG.SourceDocuments (doc_id, text_content) VALUES (?, ?)",
    [("a", "..."), ("b", "...")],
)
```

## From a connection you already hold

```python
from iris_vector_rag.session import IRISSession

session = IRISSession.from_connection(my_connection)
```

The session uses it and never closes it. If that connection dies, you get
`CallerConnectionBroken` — the session will not replace something it does not own.

## Two pipelines, one session

```python
engine = IRISVectorEngine.from_config()
assert engine.session is session_for()        # one session per identity per process
```

## In a unit test: no driver, no network

```python
from iris_vector_rag.testing.fake_driver import fake_session

def test_storage_writes_the_expected_sql():
    session = fake_session(rows=[("doc-1", "hello")])          # one line (SC-008)
    store = IRISVectorStore(session=session)

    store.fetch("doc-1")

    assert session.driver_name == "fake"
    assert "SELECT" in session.driver.statements[0]
```

`fake_session()` builds a real `IRISSession` around a `FakeDriver`, so the health
cadence, the reconnect path and the cursor discipline are the ones production uses —
only the socket is missing. `FakeDriver` records every statement, can be told to fail
a connect, to expire a password, or to drop a connection after N statements, which is
how the failure-mode tests reach those branches without a container.

## What replaces what

| Before | Now |
| --- | --- |
| `get_iris_connection()` | `session_for().connection_for_adapter()` |
| `ConnectionManager().get_connection()` | `session_for()` and its `execute()` |
| `IRISConnectionManager()` | `session_for()` |
| `get_iris_dbapi_connection()` | `session_for().connection_for_adapter()` |
| hand-rolled `cursor = conn.cursor(); …; cursor.close()` | `session.execute(...)` |

The old names still work in 0.15 and warn once each. They are gone in 0.16.

## Tuning

| Variable | Default | Effect |
| --- | --- | --- |
| `IRIS_HEALTH_INTERVAL` | `30` | seconds between health probes per thread |
| `IRIS_CONNECT_TIMEOUT` | `10` | budget for one driver handshake |
| `IRIS_CONNECT_BACKOFF` | `30` | window during which attempts fail immediately after a hung or failed handshake |
