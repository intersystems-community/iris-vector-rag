# Contract: SqlExecutor Protocol

## Overview
`SqlExecutor` is the portability bridge between `iris_vector_rag` pipelines and any SQL-capable connection. It decouples the pipeline from DBAPI2 internals, enabling injection of mock executors for testing and shared connections from consumer packages.

## Location
`iris_vector_rag/executor.py`

## Interface

```python
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class SqlExecutor(Protocol):
    def execute(self, sql: str, params: Any = None) -> list[dict]:
        """
        Execute a SQL statement and return results as a list of dicts.

        Args:
            sql: SQL query string with ? placeholders
            params: List of parameter values, or None

        Returns:
            List of dicts, one per row, with column names as keys.
            Returns empty list for queries that return no rows.

        Raises:
            Any exception from the underlying connection (not swallowed).
        """
        ...
```

## Contract Rules

1. **Column names**: Keys in returned dicts MUST match column names exactly as returned by the database (case-sensitive per IRIS conventions).
2. **Empty results**: Return `[]` — never `None`.
3. **Params format**: Accept `None` (no params), a `list`, or a `tuple`. Do not require a specific container type.
4. **Exception passthrough**: Do NOT catch or swallow database exceptions. Let them propagate to the pipeline.
5. **No transactions**: `SqlExecutor` is read-only in all current usages. Write operations (DDL, INSERT) are out of scope for this protocol.

## Verified Implementors

| Implementor | Location | Notes |
|---|---|---|
| `MockSqlExecutor` | `tests/unit/test_sql_executor.py` | In-memory; keyed by SQL fragment |
| `IrisSyncWrapperExecutor` | `ai-hub` (spec 014) | Wraps `iris_sync_wrapper.execute_sql_query_dict` |

> **Note**: A `DbApiCursorExecutor` (wrapping a raw DBAPI2 cursor) is a natural future implementor but is explicitly out of scope for feature 065. Add a task in a future spec if needed.

## Export
`SqlExecutor` MUST be importable from `iris_vector_rag` directly:
```python
from iris_vector_rag import SqlExecutor  # must work without iris_llm
from iris_vector_rag.executor import SqlExecutor  # also valid
```
