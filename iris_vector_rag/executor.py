"""
SqlExecutor Protocol — connection-free abstraction over IRIS SQL.

Consumers of GraphRAGPipeline inject an executor so the pipeline can be
unit-tested without a live IRIS connection.  The default path (no executor)
is unchanged and continues to use the DBAPI cursor directly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SqlExecutor(Protocol):
    """
    Minimal protocol for executing parameterised SQL queries.

    An implementation must:
    - Accept a SQL string and an optional sequence/mapping of parameters.
    - Return results as a list of row-dicts (column name → value).
    - Raise on error (do not swallow exceptions).
    """

    def execute(self, sql: str, params: Any = None) -> list[dict]:
        """Execute *sql* with *params* and return rows as a list of dicts."""
        ...
