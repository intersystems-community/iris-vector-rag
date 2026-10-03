"""Storing documents into a populated table never destroys what is there (feature 086).

`store_documents` probes the table with `SELECT COUNT(*) ... WHERE 1=0` and calls
`initialize_schema()` on *any* exception. That probe cannot tell an absent table from a
locked, licence-starved or disconnected one, so anything destructive reachable from it is
reached by accident. Before this feature, `initialize_schema` dropped the table
unconditionally and emptied it when the drop failed.

The unit suite asserts no DDL reaches the cursor. This asserts the rows are still in IRIS
afterwards, which is the claim that actually matters.

Container: `iris-vector-rag-iris` on 51972, per CLAUDE.md. Never another one.
"""

from __future__ import annotations

import uuid

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Document
from iris_vector_rag.storage.enterprise_storage import IRISStorage
from iris_vector_rag.storage.schema_manager import SchemaManager

pytestmark = pytest.mark.integration

TABLE = "RAG.SourceDocuments"
EXISTING_ROWS = 3


class _CursorFailsFirstProbe:
    """A real cursor whose first `WHERE 1=0` probe raises, exactly once."""

    def __init__(self, cursor, state):
        self._cursor = cursor
        self._state = state

    def execute(self, sql, *args, **kwargs):
        if "WHERE 1=0" in str(sql) and not self._state["probe_failed"]:
            self._state["probe_failed"] = True
            raise RuntimeError("simulated transient failure: table momentarily locked")
        return self._cursor.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._cursor, name)


class _ConnectionFailsFirstProbe:
    """A real connection handing out cursors that fail the first probe."""

    def __init__(self, connection, state):
        self._connection = connection
        self._state = state

    def cursor(self):
        return _CursorFailsFirstProbe(self._connection.cursor(), self._state)

    def __getattr__(self, name):
        return getattr(self._connection, name)


@pytest.fixture
def populated_table():
    """Ensure the document table exists and seed it with rows we can recognise."""
    from iris_vector_rag.common.iris_connection import get_iris_connection

    config_manager = ConfigurationManager()
    SchemaManager(config_manager=config_manager).ensure_table_schema("SourceDocuments")

    marker = f"t086_{uuid.uuid4().hex[:8]}"
    connection = get_iris_connection()
    cursor = connection.cursor()
    for index in range(EXISTING_ROWS):
        cursor.execute(
            f"INSERT INTO {TABLE} (doc_id, title, text_content) VALUES (?, ?, ?)",
            [f"{marker}_seed_{index}", "seeded", f"row {index} was here first"],
        )
    connection.commit()
    cursor.close()

    try:
        yield marker, config_manager
    finally:
        cursor = connection.cursor()
        try:
            cursor.execute(f"DELETE FROM {TABLE} WHERE doc_id LIKE ?", [f"{marker}%"])
            connection.commit()
        except Exception:  # noqa: BLE001 - teardown is advisory
            pass
        cursor.close()
        # Do not close: get_iris_connection hands out a cached connection other
        # tests in the same session reuse.


def _count_rows(connection, predicate="1=1", params=None):
    cursor = connection.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE {predicate}", params or [])
        return cursor.fetchone()[0]
    finally:
        cursor.close()


def test_failed_probe_does_not_empty_a_populated_table(populated_table):
    """The rows that were there before storing a document are there afterwards."""
    marker, config_manager = populated_table
    from iris_vector_rag.common.iris_connection import get_iris_connection

    reader = get_iris_connection()
    rows_before = _count_rows(reader)
    seeded_before = _count_rows(reader, "doc_id LIKE ?", [f"{marker}_seed_%"])
    assert seeded_before == EXISTING_ROWS

    storage = IRISStorage(ConnectionManager(config_manager), config_manager)
    state = {"probe_failed": False}
    storage._connection = _ConnectionFailsFirstProbe(get_iris_connection(), state)

    result = storage.store_documents(
        [
            Document(
                id=f"{marker}_new",
                page_content="stored after the probe failed",
                metadata={"title": "new"},
            )
        ]
    )

    assert state["probe_failed"], "the test never exercised the failing-probe path"
    assert result["documents_stored"] == 1

    assert _count_rows(reader, "doc_id LIKE ?", [f"{marker}_seed_%"]) == EXISTING_ROWS
    assert _count_rows(reader) == rows_before + 1
