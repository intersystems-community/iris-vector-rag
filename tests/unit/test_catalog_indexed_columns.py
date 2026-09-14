"""Which columns of a table are indexed (feature 082, T054).

`index_names` answers "which indexes exist", which is what index creation needs.
Attaching a corpus asks a different question -- "is this one column indexed" -- and
used to answer it with its own `INFORMATION_SCHEMA.INDEXES` query inside the
pipeline. The query belongs here, where every other catalog read lives (FR-015).

The SQL is asserted as text because that is the whole of the behaviour: the live
run against the recorded shapes lives in the integration suite.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from iris_vector_rag.storage.catalog import IRISCatalogReader
from iris_vector_rag.storage.table_shape import CatalogUnavailable


class RecordingCursor:
    """One scripted result set, and every statement it was given."""

    def __init__(self, rows=(), fail: bool = False):
        self._rows = list(rows)
        self._fail = fail
        self.statements: list[tuple[str, object]] = []
        self.closed = False
        self.description = [("COLUMN_NAME",)]

    def execute(self, sql, params=None):
        self.statements.append((" ".join(str(sql).split()), params))
        if self._fail:
            raise RuntimeError("indexes view unavailable")

    def fetchall(self):
        return list(self._rows)

    def close(self):
        self.closed = True


def reader_for(cursor) -> IRISCatalogReader:
    connection = Mock()
    connection.cursor.return_value = cursor
    return IRISCatalogReader(connection)


def test_the_indexed_columns_are_returned_upper_cased():
    """Upper case, like `index_names`: IRIS reports catalog names its own way."""
    cursor = RecordingCursor(rows=[("doc_id",), ("embedding",)])

    assert reader_for(cursor).indexed_columns("RAG", "SourceDocuments") == {
        "DOC_ID",
        "EMBEDDING",
    }


def test_the_query_is_scoped_to_the_schema_and_table():
    """Two schemas can carry the same table name; the answer must not merge them."""
    cursor = RecordingCursor(rows=[])

    reader_for(cursor).indexed_columns("ShapeTest", "ShapeCanonical")

    sql, params = cursor.statements[0]
    assert "INFORMATION_SCHEMA.INDEXES" in sql
    assert "TABLE_SCHEMA = ?" in sql and "TABLE_NAME = ?" in sql
    assert params == ["ShapeTest", "ShapeCanonical"]
    assert cursor.closed


def test_a_table_with_no_indexes_gives_an_empty_set():
    """Empty, not None: "no indexed columns" is an answer, not a failure."""
    assert (
        reader_for(RecordingCursor(rows=[])).indexed_columns("RAG", "Nothing") == set()
    )


def test_a_driver_failure_is_reported_as_catalog_unavailable():
    """Same contract as the rest of the module: the driver error becomes the cause."""
    cursor = RecordingCursor(fail=True)

    with pytest.raises(CatalogUnavailable, match="index"):
        reader_for(cursor).indexed_columns("RAG", "SourceDocuments")

    assert cursor.closed
