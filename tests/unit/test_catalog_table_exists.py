"""Whether a table exists is a catalog read, so it lives in `catalog.py`.

The note-search concept index asked `INFORMATION_SCHEMA.TABLES` itself, which the
`catalog-read-outside-shape` detector rejects. `table_exists` answers it the way
`snapshot` identifies a table: schema and name compared without regard to case.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from iris_vector_rag.storage.catalog import IRISCatalogReader
from iris_vector_rag.storage.table_shape import CatalogUnavailable


class ScriptedCursor:
    def __init__(self, rows=(), fail: bool = False):
        self._rows = list(rows)
        self._fail = fail
        self.statements: list[tuple[str, object]] = []
        self.closed = False
        self.description = [("TABLE_SCHEMA",), ("TABLE_NAME",)]

    def execute(self, sql, params=None):
        self.statements.append((" ".join(str(sql).split()), params))
        if self._fail:
            raise RuntimeError("tables view unavailable")

    def fetchall(self):
        return list(self._rows)

    def close(self):
        self.closed = True


def reader_for(cursor) -> IRISCatalogReader:
    connection = Mock()
    connection.cursor.return_value = cursor
    return IRISCatalogReader(connection)


def test_a_table_the_catalog_lists_exists():
    cursor = ScriptedCursor(rows=[("Notes", "ConceptPostings")])

    assert reader_for(cursor).table_exists("notes", "conceptpostings") is True


def test_a_table_the_catalog_does_not_list_does_not_exist():
    assert reader_for(ScriptedCursor()).table_exists("Notes", "Missing") is False


def test_the_names_are_compared_without_regard_to_case():
    cursor = ScriptedCursor()

    reader_for(cursor).table_exists("Notes", "ConceptPostings")

    sql, params = cursor.statements[0]
    assert "UPPER(TABLE_SCHEMA) = UPPER(?)" in sql
    assert "UPPER(TABLE_NAME) = UPPER(?)" in sql
    assert params == ["Notes", "ConceptPostings"]


def test_a_given_cursor_is_used_and_left_open():
    """Callers that are mid-transaction pass their cursor, as with `index_names`."""
    own = ScriptedCursor(rows=[("Notes", "T")])
    connection = Mock()

    assert IRISCatalogReader(connection).table_exists("Notes", "T", own) is True
    connection.cursor.assert_not_called()
    assert own.closed is False


def test_a_cursor_it_opens_is_closed():
    cursor = ScriptedCursor()

    reader_for(cursor).table_exists("Notes", "T")

    assert cursor.closed is True


def test_a_failed_read_says_which_table():
    with pytest.raises(CatalogUnavailable) as raised:
        reader_for(ScriptedCursor(fail=True)).table_exists("Notes", "T")

    assert "tables view unavailable" in str(raised.value)
