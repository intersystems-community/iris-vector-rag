"""SectionStore and ensure_section_column with the opt-in section_label column.

A fake cursor and a fake shape stand in for IRIS: the statements are the behaviour
under test. The same paths run against a real table in
tests/integration/test_note_search_column_iris.py.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from iris_vector_rag.note_search.sections import Section
from iris_vector_rag.note_search.store import (
    SECTION_COLUMN,
    SectionStore,
    SectionStoreError,
    ensure_section_column,
)
from iris_vector_rag.storage.table_shape import ColumnRole


class FakeShape:
    schema = "NSCOL"
    table = "DocumentChunks"
    qualified_name = "NSCOL.DocumentChunks"

    def __init__(self, *, with_label_column: bool):
        names = ["doc_id", "document_id", "chunk_text", "metadata"]
        if with_label_column:
            names.append(SECTION_COLUMN)
        self.columns = {name: SimpleNamespace(name=name) for name in names}

    def column_for(self, role):
        assert role is ColumnRole.IDENTIFIER or role is ColumnRole.TEXT
        return SimpleNamespace(
            name="doc_id" if role is ColumnRole.IDENTIFIER else "chunk_text"
        )

    def optional_column_for(self, role):
        return SimpleNamespace(name="metadata")


class FakeCursor:
    def __init__(self, *, metadata_rows=(), labels=(), indexes=()):
        self.executed: list[tuple[str, object]] = []
        self._metadata_rows = list(metadata_rows)
        self._labels = list(labels)
        self._indexes = list(indexes)
        self._result: list = []
        self.description = None

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        self.description = None
        if "INFORMATION_SCHEMA.INDEXES" in sql:
            self._result = list(self._indexes)
        elif sql.startswith("SELECT") and SECTION_COLUMN in sql.split("FROM")[0]:
            self._result = list(self._labels)
        elif sql.startswith("SELECT"):
            self._result = list(self._metadata_rows)
        else:
            self._result = []

    def fetchall(self):
        return self._result

    def close(self):
        pass

    def sql(self, prefix):
        return [sql for sql, _ in self.executed if sql.lstrip().startswith(prefix)]


def make_vector_store(shape, cursor):
    connection = MagicMock()
    connection.cursor.return_value = cursor
    store = MagicMock()
    store.shape.return_value = shape
    store.table_name = "NSCOL.DocumentChunks"
    store._get_connection.return_value = connection
    store._connection = connection
    return store, connection


def _section(label="hpi", ordinal=0):
    return Section(label=label, start=0, end=4, ordinal=ordinal, text="text")


# --- SectionStore.store --------------------------------------------------------


def test_without_the_column_the_insert_is_the_four_column_form_it_always_was():
    shape = FakeShape(with_label_column=False)
    cursor = FakeCursor()
    store, _ = make_vector_store(shape, cursor)

    _store_one(store, shape, cursor)

    inserts = cursor.sql("INSERT")
    assert len(inserts) == 1
    assert SECTION_COLUMN not in inserts[0]
    assert inserts[0].count("?") == 4


def test_with_the_column_the_label_is_written_in_the_same_insert():
    shape = FakeShape(with_label_column=True)
    cursor = FakeCursor()
    store, _ = make_vector_store(shape, cursor)

    _store_one(store, shape, cursor)

    (insert,) = cursor.sql("INSERT")
    assert SECTION_COLUMN in insert
    assert insert.count("?") == 5
    params = [p for sql, p in cursor.executed if sql.startswith("INSERT")][0]
    assert params[-1] == "hpi"
    assert json.loads(params[3])["section"] == "hpi"


def test_a_label_over_100_characters_is_refused_before_any_write():
    shape = FakeShape(with_label_column=True)
    cursor = FakeCursor()
    store, _ = make_vector_store(shape, cursor)
    long_label = "x" * 101

    with pytest.raises(SectionStoreError):
        _store_one(store, shape, cursor, label=long_label)

    assert cursor.sql("INSERT") == []
    assert cursor.sql("DELETE") == []


# --- ensure_section_column -----------------------------------------------------


def test_a_label_over_100_characters_raises_before_any_alter():
    shape = FakeShape(with_label_column=False)
    rows = [("N-1#0", json.dumps({"section": "x" * 101}))]
    cursor = FakeCursor(metadata_rows=rows)
    store, _ = make_vector_store(shape, cursor)

    with pytest.raises(SectionStoreError):
        ensure_section_column(store)

    assert cursor.sql("ALTER") == []
    assert cursor.sql("CREATE") == []


def test_unparsable_metadata_is_skipped_not_fatal():
    shape = FakeShape(with_label_column=False)
    rows = [("a", "{not json"), ("b", json.dumps({"section": "hpi"}))]
    cursor = FakeCursor(metadata_rows=rows)
    store, _ = make_vector_store(shape, cursor)

    assert ensure_section_column(store) is True

    updates = cursor.sql("UPDATE")
    assert len(updates) == 1


def test_a_table_already_complete_changes_nothing():
    shape = FakeShape(with_label_column=True)
    rows = [("N-1#0", json.dumps({"section": "hpi"}))]
    cursor = FakeCursor(
        metadata_rows=rows,
        labels=[("N-1#0", "hpi")],
        indexes=[("SECTIONLABELIDX", SECTION_COLUMN)],
    )
    store, _ = make_vector_store(shape, cursor)

    assert ensure_section_column(store) is False
    assert cursor.sql("ALTER") == []
    assert cursor.sql("UPDATE") == []


def _store_one(store, shape, cursor, label="hpi"):
    """Run SectionStore.store for one section through the fakes."""
    section_store = SectionStore(store)
    with patch(
        "iris_vector_rag.note_search.store.split_sections",
        return_value=[_section(label=label)],
    ):
        return section_store.store("N-1", "HPI: text.", lambda text: [])
