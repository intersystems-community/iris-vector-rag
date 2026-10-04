"""A legacy entity table lacks `confidence`; the write path adds it, never drops data.

The entity writers always INSERT a `confidence` column. A table created by an older
release does not have one, and every write then failed with SQLCODE -29. The fix asks
the catalog which columns exist and adds only what is missing, with ALTER TABLE.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from iris_vector_rag.storage.catalog import CatalogSnapshot, RecordedCatalog
from iris_vector_rag.storage.legacy_columns import ensure_confidence_column


def _snapshot(*names: str) -> CatalogSnapshot:
    return CatalogSnapshot(
        schema="RAG",
        table="Entities",
        columns=[{"COLUMN_NAME": n} for n in names],
    )


def _connection():
    cursor = Mock()
    conn = Mock()
    conn.cursor.return_value = cursor
    return conn, cursor


LEGACY = ("id", "entity_name", "entity_type", "entity_id", "description")


def test_adds_the_missing_column_with_a_default():
    conn, cursor = _connection()
    reader = RecordedCatalog({("RAG", "Entities"): _snapshot(*LEGACY)})

    assert ensure_confidence_column(conn, "RAG.Entities", reader=reader) is True

    (sql,), _ = cursor.execute.call_args
    assert sql == "ALTER TABLE RAG.Entities ADD confidence DOUBLE DEFAULT 1.0"
    conn.commit.assert_called_once()


def test_statements_are_additive_only():
    conn, cursor = _connection()
    reader = RecordedCatalog({("RAG", "Entities"): _snapshot(*LEGACY)})
    ensure_confidence_column(conn, "RAG.Entities", reader=reader)
    for call in cursor.execute.call_args_list:
        text = call.args[0].upper()
        assert text.startswith("ALTER TABLE") and " ADD " in text
        assert "DROP" not in text and "DELETE" not in text


def test_a_table_that_has_the_column_is_left_alone():
    conn, cursor = _connection()
    reader = RecordedCatalog({("RAG", "Entities"): _snapshot(*LEGACY, "CONFIDENCE")})

    assert ensure_confidence_column(conn, "RAG.Entities", reader=reader) is False
    cursor.execute.assert_not_called()


def test_a_table_that_does_not_exist_is_left_to_the_schema_manager():
    conn, cursor = _connection()
    reader = RecordedCatalog({})

    assert ensure_confidence_column(conn, "RAG.Entities", reader=reader) is False
    cursor.execute.assert_not_called()


def test_a_failing_alter_raises_and_rolls_back():
    conn, cursor = _connection()
    cursor.execute.side_effect = RuntimeError("no privilege")
    reader = RecordedCatalog({("RAG", "Entities"): _snapshot(*LEGACY)})

    with pytest.raises(RuntimeError, match="no privilege"):
        ensure_confidence_column(conn, "RAG.Entities", reader=reader)
    conn.rollback.assert_called_once()
