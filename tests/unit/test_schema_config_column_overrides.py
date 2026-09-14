"""Column names in `SchemaMetadata.configuration` are ignored, loudly (T056).

Before this feature a table's stored configuration could name the columns the store
should read and write (`id_column`, `text_column`, `metadata_column`). Two accounts
of one fact: the configuration said one thing and the table said another, and which
one won depended on which code path you were in.

The database now decides -- `TableShape` is read from the catalog and nothing
consults these keys. They are not silently dropped: a configuration that still
carries them gets a `DeprecationWarning` naming them, so the operator finds out from
the warning rather than from a query that returns nothing (FR-001).
"""

from __future__ import annotations

import json
import warnings
from unittest.mock import MagicMock

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.storage.schema_manager import SchemaManager


class RecordingCursor:
    def __init__(self, row):
        self._row = row
        self.statements: list[tuple[str, object]] = []

    def execute(self, sql, params=None):
        self.statements.append((" ".join(str(sql).split()), params))

    def fetchone(self):
        return self._row

    def close(self):
        pass


def manager_with(configuration: dict) -> tuple[SchemaManager, RecordingCursor]:
    """A schema manager whose metadata row carries the given configuration."""
    row = (3, 384, "all-MiniLM-L6-v2", json.dumps(configuration))
    cursor = RecordingCursor(row)
    connection = MagicMock()
    connection.cursor.return_value = cursor
    connections = MagicMock()
    connections.get_connection.return_value = connection

    manager = SchemaManager.__new__(SchemaManager)
    manager.connection_manager = connections
    manager.config_manager = MagicMock(spec=ConfigurationManager)
    manager.schema_prefix = "RAG"
    return manager, cursor


def test_a_stored_text_column_is_reported_as_deprecated():
    """The key is named, and so is the table, because the fix is to delete it."""
    manager, _ = manager_with({"text_column": "content"})

    with pytest.warns(DeprecationWarning, match="text_column"):
        config = manager.get_current_schema_config("SourceDocuments")

    assert "text_column" not in config["configuration"]


def test_every_column_key_is_named_in_one_warning():
    """Three stale keys, one warning: the operator deletes them together."""
    manager, _ = manager_with(
        {
            "id_column": "doc_id",
            "text_column": "content",
            "metadata_column": "metadata_json",
            "vector_dimension": 384,
        }
    )

    with pytest.warns(DeprecationWarning) as recorded:
        config = manager.get_current_schema_config("SourceDocuments")

    assert len(recorded) == 1, [str(w.message) for w in recorded]
    message = str(recorded[0].message)
    for key in ("id_column", "text_column", "metadata_column"):
        assert key in message
    # Everything that is not a column name survives untouched.
    assert config["configuration"] == {"vector_dimension": 384}


def test_a_configuration_without_column_keys_warns_about_nothing():
    """The common case stays quiet, or the warning becomes noise to be filtered."""
    manager, _ = manager_with({"vector_data_type": "FLOAT"})

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        config = manager.get_current_schema_config("SourceDocuments")

    assert config["vector_data_type"] == "FLOAT"
