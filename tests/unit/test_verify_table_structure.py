"""`verify_table_structure` asks about the manager's own schema (feature 082).

The default used to be the literal `RAG`, so a deployment configured with any other
schema prefix got `{}` back -- "no such table" -- for a table that was right there.
Feature 082 changed the default to the manager's own prefix and shipped no test, which
is how the change went unnoticed; this is that test.

`{}` still means "cannot tell": callers use this to describe a table, and a catalog
that will not answer is not worth an exception here.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.storage.catalog import CatalogSnapshot
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.table_shape import CatalogUnavailable


def _config(schema_prefix: str = "RAG") -> MagicMock:
    config = MagicMock(spec=ConfigurationManager)
    cloud = MagicMock()
    cloud.vector.vector_dimension = 384
    config.get_cloud_config.return_value = cloud
    config.get.side_effect = lambda key, default=None: (
        "sentence-transformers/all-MiniLM-L6-v2"
        if key == "embedding_model.name"
        else ({} if key == "storage:iris" else default)
    )
    config.get_schema_prefix.return_value = schema_prefix
    return config


def _manager(schema_prefix: str = "RAG") -> SchemaManager:
    return SchemaManager(
        MagicMock(spec=ConnectionManager), _config(schema_prefix), schema_prefix
    )


def _snapshot(schema: str, table: str) -> CatalogSnapshot:
    return CatalogSnapshot(
        schema=schema,
        table=table,
        columns=[
            {"COLUMN_NAME": "doc_id", "DATA_TYPE": "varchar"},
            {"COLUMN_NAME": "embedding", "DATA_TYPE": "varchar"},
        ],
    )


def test_the_default_schema_is_the_managers_own_prefix(monkeypatch):
    manager = _manager("TENANT_A")
    asked: list[tuple[str, str | None]] = []

    def fake_snapshot(table_name, *, schema=None):
        asked.append((table_name, schema))
        return _snapshot(schema or "TENANT_A", table_name)

    monkeypatch.setattr(manager, "_snapshot", fake_snapshot)

    assert manager.verify_table_structure("SourceDocuments") == {
        "doc_id": "varchar",
        "embedding": "varchar",
    }
    # None reaches `_snapshot`, which fills in the manager's prefix. The old default
    # was the literal "RAG", which is what made this wrong for any other prefix.
    assert asked == [("SourceDocuments", None)]


def test_a_prefixed_manager_reads_its_own_schema():
    """End to end through `_snapshot`: the reader is asked about TENANT_A."""
    manager = _manager("TENANT_A")
    reader = MagicMock()
    reader.snapshot.return_value = _snapshot("TENANT_A", "SourceDocuments")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "iris_vector_rag.storage.schema_manager.IRISCatalogReader",
            lambda connection: reader,
        )
        assert manager.verify_table_structure("SourceDocuments")

    reader.snapshot.assert_called_once_with("TENANT_A", "SourceDocuments")


def test_an_explicit_schema_still_wins(monkeypatch):
    manager = _manager("TENANT_A")
    monkeypatch.setattr(
        manager,
        "_snapshot",
        lambda table_name, *, schema=None: _snapshot(schema or "", table_name),
    )

    assert manager.verify_table_structure("SourceDocuments", schema_name="RAG")


def test_a_table_that_is_not_there_reports_nothing(monkeypatch):
    manager = _manager()
    monkeypatch.setattr(manager, "_snapshot", lambda *a, **k: None)

    assert manager.verify_table_structure("NoSuchTable") == {}


def test_a_catalog_that_will_not_answer_reports_nothing(monkeypatch):
    """`{}` means "cannot tell"; the caller is describing a table, not asserting one."""
    manager = _manager()

    def unavailable(*_args, **_kwargs):
        raise CatalogUnavailable("connection refused", "RAG", "SourceDocuments")

    monkeypatch.setattr(manager, "_snapshot", unavailable)

    assert manager.verify_table_structure("SourceDocuments") == {}
