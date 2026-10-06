#!/usr/bin/env python3
"""
Regression tests for three SchemaManager bugs:

1. Schema validation ran again on every call (no caching).
2. ``base_embedding_dimension`` was missing when the class-level config flag was set.
3. The Entities foreign key referenced the wrong SourceDocuments column.

The validation cache is per instance by design (``__init__``: "not class-level, so
different prefixes don't collide"), so the first test asserts that an instance
validates a table once however often it is asked, and that two instances with
different prefixes do not share verdicts. Only the configuration-loaded flag is
class-level. The manager takes a connection manager first and a configuration
manager second; the database is a mock here because these are unit-level checks of
the manager's own bookkeeping.
"""

from unittest.mock import MagicMock, patch

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.storage.schema_manager import SchemaManager

CONFIG = """
database:
  iris:
    host: "localhost"
    port: 1972
    namespace: "USER"
    username: "_SYSTEM"
    password: "SYS"

embedding_model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
"""


@pytest.fixture
def config_manager(tmp_path):
    path = tmp_path / "test_config.yaml"
    path.write_text(CONFIG)
    return ConfigurationManager(str(path))


@pytest.fixture(autouse=True)
def fresh_class_state():
    """`_config_loaded` is class-level; each test starts and leaves it unset."""
    SchemaManager._config_loaded = False
    yield
    SchemaManager._config_loaded = False


def make_manager(config_manager, **kwargs):
    kwargs.setdefault("base_embedding_dimension", 384)
    return SchemaManager(MagicMock(), config_manager, **kwargs)


class TestSchemaManagerCachingPerformance:
    def test_validation_cache_is_per_instance(self, config_manager):
        """Instances with different prefixes must not share schema verdicts."""
        first = make_manager(config_manager)
        second = make_manager(config_manager, schema_prefix="OTHER")

        assert first._schema_validation_cache is not second._schema_validation_cache
        first._schema_validation_cache["SourceDocuments:default"] = False
        assert "SourceDocuments:default" not in second._schema_validation_cache

    def test_needs_migration_validates_once_per_instance(self, config_manager):
        """A second ask for the same table is answered from the cache."""
        manager = make_manager(config_manager)

        with patch.object(
            manager, "get_current_schema_config", return_value=None
        ) as current:
            result1 = manager.needs_migration("SourceDocuments")
            manager._dimension_cache = {}
            result2 = manager.needs_migration("SourceDocuments")

        assert result1 == result2 is True
        assert current.call_count == 1
        assert "SourceDocuments:default" in manager._schema_validation_cache


class TestSchemaManagerAttributeInitialization:
    def test_base_embedding_dimension_set_when_config_already_loaded(
        self, config_manager
    ):
        """BUG #2: the branch taken when `_config_loaded` is set skipped the model."""
        first = make_manager(config_manager)
        assert SchemaManager._config_loaded
        assert first.base_embedding_model
        assert first.base_embedding_dimension == 384

        second = make_manager(config_manager)
        assert second.base_embedding_model == first.base_embedding_model
        assert second.base_embedding_dimension == 384
        assert second._model_dimensions[second.base_embedding_model] == 384
        assert "SourceDocuments" in second._table_configs


class TestSchemaManagerForeignKeyConstraints:
    @staticmethod
    def created_sql(migrate):
        cursor = MagicMock()
        migrate(cursor)
        return [
            str(call.args[0])
            for call in cursor.execute.call_args_list
            if call.args and "CREATE TABLE" in str(call.args[0])
        ]

    def test_entities_table_foreign_key_references_doc_id(self, config_manager):
        """BUG #3: the FK named SourceDocuments(id), a column that does not exist."""
        manager = make_manager(config_manager)
        expected = {"vector_dimension": 384, "vector_data_type": "FLOAT"}

        creates = self.created_sql(
            lambda cursor: manager._migrate_entities_table(cursor, expected, False)
        )

        assert len(creates) == 1
        assert "REFERENCES RAG.SourceDocuments(doc_id)" in creates[0]
        assert "REFERENCES RAG.SourceDocuments(id)" not in creates[0]

    def test_source_documents_table_has_doc_id_primary_key(self, config_manager):
        """Entities.source_doc_id references the document table's `doc_id`."""
        manager = make_manager(config_manager)
        expected = {"vector_dimension": 384, "vector_data_type": "FLOAT"}

        with patch.object(manager, "_snapshot", return_value=None):
            creates = self.created_sql(
                lambda cursor: manager._migrate_source_documents_table(
                    cursor, expected, False
                )
            )

        assert len(creates) == 1
        definition = creates[0]
        import re

        assert re.search(r"doc_id\s+VARCHAR\(\d+\)\s+PRIMARY KEY", definition)
        assert not re.search(r"(?<![a-z_])id\s+VARCHAR", definition)


class TestSchemaManagerIntegrationScenarios:
    def test_repeated_needs_migration_validates_once(self, config_manager):
        """A long-lived manager (one pipeline, many batches) validates a table once."""
        manager = make_manager(config_manager)

        with patch.object(
            manager, "get_current_schema_config", return_value=None
        ) as current:
            for _ in range(100):
                manager.needs_migration("Entities")

        assert current.call_count == 1

    def test_config_loaded_once_across_instances(self, config_manager):
        """`_config_loaded` stops later instances from loading configuration again."""
        with patch.object(SchemaManager, "_load_and_validate_config") as load:
            for _ in range(10):
                make_manager(config_manager)

        assert load.call_count == 1
