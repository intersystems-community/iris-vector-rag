"""COLLATE EXACT on the type columns: DDL text and catalog parsing (no database).

The behaviour against a real IRIS is in tests/integration/test_exact_collation_iris.py.
"""

from unittest.mock import MagicMock

import pytest

from iris_vector_rag.storage.catalog import CatalogSnapshot
from iris_vector_rag.storage.collation import (
    EXACT_CASE_COLUMNS,
    collation_of,
    columns_needing_exact,
)
from iris_vector_rag.storage.schema_manager import SchemaManager


def manager():
    return SchemaManager(MagicMock(), MagicMock(), base_embedding_dimension=384)


def executed_sql(create):
    cursor = MagicMock()
    create(cursor)
    return "\n".join(str(c.args[0]) for c in cursor.execute.call_args_list if c.args)


def params(collation):
    return (
        "CALCSELECTIVITY,1,CAPTION,,COLLATION,"
        f"{collation},COLLATIONTARGET,,CONTENT,STRING,MAXLEN,100"
    )


def snapshot(collation, data_type="varchar", length=100):
    return CatalogSnapshot(
        schema="RAG",
        table="Entities",
        columns=[
            {
                "COLUMN_NAME": "entity_type",
                "DATA_TYPE": data_type,
                "CHARACTER_MAXIMUM_LENGTH": length,
            },
            {
                "COLUMN_NAME": "entity_name",
                "DATA_TYPE": "varchar",
                "CHARACTER_MAXIMUM_LENGTH": 500,
            },
        ],
        properties=[
            {
                "SqlFieldName": "entity_type",
                "Type": "%String",
                "Parameters": params(collation),
            },
            {
                "SqlFieldName": "entity_name",
                "Type": "%String",
                "Parameters": params(""),
            },
        ],
    )


class TestCanonicalDdl:
    @pytest.mark.parametrize(
        "create, column",
        [
            (lambda m, c: m._create_entities_table(c), "entity_type"),
            (lambda m, c: m._create_entity_relationships_table(c), "relationship_type"),
        ],
    )
    def test_create_declares_the_type_column_exact(self, create, column):
        sql = executed_sql(lambda cursor: create(manager(), cursor))
        assert f"{column} VARCHAR" in sql
        line = next(l for l in sql.splitlines() if l.strip().startswith(column))
        assert "COLLATE EXACT" in line

    def test_migrate_paths_declare_the_type_columns_exact(self):
        expected = {"vector_dimension": 384, "vector_data_type": "FLOAT"}
        sql = executed_sql(
            lambda cursor: (
                manager()._migrate_entities_table(cursor, expected, False),
                manager()._migrate_entity_relationships_table(cursor, expected, False),
            )
        )
        for column in ("entity_type", "relationship_type"):
            line = next(l for l in sql.splitlines() if l.strip().startswith(column))
            assert "COLLATE EXACT" in line

    def test_only_the_two_type_columns_are_exact(self):
        assert {t: tuple(c) for t, c in EXACT_CASE_COLUMNS.items()} == {
            "Entities": ("entity_type",),
            "EntityRelationships": ("relationship_type",),
        }
        sql = executed_sql(lambda cursor: manager()._create_entities_table(cursor))
        assert sql.count("COLLATE EXACT") == 1


class TestCatalogReading:
    def test_empty_parameter_is_the_default(self):
        assert collation_of(snapshot(""), "entity_type") == "SQLUPPER"

    def test_exact_is_read_from_the_parameters(self):
        assert collation_of(snapshot("EXACT"), "entity_type") == "EXACT"

    def test_unknown_column_is_none(self):
        assert collation_of(snapshot(""), "nope") is None

    def test_default_column_needs_the_alter_with_its_length(self):
        assert columns_needing_exact(snapshot(""), ["entity_type"]) == {
            "entity_type": 100
        }

    def test_exact_column_needs_nothing(self):
        assert columns_needing_exact(snapshot("EXACT"), ["entity_type"]) == {}

    def test_missing_and_non_varchar_columns_are_left_alone(self):
        assert (
            columns_needing_exact(snapshot("", data_type="integer"), ["entity_type"])
            == {}
        )
        assert columns_needing_exact(snapshot(""), ["absent"]) == {}
