"""Entity and relationship types keep their case, on a real IRIS.

Tables are made (or repaired) through the library in a private schema, so nothing here
touches RAG.*. Without COLLATE EXACT, `GROUP BY entity_type` returns `DISEASE` for a
stored `Disease`.
"""

import time

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.storage.catalog import IRISCatalogReader
from iris_vector_rag.storage.collation import collation_of, ensure_exact_collation
from iris_vector_rag.storage.schema_manager import SchemaManager
from tests.support.iris_environment import connect

SCHEMA = "COLLTEST"


@pytest.fixture
def connection_manager(iris_connection):
    cursor = iris_connection.cursor()
    for table in ("EntityRelationships", "Entities"):
        cursor.execute(f"DROP TABLE IF EXISTS {SCHEMA}.{table}")
    iris_connection.commit()
    manager = ConnectionManager(ConfigurationManager())
    yield manager
    cursor = iris_connection.cursor()
    iris_connection.rollback()
    for table in ("EntityRelationships", "Entities"):
        cursor.execute(f"DROP TABLE IF EXISTS {SCHEMA}.{table}")
    iris_connection.commit()


def schema_manager(connection_manager):
    return SchemaManager(
        connection_manager,
        ConfigurationManager(),
        schema_prefix=SCHEMA,
        base_embedding_dimension=384,
    )


def collation(connection, table, column):
    snapshot = IRISCatalogReader(connection).snapshot(SCHEMA, table)
    value = collation_of(snapshot, column)
    connection.commit()
    return value


def groups(connection, table, column):
    cursor = connection.cursor()
    cursor.execute(f"SELECT {column}, COUNT(*) FROM {SCHEMA}.{table} GROUP BY {column}")
    rows = {str(r[0]): int(r[1]) for r in cursor.fetchall()}
    connection.commit()
    return rows


class TestNewTables:
    def test_group_by_returns_the_stored_case(
        self, iris_connection, connection_manager
    ):
        manager = schema_manager(connection_manager)
        assert manager.ensure_table_schema("Entities")
        assert manager.ensure_table_schema("EntityRelationships")

        cursor = iris_connection.cursor()
        cursor.execute(
            f"INSERT INTO {SCHEMA}.Entities (entity_id, entity_name, entity_type) "
            "VALUES ('e1', 'Diabetes', 'Disease')"
        )
        cursor.execute(
            f"INSERT INTO {SCHEMA}.Entities (entity_id, entity_name, entity_type) "
            "VALUES ('e2', 'Flu', 'Disease')"
        )
        cursor.execute(
            f"INSERT INTO {SCHEMA}.EntityRelationships "
            "(relationship_id, source_entity_id, target_entity_id, relationship_type) "
            "VALUES ('r1', 'e1', 'e2', 'Treats')"
        )
        iris_connection.commit()

        assert groups(iris_connection, "Entities", "entity_type") == {"Disease": 2}
        assert groups(iris_connection, "EntityRelationships", "relationship_type") == {
            "Treats": 1
        }
        assert collation(iris_connection, "Entities", "entity_type") == "EXACT"
        assert (
            collation(iris_connection, "EntityRelationships", "relationship_type")
            == "EXACT"
        )
        # Other columns keep the default: lookups by name still fold case.
        assert collation(iris_connection, "Entities", "entity_name") == "SQLUPPER"


LEGACY_ENTITIES = f"""
CREATE TABLE {SCHEMA}.Entities (
    entity_id VARCHAR(255) NOT NULL,
    entity_name VARCHAR(500) NOT NULL,
    entity_type VARCHAR(100) NOT NULL,
    description VARCHAR(2000),
    confidence DOUBLE DEFAULT 1.0,
    source_doc_id VARCHAR(255),
    created_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (entity_id)
)
"""


class TestLegacyTable:
    @pytest.fixture
    def legacy(self, iris_connection, connection_manager):
        cursor = iris_connection.cursor()
        cursor.execute(LEGACY_ENTITIES)
        cursor.execute(
            f"CREATE INDEX idx_entities_type ON {SCHEMA}.Entities (entity_type)"
        )
        for i, kind in enumerate(["Disease", "Disease", "disease", "Drug"]):
            cursor.execute(
                f"INSERT INTO {SCHEMA}.Entities (entity_id, entity_name, entity_type) "
                "VALUES (?, ?, ?)",
                [f"e{i}", f"name{i}", kind],
            )
        iris_connection.commit()
        return connection_manager

    def test_default_collation_is_the_bug(self, iris_connection, legacy):
        assert collation(iris_connection, "Entities", "entity_type") == "SQLUPPER"
        assert groups(iris_connection, "Entities", "entity_type") == {
            "DISEASE": 3,
            "DRUG": 1,
        }

    def test_ensure_repairs_it_without_losing_data_or_the_index(
        self, iris_connection, legacy
    ):
        assert schema_manager(legacy).ensure_table_schema("Entities")

        assert collation(iris_connection, "Entities", "entity_type") == "EXACT"
        assert groups(iris_connection, "Entities", "entity_type") == {
            "Disease": 2,
            "disease": 1,
            "Drug": 1,
        }
        cursor = iris_connection.cursor()
        cursor.execute(
            f"SELECT entity_id FROM {SCHEMA}.Entities WHERE entity_type = 'disease'"
        )
        assert [r[0] for r in cursor.fetchall()] == ["e2"]
        cursor.execute(f"SELECT COUNT(*) FROM {SCHEMA}.Entities")
        assert cursor.fetchone()[0] == 4
        iris_connection.commit()
        assert "IDX_ENTITIES_TYPE" in IRISCatalogReader(iris_connection).index_names(
            SCHEMA, "Entities"
        )
        # NOT NULL survived the ALTER.
        with pytest.raises(Exception, match="required"):
            cursor.execute(
                f"INSERT INTO {SCHEMA}.Entities (entity_id, entity_name) VALUES ('x', 'y')"
            )
        iris_connection.rollback()

    def test_ensure_is_idempotent(self, iris_connection, legacy):
        assert ensure_exact_collation(legacy, SCHEMA, "Entities") == {
            "entity_type": 100
        }
        assert ensure_exact_collation(legacy, SCHEMA, "Entities") == {}

    def test_a_table_that_does_not_exist_is_left_alone(self, connection_manager):
        assert ensure_exact_collation(connection_manager, SCHEMA, "Entities") == {}

    def test_a_blocked_alter_gives_up_and_a_retry_repairs(
        self, iris_connection, iris_target, legacy, caplog
    ):
        """Another connection's table lock must not hang the caller or change the table later."""
        blocker = connect(iris_target)
        try:
            blocker.cursor().execute(f"LOCK TABLE {SCHEMA}.Entities IN EXCLUSIVE MODE")

            started = time.time()
            with caplog.at_level("WARNING"):
                altered = ensure_exact_collation(legacy, SCHEMA, "Entities", timeout=3)
            assert altered == {}
            assert time.time() - started < 20
            assert "left at the default" in caplog.text
        finally:
            blocker.rollback()
            blocker.close()

        assert collation(iris_connection, "Entities", "entity_type") == "SQLUPPER"
        assert ensure_exact_collation(legacy, SCHEMA, "Entities") == {
            "entity_type": 100
        }
        assert collation(iris_connection, "Entities", "entity_type") == "EXACT"
        assert groups(iris_connection, "Entities", "entity_type") == {
            "Disease": 2,
            "disease": 1,
            "Drug": 1,
        }
