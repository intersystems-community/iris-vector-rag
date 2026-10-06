"""An entity table from an older release gets `confidence` added, and keeps its rows.

The dev database that motivated this had `RAG.Entities` with columns id, entity_name,
entity_type, properties, created_at, test_run_id, entity_id, source_doc_id, doc_id and
description: no `confidence`. Both entity writers INSERT it, so every write failed with
SQLCODE -29. Works in a schema of its own (`LEGENT`); DDL autocommits.
"""

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Entity, Relationship
from iris_vector_rag.services.batch_entity_processor import BatchEntityProcessor
from iris_vector_rag.services.storage import EntityStorageAdapter
from iris_vector_rag.storage.schema_manager import SchemaManager

pytestmark = pytest.mark.integration

SCHEMA = "LEGENT"
TABLES = ("EntityRelationships", "Entities", "DocumentChunks", "SourceDocuments")

LEGACY_IDENTITY = """
    CREATE TABLE {schema}.Entities (
        id INTEGER IDENTITY PRIMARY KEY,
        entity_name VARCHAR(255), entity_type VARCHAR(100), properties VARCHAR(4000),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, test_run_id VARCHAR(255),
        entity_id VARCHAR(255), source_doc_id VARCHAR(255), doc_id VARCHAR(255),
        description VARCHAR(4000)
    )
"""


def _run(connection, sql, params=None):
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params or [])
        connection.commit()
    finally:
        cursor.close()


def _scalar(connection, sql, params=None):
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params or [])
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        cursor.close()


def _columns(connection):
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = 'Entities'",
            [SCHEMA],
        )
        return {str(row[0]).lower() for row in cursor.fetchall()}
    finally:
        cursor.close()


def _drop_all(connection):
    for table in TABLES:
        _run(connection, f"DROP TABLE IF EXISTS {SCHEMA}.{table}")


@pytest.fixture
def legacy(iris_connection):
    config = ConfigurationManager()
    config._config.setdefault("entity_extraction", {})["storage"] = {
        "entities_table": f"{SCHEMA}.Entities",
        "relationships_table": f"{SCHEMA}.EntityRelationships",
    }
    connections = ConnectionManager(config)
    _drop_all(iris_connection)
    assert SchemaManager(connections, config, schema_prefix=SCHEMA).ensure_table_schema(
        "SourceDocuments"
    )
    _run(
        iris_connection,
        "INSERT INTO LEGENT.SourceDocuments (doc_id, text_content) VALUES ('d1', 'text')",
    )
    _run(iris_connection, LEGACY_IDENTITY.format(schema=SCHEMA))
    assert "confidence" not in _columns(iris_connection)
    yield config, connections
    _drop_all(iris_connection)


def _entity(n):
    return Entity(
        id=f"e{n}",
        text=f"name{n}",
        entity_type="PERSON",
        confidence=0.8,
        start_offset=0,
        end_offset=1,
        source_document_id="d1",
    )


def test_the_adapter_adds_confidence_and_writes(legacy, iris_connection):
    config, connections = legacy
    adapter = EntityStorageAdapter(connections, config._config)

    assert adapter.store_entity(_entity(1)) is True

    assert "confidence" in _columns(iris_connection)
    assert _scalar(
        iris_connection,
        f"SELECT confidence FROM {SCHEMA}.Entities WHERE entity_id = 'e1'",
    ) == pytest.approx(0.8)


def test_the_batch_processor_adds_confidence_and_writes(legacy, iris_connection):
    config, connections = legacy
    processor = BatchEntityProcessor(connections, config._config)

    result = processor.store_entities_batch([_entity(1), _entity(2)])

    assert result["validation_passed"] is True, result
    assert result["entities_stored"] == 2
    assert "confidence" in _columns(iris_connection)


def test_existing_rows_survive_and_read_back(legacy, iris_connection):
    config, connections = legacy
    _run(
        iris_connection,
        f"INSERT INTO {SCHEMA}.Entities "
        "(entity_name, entity_type, entity_id, source_doc_id, description) "
        "VALUES (?, ?, ?, ?, ?)",
        ["Old", "PERSON", "old", "d1", "written by the old release"],
    )
    adapter = EntityStorageAdapter(connections, config._config)

    assert adapter.store_entity(_entity(1)) is True

    assert _scalar(iris_connection, f"SELECT COUNT(*) FROM {SCHEMA}.Entities") == 2
    assert (
        _scalar(
            iris_connection,
            f"SELECT description FROM {SCHEMA}.Entities WHERE entity_id = 'old'",
        )
        == "written by the old release"
    )
    # The old row is not rewritten: its confidence stays unknown (NULL).
    assert (
        _scalar(
            iris_connection,
            f"SELECT confidence FROM {SCHEMA}.Entities WHERE entity_id = 'old'",
        )
        is None
    )
    stored = adapter.get_entities_by_document("d1")
    assert {e.id for e in stored} == {"old", "e1"}


def test_relationships_are_written_too(legacy, iris_connection):
    config, connections = legacy
    adapter = EntityStorageAdapter(connections, config._config)
    assert adapter.store_entity(_entity(1)) and adapter.store_entity(_entity(2))

    assert (
        adapter.store_relationship(
            Relationship(
                id="r1",
                source_entity_id="e1",
                target_entity_id="e2",
                relationship_type="KNOWS",
                confidence=0.7,
                source_document_id="d1",
            )
        )
        is True
    )
