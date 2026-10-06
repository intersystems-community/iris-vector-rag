"""Storage correctness against a real IRIS (feature `storage-correctness`).

Each test works in schemas of its own (`SCFIXA`, `SCFIXB`): DDL autocommits, so nothing
here can be rolled back and nothing here touches `RAG`. Runs on the ephemeral container
`iris_target` starts.
"""

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.exceptions import DestructiveMigrationError, RelationNotFoundError
from iris_vector_rag.storage.relation_embedding_store import RelationEmbeddingStore
from iris_vector_rag.storage.schema_manager import SchemaManager

pytestmark = pytest.mark.integration

A, B = "SCFIXA", "SCFIXB"
TABLES = (
    "EntityRelationships",
    "Entities",
    "DocumentChunks",
    "SourceDocuments",
    "SchemaMetadata",
)


def _drop_all(connection):
    cursor = connection.cursor()
    try:
        for schema in (A, B):
            for table in TABLES:
                cursor.execute(f"DROP TABLE IF EXISTS {schema}.{table}")
        connection.commit()
    finally:
        cursor.close()


@pytest.fixture
def stack(iris_connection):
    config = ConfigurationManager()
    connections = ConnectionManager(config)
    _drop_all(iris_connection)
    yield config, connections
    _drop_all(iris_connection)


def _manager(stack, prefix):
    config, connections = stack
    return SchemaManager(connections, config, schema_prefix=prefix)


def _scalar(connection, sql, params=None):
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params or [])
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        cursor.close()


def _run(connection, sql, params=None):
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params or [])
        connection.commit()
    finally:
        cursor.close()


def _populate(connection, manager, prefix):
    assert manager.ensure_table_schema("SourceDocuments")
    assert manager.ensure_table_schema("Entities")
    _run(
        connection,
        f"INSERT INTO {prefix}.SourceDocuments (doc_id, text_content) VALUES (?, ?)",
        ["d1", "hello"],
    )
    _run(
        connection,
        f"INSERT INTO {prefix}.Entities (entity_id, entity_name, entity_type, source_doc_id)"
        " VALUES (?, ?, ?, ?)",
        ["e1", "Ada", "PERSON", "d1"],
    )


# --- Story 1: a populated table survives migrate_table ---------------------------------


def test_a_populated_table_survives_migrate_table_without_the_flag(
    stack, iris_connection
):
    manager = _manager(stack, A)
    _populate(iris_connection, manager, A)

    with pytest.raises(DestructiveMigrationError, match="allow_destructive"):
        manager.migrate_table("Entities")
    with pytest.raises(DestructiveMigrationError, match="allow_destructive"):
        manager.migrate_table("SourceDocuments")

    assert _scalar(iris_connection, f"SELECT COUNT(*) FROM {A}.Entities") == 1
    assert _scalar(iris_connection, f"SELECT COUNT(*) FROM {A}.SourceDocuments") == 1


def test_allow_destructive_rebuilds_the_populated_table(stack, iris_connection):
    manager = _manager(stack, A)
    _populate(iris_connection, manager, A)

    assert manager.migrate_table("Entities", allow_destructive=True) is True

    assert _scalar(iris_connection, f"SELECT COUNT(*) FROM {A}.Entities") == 0


def test_an_empty_table_migrates_without_the_flag_and_stays_in_its_schema(
    stack, iris_connection
):
    manager = _manager(stack, A)
    assert manager.ensure_table_schema("SourceDocuments")

    assert manager.migrate_table("Entities") is True

    assert manager.table_exists("Entities") is True
    assert manager.table_exists(f"{A}.Entities") is True
    # the bare name was never created in the connecting user's schema
    assert (
        _scalar(
            iris_connection,
            "SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES "
            "WHERE UPPER(TABLE_NAME) = 'ENTITIES' AND UPPER(TABLE_SCHEMA) = 'SQLUSER'",
        )
        == 0
    )


def test_table_exists_honours_the_managers_schema(stack, iris_connection):
    a, b = _manager(stack, A), _manager(stack, B)
    assert a.ensure_table_schema("SourceDocuments")
    assert a.table_exists("SourceDocuments") is True
    assert b.table_exists("SourceDocuments") is False


# --- Story 2: index ensuring is schema-scoped and reports truth ------------------------


def _indexes(connection, schema, table):
    cursor = connection.cursor()
    try:
        cursor.execute(
            "SELECT INDEX_NAME FROM INFORMATION_SCHEMA.INDEXES "
            "WHERE UPPER(TABLE_SCHEMA) = UPPER(?) AND UPPER(TABLE_NAME) = UPPER(?)",
            [schema, table],
        )
        return {str(r[0]).upper() for r in cursor.fetchall()}
    finally:
        cursor.close()


def test_two_schemas_with_the_same_table_and_index_name_both_get_their_index(
    stack, iris_connection
):
    a, b = _manager(stack, A), _manager(stack, B)
    for manager in (a, b):
        assert manager.ensure_table_schema("SourceDocuments")
        # the migrated Entities table carries an embedding column; the standard one does not
        assert manager.migrate_table("Entities") is True

    assert a.ensure_all_vector_indexes() == [
        "idx_SourceDocuments_embedding",
        "idx_Entities_embedding",
    ]
    assert b.ensure_all_vector_indexes() == [
        "idx_SourceDocuments_embedding",
        "idx_Entities_embedding",
    ]

    for schema in (A, B):
        assert "IDX_SOURCEDOCUMENTS_EMBEDDING" in _indexes(
            iris_connection, schema, "SourceDocuments"
        )
        assert "IDX_ENTITIES_EMBEDDING" in _indexes(iris_connection, schema, "Entities")


def test_the_standard_entities_table_has_no_vector_column_so_no_index(
    stack, iris_connection
):
    manager = _manager(stack, A)
    assert manager.ensure_table_schema("SourceDocuments")
    assert manager.ensure_table_schema("Entities")
    assert manager.ensure_all_vector_indexes() == ["idx_SourceDocuments_embedding"]


def test_a_missing_table_makes_ensure_all_vector_indexes_raise(stack, iris_connection):
    manager = _manager(stack, A)
    assert manager.ensure_table_schema("SourceDocuments")  # no Entities

    from iris_vector_rag.exceptions import SchemaIndexError

    with pytest.raises(SchemaIndexError, match="idx_Entities_embedding"):
        manager.ensure_all_vector_indexes()


def test_ensure_index_really_creates_a_plain_index(stack, iris_connection):
    manager = _manager(stack, A)
    assert manager.ensure_table_schema("SourceDocuments")
    assert manager.ensure_table_schema("Entities")
    assert manager.ensure_table_schema("EntityRelationships")
    config = {
        "name": "idx_sc_rel_type",
        "table": "EntityRelationships",
        "columns": ["relationship_type"],
    }

    assert manager._ensure_index(config) is True
    assert "IDX_SC_REL_TYPE" in _indexes(iris_connection, A, "EntityRelationships")
    assert manager._ensure_index(config) is True  # idempotent


def test_new_tables_have_no_text_columns(stack, iris_connection):
    manager = _manager(stack, A)
    assert manager.migrate_table("SourceDocuments") is True
    assert manager.migrate_table("Entities") is True
    assert manager.migrate_table("EntityRelationships") is True
    text_columns = _scalar(
        iris_connection,
        "SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE UPPER(TABLE_SCHEMA) = UPPER(?) "
        "AND UPPER(TABLE_NAME) IN ('ENTITIES', 'ENTITYRELATIONSHIPS') "
        "AND UPPER(DATA_TYPE) IN ('TEXT', 'LONGVARCHAR')",
        [A],
    )
    assert text_columns == 0


# --- Story 3: the relation embedding store ---------------------------------------------


def test_relation_store_raises_on_a_missing_id_and_stores_a_present_one(
    stack, iris_connection
):
    config, connections = stack
    manager = _manager(stack, A)
    for table in ("SourceDocuments", "Entities", "EntityRelationships"):
        assert manager.ensure_table_schema(table)
    _populate(iris_connection, manager, A)
    _run(
        iris_connection,
        f"INSERT INTO {A}.Entities (entity_id, entity_name, entity_type, source_doc_id)"
        " VALUES (?, ?, ?, ?)",
        ["e2", "Bob", "PERSON", "d1"],
    )
    _run(
        iris_connection,
        f"INSERT INTO {A}.EntityRelationships "
        "(relationship_id, source_entity_id, target_entity_id, relationship_type) "
        "VALUES (?, ?, ?, ?)",
        ["r1", "e1", "e2", "KNOWS"],
    )

    store = RelationEmbeddingStore(connections, config, schema_prefix=A)
    assert store.table == f"{A}.EntityRelationships"
    store._ensure_schema()

    with pytest.raises(RelationNotFoundError, match="no-such-id"):
        store.embed_and_store("no-such-id", "KNOWS", "Ada", "Bob")
    assert store.count_embedded() == 0

    store.embed_and_store("r1", "KNOWS", "Ada", "Bob", "friends")
    assert store.count_embedded() == 1
