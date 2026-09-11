"""The store behaves the same on every recorded shape (feature 082, US1).

Each test builds a real `IRISVectorStore` on a `TableShapeResolver` over the six
recorded catalog snapshots, so the shape under test is what IRIS actually reported
for that DDL -- no database, and no hand-written catalog query text (FR-015).

The cursor here answers data queries only. If a test sees a catalog query in
`cursor.sql`, that is the failure: after this feature the only catalog reader is
`IRISCatalogReader`, and the store asks it once (FR-005).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Document
from iris_vector_rag.exceptions import VectorStoreDataError
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.shape_resolver import TableShapeResolver
from iris_vector_rag.storage.table_shape import (
    ColumnRole,
    IdentifierNotWritable,
    NotVectorSearchable,
)
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from tests.fixtures.shapes.ddl import SHAPE_SCHEMA, SHAPE_TABLES, VECTOR_DIMENSION

DIM = VECTOR_DIMENSION
EMBEDDING = [0.1] * DIM


class FakeCursor:
    """Answers the data queries the store sends, and records every one of them."""

    rowcount = 1

    def __init__(self, *, ranked_rows=None, metadata_rows=None):
        self.executed: list[tuple[str, object]] = []
        self._ranked_rows = list(ranked_rows or [])
        self._metadata_rows = list(metadata_rows or [])
        self._result: list = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        upper = " ".join(str(sql).upper().split())
        if "INFORMATION_SCHEMA" in upper or "%DICTIONARY" in upper:
            raise AssertionError(f"the store read the catalog itself: {sql}")
        if "VECTOR_DOT_PRODUCT" in upper:
            self._result = list(self._ranked_rows)
        elif "COUNT(*)" in upper:
            self._result = [(0,)]
        elif upper.startswith("SELECT") and " IN (" in upper:
            self._result = list(self._metadata_rows)
        else:
            self._result = []

    def fetchall(self):
        return self._result

    def fetchone(self):
        return self._result[0] if self._result else None

    def close(self):
        pass

    # -- assertions ----------------------------------------------------------

    @property
    def sql(self) -> list[str]:
        return [sql for sql, _ in self.executed]

    def only(self, needle: str) -> str:
        """The one statement containing `needle`; fails if there is not exactly one."""
        matches = [sql for sql in self.sql if needle.lower() in sql.lower()]
        assert len(matches) == 1, f"{needle!r} in {len(matches)} statements: {self.sql}"
        return matches[0]


def insert_columns(sql: str) -> list[str]:
    """The column list of an INSERT, as written."""
    inner = sql[sql.index("(") + 1 : sql.index(")")]
    return [column.strip() for column in inner.split(",")]


def make_store(shape: str, cursor: FakeCursor, recorded_catalog) -> IRISVectorStore:
    """A store pointed at one recorded shape, with the catalog answered from disk."""
    table = f"{SHAPE_SCHEMA}.{SHAPE_TABLES[shape]}"

    connection = MagicMock()
    connection.cursor.return_value = cursor
    connections = MagicMock(spec=ConnectionManager)
    connections.get_connection.return_value = connection

    config = MagicMock(spec=ConfigurationManager)

    def config_get(key, default=None):
        if key == "storage:iris":
            return {"table_name": table}
        if key == "embedding_model.name":
            return "sentence-transformers/all-MiniLM-L6-v2"
        return default

    config.get.side_effect = config_get
    config.get_embedding_config.return_value = {"dimension": DIM}

    schema_manager = MagicMock(spec=SchemaManager)
    schema_manager.get_vector_dimension.return_value = DIM
    # Deliberately wrong: the database, not the configuration, decides which column
    # fills which role (FR-001). If any of these were consulted the tests would fail.
    schema_manager.get_current_schema_config.return_value = {
        "id_column": "wrong_id",
        "text_column": "wrong_text",
        "metadata_column": "wrong_metadata",
    }

    return IRISVectorStore(
        connections,
        config,
        schema_manager=schema_manager,
        shapes=TableShapeResolver(recorded_catalog),
    )


def document(doc_id="doc-1", text="the text", **metadata) -> Document:
    return Document(id=doc_id, page_content=text, metadata=metadata)


# --- T017: one text column, whatever the shape --------------------------------


def test_mixed_shape_writes_the_canonical_text_column(recorded_catalog):
    cursor = FakeCursor()
    store = make_store("mixed", cursor, recorded_catalog)

    store._store_documents([document()], [EMBEDDING])

    columns = insert_columns(cursor.only("INSERT INTO"))
    assert "text_content" in columns
    # Not both: two columns holding the same fact can disagree after an update, and
    # FR-003 makes the canonical column the one that counts.
    assert "content" not in columns


def test_mixed_shape_reads_the_column_it_wrote(recorded_catalog):
    """The defect this feature exists to remove: write text_content, read content."""
    write_cursor = FakeCursor()
    write_store = make_store("mixed", write_cursor, recorded_catalog)
    write_store._store_documents([document()], [EMBEDDING])

    read_cursor = FakeCursor(
        ranked_rows=[("doc-1", "the text", None, 0.9)],
        metadata_rows=[("doc-1", "{}")],
    )
    read_store = make_store("mixed", read_cursor, recorded_catalog)
    results = read_store.similarity_search_by_embedding(EMBEDDING, 5)

    written = write_store.shape().column_for(ColumnRole.TEXT).name
    ranked = read_cursor.only("VECTOR_DOT_PRODUCT")
    assert written == "text_content"
    assert f"SELECT TOP 5 doc_id, {written}" in ranked
    assert results[0][0].page_content == "the text"


def test_mixed_shape_falls_back_per_row_to_the_legacy_column(recorded_catalog):
    """FR-004: one presented form, even where only the old column was populated."""
    cursor = FakeCursor(
        ranked_rows=[
            ("new-row", "written by 0.15", "written by 0.13", 0.9),
            ("old-row", None, "written by 0.13", 0.8),
        ],
        metadata_rows=[("new-row", "{}"), ("old-row", "{}")],
    )
    store = make_store("mixed", cursor, recorded_catalog)

    results = store.similarity_search_by_embedding(EMBEDDING, 5)

    ranked = cursor.only("VECTOR_DOT_PRODUCT")
    assert "text_content" in ranked and "content" in ranked
    assert [doc.page_content for doc, _ in results] == [
        "written by 0.15",
        "written by 0.13",
    ]


def test_a_single_family_shape_selects_one_text_column(recorded_catalog):
    """No fallback column where the table has no second spelling of the same fact."""
    cursor = FakeCursor(
        ranked_rows=[("doc-1", "the text", 0.9)], metadata_rows=[("doc-1", "{}")]
    )
    store = make_store("fixture", cursor, recorded_catalog)

    results = store.similarity_search_by_embedding(EMBEDDING, 5)

    assert "SELECT TOP 5 doc_id, content," in cursor.only("VECTOR_DOT_PRODUCT")
    assert results[0][0].page_content == "the text"


# --- T017: stream metadata columns are filtered, not silently dropped ---------


def test_a_stream_metadata_column_is_filtered_with_substring(recorded_catalog):
    """FR-009. `LIKE` on a LONGVARCHAR column matches nothing, silently.

    The shape with a stream metadata column (`metadata VARCHAR(MAX)`) is the one
    without an embedding column, so this is asserted on the expression the filter
    is built from rather than through a vector search.
    """
    cursor = FakeCursor()
    store = make_store("legacy_id", cursor, recorded_catalog)
    metadata = store.shape().column_for(ColumnRole.METADATA)

    assert metadata.filterable_with_like is False
    assert store._metadata_filter_column_expr(metadata) == (
        "SUBSTRING(metadata,1,32000)"
    )
    assert cursor.sql == []


def test_a_string_metadata_column_is_filtered_directly(recorded_catalog):
    cursor = FakeCursor(
        ranked_rows=[("doc-1", "the text", 0.9)],
        metadata_rows=[("doc-1", '{"collection_id": "acme"}')],
    )
    store = make_store("canonical", cursor, recorded_catalog)
    metadata = store.shape().column_for(ColumnRole.METADATA)

    assert metadata.filterable_with_like is True
    assert store._metadata_filter_column_expr(metadata) == "metadata"

    store.similarity_search_by_embedding(EMBEDDING, 5, {"collection_id": "acme"})
    ranked = cursor.only("VECTOR_DOT_PRODUCT")
    assert "SUBSTRING" not in ranked
    assert "metadata LIKE" in ranked


# --- T017: the table's own vector type ---------------------------------------


def test_double_precision_shape_carries_its_element_type_into_to_vector(
    recorded_catalog,
):
    """FR-010: the element type comes from the column, not from an environment guess."""
    read_cursor = FakeCursor(
        ranked_rows=[("doc-1", "the text", 0.9)], metadata_rows=[("doc-1", "{}")]
    )
    read_store = make_store("double_vector", read_cursor, recorded_catalog)
    read_store.similarity_search_by_embedding(EMBEDDING, 5)
    assert f"DOUBLE, {DIM}" in read_cursor.only("VECTOR_DOT_PRODUCT")

    write_cursor = FakeCursor()
    write_store = make_store("double_vector", write_cursor, recorded_catalog)
    write_store._store_documents([document()], [EMBEDDING])
    assert f"TO_VECTOR(?, DOUBLE, {DIM})" in write_cursor.only("INSERT INTO")


def test_canonical_shape_carries_float(recorded_catalog):
    cursor = FakeCursor(
        ranked_rows=[("doc-1", "the text", 0.9)], metadata_rows=[("doc-1", "{}")]
    )
    store = make_store("canonical", cursor, recorded_catalog)
    store.similarity_search_by_embedding(EMBEDDING, 5)
    assert f"FLOAT, {DIM}" in cursor.only("VECTOR_DOT_PRODUCT")


def test_the_environment_cannot_override_the_element_type(
    recorded_catalog, monkeypatch
):
    """The fallback that used to turn a mistyped table name into a plausible FLOAT."""
    monkeypatch.setenv("IRIS_VECTOR_DATA_TYPE", "DECIMAL")
    cursor = FakeCursor(
        ranked_rows=[("doc-1", "the text", 0.9)], metadata_rows=[("doc-1", "{}")]
    )
    store = make_store("double_vector", cursor, recorded_catalog)
    store.similarity_search_by_embedding(EMBEDDING, 5)
    assert "DECIMAL" not in cursor.only("VECTOR_DOT_PRODUCT")


# --- T017: a generated identifier cannot be written -------------------------


def test_caller_supplied_identifier_on_a_generated_key_is_refused(recorded_catalog):
    """FR-008, and before any SQL: the error must name the real cause."""
    cursor = FakeCursor()
    store = make_store("legacy_serial", cursor, recorded_catalog)

    with pytest.raises(IdentifierNotWritable) as raised:
        store._store_documents([document()], [EMBEDDING])

    message = str(raised.value)
    assert f"{SHAPE_SCHEMA}.{SHAPE_TABLES['legacy_serial']}" in message
    assert "id" in message
    assert cursor.sql == []


def test_a_writable_identifier_shape_is_not_refused(recorded_catalog):
    cursor = FakeCursor()
    store = make_store("legacy_id", cursor, recorded_catalog)

    assert store._store_documents([document()]) == ["doc-1"]
    insert = cursor.only("INSERT INTO")
    assert "id" in insert and "content" in insert


# --- T018: the dimension check happens before the query ---------------------


def test_a_query_vector_of_the_wrong_length_is_rejected_naming_both_lengths(
    recorded_catalog,
):
    cursor = FakeCursor()
    store = make_store("canonical", cursor, recorded_catalog)

    with pytest.raises(VectorStoreDataError) as raised:
        store.similarity_search_by_embedding([0.1, 0.2, 0.3], 5)

    message = str(raised.value)
    assert "3" in message
    assert str(DIM) in message
    assert f"{SHAPE_SCHEMA}.{SHAPE_TABLES['canonical']}" in message
    assert cursor.sql == []


def test_the_expected_dimension_comes_from_the_column(recorded_catalog):
    cursor = FakeCursor()
    store = make_store("canonical", cursor, recorded_catalog)
    assert store.shape().vector_spec().dimension == DIM


# --- T019: a table with no embedding column is not searchable ---------------


def test_a_table_with_no_embedding_column_is_not_vector_searchable(recorded_catalog):
    """Replaces today's silent FLOAT fallback and the failure a layer away."""
    cursor = FakeCursor()
    store = make_store("legacy_id", cursor, recorded_catalog)

    with pytest.raises(NotVectorSearchable) as raised:
        store.similarity_search_by_embedding(EMBEDDING, 5)

    assert f"{SHAPE_SCHEMA}.{SHAPE_TABLES['legacy_id']}" in str(raised.value)
    assert cursor.sql == []


# --- the rest of the surface, on every shape ---------------------------------


@pytest.mark.parametrize(
    "shape,identifier",
    [
        ("canonical", "doc_id"),
        ("legacy_serial", "id"),
        ("legacy_id", "id"),
        ("fixture", "doc_id"),
        ("mixed", "doc_id"),
        ("double_vector", "doc_id"),
    ],
)
def test_delete_uses_the_shape_identifier(shape, identifier, recorded_catalog):
    cursor = FakeCursor()
    store = make_store(shape, cursor, recorded_catalog)

    store.delete_documents(["a", "b"])

    delete = cursor.only("DELETE FROM")
    assert f"WHERE {identifier} IN (?,?)" in delete


@pytest.mark.parametrize("shape", sorted(SHAPE_TABLES))
def test_fetch_documents_by_ids_needs_no_retry(shape, recorded_catalog):
    """One statement, built from the shape -- not canonical-then-legacy guessing."""
    cursor = FakeCursor()
    store = make_store(shape, cursor, recorded_catalog)

    store.fetch_documents_by_ids(["a"])

    identifier = store.shape().column_for(ColumnRole.IDENTIFIER).name
    text = store.shape().column_for(ColumnRole.TEXT).name
    assert len(cursor.sql) == 1
    assert f"SELECT {identifier}, {text}" in cursor.sql[0]


@pytest.mark.parametrize("shape", sorted(SHAPE_TABLES))
def test_get_all_documents_needs_no_retry(shape, recorded_catalog):
    cursor = FakeCursor()
    store = make_store(shape, cursor, recorded_catalog)

    store.get_all_documents()

    identifier = store.shape().column_for(ColumnRole.IDENTIFIER).name
    assert len(cursor.sql) == 1
    assert f"SELECT {identifier}," in cursor.sql[0]


def test_the_shape_is_resolved_once_per_store(recorded_catalog):
    """SC-002 at the unit level; Phase 4 asserts it over a live catalog."""
    cursor = FakeCursor(
        ranked_rows=[("doc-1", "the text", 0.9)], metadata_rows=[("doc-1", "{}")]
    )
    store = make_store("canonical", cursor, recorded_catalog)

    for _ in range(10):
        store.similarity_search_by_embedding(EMBEDDING, 5)

    assert store.shapes.reads == 1
