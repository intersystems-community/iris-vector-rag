"""The recorded shapes must still describe what IRIS reports (feature 082, T015).

`tests/unit/test_table_shape.py` runs against recorded catalog rows, which is only
sound while the recording is honest. This suite creates all six shapes for real,
snapshots them with `IRISCatalogReader`, and asserts the resulting `TableShape`
equals the one built from the committed fixture.

So a driver upgrade that changes how a VECTOR column is reported, or a schema change
that renames a column, fails here rather than being silently absorbed by fixtures
that no longer match anything real.

Container: `iris-vector-rag-iris` on 51972, per CLAUDE.md. Never another one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iris_vector_rag.storage.catalog import IRISCatalogReader, RecordedCatalog
from iris_vector_rag.storage.shape_resolver import TableShapeResolver
from iris_vector_rag.storage.table_shape import ColumnRole, ShapeKind, TableShape
from tests.fixtures.shapes.ddl import (
    SHAPE_SCHEMA,
    SHAPE_TABLES,
    SIX_SHAPES,
    VECTOR_DIMENSION,
    create_statement,
    drop_statement,
)

pytestmark = pytest.mark.integration

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "shapes"


@pytest.fixture(scope="module")
def shape_connection():
    """A raw DBAPI connection: the catalog reader needs `cursor()`, not SQLAlchemy."""
    from iris_vector_rag.common.iris_connection import get_iris_connection

    conn = get_iris_connection()
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def live_shapes(shape_connection):
    """Create all six shapes, snapshot each one, drop them again."""
    cursor = shape_connection.cursor()
    for shape in SIX_SHAPES:
        cursor.execute(drop_statement(shape, SHAPE_SCHEMA))
        cursor.execute(create_statement(shape, SHAPE_SCHEMA))
    shape_connection.commit()
    cursor.close()

    reader = IRISCatalogReader(shape_connection)
    try:
        yield {
            shape: TableShape.from_snapshot(
                reader.snapshot(SHAPE_SCHEMA, SHAPE_TABLES[shape])
            )
            for shape in SIX_SHAPES
        }
    finally:
        cursor = shape_connection.cursor()
        for shape in SIX_SHAPES:
            try:
                cursor.execute(drop_statement(shape, SHAPE_SCHEMA))
            except Exception:  # noqa: BLE001 - teardown is advisory
                pass
        shape_connection.commit()
        cursor.close()


@pytest.fixture(scope="module")
def recorded_shapes():
    catalog = RecordedCatalog.from_fixture_dir(FIXTURE_DIR)
    return {
        shape: TableShape.from_snapshot(
            catalog.snapshot(SHAPE_SCHEMA, SHAPE_TABLES[shape])
        )
        for shape in SIX_SHAPES
    }


@pytest.mark.parametrize("shape", sorted(SIX_SHAPES))
def test_recorded_fixtures_match_live_catalog(live_shapes, recorded_shapes, shape):
    """The gate on the whole recorded-fixture approach."""
    live, recorded = live_shapes[shape], recorded_shapes[shape]
    assert live.kind is recorded.kind
    assert live.roles == recorded.roles
    assert sorted(live.columns) == sorted(recorded.columns)
    for name, column in recorded.columns.items():
        assert live.columns[name] == column, name
    assert live == recorded


def test_live_vector_columns_report_their_element_type(live_shapes):
    """The standing gotcha, against a real column: SQL says varchar, class says vector."""
    canonical = live_shapes["canonical"].column_for(ColumnRole.EMBEDDING)
    assert canonical.sql_type == "varchar"
    assert canonical.class_type == "%Library.Vector"
    assert canonical.vector.element_type == "FLOAT"
    assert canonical.vector.dimension == VECTOR_DIMENSION
    assert live_shapes["double_vector"].vector_spec().element_type == "DOUBLE"


def test_live_identity_column_is_recognised(live_shapes):
    """`INTEGER IDENTITY PRIMARY KEY` must not look writable."""
    identifier = live_shapes["legacy_serial"].column_for(ColumnRole.IDENTIFIER)
    assert identifier.is_identity is True
    assert identifier.is_key is True


def test_vector_cosine_confirms_the_column_independently(shape_connection, live_shapes):
    """A second, mechanism-independent check that the vector column is really a vector.

    The shape reads the class view; this reads the SQL engine. Agreement between two
    unrelated mechanisms is what makes the class-view answer trustworthy.
    """
    table = f"{SHAPE_SCHEMA}.{SHAPE_TABLES['canonical']}"
    vector = ",".join(["0.1"] * VECTOR_DIMENSION)
    cursor = shape_connection.cursor()
    try:
        cursor.execute(
            f"INSERT INTO {table} (doc_id, text_content, embedding) "
            f"VALUES (?, ?, TO_VECTOR(?, FLOAT, {VECTOR_DIMENSION}))",
            ["shape-probe", "probe text", vector],
        )
        shape_connection.commit()
        cursor.execute(
            f"SELECT TOP 1 VECTOR_COSINE(embedding, embedding) FROM {table} "
            "WHERE doc_id = ?",
            ["shape-probe"],
        )
        row = cursor.fetchone()
        similarity = float(row[0])
    finally:
        try:
            cursor.execute(f"DELETE FROM {table} WHERE doc_id = ?", ["shape-probe"])
            shape_connection.commit()
        except Exception:  # noqa: BLE001 - cleanup is advisory
            pass
        cursor.close()
    assert similarity == pytest.approx(1.0, abs=1e-4)


def test_resolver_reads_the_live_catalog_once(shape_connection, live_shapes):
    """SC-002 against the real catalog, not a double."""
    resolver = TableShapeResolver(IRISCatalogReader(shape_connection))
    first = resolver.resolve(SHAPE_SCHEMA, SHAPE_TABLES["canonical"])
    for _ in range(20):
        assert resolver.resolve(SHAPE_SCHEMA, SHAPE_TABLES["canonical"]) is first
    assert resolver.reads == 1
    assert first.kind is ShapeKind.CANONICAL


def test_absent_table_is_reported_as_absent(shape_connection):
    reader = IRISCatalogReader(shape_connection)
    assert reader.snapshot(SHAPE_SCHEMA, "NoSuchShapeTable") is None
