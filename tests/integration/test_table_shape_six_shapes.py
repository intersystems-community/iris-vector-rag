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

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Document
from iris_vector_rag.storage.catalog import IRISCatalogReader, RecordedCatalog
from iris_vector_rag.storage.shape_resolver import TableShapeResolver
from iris_vector_rag.storage.table_shape import (
    ColumnRole,
    IdentifierNotWritable,
    NotVectorSearchable,
    ShapeKind,
    TableShape,
)
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
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


# --- US1: the same behaviour on every shape ----------------------------------

#: Shapes that can be both written with a caller-supplied identifier and searched
#: by vector. The other two are excluded for reasons the suite asserts directly:
#: `legacy_serial` generates its own identifiers (FR-008) and `legacy_id` has no
#: embedding column at all (FR-011).
SEARCHABLE_SHAPES = ("canonical", "fixture", "mixed", "double_vector")

DOCUMENTS = [
    Document(
        id="parity-1",
        page_content="vector search over a document table",
        metadata={"source": "one"},
    ),
    Document(
        id="parity-2",
        page_content="the shape of a table decides its columns",
        metadata={"source": "two"},
    ),
    Document(
        id="parity-3",
        page_content="unrelated notes about the weather",
        metadata={"source": "three"},
    ),
]


def shape_vector(seed: int) -> list[float]:
    """A deterministic vector, so ranking is a property of the store, not chance."""
    return [((seed * 7 + i) % 13) / 13.0 for i in range(VECTOR_DIMENSION)]


EMBEDDINGS = [shape_vector(n) for n in range(1, len(DOCUMENTS) + 1)]
QUERY = shape_vector(1)


@pytest.fixture
def store_for(monkeypatch, live_shapes):
    """A real store pointed at one shape's table, configured the way callers do.

    The table name arrives through configuration and nothing else: no column name
    is named by this test, which is the whole point of US1.
    """

    def make(shape: str) -> IRISVectorStore:
        monkeypatch.setenv(
            "RAG_STORAGE__IRIS__TABLE_NAME", f"{SHAPE_SCHEMA}.{SHAPE_TABLES[shape]}"
        )
        config = ConfigurationManager()
        return IRISVectorStore(ConnectionManager(config), config)

    return make


@pytest.fixture
def clean_rows(shape_connection):
    """Remove the documents a test loaded, whatever the identifier column is."""
    loaded: list[IRISVectorStore] = []
    yield loaded
    for store in loaded:
        try:
            store.delete_documents([doc.id for doc in DOCUMENTS])
        except Exception:  # noqa: BLE001 - cleanup is advisory
            pass
    shape_connection.commit()


def presented(results):
    """(identifier, text) per hit, in rank order."""
    return [(document.id, document.page_content) for document, _ in results]


@pytest.mark.parametrize("shape", SEARCHABLE_SHAPES)
def test_search_filter_write_delete_parity(shape, store_for, clean_rows):
    """FR-002: identical documents in, identical results out, on every shape."""
    baseline_store = store_for("canonical")
    clean_rows.append(baseline_store)
    baseline_store.add_documents(DOCUMENTS, EMBEDDINGS)
    baseline = presented(baseline_store.similarity_search_by_embedding(QUERY, 3))
    assert baseline[0][0] == "parity-1", baseline

    store = store_for(shape)
    clean_rows.append(store)
    store.add_documents(DOCUMENTS, EMBEDDINGS)

    assert presented(store.similarity_search_by_embedding(QUERY, 3)) == baseline

    filtered = store.similarity_search_by_embedding(QUERY, 3, {"source": "two"})
    assert presented(filtered) == [("parity-2", DOCUMENTS[1].page_content)]

    assert store.delete_documents(["parity-1"]) is True
    remaining = {
        doc.id for doc in store.fetch_documents_by_ids([d.id for d in DOCUMENTS])
    }
    assert remaining == {"parity-2", "parity-3"}


def test_scores_agree_across_shapes(store_for, clean_rows):
    """A double-precision vector column must not change the ranking or the scores.

    Compared relatively: a DOUBLE column keeps more of each component than a FLOAT
    one, so the dot products differ in their last few digits -- about 1e-6 relative
    on these vectors. That is the storage precision, not a difference in behaviour.
    """
    scores = {}
    for shape in SEARCHABLE_SHAPES:
        store = store_for(shape)
        clean_rows.append(store)
        store.add_documents(DOCUMENTS, EMBEDDINGS)
        scores[shape] = [
            score for _, score in store.similarity_search_by_embedding(QUERY, 3)
        ]

    for shape in SEARCHABLE_SHAPES[1:]:
        assert scores[shape] == pytest.approx(scores["canonical"], rel=1e-5), shape


def test_a_generated_identifier_shape_refuses_a_caller_supplied_id(store_for):
    """FR-008: the refusal names the real cause instead of failing at insert time."""
    store = store_for("legacy_serial")
    with pytest.raises(IdentifierNotWritable) as raised:
        store.add_documents(DOCUMENTS, EMBEDDINGS)
    assert SHAPE_TABLES["legacy_serial"] in str(raised.value)


def test_a_table_without_an_embedding_column_round_trips_but_cannot_be_searched(
    store_for, clean_rows
):
    """FR-011: the failure names the missing column family, not a dimension mismatch."""
    store = store_for("legacy_id")
    clean_rows.append(store)
    # No embeddings: this shape has no column to hold them, and `_store_documents`
    # is the write path `add_documents` delegates to once embedding is out of the way.
    store._store_documents(DOCUMENTS)

    fetched = {
        doc.id: doc.page_content for doc in store.fetch_documents_by_ids(["parity-2"])
    }
    assert fetched == {"parity-2": DOCUMENTS[1].page_content}

    with pytest.raises(NotVectorSearchable) as raised:
        store.similarity_search_by_embedding(QUERY, 3)
    assert SHAPE_TABLES["legacy_id"] in str(raised.value)


def test_mixed_table_round_trip(store_for, shape_connection, clean_rows):
    """The defect US1 exists to remove, against a real table carrying both columns."""
    store = store_for("mixed")
    clean_rows.append(store)
    store.add_documents(DOCUMENTS[:1], EMBEDDINGS[:1])

    results = store.similarity_search_by_embedding(QUERY, 3)
    assert presented(results)[0] == ("parity-1", DOCUMENTS[0].page_content)

    # A row as an older release left it: text in `content`, nothing in `text_content`.
    table = f"{SHAPE_SCHEMA}.{SHAPE_TABLES['mixed']}"
    legacy_text = "written by 0.13"
    cursor = shape_connection.cursor()
    try:
        cursor.execute(
            f"INSERT INTO {table} (doc_id, content, embedding) "
            f"VALUES (?, ?, TO_VECTOR(?, FLOAT, {VECTOR_DIMENSION}))",
            ["parity-legacy", legacy_text, ",".join(str(v) for v in EMBEDDINGS[0])],
        )
        shape_connection.commit()
        cursor.close()

        presented_texts = {
            document.id: document.page_content
            for document, _ in store.similarity_search_by_embedding(QUERY, 3)
        }
        assert presented_texts["parity-legacy"] == legacy_text
        assert presented_texts["parity-1"] == DOCUMENTS[0].page_content
    finally:
        cursor = shape_connection.cursor()
        try:
            cursor.execute(f"DELETE FROM {table} WHERE doc_id = ?", ["parity-legacy"])
            shape_connection.commit()
        except Exception:  # noqa: BLE001 - cleanup is advisory
            pass
        cursor.close()
