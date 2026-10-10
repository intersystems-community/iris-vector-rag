"""The opt-in section_label column against real IRIS (spec 089).

The synthetic notes of tests/fixtures/note_search/vectors.json are stored as sections
first without the column (the metadata form), then the column is added with
`ensure_section_column`, and phrase_search must return the same answers before and
after. Nothing is mocked and no embedder runs.

Each test works in the NSCOL schema; nothing here touches RAG.
"""

import json

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.note_search.sections import default_heading_detector
from iris_vector_rag.note_search.store import (
    SECTION_COLUMN,
    SectionStore,
    SectionStoreError,
    ensure_section_column,
)
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from tests.fixtures.note_search.loader import load_vectors

pytestmark = pytest.mark.integration

SCHEMA = "NSCOL"
TABLE = "DocumentChunks"
INDEX = "SectionLabelIdx"

VECTORS = load_vectors()
NOTES = {note["id"]: note for note in VECTORS["notes"]}
QUERIES = {query["id"]: query for query in VECTORS["queries"]}
# The queries that name a section, the ones the column filter applies to.
SECTIONED_QUERIES = [
    query["id"]
    for query in VECTORS["queries"]
    if "parents" in query and query.get("section") is not None
]


def _run(connection, sql, params=None):
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params or [])
        rows = cursor.fetchall() if cursor.description else None
        connection.commit()
        return rows
    finally:
        cursor.close()


def _drop(connection):
    _run(connection, f"DROP TABLE IF EXISTS {SCHEMA}.{TABLE}")


@pytest.fixture
def clean(iris_connection):
    _drop(iris_connection)
    yield iris_connection
    _drop(iris_connection)


@pytest.fixture
def stack():
    config = ConfigurationManager()
    return config, ConnectionManager(config)


def _store(stack, monkeypatch):
    monkeypatch.setenv("IRIS_SCHEMA_PREFIX", SCHEMA)
    monkeypatch.setenv("RAG_STORAGE__IRIS__TABLE_NAME", f"{SCHEMA}.{TABLE}")
    config = ConfigurationManager()
    return IRISVectorStore(ConnectionManager(config), config)


@pytest.fixture
def chunk_table(stack, clean):
    config, connections = stack
    manager = SchemaManager(connections, config, schema_prefix=SCHEMA)
    assert manager.ensure_table_schema(TABLE) is True
    manager.ensure_text_index(TABLE)
    return manager


@pytest.fixture
def vector_store(stack, chunk_table, monkeypatch):
    return _store(stack, monkeypatch)


@pytest.fixture
def metadata_only(vector_store):
    """The vectors corpus stored as sections, before the column exists."""
    section_store = SectionStore(vector_store)
    for note in VECTORS["notes"]:
        section_store.store(note["id"], note["text"], default_heading_detector)
    return vector_store


def _columns(connection):
    rows = _run(
        connection,
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?",
        [SCHEMA, TABLE],
    )
    return {str(row[0]).lower() for row in rows or []}


def _indexes(connection):
    rows = _run(
        connection,
        "SELECT INDEX_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.INDEXES "
        "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?",
        [SCHEMA, TABLE],
    )
    return {str(name).upper(): str(column).lower() for name, column in rows or []}


def _section_rows(connection):
    """(document_id, metadata section, section_label column) for every section row."""
    rows = _run(
        connection,
        f"SELECT document_id, metadata, {SECTION_COLUMN} FROM {SCHEMA}.{TABLE}",
    )
    return sorted(
        (doc, json.loads(meta)["section"], label)
        for doc, meta, label in rows
        if json.loads(meta).get("section") is not None
    )


def _summary(hits):
    """Parent id -> (polarity, sorted matched section labels), for comparison."""
    return {
        hit.id: (
            hit.metadata["polarity"],
            sorted(section["label"] for section in hit.metadata["matched_sections"]),
        )
        for hit in hits
    }


def _expected(query):
    return {
        parent["id"]: (parent["polarity"], sorted(parent["sections"]))
        for parent in query["parents"]
    }


# --- backfill, index, idempotence ---------------------------------------------


def test_ensure_backfills_a_metadata_only_table(clean, metadata_only):
    assert SECTION_COLUMN not in _columns(clean)

    assert ensure_section_column(metadata_only) is True

    rows = _section_rows(clean)
    assert len(rows) == sum(len(note["sections"]) for note in VECTORS["notes"])
    assert all(label == section for _, section, label in rows)
    assert SECTION_COLUMN in _columns(clean)


def test_ensure_creates_the_bitmap_index_on_the_column(clean, metadata_only):
    ensure_section_column(metadata_only)

    assert _indexes(clean)[INDEX.upper()] == SECTION_COLUMN


def test_ensure_is_idempotent(clean, metadata_only):
    assert ensure_section_column(metadata_only) is True
    before = _section_rows(clean)

    assert ensure_section_column(metadata_only) is False

    assert _section_rows(clean) == before
    assert [name for name in _indexes(clean) if name == INDEX.upper()] == [
        INDEX.upper()
    ]


def test_a_long_label_is_refused_before_the_table_is_altered(clean, vector_store):
    SectionStore(vector_store).store(
        "N-long",
        "HPI: chest pain.",
        default_heading_detector,
        label_map={"hpi": "h" * 101},
    )

    with pytest.raises(SectionStoreError):
        ensure_section_column(vector_store)

    assert SECTION_COLUMN not in _columns(clean)
    assert INDEX.upper() not in _indexes(clean)


# --- store after the column exists -------------------------------------------


def test_a_store_after_the_column_exists_fills_metadata_and_column(clean, vector_store):
    ensure_section_column(vector_store)
    note = NOTES["N-001"]

    SectionStore(vector_store).store(note["id"], note["text"], default_heading_detector)

    rows = _section_rows(clean)
    assert sorted((doc, section) for doc, section, _ in rows) == sorted(
        (note["id"], section["label"]) for section in note["sections"]
    )
    assert all(label == section for _, section, label in rows)


def test_a_long_label_is_refused_by_store_with_the_column_and_nothing_is_written(
    clean, vector_store
):
    ensure_section_column(vector_store)
    section_store = SectionStore(vector_store)
    section_store.store("N-keep", "HPI: kept.", default_heading_detector)
    before = _section_rows(clean)

    with pytest.raises(SectionStoreError):
        section_store.store(
            "N-keep",
            "HPI: replaced.",
            default_heading_detector,
            label_map={"hpi": "h" * 101},
        )

    assert _section_rows(clean) == before


# --- phrase search: the same answers with and without the column --------------


@pytest.mark.parametrize("query_id", SECTIONED_QUERIES)
def test_the_column_gives_the_same_answers_as_the_metadata(metadata_only, query_id):
    query = QUERIES[query_id]
    before = metadata_only.phrase_search(
        query["phrase"], top_k=query["k"], section=query["section"]
    )

    ensure_section_column(metadata_only)
    assert SECTION_COLUMN in metadata_only.shape().columns
    after = metadata_only.phrase_search(
        query["phrase"], top_k=query["k"], section=query["section"]
    )

    assert _summary(after) == _summary(before)
    assert _summary(after) == _expected(query)


def test_an_unsectioned_search_is_unchanged_by_the_column(metadata_only):
    query = QUERIES["V1"]
    before = metadata_only.phrase_search(query["phrase"], top_k=query["k"])

    ensure_section_column(metadata_only)
    after = metadata_only.phrase_search(query["phrase"], top_k=query["k"])

    assert _summary(after) == _summary(before)


def test_a_table_without_the_column_still_filters_by_section(metadata_only):
    assert SECTION_COLUMN not in metadata_only.shape().columns
    query = QUERIES["V2"]

    hits = metadata_only.phrase_search(
        query["phrase"], top_k=query["k"], section=query["section"]
    )

    assert _summary(hits) == _expected(query)


# --- a label with an apostrophe ----------------------------------------------


def test_a_label_with_an_apostrophe_is_stored_filtered_and_found(clean, vector_store):
    ensure_section_column(vector_store)
    label = "patient's history"
    SectionStore(vector_store).store(
        "N-apos",
        "HPI: chest pain noted.",
        default_heading_detector,
        label_map={"hpi": label},
    )

    assert _section_rows(clean) == [("N-apos", label, label)]
    hits = vector_store.phrase_search("chest pain", top_k=5, section=label)
    assert [hit.id for hit in hits] == ["N-apos"]
    assert _summary(hits)["N-apos"][1] == [label]
