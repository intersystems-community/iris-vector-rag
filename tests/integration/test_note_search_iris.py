"""Note search against real IRIS: section store, phrase search, roll-up and polarity.

The synthetic notes of tests/fixtures/note_search/vectors.json are split with the
default heading detector and stored through SectionStore into a chunk-shaped table.
What IRIS then holds is read back with SQL, and what IRIS returns for each query is
compared with the known answers in the same file. Nothing is mocked and no embedder
runs: section rows carry no embedding, and the text index reads chunk_text alone.

Each test works in the NOTESRCH schema; nothing here touches RAG.
"""

import json

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.query_options import QueryOptions
from iris_vector_rag.exceptions import TextIndexMissingError
from iris_vector_rag.note_search.polarity import ReferenceNegExDetector
from iris_vector_rag.note_search.sections import default_heading_detector
from iris_vector_rag.note_search.store import SectionStore, SectionStoreError
from iris_vector_rag.retrieval.engine import RetrievalEngine
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.table_shape import (
    CANONICAL_CHUNK_SHAPE,
    create_table_statement,
)
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from tests.fixtures.note_search.loader import load_vectors

pytestmark = pytest.mark.integration

SCHEMA = "NOTESRCH"
TABLE = "DocumentChunks"
FLAT = "FlatNotes"
UNINDEXED = "UnindexedNotes"

VECTORS = load_vectors()
NOTES = {note["id"]: note for note in VECTORS["notes"]}
QUERIES = {query["id"]: query for query in VECTORS["queries"]}
ERRORS = {"ValueError": ValueError}


def _run(connection, sql, params=None):
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params or [])
        rows = cursor.fetchall() if cursor.description else None
        connection.commit()
        return rows
    finally:
        cursor.close()


def _drop_all(connection):
    for table in (TABLE, FLAT, UNINDEXED):
        _run(connection, f"DROP TABLE IF EXISTS {SCHEMA}.{table}")


@pytest.fixture
def clean(iris_connection):
    _drop_all(iris_connection)
    yield iris_connection
    _drop_all(iris_connection)


@pytest.fixture
def stack():
    config = ConfigurationManager()
    return config, ConnectionManager(config)


def _manager(stack):
    config, connections = stack
    return SchemaManager(connections, config, schema_prefix=SCHEMA)


def _store(stack, monkeypatch, table=TABLE):
    monkeypatch.setenv("IRIS_SCHEMA_PREFIX", SCHEMA)
    monkeypatch.setenv("RAG_STORAGE__IRIS__TABLE_NAME", f"{SCHEMA}.{table}")
    config = ConfigurationManager()
    return IRISVectorStore(ConnectionManager(config), config)


@pytest.fixture
def chunk_table(stack, clean):
    manager = _manager(stack)
    assert manager.ensure_table_schema(TABLE) is True
    manager.ensure_text_index(TABLE)
    return manager


@pytest.fixture
def vector_store(stack, chunk_table, monkeypatch):
    return _store(stack, monkeypatch)


@pytest.fixture
def section_store(vector_store):
    return SectionStore(vector_store)


@pytest.fixture
def sections(section_store, vector_store):
    """The vectors corpus stored as sections of its parents; returns the store."""
    for note in VECTORS["notes"]:
        section_store.store(note["id"], note["text"], default_heading_detector)
    return vector_store


def _rows(connection, parent=None):
    """Every stored section row, read with SQL, sorted by parent then ordinal."""
    sql = (
        "SELECT doc_id, document_id, chunk_text, metadata, "
        "CASE WHEN chunk_embedding IS NULL THEN 1 ELSE 0 END "
        f"FROM {SCHEMA}.{TABLE}"
    )
    params = []
    if parent is not None:
        sql += " WHERE document_id = ?"
        params.append(parent)
    rows = _run(connection, sql, params) or []
    stored = [
        {
            "doc_id": doc_id,
            "document_id": document_id,
            "chunk_text": text,
            "meta": json.loads(metadata),
            "embedding_is_null": bool(is_null),
        }
        for doc_id, document_id, text, metadata, is_null in rows
    ]
    return sorted(stored, key=lambda row: (row["document_id"], row["meta"]["ordinal"]))


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


# --- storing sections ----------------------------------------------------------


def test_each_note_reads_back_as_its_known_sections(clean, sections):
    """US1: labels, texts, ids, offsets and the null embedding match the fixture."""
    rows = _rows(clean)
    assert len(rows) == sum(len(note["sections"]) for note in VECTORS["notes"])

    for note in VECTORS["notes"]:
        stored = [row for row in rows if row["document_id"] == note["id"]]
        assert [row["meta"]["section"] for row in stored] == [
            section["label"] for section in note["sections"]
        ]
        assert [row["chunk_text"] for row in stored] == [
            section["text"] for section in note["sections"]
        ]
        assert [row["doc_id"] for row in stored] == [
            f"{note['id']}#{n}" for n in range(len(stored))
        ]
        assert all(row["meta"]["parent_id"] == note["id"] for row in stored)
        for row in stored:
            meta = row["meta"]
            assert note["text"][meta["start"] : meta["end"]] == row["chunk_text"]
        assert all(row["embedding_is_null"] for row in stored)


def test_store_returns_the_sections_it_wrote(section_store):
    written = section_store.store(
        "N-009", NOTES["N-009"]["text"], default_heading_detector
    )

    assert [section.label for section in written] == ["hpi"]
    assert [section.text for section in written] == ["Chest-pain noted."]


def test_storing_a_parent_twice_leaves_one_set_of_rows(clean, section_store):
    note = NOTES["N-001"]
    section_store.store(note["id"], note["text"], default_heading_detector)
    section_store.store(note["id"], note["text"], default_heading_detector)

    assert len(_rows(clean, note["id"])) == len(note["sections"])


def test_restoring_a_parent_replaces_its_sections(clean, section_store):
    section_store.store("N-x", "HPI: old text.", default_heading_detector)
    section_store.store(
        "N-x", "HPI: new text.\nALLERGIES: none.", default_heading_detector
    )

    rows = _rows(clean, "N-x")
    assert [row["chunk_text"] for row in rows] == ["new text.", "none."]


def test_a_parent_id_with_a_hash_is_refused_before_anything_is_written(
    clean, section_store
):
    with pytest.raises(SectionStoreError):
        section_store.store("N#1", NOTES["N-003"]["text"], default_heading_detector)

    assert _rows(clean) == []


def test_a_failed_write_keeps_the_earlier_rows_and_raises(clean, section_store):
    """Rollback: a store that fails midway leaves the parent's previous rows intact.

    IRIS rejects a doc_id longer than 255 characters (SQLCODE -104). With a
    253-character parent id, `<parent>#<n>` fits for ordinals 0 to 9 and not from 10
    on, so a note of eleven sections fails after its first ten rows are written.
    """
    parent = "P" * 253
    ten = "\n".join(f"HPI: visit note {n}." for n in range(10))
    eleven = "\n".join(f"HPI: visit note {n}." for n in range(11))
    section_store.store(parent, ten, default_heading_detector)
    before = _rows(clean, parent)
    assert len(before) == 10

    with pytest.raises(SectionStoreError):
        section_store.store(parent, eleven, default_heading_detector)

    assert _rows(clean, parent) == before


def test_an_empty_note_writes_nothing(clean, section_store):
    assert section_store.store("N-empty", "", default_heading_detector) == []
    assert section_store.store("N-blank", " \n\t", default_heading_detector) == []

    assert _rows(clean, "N-empty") == []
    assert _rows(clean, "N-blank") == []


# --- phrase search over the fixture ------------------------------------------

PARENT_QUERIES = [query["id"] for query in VECTORS["queries"] if "parents" in query]
SAME_AS_QUERIES = [query["id"] for query in VECTORS["queries"] if "same_as" in query]
ERROR_QUERIES = [query["id"] for query in VECTORS["queries"] if "error" in query]


@pytest.mark.parametrize("query_id", PARENT_QUERIES)
def test_phrase_search_returns_the_known_parents(sections, query_id):
    query = QUERIES[query_id]

    hits = sections.phrase_search(
        query["phrase"], top_k=query["k"], section=query["section"]
    )

    assert sorted(hit.id for hit in hits) == sorted(
        parent["id"] for parent in query["parents"]
    )
    assert _summary(hits) == _expected(query)


@pytest.mark.parametrize("query_id", SAME_AS_QUERIES)
def test_a_query_the_fixture_says_is_the_same_gives_the_same_answer(sections, query_id):
    query = QUERIES[query_id]
    base = QUERIES[query["same_as"]]

    hits = sections.phrase_search(query["phrase"], top_k=query["k"])
    base_hits = sections.phrase_search(base["phrase"], top_k=base["k"])

    assert _summary(hits) == _summary(base_hits)


def test_top_k_three_returns_three_distinct_parents_from_v1(sections):
    """V10: a page of 3 is 3 distinct parents, all among V1's, though N-008 has four."""
    query = QUERIES["V10"]
    v1 = QUERIES[query["subset_of"]]

    hits = sections.phrase_search(query["phrase"], top_k=query["k"])
    ids = [hit.id for hit in hits]
    v1_ids = {hit.id for hit in sections.phrase_search(v1["phrase"], top_k=v1["k"])}

    assert len(hits) == query["count"]
    assert len(set(ids)) == len(ids)
    assert set(ids) <= v1_ids


def test_a_short_note_with_three_mentions_ranks_before_a_long_one(sections):
    query = QUERIES["V13"]
    ids = [hit.id for hit in sections.phrase_search(query["phrase"], top_k=query["k"])]

    first, second = query["ranked_before"]
    assert ids.index(first) < ids.index(second)


@pytest.mark.parametrize("query_id", ERROR_QUERIES)
def test_an_empty_or_punctuation_phrase_is_a_value_error(sections, query_id):
    query = QUERIES[query_id]

    with pytest.raises(ERRORS[query["error"]]):
        sections.phrase_search(
            query["phrase"], top_k=query["k"], section=query["section"]
        )


def test_each_hit_is_rolled_up_with_its_best_section_first(sections):
    hits = sections.phrase_search("chest pain", top_k=10)

    for hit in hits:
        matched = hit.metadata["matched_sections"]
        scores = [section["text_score"] for section in matched]
        assert scores == sorted(scores, reverse=True)
        assert hit.metadata["text_score"] == scores[0]
    hit_scores = [hit.metadata["text_score"] for hit in hits]
    assert hit_scores == sorted(hit_scores, reverse=True)


def test_three_runs_give_identical_results(sections):
    runs = [
        [
            (hit.id, hit.metadata["polarity"], hit.metadata["matched_sections"])
            for hit in sections.phrase_search("chest pain", top_k=10)
        ]
        for _ in range(3)
    ]

    assert runs[0] == runs[1] == runs[2]


def test_a_similarity_threshold_is_refused_for_phrase(sections):
    engine = RetrievalEngine(sections)

    with pytest.raises(ValueError):
        engine.retrieve(
            QueryOptions(
                query="chest pain",
                retrieval="phrase",
                similarity_threshold=0.5,
            )
        )


# --- polarity, the caller's detector, flat tables, missing index ---------------


def test_a_caller_detector_replaces_the_default(sections):
    def always_negated(text, span):
        return "negated"

    hits = sections.phrase_search("chest pain", top_k=10, detector=always_negated)

    assert {hit.metadata["polarity"] for hit in hits} == {"negated"}
    assert {
        section["polarity"]
        for hit in hits
        for section in hit.metadata["matched_sections"]
    } == {"negated"}


def test_passing_the_reference_detector_is_the_same_as_the_default(sections):
    default = sections.phrase_search("chest pain", top_k=10)
    explicit = sections.phrase_search(
        "chest pain", top_k=10, detector=ReferenceNegExDetector()
    )

    assert _summary(explicit) == _summary(default)


def test_a_table_of_plain_documents_searches_flat(stack, clean, monkeypatch):
    _run(
        clean, create_table_statement(CANONICAL_CHUNK_SHAPE, schema=SCHEMA, table=FLAT)
    )
    documents = [
        ("f1", "Patient reports chest pain."),
        ("f2", "Denies chest pain."),
        ("f3", "Penicillin given."),
    ]
    for doc_id, text in documents:
        _run(
            clean,
            f"INSERT INTO {SCHEMA}.{FLAT} (doc_id, document_id, chunk_text, metadata) "
            "VALUES (?, ?, ?, ?)",
            [doc_id, doc_id, text, "{}"],
        )
    _manager(stack).ensure_text_index(FLAT)

    hits = _store(stack, monkeypatch, FLAT).phrase_search("chest pain", top_k=10)

    assert {hit.id: hit.metadata["polarity"] for hit in hits} == {
        "f1": "affirmed",
        "f2": "negated",
    }
    assert {hit.id: hit.page_content for hit in hits} == {
        "f1": "Patient reports chest pain.",
        "f2": "Denies chest pain.",
    }


def test_a_table_without_the_text_index_raises_and_names_the_fix(
    stack, clean, monkeypatch
):
    _run(
        clean,
        create_table_statement(CANONICAL_CHUNK_SHAPE, schema=SCHEMA, table=UNINDEXED),
    )
    _run(
        clean,
        f"INSERT INTO {SCHEMA}.{UNINDEXED} (doc_id, document_id, chunk_text, metadata) "
        "VALUES (?, ?, ?, ?)",
        ["u1", "u1", "chest pain", "{}"],
    )
    store = _store(stack, monkeypatch, UNINDEXED)

    with pytest.raises(TextIndexMissingError) as caught:
        store.phrase_search("chest pain")

    message = str(caught.value)
    assert f"{SCHEMA}.{UNINDEXED}" in message
    assert "ensure_text_index" in message
