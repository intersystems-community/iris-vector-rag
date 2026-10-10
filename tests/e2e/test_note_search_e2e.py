"""Note search end to end, one test per user story, against real IRIS.

The fixture notes of tests/fixtures/note_search/vectors.json are ingested with
SectionStore and then queried only through QueryOptions and RetrievalEngine.retrieve,
the way a caller uses the feature. Expected answers come from the same vectors file.
"""

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.query_options import QueryOptions
from iris_vector_rag.note_search.sections import default_heading_detector
from iris_vector_rag.note_search.store import SectionStore
from iris_vector_rag.retrieval.engine import RetrievalEngine
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from tests.fixtures.note_search.loader import load_vectors

pytestmark = pytest.mark.e2e

SCHEMA = "NOTESRCHE"
TABLE = "DocumentChunks"

VECTORS = load_vectors()
NOTES = {note["id"]: note for note in VECTORS["notes"]}
QUERIES = {query["id"]: query for query in VECTORS["queries"]}
ERRORS = {"ValueError": ValueError}


def _run(connection, sql):
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        connection.commit()
    finally:
        cursor.close()


@pytest.fixture
def ingested(iris_connection, monkeypatch):
    """A fresh chunk table, indexed, holding the fixture notes as sections.

    Returns the RetrievalEngine a caller would build over the store. The configuration
    is read after the environment points at the NOTESRCHE table, as in the integration
    suite.
    """
    _run(iris_connection, f"DROP TABLE IF EXISTS {SCHEMA}.{TABLE}")
    monkeypatch.setenv("IRIS_SCHEMA_PREFIX", SCHEMA)
    monkeypatch.setenv("RAG_STORAGE__IRIS__TABLE_NAME", f"{SCHEMA}.{TABLE}")
    config = ConfigurationManager()
    connections = ConnectionManager(config)

    manager = SchemaManager(connections, config, schema_prefix=SCHEMA)
    assert manager.ensure_table_schema(TABLE) is True
    manager.ensure_text_index(TABLE)

    store = IRISVectorStore(ConnectionManager(config), config)
    section_store = SectionStore(store)
    for note in VECTORS["notes"]:
        section_store.store(note["id"], note["text"], default_heading_detector)

    yield RetrievalEngine(store)
    _run(iris_connection, f"DROP TABLE IF EXISTS {SCHEMA}.{TABLE}")


def _search(engine, query, section=None, top_k=10):
    return engine.retrieve(
        QueryOptions(
            query=query,
            retrieval="phrase",
            top_k=top_k,
            section=section,
        )
    )


def _actual(hits):
    """Parent id -> (polarity, sorted matched section labels)."""
    return {
        hit.id: (
            hit.metadata["polarity"],
            sorted(section["label"] for section in hit.metadata["matched_sections"]),
        )
        for hit in hits
    }


def _expected(query_id):
    query = QUERIES[query_id]
    return {
        parent["id"]: (parent["polarity"], sorted(parent["sections"]))
        for parent in query["parents"]
    }


# --- US1: a sectioned note is stored and its sections are what the search finds ---


def test_us1_a_stored_section_is_found_by_its_phrase(ingested):
    """V4: 'penicillin' is found in the allergy and history sections of three notes."""
    hits = _search(ingested, QUERIES["V4"]["phrase"])

    assert _actual(hits) == _expected("V4")
    for hit in hits:
        stored_texts = {section["text"] for section in NOTES[hit.id]["sections"]}
        assert hit.page_content in stored_texts


# --- US2: each hit says whether the phrase was affirmed, negated or unknown ----


@pytest.mark.parametrize("query_id", ["V1", "V8", "V9"])
def test_us2_each_hit_carries_its_polarity(ingested, query_id):
    query = QUERIES[query_id]

    hits = _search(ingested, query["phrase"], section=query["section"])

    assert _actual(hits) == _expected(query_id)
    polarities = {hit.id: hit.metadata["polarity"] for hit in hits}
    assert set(polarities.values()) <= {"affirmed", "negated", "unknown"}


def test_us2_a_negated_hit_is_returned_not_dropped(ingested):
    """V1: N-002 says 'denies chest pain' and is still a hit, marked negated."""
    hits = _search(ingested, "chest pain")

    assert {hit.id: hit.metadata["polarity"] for hit in hits}["N-002"] == "negated"


@pytest.mark.parametrize("query_id", ["V16a", "V16b", "V16c"])
def test_us2_an_empty_phrase_is_an_error_not_no_match(ingested, query_id):
    query = QUERIES[query_id]

    with pytest.raises(ERRORS[query["error"]]):
        _search(ingested, query["phrase"], top_k=query["k"])


# --- US3: hits roll up to one per note, with the matched sections listed -----------


def test_us3_one_hit_per_note_with_its_matched_sections(ingested):
    """V1: N-008 has four matching sections and one hit, listing all four."""
    hits = _search(ingested, QUERIES["V1"]["phrase"])

    ids = [hit.id for hit in hits]
    assert len(ids) == len(set(ids))
    note = next(hit for hit in hits if hit.id == "N-008")
    matched = note.metadata["matched_sections"]
    assert sorted(section["label"] for section in matched) == [
        "disposition",
        "hpi",
        "plan",
        "ros",
    ]
    assert note.metadata["text_score"] == max(
        section["text_score"] for section in matched
    )


def test_us3_a_page_holds_distinct_notes_from_the_full_result(ingested):
    """V10: a page of 3 is 3 distinct notes, all among the full V1 result."""
    query = QUERIES["V10"]

    page = _search(ingested, query["phrase"], top_k=query["k"])
    full_ids = {hit.id for hit in _search(ingested, QUERIES["V1"]["phrase"])}

    ids = [hit.id for hit in page]
    assert len(page) == query["count"]
    assert len(set(ids)) == len(ids)
    assert set(ids) <= full_ids


# --- US4: a section filter narrows the search to one kind of section --------------


@pytest.mark.parametrize("query_id", ["V2", "V3", "V5", "V14", "V15"])
def test_us4_the_section_filter_limits_the_notes(ingested, query_id):
    query = QUERIES[query_id]

    hits = _search(ingested, query["phrase"], section=query["section"])

    assert _actual(hits) == _expected(query_id)


def test_us4_without_the_filter_a_note_outside_the_section_comes_back(ingested):
    """N-007 has 'penicillin' only in its history, so the allergies filter drops it."""
    everything = {hit.id for hit in _search(ingested, "penicillin")}
    allergies = {hit.id for hit in _search(ingested, "penicillin", section="allergies")}

    assert "N-007" in everything
    assert "N-007" not in allergies
    assert allergies == {"N-001", "N-004"}
