"""Concept postings end to end against real IRIS (spec 091).

The fixture notes are ingested with SectionStore, indexed with ConceptIndex and a small
invented dictionary, then queried only through QueryOptions and RetrievalEngine.retrieve,
the way a caller uses the feature. Expected parents are the ones worked out by hand in
tests/integration/test_note_search_concepts_iris.py.
"""

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.query_options import QueryOptions
from iris_vector_rag.note_search.concept_index import ConceptIndex
from iris_vector_rag.note_search.concepts import DictionaryTagger
from iris_vector_rag.note_search.sections import default_heading_detector
from iris_vector_rag.note_search.store import SectionStore
from iris_vector_rag.retrieval.engine import RetrievalEngine
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from tests.fixtures.note_search.loader import load_vectors

pytestmark = pytest.mark.e2e

SCHEMA = "NSCON"
TABLE = "DocumentChunks"
CONCEPTS = "SectionConcepts"

DICTIONARY = {
    "C-CHEST-PAIN": ["chest pain", "chest discomfort"],
    "C-SOB": ["shortness of breath", "sob", "dyspnea"],
    "C-PCN": ["penicillin"],
    "C-STROKE": ["stroke", "cva"],
}

VECTORS = load_vectors()
NOTES = {note["id"]: note for note in VECTORS["notes"]}


def _run(connection, sql):
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        connection.commit()
    finally:
        cursor.close()


@pytest.fixture
def ingested(iris_connection, monkeypatch):
    """Fresh tables holding the fixture notes as sections, with their postings indexed.

    Returns the RetrievalEngine a caller would build over the stores.
    """
    _run(iris_connection, f"DROP TABLE IF EXISTS {SCHEMA}.{CONCEPTS}")
    _run(iris_connection, f"DROP TABLE IF EXISTS {SCHEMA}.{TABLE}")
    monkeypatch.setenv("IRIS_SCHEMA_PREFIX", SCHEMA)
    monkeypatch.setenv("RAG_STORAGE__IRIS__TABLE_NAME", f"{SCHEMA}.{TABLE}")
    config = ConfigurationManager()
    connections = ConnectionManager(config)

    manager = SchemaManager(connections, config, schema_prefix=SCHEMA)
    assert manager.ensure_table_schema(TABLE) is True

    store = IRISVectorStore(ConnectionManager(config), config)
    tagger = DictionaryTagger(DICTIONARY)
    concept_index = ConceptIndex(store)
    concept_index.ensure_table()
    section_store = SectionStore(store)
    for note in VECTORS["notes"]:
        sections = section_store.store(
            note["id"], note["text"], default_heading_detector
        )
        concept_index.index_parent(note["id"], sections, tagger)

    yield RetrievalEngine(store, concept_index=concept_index, concept_tagger=tagger)
    _run(iris_connection, f"DROP TABLE IF EXISTS {SCHEMA}.{CONCEPTS}")
    _run(iris_connection, f"DROP TABLE IF EXISTS {SCHEMA}.{TABLE}")


def _search(engine, query, polarity=None, section=None, top_k=10):
    return engine.retrieve(
        QueryOptions(
            query=query,
            retrieval="concept",
            top_k=top_k,
            polarity=polarity,
            section=section,
        )
    )


# --- US1: a concept named in the query finds the notes that have it -----------------


def test_us1_a_concept_finds_its_notes_and_the_hit_reads_its_best_section(ingested):
    hits = _search(ingested, "penicillin")

    assert {hit.id for hit in hits} == {"N-001", "N-004", "N-007"}
    for hit in hits:
        stored_texts = {section["text"] for section in NOTES[hit.id]["sections"]}
        assert hit.page_content in stored_texts


# --- US2: polarity decides which mentions count -----------------------------------


def test_us2_affirmed_chest_pain_is_the_affirmed_notes(ingested):
    hits = _search(ingested, "chest pain", polarity="affirmed")

    assert {hit.id for hit in hits} == {"N-001", "N-003", "N-005", "N-006", "N-008"}
    assert {hit.metadata["polarity"] for hit in hits} == {"affirmed"}


def test_us2_negated_chest_pain_is_only_the_denial(ingested):
    hits = _search(ingested, "chest pain", polarity="negated")

    assert [hit.id for hit in hits] == ["N-002"]
    assert hits[0].metadata["polarity"] == "negated"


def test_us2_a_hedged_chest_pain_is_unknown(ingested):
    hits = _search(ingested, "chest pain", polarity="unknown")

    assert [hit.id for hit in hits] == ["N-004"]


# --- US3: one hit per note, the matched sections listed best first --------------------


def test_us3_one_hit_per_note_with_its_matched_sections(ingested):
    hits = _search(ingested, "chest pain", polarity="affirmed")

    note = next(hit for hit in hits if hit.id == "N-008")
    matched = note.metadata["matched_sections"]
    assert sorted(section["label"] for section in matched) == [
        "disposition",
        "hpi",
        "plan",
        "ros",
    ]
    assert note.metadata["text_score"] == 4
    assert note.page_content == "Chest pain since noon."


def test_us3_a_page_of_three_is_the_top_three_by_mentions(ingested):
    hits = _search(ingested, "chest pain", polarity="affirmed", top_k=3)

    assert [hit.id for hit in hits] == ["N-008", "N-006", "N-001"]


# --- US4: a section filter and several concepts in one query ---------------------------


def test_us4_the_section_filter_limits_the_notes(ingested):
    hits = _search(ingested, "penicillin", section="allergies")

    assert {hit.id for hit in hits} == {"N-001", "N-004"}


def test_us4_two_concepts_in_one_query_need_both_in_one_note(ingested):
    """N-001 has chest pain and shortness of breath; N-002 has both too, in other
    sections. N-001 scores higher (3 against 2) so it comes first."""
    hits = _search(ingested, "shortness of breath with chest pain")

    assert [hit.id for hit in hits] == ["N-001", "N-002"]


def test_us4_a_query_naming_no_concept_is_an_error_not_no_match(ingested):
    with pytest.raises(ValueError):
        _search(ingested, "headache")
