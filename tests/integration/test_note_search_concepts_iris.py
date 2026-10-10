"""Concept postings against real IRIS (spec 091).

The synthetic notes of tests/fixtures/note_search/vectors.json are split with the default
heading detector, stored through SectionStore, and tagged with an invented dictionary
(no clinical vocabulary ships). The postings table is `NSCON.SectionConcepts`, next to
the chunk table `NSCON.DocumentChunks`; nothing here touches RAG.

Expected parents are derived by hand from the note text (see the comments on each
dictionary entry below) and then checked against what IRIS returns.
"""

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.note_search.concept_index import (
    ConceptIndex,
    ConceptIndexError,
    ConceptIndexMissingError,
)
from iris_vector_rag.note_search.concepts import DictionaryTagger
from iris_vector_rag.note_search.sections import Section, default_heading_detector
from iris_vector_rag.note_search.store import SectionStore
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from tests.fixtures.note_search.loader import load_vectors

pytestmark = pytest.mark.integration

SCHEMA = "NSCON"
TABLE = "DocumentChunks"
CONCEPTS = "SectionConcepts"

CHEST_PAIN = "C-CHEST-PAIN"
SOB = "C-SOB"
PCN = "C-PCN"
STROKE = "C-STROKE"

# Invented surface forms, chosen to exercise the tagger over the fixture notes.
DICTIONARY = {
    CHEST_PAIN: ["chest pain", "chest discomfort"],
    SOB: ["shortness of breath", "sob", "dyspnea"],
    PCN: ["penicillin"],
    STROKE: ["stroke", "cva"],
}

VECTORS = load_vectors()
NOTES = {note["id"]: note for note in VECTORS["notes"]}


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
    for table in (TABLE, CONCEPTS, "NoSuchConcepts"):
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


def _store(stack, monkeypatch):
    monkeypatch.setenv("IRIS_SCHEMA_PREFIX", SCHEMA)
    monkeypatch.setenv("RAG_STORAGE__IRIS__TABLE_NAME", f"{SCHEMA}.{TABLE}")
    config = ConfigurationManager()
    return IRISVectorStore(ConnectionManager(config), config)


@pytest.fixture
def vector_store(stack, clean, monkeypatch):
    config, connections = stack
    manager = SchemaManager(connections, config, schema_prefix=SCHEMA)
    assert manager.ensure_table_schema(TABLE) is True
    return _store(stack, monkeypatch)


@pytest.fixture
def tagger():
    return DictionaryTagger(DICTIONARY)


@pytest.fixture
def index(vector_store):
    concepts = ConceptIndex(vector_store)
    concepts.ensure_table()
    return concepts


@pytest.fixture
def indexed(vector_store, index, tagger):
    """The fixture notes stored as sections and their postings indexed.

    Returns (concept index, {parent id: [Section]}).
    """
    section_store = SectionStore(vector_store)
    stored = {}
    for note in VECTORS["notes"]:
        sections = section_store.store(
            note["id"], note["text"], default_heading_detector
        )
        index.index_parent(note["id"], sections, tagger)
        stored[note["id"]] = sections
    return index, stored


def _ids(hits):
    return {hit.id for hit in hits}


def _rows(connection, parent=None):
    sql = (
        "SELECT section_id, parent_id, concept_id, polarity, mentions, section_label "
        f"FROM {SCHEMA}.{CONCEPTS}"
    )
    params = []
    if parent is not None:
        sql += " WHERE parent_id = ?"
        params.append(parent)
    return sorted(_run(connection, sql, params) or [])


# --- the table ------------------------------------------------------------------------


def test_ensure_table_creates_the_postings_columns_and_is_idempotent(clean, index):
    index.ensure_table()

    assert _rows(clean) == []
    columns = _run(
        clean,
        "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
        "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?",
        [SCHEMA, CONCEPTS],
    )
    names = {row[0].lower() for row in columns}
    assert {
        "section_id",
        "parent_id",
        "concept_id",
        "polarity",
        "mentions",
        "section_label",
    } <= names


def test_a_search_against_a_missing_table_names_ensure_table(vector_store):
    missing = ConceptIndex(vector_store, table=f"{SCHEMA}.NoSuchConcepts")

    with pytest.raises(ConceptIndexMissingError) as caught:
        missing.search([CHEST_PAIN])

    assert "ensure_table" in str(caught.value)


# --- the postings and the search ---------------------------------------------------------


def test_each_note_section_has_the_postings_the_text_gives(clean, indexed):
    rows = _rows(clean, "N-001")

    assert [(row[2], row[3], row[4], row[5]) for row in rows] == [
        (CHEST_PAIN, "affirmed", 1, "chief complaint"),
        (CHEST_PAIN, "affirmed", 1, "hpi"),
        (SOB, "negated", 1, "hpi"),
        (PCN, "affirmed", 1, "allergies"),
    ]


def test_affirmed_chest_pain_finds_the_notes_that_affirm_it(indexed):
    """Affirmed: N-001 (both), N-003 (rest), N-005 (affirmed 'chest pain with
    exertion' outweighs 'no chest pain at rest'), N-006 (both), N-008 (all four)."""
    index, _ = indexed

    hits = index.search([CHEST_PAIN], polarity="affirmed", top_k=10)

    assert _ids(hits) == {"N-001", "N-003", "N-005", "N-006", "N-008"}


def test_negated_chest_pain_finds_only_the_denial(indexed):
    """Negated: N-002 ('Patient denies chest pain'). N-005's posting is affirmed because
    the section polarity is affirmed if any mention is, so it does not count here."""
    index, _ = indexed

    hits = index.search([CHEST_PAIN], polarity="negated", top_k=10)

    assert _ids(hits) == {"N-002"}


def test_unknown_chest_pain_finds_the_hedged_mention(indexed):
    """Unknown: N-004 ('Possible chest pain, rule out MI')."""
    index, _ = indexed

    hits = index.search([CHEST_PAIN], polarity="unknown", top_k=10)

    assert _ids(hits) == {"N-004"}


def test_without_a_polarity_every_parent_with_the_concept_comes_back(indexed):
    index, _ = indexed

    hits = index.search([CHEST_PAIN], top_k=10)

    assert _ids(hits) == {"N-001", "N-002", "N-003", "N-004", "N-005", "N-006", "N-008"}


def test_penicillin_in_the_allergies_section_finds_two_notes(indexed):
    """N-007 has penicillin only in its history, so the allergies filter drops it."""
    index, _ = indexed

    hits = index.search([PCN], section="allergies", top_k=10)

    assert _ids(hits) == {"N-001", "N-004"}


def test_a_stroke_word_with_no_stroke_in_the_corpus_finds_nothing(indexed):
    index, _ = indexed

    assert index.search([STROKE], top_k=10) == []


def test_an_unknown_concept_id_matches_nothing(indexed):
    index, _ = indexed

    assert index.search(["C-NOT-IN-THE-DICTIONARY"], top_k=10) == []


def test_two_concepts_match_a_parent_only_when_it_has_both(indexed):
    """N-001 has chest pain (hpi, chief complaint) and shortness of breath (negated, hpi).
    N-002 has a denied chest pain in hpi and shortness of breath in its chief complaint,
    so with no polarity it matches too; N-001 has the larger score (3 against 2)."""
    index, _ = indexed

    hits = index.search([CHEST_PAIN, SOB], top_k=10)

    assert [hit.id for hit in hits] == ["N-001", "N-002"]
    assert [hit.metadata["text_score"] for hit in hits] == [3, 2]


def test_two_concepts_with_a_polarity_need_a_posting_of_that_polarity_for_each(indexed):
    index, _ = indexed

    assert index.search([CHEST_PAIN, SOB], polarity="affirmed", top_k=10) == []
    assert index.search([CHEST_PAIN, SOB], polarity="negated", top_k=10) == []


def test_top_k_keeps_the_parents_with_the_most_mentions(indexed):
    """Affirmed chest pain scores: N-008 4, N-006 3, N-001 2, N-005 2, N-003 1."""
    index, _ = indexed

    assert [h.id for h in index.search([CHEST_PAIN], polarity="affirmed", top_k=2)] == [
        "N-008",
        "N-006",
    ]
    assert [h.id for h in index.search([CHEST_PAIN], polarity="affirmed", top_k=3)] == [
        "N-008",
        "N-006",
        "N-001",
    ]


def test_a_hit_holds_its_best_section_and_lists_all_matched_sections(indexed):
    index, _ = indexed

    hits = {h.id: h for h in index.search([CHEST_PAIN], polarity="affirmed", top_k=10)}

    # N-006: the hpi section has two mentions, the chief complaint one.
    assert hits["N-006"].page_content == NOTES["N-006"]["sections"][1]["text"]
    # N-008: four sections with one mention each; the lowest ordinal wins the tie.
    assert hits["N-008"].page_content == "Chest pain since noon."
    matched = hits["N-008"].metadata["matched_sections"]
    assert [m["label"] for m in matched] == ["hpi", "ros", "plan", "disposition"]
    assert {m["concept_id"] for m in matched} == {CHEST_PAIN}
    assert {m["polarity"] for m in matched} == {"affirmed"}
    assert hits["N-008"].metadata["text_score"] == 4
    assert hits["N-008"].metadata["polarity"] == "affirmed"


def test_the_parent_polarity_is_affirmed_when_any_matched_posting_is(indexed):
    """N-002 has a negated chest pain posting and an affirmed shortness-of-breath
    posting. The parent polarity follows the any-affirmed rule, so it reads affirmed."""
    index, _ = indexed

    hits = {h.id: h for h in index.search([CHEST_PAIN, SOB], top_k=10)}

    assert hits["N-002"].metadata["polarity"] == "affirmed"
    assert hits["N-001"].metadata["polarity"] == "affirmed"


def test_an_empty_concept_list_or_a_bad_polarity_is_a_value_error(indexed):
    index, _ = indexed

    with pytest.raises(ValueError):
        index.search([])
    with pytest.raises(ValueError):
        index.search([CHEST_PAIN], polarity="maybe")


def test_three_runs_give_identical_results(indexed):
    index, _ = indexed

    runs = [
        [
            (h.id, h.page_content, h.metadata)
            for h in index.search([CHEST_PAIN, SOB], top_k=10)
        ]
        for _ in range(3)
    ]

    assert runs[0] == runs[1] == runs[2]


# --- replacing and failing ---------------------------------------------------------------


def test_indexing_a_parent_again_leaves_one_set_of_postings(clean, index, tagger):
    section_store = SectionStore(index.vector_store)
    note = NOTES["N-001"]
    sections = section_store.store(note["id"], note["text"], default_heading_detector)

    index.index_parent(note["id"], sections, tagger)
    first = _rows(clean, note["id"])
    index.index_parent(note["id"], sections, tagger)

    assert len(first) == 4
    assert _rows(clean, note["id"]) == first


def test_reindexing_with_new_text_replaces_the_old_postings(clean, index, tagger):
    old = [Section(ordinal=0, label="hpi", text="Penicillin given.", start=0, end=16)]
    new = [Section(ordinal=0, label="hpi", text="Chest pain.", start=0, end=11)]

    index.index_parent("N-x", old, tagger)
    index.index_parent("N-x", new, tagger)

    assert [(row[2], row[3]) for row in _rows(clean, "N-x")] == [
        (CHEST_PAIN, "affirmed")
    ]


def test_a_failed_write_rolls_back_and_keeps_the_earlier_postings(clean, index):
    tagger = DictionaryTagger({"C-" + "X" * 150: ["penicillin"]})
    section_store = SectionStore(index.vector_store)
    note = NOTES["N-004"]
    sections = section_store.store(note["id"], note["text"], default_heading_detector)
    index.index_parent(note["id"], sections, DictionaryTagger(DICTIONARY))
    before = _rows(clean, note["id"])
    assert before

    with pytest.raises(ConceptIndexError):
        index.index_parent(note["id"], sections, tagger)

    assert _rows(clean, note["id"]) == before
