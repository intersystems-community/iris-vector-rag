"""Note search contract, no IRIS: the phrase mode, QueryOptions and the vectors file.

Covers what a second implementation (fhir-017) depends on: the `phrase` mode's
registration, the `section` field on QueryOptions, the shape of
tests/fixtures/note_search/vectors.json, a loader with no iris_vector_rag import, and
SC-004: no retrieval module carries the reference detector's trigger words or the
fixture's heading names.
"""

import ast
import dataclasses
from pathlib import Path

import pytest

from iris_vector_rag.core.query_options import QueryOptions
from iris_vector_rag.retrieval.modes import get_mode, list_modes
from tests.fixtures.note_search.loader import load_vectors

REPO = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO / "tests" / "fixtures" / "note_search"
LOADER = FIXTURE_DIR / "loader.py"
RETRIEVAL_DIR = REPO / "iris_vector_rag" / "retrieval"

EXPECTATIONS = ("parents", "count", "same_as", "error", "ranked_before")
POLARITIES = {"affirmed", "negated", "unknown"}
KNOWN_ERRORS = {"ValueError"}
TRIGGER_WORDS = {
    "denies",
    "denied",
    "without",
    "unlikely",
    "questionable",
    "possibly",
    "ruled out",
    "negative for",
    "absence of",
}
HEADING_NAMES = {"chief complaint", "allergies", "medications given"}

VECTORS = load_vectors()


# --- the phrase mode and QueryOptions ----------------------------------------------


def test_the_phrase_mode_is_registered_over_the_text_source():
    mode = get_mode("phrase")

    assert mode.name == "phrase"
    assert mode.sources == ["text"]
    assert mode.requires == ["text_index"]
    assert mode.fusion is None


def test_the_phrase_mode_is_listed():
    assert "phrase" in list_modes()


def test_query_options_takes_an_optional_section_defaulting_to_none():
    names = {field.name for field in dataclasses.fields(QueryOptions)}

    assert "section" in names
    assert QueryOptions(query="chest pain").section is None
    assert QueryOptions(query="chest pain", section="hpi").section == "hpi"


# --- the vectors file ----------------------------------------------------------------


def test_the_vectors_file_has_its_top_level_keys():
    assert {"version", "notes", "queries", "polarity"} <= set(VECTORS)


def test_every_query_has_id_phrase_section_and_k():
    for query in VECTORS["queries"]:
        assert {"id", "phrase", "section", "k"} <= set(query), query.get("id")


def test_every_query_has_exactly_one_expectation():
    for query in VECTORS["queries"]:
        present = [key for key in EXPECTATIONS if key in query]
        assert len(present) == 1, f"{query['id']} has {present}"


def test_query_ids_are_unique():
    ids = [query["id"] for query in VECTORS["queries"]]

    assert len(ids) == len(set(ids))


def test_every_same_as_and_subset_of_names_an_existing_query():
    ids = {query["id"] for query in VECTORS["queries"]}
    for query in VECTORS["queries"]:
        for key in ("same_as", "subset_of"):
            if key in query:
                assert query[key] in ids, f"{query['id']} {key} -> {query[key]}"


def test_note_ids_are_unique_and_each_note_has_sections():
    ids = [note["id"] for note in VECTORS["notes"]]

    assert len(ids) == len(set(ids))
    for note in VECTORS["notes"]:
        assert note["sections"], note["id"]
        assert all(section["label"] and section["text"] for section in note["sections"])


def test_each_expected_parent_is_a_note_with_those_sections():
    notes = {note["id"]: note for note in VECTORS["notes"]}
    for query in VECTORS["queries"]:
        for parent in query.get("parents", []):
            assert parent["id"] in notes, f"{query['id']} names {parent['id']}"
            assert parent["polarity"] in POLARITIES
            labels = {section["label"] for section in notes[parent["id"]]["sections"]}
            assert set(parent["sections"]) <= labels, query["id"]


def test_error_queries_name_a_known_error():
    for query in VECTORS["queries"]:
        if "error" in query:
            assert query["error"] in KNOWN_ERRORS, query["id"]


def test_ranked_before_names_notes_that_exist():
    notes = {note["id"] for note in VECTORS["notes"]}
    for query in VECTORS["queries"]:
        for note_id in query.get("ranked_before", []):
            assert note_id in notes, query["id"]


def test_polarity_vectors_have_text_phrase_mention_and_a_known_value():
    for case in VECTORS["polarity"]:
        assert {"text", "phrase", "mention", "expected"} <= set(case)
        assert case["expected"] in POLARITIES
        assert isinstance(case["mention"], int)


# --- the loader and the no-trigger-words check ----------------------------------------


def test_the_loader_imports_nothing_from_iris_vector_rag():
    tree = ast.parse(LOADER.read_text(encoding="utf-8"), filename=str(LOADER))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")

    assert not [name for name in imported if name.split(".")[0] == "iris_vector_rag"]


def _normalised_literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield " ".join(node.value.lower().split()).rstrip(":")


def test_no_retrieval_module_holds_a_trigger_word_or_a_heading_name():
    """SC-004: the reference detector's words live in note_search/polarity.py only."""
    modules = sorted(RETRIEVAL_DIR.rglob("*.py"))
    assert modules, "no retrieval modules found"

    forbidden = TRIGGER_WORDS | HEADING_NAMES
    offenders = [
        f"{path.relative_to(REPO)}: {literal!r}"
        for path in modules
        for literal in _normalised_literals(path)
        if literal in forbidden
    ]

    assert offenders == []


@pytest.mark.parametrize("word", sorted(TRIGGER_WORDS | HEADING_NAMES))
def test_the_check_would_catch_each_listed_word(tmp_path, word):
    """The grep above is only as good as its reach: each listed word must be caught."""
    probe = tmp_path / "probe.py"
    probe.write_text(f"WORDS = [{word!r}]\n", encoding="utf-8")

    assert word in set(_normalised_literals(probe))
