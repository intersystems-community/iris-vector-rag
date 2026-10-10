"""Polarity (spec 088, US2; contracts/query-semantics.md section 5)."""

import pytest

from iris_vector_rag.note_search.phrase import find_mentions
from iris_vector_rag.note_search.polarity import (
    AFFIRMED,
    NEGATED,
    UNKNOWN,
    ReferenceNegExDetector,
    section_polarity,
)
from tests.fixtures.note_search.loader import load_vectors

VECTORS = load_vectors()["polarity"]


@pytest.mark.parametrize(
    "v",
    VECTORS,
    ids=[f'{v["text"][:30]}|{v["phrase"]}|{v["mention"]}' for v in VECTORS],
)
def test_reference_detector_on_the_shared_vectors(v):
    spans = find_mentions(v["text"], v["phrase"])
    assert ReferenceNegExDetector()(v["text"], spans[v["mention"]]) == v["expected"]


def test_a_trigger_does_not_reach_across_a_sentence_end():
    text = "Denies fever. Reports chest pain."
    (span,) = find_mentions(text, "chest pain")
    assert ReferenceNegExDetector()(text, span) == AFFIRMED


def test_a_decimal_point_does_not_end_a_sentence():
    text = "No 3.5 chest pain."
    (span,) = find_mentions(text, "chest pain")
    assert ReferenceNegExDetector()(text, span) == NEGATED


def test_a_newline_ends_a_sentence():
    text = "No fever\nchest pain"
    (span,) = find_mentions(text, "chest pain")
    assert ReferenceNegExDetector()(text, span) == AFFIRMED


def test_a_comma_does_not_end_a_sentence():
    text = "Denies fever, cough, chest pain"
    (span,) = find_mentions(text, "chest pain")
    assert ReferenceNegExDetector()(text, span) == NEGATED


def test_a_post_negation_trigger_negates():
    text = "Chest pain unlikely."
    (span,) = find_mentions(text, "chest pain")
    assert ReferenceNegExDetector()(text, span) == NEGATED


def test_a_terminator_stops_a_post_negation_trigger():
    text = "Chest pain, but pneumonia ruled out."
    (span,) = find_mentions(text, "chest pain")
    assert ReferenceNegExDetector()(text, span) == AFFIRMED


def test_a_mention_across_a_sentence_end_is_judged_where_it_starts():
    text = "No chest pain. Chest pain worse"
    spans = find_mentions(text, "pain chest")
    assert ReferenceNegExDetector()(text, spans[0]) == NEGATED


@pytest.mark.parametrize(
    "polarities,expected",
    [
        ([AFFIRMED, NEGATED], AFFIRMED),
        ([NEGATED, UNKNOWN], NEGATED),
        ([UNKNOWN], UNKNOWN),
        ([NEGATED, NEGATED], NEGATED),
        ([UNKNOWN, AFFIRMED, NEGATED], AFFIRMED),
        ([], UNKNOWN),
    ],
)
def test_section_polarity_is_affirmed_if_any_is(polarities, expected):
    assert section_polarity(polarities) == expected
