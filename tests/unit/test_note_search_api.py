"""The embedded-Python entry points for callers outside this package (spec 088).

Baystate calls the detector from ObjectScript: a section's text and the character spans
(0-based, end exclusive, as Python slices) in, plain values out.
"""

import json

import pytest

from iris_vector_rag.note_search.api import classify, judge_section, judge_section_json
from iris_vector_rag.note_search.polarity import AFFIRMED, NEGATED, UNKNOWN


def test_classify_judges_each_span_in_order():
    text = "Denies chest pain. Reports chest pain."
    spans = [(7, 17), (27, 37)]
    assert text[7:17] == "chest pain" and text[27:37] == "chest pain"
    assert classify(text, spans) == [NEGATED, AFFIRMED]


def test_classify_takes_lists_as_well_as_tuples():
    assert classify("No fever.", [[3, 8]]) == [NEGATED]


def test_classify_with_no_spans_is_empty():
    assert classify("anything", []) == []


def test_classify_uses_the_detector_it_is_given():
    assert classify("x y", [(0, 1)], detector=lambda text, span: UNKNOWN) == [UNKNOWN]


@pytest.mark.parametrize("span", [(-1, 3), (3, 2), (0, 99)])
def test_classify_refuses_a_span_outside_the_text(span):
    with pytest.raises(ValueError, match="span"):
        classify("short", [span])


def test_judge_section_finds_every_mention_and_judges_it():
    text = "No chest pain at rest. Chest pain with exertion."
    result = judge_section(text, "chest pain")
    assert result["polarity"] == AFFIRMED
    assert [(m["start"], m["end"], m["polarity"]) for m in result["mentions"]] == [
        (3, 13, NEGATED),
        (23, 33, AFFIRMED),
    ]


def test_judge_section_with_no_mention_says_so():
    # The index can propose a section the phrase is not really in (reverse order at the
    # start of a text); the caller drops a section whose mentions list is empty.
    result = judge_section("Chest pain here.", "pain chest")
    assert result == {"polarity": UNKNOWN, "mentions": []}


def test_judge_section_refuses_a_phrase_without_a_letter_or_digit():
    with pytest.raises(ValueError, match="letter or digit"):
        judge_section("text", "...")


def test_judge_section_json_is_the_same_answer_as_a_string():
    text = "Denies chest pain."
    assert json.loads(judge_section_json(text, "chest pain")) == judge_section(
        text, "chest pain"
    )


def test_judge_section_uses_the_detector_it_is_given():
    result = judge_section("chest pain", "chest pain", detector=lambda t, s: NEGATED)
    assert result["polarity"] == NEGATED
