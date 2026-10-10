"""The embedded-Python entry points for callers outside this package (spec 088).

Baystate calls the detector from ObjectScript: a section's text and the character spans
(0-based, end exclusive, as Python slices) in, plain values out.
"""

import json

import pytest

from iris_vector_rag.note_search.api import (
    classify,
    classify_context,
    judge_section,
    judge_section_json,
)
from iris_vector_rag.note_search.context import (
    CURRENT,
    HISTORICAL,
    HYPOTHETICAL,
    OTHER,
    PATIENT,
    Context,
)
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
    assert result == {
        "polarity": UNKNOWN,
        "experiencer": UNKNOWN,
        "temporality": UNKNOWN,
        "mentions": [],
    }


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


def test_judge_section_adds_experiencer_and_temporality_to_each_mention():
    text = "Mother had chest pain. Chest pain resolved."
    result = judge_section(text, "chest pain")
    assert [(m["experiencer"], m["temporality"]) for m in result["mentions"]] == [
        (OTHER, CURRENT),
        (PATIENT, HISTORICAL),
    ]


def test_judge_section_rolls_context_up_over_the_mentions():
    text = "Mother had chest pain. Chest pain resolved."
    result = judge_section(text, "chest pain")
    assert result["experiencer"] == PATIENT
    assert result["temporality"] == CURRENT


def test_judge_section_keeps_polarity_separate_from_context():
    result = judge_section("Mother had a stroke.", "stroke")
    assert result["polarity"] == AFFIRMED
    assert result["experiencer"] == OTHER


def test_judge_section_uses_the_context_detector_it_is_given():
    def detector(text, span):
        return Context(OTHER, HYPOTHETICAL)

    result = judge_section("chest pain", "chest pain", context_detector=detector)
    assert result["experiencer"] == OTHER
    assert result["temporality"] == HYPOTHETICAL
    assert result["mentions"][0]["temporality"] == HYPOTHETICAL


def test_judge_section_json_carries_the_context_keys():
    decoded = json.loads(judge_section_json("Mother had chest pain.", "chest pain"))
    assert decoded["experiencer"] == OTHER
    assert decoded["temporality"] == CURRENT


def test_classify_context_gives_one_answer_per_span_in_order():
    text = "Denies chest pain. Father had chest pain."
    spans = [(7, 17), (30, 40)]
    assert text[7:17] == "chest pain" and text[30:40] == "chest pain"
    assert classify_context(text, spans) == [
        {"experiencer": PATIENT, "temporality": CURRENT},
        {"experiencer": OTHER, "temporality": CURRENT},
    ]


def test_classify_context_uses_the_context_detector_it_is_given():
    answer = classify_context(
        "x y", [(0, 1)], context_detector=lambda text, span: Context(OTHER, UNKNOWN)
    )
    assert answer == [{"experiencer": OTHER, "temporality": UNKNOWN}]


@pytest.mark.parametrize("span", [(-1, 3), (3, 2), (0, 99)])
def test_classify_context_refuses_a_span_outside_the_text(span):
    with pytest.raises(ValueError, match="span"):
        classify_context("short", [span])


# --- batch forms: one call for many sections -------------------------------------


def test_judge_sections_answers_each_text_in_order():
    from iris_vector_rag.note_search.api import judge_sections

    texts = ["Denies chest pain.", "Reports chest pain.", "No fever."]
    answers = judge_sections(texts, "chest pain")
    assert [a["polarity"] for a in answers] == [NEGATED, AFFIRMED, UNKNOWN]
    assert answers == [judge_section(t, "chest pain") for t in texts]


def test_judge_sections_with_no_texts_is_empty():
    from iris_vector_rag.note_search.api import judge_sections

    assert judge_sections([], "chest pain") == []


def test_judge_sections_refuses_a_bad_phrase_once_even_with_no_texts():
    from iris_vector_rag.note_search.api import judge_sections

    with pytest.raises(ValueError, match="letter or digit"):
        judge_sections([], "...")


def test_judge_sections_json_takes_and_returns_json_strings():
    from iris_vector_rag.note_search.api import judge_sections_json

    out = json.loads(
        judge_sections_json(json.dumps(["Denies chest pain."]), "chest pain")
    )
    assert out[0]["polarity"] == NEGATED


def test_judge_sections_json_refuses_something_that_is_not_a_list_of_strings():
    from iris_vector_rag.note_search.api import judge_sections_json

    for bad in ('{"a": 1}', "[1, 2]", "not json"):
        with pytest.raises(ValueError):
            judge_sections_json(bad, "chest pain")
