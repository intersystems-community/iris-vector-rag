"""Concept tagging and per-section postings (spec 091), no IRIS and no model.

The dictionary here is invented for these tests; no clinical vocabulary is shipped.
"""

import dataclasses

import pytest

from iris_vector_rag.note_search.concepts import (
    ConceptMention,
    DictionaryTagger,
    postings_for_section,
)

CHEST_PAIN = "C-CHEST-PAIN"
SOB = "C-SOB"
PCN = "C-PCN"
STROKE = "C-STROKE"

DICTIONARY = {
    CHEST_PAIN: ["chest pain", "chest discomfort"],
    SOB: ["shortness of breath", "sob", "dyspnea"],
    PCN: ["penicillin"],
    STROKE: ["stroke", "cva"],
}


@pytest.fixture
def tagger():
    return DictionaryTagger(DICTIONARY)


def _spans(text, mentions):
    return [(m.concept_id, text[m.start : m.end]) for m in mentions]


# --- the tagger: matching -------------------------------------------------------------


def test_a_mention_has_the_character_span_of_its_matched_tokens(tagger):
    text = "Patient has Chest Pain today."

    mentions = tagger(text)

    assert mentions == [ConceptMention(CHEST_PAIN, 12, 22)]
    assert text[12:22] == "Chest Pain"


def test_matching_ignores_case(tagger):
    assert _spans("CHEST PAIN", tagger("CHEST PAIN")) == [(CHEST_PAIN, "CHEST PAIN")]


def test_a_longer_surface_form_wins_over_a_shorter_one_it_contains():
    tagger = DictionaryTagger({"C-CHEST": ["chest"], "C-CHEST-PAIN": ["chest pain"]})
    text = "chest pain"

    assert _spans(text, tagger(text)) == [("C-CHEST-PAIN", "chest pain")]


def test_overlapping_surface_forms_do_not_both_match():
    tagger = DictionaryTagger(
        {"C-SOB": ["shortness of breath"], "C-B": ["breath sounds"]}
    )
    text = "shortness of breath sounds"

    assert _spans(text, tagger(text)) == [("C-SOB", "shortness of breath")]


def test_a_surface_form_matches_on_token_boundaries_only(tagger):
    assert tagger("He stroked the cat.") == []
    assert tagger("Strokes are rare.") == []
    assert _spans("Prior stroke.", tagger("Prior stroke.")) == [(STROKE, "stroke")]


def test_punctuation_at_a_token_edge_does_not_block_a_match(tagger):
    text = "CVA, then sob."

    assert _spans(text, tagger(text)) == [(STROKE, "CVA"), (SOB, "sob")]


def test_punctuation_inside_a_word_is_part_of_the_token(tagger):
    assert tagger("Chest-pain noted.") == []


def test_mentions_come_back_in_text_order(tagger):
    text = "Stroke first, then penicillin, then CVA."

    assert [m.concept_id for m in tagger(text)] == [STROKE, PCN, STROKE]


def test_text_with_no_known_phrase_has_no_mentions(tagger):
    assert tagger("Afebrile, well appearing.") == []
    assert tagger("") == []


def test_repeated_mentions_are_each_returned(tagger):
    assert len(tagger("chest pain, chest pain again")) == 2


def test_a_mention_is_a_frozen_value():
    mention = ConceptMention("C-X", 0, 3)

    with pytest.raises(dataclasses.FrozenInstanceError):
        mention.start = 1  # type: ignore[misc]


# --- the tagger: dictionary errors ------------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_a_blank_surface_form_is_a_value_error(blank):
    with pytest.raises(ValueError):
        DictionaryTagger({"C-X": ["chest pain", blank]})


def test_one_surface_form_for_two_concepts_is_a_value_error_naming_both():
    with pytest.raises(ValueError) as caught:
        DictionaryTagger({"C-A": ["chest pain"], "C-B": ["Chest  Pain"]})

    assert "C-A" in str(caught.value)
    assert "C-B" in str(caught.value)


# --- postings for a section -----------------------------------------------------------


def test_a_denied_mention_is_a_negated_posting():
    assert postings_for_section("Denies chest pain.", DictionaryTagger(DICTIONARY)) == [
        {"concept_id": CHEST_PAIN, "polarity": "negated", "mentions": 1}
    ]


def test_one_affirmed_mention_makes_the_concept_affirmed_with_both_counted(tagger):
    postings = postings_for_section(
        "Denies chest pain. Chest discomfort on exertion.", tagger
    )

    assert postings == [
        {"concept_id": CHEST_PAIN, "polarity": "affirmed", "mentions": 2}
    ]


def test_postings_follow_the_order_in_which_concepts_first_appear(tagger):
    postings = postings_for_section(
        "Denies shortness of breath. Penicillin given.", tagger
    )

    assert postings == [
        {"concept_id": SOB, "polarity": "negated", "mentions": 1},
        {"concept_id": PCN, "polarity": "affirmed", "mentions": 1},
    ]


def test_a_section_with_no_concept_has_no_postings(tagger):
    assert postings_for_section("Well appearing.", tagger) == []


def test_a_stroked_word_gives_no_stroke_posting(tagger):
    assert postings_for_section("Stroked as a child.", tagger) == []


def test_a_caller_detector_judges_the_mentions_instead():
    tagger = DictionaryTagger(DICTIONARY)

    postings = postings_for_section(
        "Chest pain.", tagger, detector=lambda text, span: "unknown"
    )

    assert postings == [
        {"concept_id": CHEST_PAIN, "polarity": "unknown", "mentions": 1}
    ]


def test_the_detector_sees_the_span_of_each_mention_in_the_section_text():
    seen = []

    def record(text, span):
        seen.append(text[span[0] : span[1]])
        return "affirmed"

    postings_for_section(
        "Chest pain. Penicillin.", DictionaryTagger(DICTIONARY), detector=record
    )

    assert seen == ["Chest pain", "Penicillin"]
