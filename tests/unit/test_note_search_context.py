"""Experiencer and temporality (spec 088, contracts/query-semantics.md section 5b)."""

import dataclasses

import pytest

from iris_vector_rag.note_search.context import (
    CURRENT,
    HISTORICAL,
    HYPOTHETICAL,
    OTHER,
    PATIENT,
    UNKNOWN,
    Context,
    ReferenceContextDetector,
    section_context,
)
from iris_vector_rag.note_search.phrase import find_mentions
from iris_vector_rag.note_search.polarity import AFFIRMED, ReferenceNegExDetector
from tests.fixtures.note_search.loader import load_vectors

VECTORS = load_vectors()["context"]


def _judge(text, phrase, mention=0):
    (span,) = find_mentions(text, phrase)[mention : mention + 1]
    return ReferenceContextDetector()(text, span)


@pytest.mark.parametrize(
    "v",
    VECTORS,
    ids=[f"{v['text'][:30]}|{v['phrase']}|{v['mention']}" for v in VECTORS],
)
def test_reference_detector_on_the_shared_context_vectors(v):
    spans = find_mentions(v["text"], v["phrase"])
    context = ReferenceContextDetector()(v["text"], spans[v["mention"]])
    assert context == Context(v["experiencer"], v["temporality"])


def test_the_baystate_sentence_is_other_and_current_but_still_affirmed():
    text = "Mother had a stroke at 70; patient has no neurological history."
    (span,) = find_mentions(text, "stroke")
    assert ReferenceContextDetector()(text, span) == Context(OTHER, CURRENT)
    # Context is a separate answer: polarity is unchanged by it.
    assert ReferenceNegExDetector()(text, span) == AFFIRMED


# --- experiencer --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Father has chest pain.",
        "Sister reports chest pain.",
        "Grandmother had chest pain.",
        "Parents deny chest pain in the family.",
        "Cousin with chest pain.",
    ],
)
def test_a_family_term_before_the_mention_makes_it_other(text):
    assert _judge(text, "chest pain").experiencer == OTHER


def test_a_family_term_after_the_mention_does_not_count():
    assert _judge("Chest pain like her mother.", "chest pain").experiencer == PATIENT


def test_a_family_term_in_an_earlier_sentence_does_not_count():
    text = "Mother is well. Chest pain started today."
    assert _judge(text, "chest pain").experiencer == PATIENT


@pytest.mark.parametrize("terminator", ["but", "however", "although"])
def test_a_terminator_stops_a_family_term(terminator):
    text = f"Mother is well {terminator} chest pain started."
    assert _judge(text, "chest pain").experiencer == PATIENT


def test_family_history_is_other():
    assert _judge("Family history of chest pain.", "chest pain").experiencer == OTHER


def test_fh_and_fhx_are_other():
    assert _judge("FH: chest pain.", "chest pain").experiencer == OTHER
    assert _judge("FHx chest pain.", "chest pain").experiencer == OTHER


def test_a_hyphenated_relative_is_not_a_family_term():
    # Tokens keep inner hyphens, so "brother-in-law" is not "brother".
    text = "Brother-in-law drove her; she reports chest pain."
    assert _judge(text, "chest pain").experiencer == PATIENT


def test_a_mother_with_a_history_of_is_other_and_historical():
    text = "Mother had a history of stroke."
    assert _judge(text, "stroke") == Context(OTHER, HISTORICAL)


# --- temporality --------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "If chest pain worsens, call.",
        "Should chest pain return, call.",
        "Return if chest pain worsens.",
        "Return for chest pain.",
        "Call if chest pain.",
        "In case of chest pain, call.",
        "Risk of chest pain.",
        "Monitor for chest pain.",
        "Worry about chest pain.",
        "Worried about chest pain.",
    ],
)
def test_hypothetical_triggers_before_the_mention(text):
    assert _judge(text, "chest pain").temporality == HYPOTHETICAL


@pytest.mark.parametrize(
    "text",
    [
        "Patient has a history of chest pain.",
        "H/o chest pain.",
        "Hx of chest pain.",
        "Previous chest pain.",
        "Previously chest pain.",
        "Prior chest pain.",
        "Past chest pain.",
        "Remote chest pain.",
        "Resolved chest pain.",
    ],
)
def test_historical_triggers_before_the_mention(text):
    assert _judge(text, "chest pain").temporality == HISTORICAL


def test_resolved_after_the_mention_is_historical():
    assert _judge("Chest pain resolved.", "chest pain").temporality == HISTORICAL


def test_a_terminator_stops_a_resolved_after_the_mention():
    assert _judge("Chest pain but resolved.", "chest pain").temporality == CURRENT


def test_a_trigger_after_the_mention_is_not_hypothetical():
    assert _judge("Chest pain if worse.", "chest pain").temporality == CURRENT


def test_hypothetical_outranks_historical():
    text = "If prior chest pain returns, call."
    assert _judge(text, "chest pain").temporality == HYPOTHETICAL


def test_a_terminator_stops_a_hypothetical_trigger():
    text = "If fine, but chest pain started."
    assert _judge(text, "chest pain").temporality == CURRENT


def test_a_trigger_in_an_earlier_sentence_does_not_count():
    text = "Return if stable. Chest pain started today."
    assert _judge(text, "chest pain").temporality == CURRENT


def test_plain_mention_is_current():
    assert _judge("Chest pain started today.", "chest pain") == Context(
        PATIENT, CURRENT
    )


def test_family_history_is_not_a_patient_history():
    # "history of" is part of the family-history cue, which names the experiencer.
    assert _judge("Family history of stroke.", "stroke").temporality == CURRENT


def test_a_newline_ends_the_sentence_for_context():
    text = "Mother is well\nChest pain started today."
    assert _judge(text, "chest pain").experiencer == PATIENT


# --- the value object and the aggregate ----------------------------------------------


def test_context_is_a_frozen_value():
    context = Context(PATIENT, CURRENT)
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.experiencer = OTHER


def test_constants_have_the_contract_values():
    assert (PATIENT, OTHER, CURRENT, HISTORICAL, HYPOTHETICAL, UNKNOWN) == (
        "patient",
        "other",
        "current",
        "historical",
        "hypothetical",
        "unknown",
    )


def test_section_context_of_nothing_is_unknown():
    assert section_context([]) == Context(UNKNOWN, UNKNOWN)


@pytest.mark.parametrize(
    "contexts,expected",
    [
        ([Context(PATIENT, CURRENT)], Context(PATIENT, CURRENT)),
        ([Context(OTHER, CURRENT)], Context(OTHER, CURRENT)),
        (
            [Context(OTHER, CURRENT), Context(PATIENT, HISTORICAL)],
            Context(PATIENT, CURRENT),
        ),
        (
            [Context(OTHER, HISTORICAL), Context(OTHER, HYPOTHETICAL)],
            Context(OTHER, HISTORICAL),
        ),
        (
            [Context(PATIENT, HYPOTHETICAL), Context(OTHER, HISTORICAL)],
            Context(PATIENT, HISTORICAL),
        ),
        ([Context(PATIENT, HYPOTHETICAL)], Context(PATIENT, HYPOTHETICAL)),
        ([Context(OTHER, HYPOTHETICAL)], Context(OTHER, HYPOTHETICAL)),
    ],
)
def test_section_context_prefers_patient_and_current(contexts, expected):
    assert section_context(contexts) == expected
