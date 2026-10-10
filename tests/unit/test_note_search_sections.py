"""Section splitting (spec 088, US1; contracts/query-semantics.md section 2)."""

import pytest

from iris_vector_rag.note_search.sections import (
    Heading,
    default_heading_detector,
    normalise_label,
    split_sections,
)
from tests.fixtures.note_search.loader import load_vectors

NOTES = load_vectors()["notes"]


@pytest.mark.parametrize("note", NOTES, ids=[n["id"] for n in NOTES])
def test_every_fixture_note_splits_into_its_known_sections(note):
    sections = split_sections(note["text"], default_heading_detector)
    assert [(s.label, s.text) for s in sections] == [
        (s["label"], s["text"]) for s in note["sections"]
    ]
    assert [s.ordinal for s in sections] == list(range(len(sections)))


@pytest.mark.parametrize("note", NOTES, ids=[n["id"] for n in NOTES])
def test_offsets_point_at_the_section_text_in_the_note(note):
    for s in split_sections(note["text"], default_heading_detector):
        assert note["text"][s.start : s.end] == s.text


@pytest.mark.parametrize("note", NOTES, ids=[n["id"] for n in NOTES])
def test_no_non_whitespace_character_is_lost(note):
    text = note["text"]
    sections = split_sections(text, default_heading_detector)
    headings = default_heading_detector(text)
    covered = [False] * len(text)
    for s in sections:
        for i in range(s.start, s.end):
            covered[i] = True
    for h in headings:
        for i in range(h.start, h.end):
            covered[i] = True
    lost = [i for i, c in enumerate(text) if not c.isspace() and not covered[i]]
    assert lost == []


def test_a_note_with_no_heading_is_one_rest_section():
    (s,) = split_sections("Seen for pain.", default_heading_detector)
    assert (s.label, s.text, s.start, s.end) == ("rest", "Seen for pain.", 0, 14)


def test_text_before_the_first_heading_is_a_rest_section():
    first, second = split_sections(
        "Triage note.\nHPI: Cough.", default_heading_detector
    )
    assert (first.label, first.text) == ("rest", "Triage note.")
    assert (second.label, second.text) == ("hpi", "Cough.")


def test_an_empty_note_has_no_sections():
    assert split_sections("", default_heading_detector) == []
    assert split_sections("  \n ", default_heading_detector) == []


def test_a_heading_with_no_text_yields_no_section():
    sections = split_sections("HPI:\nPLAN: Rest.", default_heading_detector)
    assert [s.label for s in sections] == ["plan"]
    assert sections[0].ordinal == 0


def test_the_caller_supplied_detector_decides_where_sections_begin():
    def detector(text):
        i = text.index("##")
        return [Heading(start=i, end=i + 2, label="Custom")]

    sections = split_sections("before ## after", detector)
    assert [(s.label, s.text) for s in sections] == [
        ("rest", "before"),
        ("custom", "after"),
    ]


def test_the_default_detector_needs_a_heading_at_the_start_of_a_line():
    assert default_heading_detector("Penicillin: rash. NOTE: x") == []
    assert [h.label for h in default_heading_detector("  hpi: x\nHPI: y")] == ["HPI"]


def test_the_label_map_renames_after_normalising():
    sections = split_sections(
        "ALLERGY: Penicillin.",
        default_heading_detector,
        label_map={"allergy": "allergies"},
    )
    assert sections[0].label == "allergies"


def test_an_unmapped_heading_keeps_its_normalised_text():
    (s,) = split_sections(
        "PAST  MEDICAL HISTORY: None.",
        default_heading_detector,
        label_map={"allergy": "allergies"},
    )
    assert s.label == "past medical history"


@pytest.mark.parametrize(
    "raw,expected",
    [("HPI:", "hpi"), ("Chief   Complaint :", "chief complaint"), ("ros", "ros")],
)
def test_normalise_label(raw, expected):
    assert normalise_label(raw) == expected


def test_the_splitter_knows_no_heading_names():
    import inspect

    from iris_vector_rag.note_search import sections

    source = inspect.getsource(sections.split_sections)
    for name in ("hpi", "allerg", "complaint", "medication"):
        assert name not in source.lower()
