"""AllowlistHeadingDetector: headings from a list of known names, in the casings given.

Baystate's notes show that a casing pattern alone over-catches (12.9 Title Case
candidates per note); a list of known headings plus the forms to accept does not. The
labels here are invented; the library ships none.
"""

import pytest

from iris_vector_rag.note_search.sections import (
    AllowlistHeadingDetector,
    Heading,
    split_sections,
)


def found(detector, text):
    return [(h.label, text[h.start : h.end]) for h in detector(text)]


def test_an_all_caps_heading_with_a_colon_matches_by_default():
    d = AllowlistHeadingDetector(["Plan"])
    assert found(d, "PLAN: rest.") == [("Plan", "PLAN:")]


def test_the_label_is_the_one_given_not_the_one_found():
    d = AllowlistHeadingDetector(["Medical Decision Making"])
    assert found(d, "MEDICAL  DECISION MAKING\nx") == [
        ("Medical Decision Making", "MEDICAL  DECISION MAKING")
    ]


def test_a_title_case_line_on_its_own_matches_when_title_is_accepted():
    d = AllowlistHeadingDetector(["Plan"], accept=("title",))
    assert found(d, "Plan\nGo home.") == [("Plan", "Plan")]


def test_a_title_case_word_starting_a_sentence_is_not_a_heading():
    d = AllowlistHeadingDetector(["Plan"], accept=("title",))
    assert found(d, "Plan to discharge today.") == []


def test_standalone_false_lets_a_colon_less_heading_share_its_line():
    d = AllowlistHeadingDetector(["Plan"], accept=("upper",), standalone=False)
    assert found(d, "PLAN go home") == [("Plan", "PLAN")]


def test_a_casing_that_is_not_accepted_does_not_match():
    d = AllowlistHeadingDetector(["Plan"], accept=("upper",))
    assert found(d, "plan:\nPlan:\n") == []


def test_any_case_accepts_every_casing():
    d = AllowlistHeadingDetector(["Plan"], accept=("any_case",))
    assert [t for _, t in found(d, "plan:\nPlan:\nPLAN:\npLaN:")] == [
        "plan:",
        "Plan:",
        "PLAN:",
        "pLaN:",
    ]


def test_a_heading_must_start_its_line():
    d = AllowlistHeadingDetector(["Plan"])
    assert found(d, "see PLAN: later") == []
    assert found(d, "   PLAN: indented") == [("Plan", "PLAN:")]


def test_the_longest_label_wins_at_a_position():
    d = AllowlistHeadingDetector(["Medical", "Medical Decision Making"])
    assert found(d, "MEDICAL DECISION MAKING:\nx") == [
        ("Medical Decision Making", "MEDICAL DECISION MAKING:")
    ]


def test_colon_required_refuses_a_colon_less_line():
    d = AllowlistHeadingDetector(["Plan"], colon="required")
    assert found(d, "PLAN\nPLAN:\n") == [("Plan", "PLAN:")]


def test_colon_forbidden_refuses_a_heading_with_a_colon():
    d = AllowlistHeadingDetector(["Plan"], colon="forbidden")
    assert found(d, "PLAN:\nPLAN\n") == [("Plan", "PLAN")]


def test_a_label_with_regex_characters_is_taken_literally():
    d = AllowlistHeadingDetector(["Hx (past)"], accept=("any_case",))
    assert found(d, "HX (PAST): none") == [("Hx (past)", "HX (PAST):")]
    assert found(d, "HX PAST: none") == []


def test_no_labels_is_an_error():
    with pytest.raises(ValueError, match="at least one"):
        AllowlistHeadingDetector([])


@pytest.mark.parametrize(
    "kwargs",
    [{"accept": ("shouting",)}, {"colon": "sometimes"}, {"accept": ()}],
)
def test_an_unknown_option_is_an_error(kwargs):
    with pytest.raises(ValueError):
        AllowlistHeadingDetector(["Plan"], **kwargs)


def test_it_drives_split_sections_and_a_label_map():
    d = AllowlistHeadingDetector(["Diagnosis", "Plan"], accept=("upper", "title"))
    text = "Triage done.\nDIAGNOSIS\nChest pain.\nPlan\nAdmit."
    sections = split_sections(text, d, label_map={"plan": "disposition"})
    assert [(s.label, s.text) for s in sections] == [
        ("rest", "Triage done."),
        ("diagnosis", "Chest pain."),
        ("disposition", "Admit."),
    ]


def test_the_detector_returns_headings_in_text_order():
    d = AllowlistHeadingDetector(["Plan", "Diagnosis"])
    hs = d("DIAGNOSIS:\nx\nPLAN:\ny")
    assert [h.label for h in hs] == ["Diagnosis", "Plan"]
    assert all(isinstance(h, Heading) for h in hs)
