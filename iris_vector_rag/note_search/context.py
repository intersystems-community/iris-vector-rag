"""Experiencer and temporality for note search (spec 088, contracts/query-semantics.md 5b).

Experiencer says who a mention is about (the patient, or someone else); temporality says
when it is (current, historical or hypothetical). Both are separate from polarity: "Mother
had a stroke" is affirmed and experiencer other.

``ReferenceContextDetector`` is a reference detector. It is not validated on
emergency-department text and no accuracy is claimed for it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from .polarity import _sentence_tokens, _triggered

PATIENT = "patient"
OTHER = "other"
CURRENT = "current"
HISTORICAL = "historical"
HYPOTHETICAL = "hypothetical"
UNKNOWN = "unknown"


@dataclass(frozen=True)
class Context:
    """The experiencer and temporality of one mention."""

    experiencer: str
    temporality: str


ContextDetector = Callable[[str, tuple[int, int]], Context]

# Trigger lists are tuples of lower-case token tuples so they are easy to replace.
# A family term before the mention, in the same sentence and with no terminator between,
# makes the experiencer other.
FAMILY_TERMS: tuple[tuple[str, ...], ...] = (
    ("mother",),
    ("father",),
    ("sister",),
    ("brother",),
    ("son",),
    ("daughter",),
    ("grandmother",),
    ("grandfather",),
    ("aunt",),
    ("uncle",),
    ("cousin",),
    ("parent",),
    ("parents",),
    ("sibling",),
    ("siblings",),
    ("family", "history"),
    ("fh",),
    ("fhx",),
)
# The words of a family-history cue. They are removed before the temporality triggers are
# checked, so "history of" inside "family history of X" does not make X historical.
FAMILY_HISTORY_CUE: tuple[str, ...] = ("family", "history")
HYPOTHETICAL_TRIGGERS: tuple[tuple[str, ...], ...] = (
    ("if",),
    ("should",),
    ("return", "for"),
    ("return", "if"),
    ("call", "if"),
    ("in", "case", "of"),
    ("risk", "of"),
    ("monitor", "for"),
    ("worry", "about"),
    ("worried", "about"),
    # Discharge-instruction wording: advice about what to do if a symptom appears.
    ("call", "your", "doctor", "for"),
    ("call", "911", "for"),
    ("return", "to", "the", "ed", "for"),
    ("return", "to", "the", "er", "for"),
    ("return", "to", "the", "emergency", "department", "for"),
    ("seek", "medical", "attention", "for"),
    ("seek", "care", "for"),
)
HISTORICAL_BEFORE_TRIGGERS: tuple[tuple[str, ...], ...] = (
    ("history", "of"),
    ("h/o",),
    ("hx", "of"),
    ("previous",),
    ("previously",),
    ("prior",),
    ("past",),
    ("remote",),
    ("resolved",),
)
HISTORICAL_AFTER_TRIGGERS: tuple[tuple[str, ...], ...] = (("resolved",),)


def _without_family_history_cue(words: list[str]) -> list[str]:
    """``words`` with each family-history cue removed, so its words cannot trigger."""
    size = len(FAMILY_HISTORY_CUE)
    kept: list[str] = []
    index = 0
    while index < len(words):
        if tuple(words[index : index + size]) == FAMILY_HISTORY_CUE:
            index += size
            continue
        kept.append(words[index])
        index += 1
    return kept


class ReferenceContextDetector:
    """Reference experiencer and temporality detector, section 5b of the query contract."""

    def __call__(self, text: str, span: tuple[int, int]) -> Context:
        """Return the experiencer and temporality of the mention at span within text."""
        start, end = span
        tokens = _sentence_tokens(text, start)
        # As in polarity: tokens inside the mention are ignored, and a mention is judged
        # in the sentence where it starts.
        pre = [token.text for token in tokens if token.end <= start]
        post = [token.text for token in tokens if token.start >= end]

        if _triggered(pre, FAMILY_TERMS, True):
            experiencer = OTHER
        else:
            experiencer = PATIENT

        pre_for_time = _without_family_history_cue(pre)
        if _triggered(pre_for_time, HYPOTHETICAL_TRIGGERS, True):
            temporality = HYPOTHETICAL
        elif _triggered(pre_for_time, HISTORICAL_BEFORE_TRIGGERS, True) or _triggered(
            post, HISTORICAL_AFTER_TRIGGERS, False
        ):
            temporality = HISTORICAL
        else:
            temporality = CURRENT
        return Context(experiencer, temporality)


def section_context(contexts: Iterable[Context]) -> Context:
    """Roll mention contexts up to a section.

    Experiencer is patient if any mention is, else other. Temporality is current if any
    mention is, else historical if any is, else hypothetical if any is. With no mentions
    both are unknown.
    """
    values = list(contexts)
    if not values:
        return Context(UNKNOWN, UNKNOWN)
    experiencers = {c.experiencer for c in values}
    temporalities = {c.temporality for c in values}
    if PATIENT in experiencers:
        experiencer = PATIENT
    else:
        experiencer = OTHER
    for candidate in (CURRENT, HISTORICAL, HYPOTHETICAL):
        if candidate in temporalities:
            temporality = candidate
            break
    else:
        temporality = UNKNOWN
    return Context(experiencer, temporality)
