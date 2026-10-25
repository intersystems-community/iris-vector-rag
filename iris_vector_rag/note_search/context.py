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

from .polarity import TERMINATORS, _sentence_tokens, _triggered

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


@dataclass(frozen=True)
class ContextLexicon:
    """The words one language uses for who a mention is about, and when it is."""

    family_terms: tuple[tuple[str, ...], ...]
    family_history_cue: tuple[str, ...]
    hypothetical: tuple[tuple[str, ...], ...]
    historical_before: tuple[tuple[str, ...], ...]
    historical_after: tuple[tuple[str, ...], ...]
    terminators: frozenset[str]


ENGLISH = ContextLexicon(
    family_terms=FAMILY_TERMS,
    family_history_cue=FAMILY_HISTORY_CUE,
    hypothetical=HYPOTHETICAL_TRIGGERS,
    historical_before=HISTORICAL_BEFORE_TRIGGERS,
    historical_after=HISTORICAL_AFTER_TRIGGERS,
    terminators=TERMINATORS,
)

#: The lexicons shipped, by language code. Another language is a caller's ``lexicon=``.
CONTEXT_LEXICONS: dict[str, ContextLexicon] = {"en": ENGLISH}


def context_lexicon(language: str) -> ContextLexicon:
    """The shipped lexicon for ``language``.

    Raises:
        ValueError: none is shipped; there is no fallback to English.
    """
    try:
        return CONTEXT_LEXICONS[language.lower()]
    except KeyError:
        raise ValueError(
            f"no context lexicon for language {language!r}; shipped: "
            f"{sorted(CONTEXT_LEXICONS)}. Pass lexicon=ContextLexicon(...) to "
            "supply one."
        ) from None


def _without_family_history_cue(
    words: list[str], cue: tuple[str, ...] = FAMILY_HISTORY_CUE
) -> list[str]:
    """``words`` with each family-history cue removed, so its words cannot trigger."""
    size = len(cue)
    kept: list[str] = []
    index = 0
    while index < len(words):
        if size and tuple(words[index : index + size]) == cue:
            index += size
            continue
        kept.append(words[index])
        index += 1
    return kept


class ReferenceContextDetector:
    """Reference experiencer and temporality detector, section 5b of the query contract.

    ``language`` picks a shipped lexicon (``CONTEXT_LEXICONS``); ``lexicon`` supplies one
    and wins over ``language``.
    """

    def __init__(
        self, language: str = "en", lexicon: ContextLexicon | None = None
    ) -> None:
        self.lexicon = lexicon or context_lexicon(language)

    def __call__(self, text: str, span: tuple[int, int]) -> Context:
        """Return the experiencer and temporality of the mention at span within text."""
        start, end = span
        tokens = _sentence_tokens(text, start)
        # As in polarity: tokens inside the mention are ignored, and a mention is judged
        # in the sentence where it starts.
        pre = [token.text for token in tokens if token.end <= start]
        post = [token.text for token in tokens if token.start >= end]

        lex = self.lexicon
        stops = lex.terminators
        if _triggered(pre, lex.family_terms, True, stops):
            experiencer = OTHER
        else:
            experiencer = PATIENT

        pre_for_time = _without_family_history_cue(pre, lex.family_history_cue)
        if _triggered(pre_for_time, lex.hypothetical, True, stops):
            temporality = HYPOTHETICAL
        elif _triggered(pre_for_time, lex.historical_before, True, stops) or _triggered(
            post, lex.historical_after, False, stops
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
