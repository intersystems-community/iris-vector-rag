"""Mention polarity for note search (spec 088, contracts/query-semantics.md section 5).

``ReferenceNegExDetector`` is a reference detector. It is not validated on
emergency-department text and no accuracy is claimed for it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from iris_vector_rag.note_search.tokens import tokenize

AFFIRMED = "affirmed"
NEGATED = "negated"
UNKNOWN = "unknown"

PolarityDetector = Callable[[str, tuple[int, int]], str]

# Trigger lists are tuples of lower-case token tuples so they are easy to replace.
PRE_NEGATION_TRIGGERS: tuple[tuple[str, ...], ...] = (
    ("no",),
    ("not",),
    ("denies",),
    ("denied",),
    ("without",),
    ("negative", "for"),
    ("absence", "of"),
)
POST_NEGATION_TRIGGERS: tuple[tuple[str, ...], ...] = (
    ("ruled", "out"),
    ("unlikely",),
)
UNCERTAINTY_TRIGGERS: tuple[tuple[str, ...], ...] = (
    ("possible",),
    ("possibly",),
    ("rule", "out"),
    ("r/o",),
    ("questionable",),
)
TERMINATORS: frozenset[str] = frozenset({"but", "however", "although"})

_SENTENCE_PUNCTUATION = ".?!;"


def _is_boundary(text: str, index: int) -> bool:
    """True if the character at index ends a sentence."""
    char = text[index]
    if char == "\n":
        return True
    if char in _SENTENCE_PUNCTUATION:
        return index + 1 == len(text) or text[index + 1].isspace()
    return False


def _sentence_bounds(text: str, pos: int) -> tuple[int, int]:
    """Return the (start, end) character range of the sentence containing pos."""
    start = 0
    for index in range(pos):
        if _is_boundary(text, index):
            start = index + 1
    end = len(text)
    for index in range(pos, len(text)):
        if _is_boundary(text, index):
            end = index
            break
    return start, end


def _triggered(
    words: list[str],
    triggers: tuple[tuple[str, ...], ...],
    before_mention: bool,
) -> bool:
    """True if a trigger occurs with no terminator between it and the mention.

    ``words`` is ordered outward from the mention: for a pre-mention list it
    ends at the mention, for a post-mention list it starts at the mention.
    """
    for trigger in triggers:
        size = len(trigger)
        for index in range(len(words) - size + 1):
            if tuple(words[index : index + size]) != trigger:
                continue
            reach = words[index + size :] if before_mention else words[:index]
            if not TERMINATORS.intersection(reach):
                return True
    return False


class ReferenceNegExDetector:
    """Reference polarity detector following section 5 of the query contract."""

    def __call__(self, text: str, span: tuple[int, int]) -> str:
        """Return the polarity of the mention at span within text."""
        start, end = span
        sent_start, sent_end = _sentence_bounds(text, start)
        tokens = [
            token
            for token in tokenize(text)
            if token.start >= sent_start and token.end <= sent_end
        ]
        # Tokens inside the mention are ignored; a mention is judged in the
        # sentence where it starts.
        pre = [token.text for token in tokens if token.end <= start]
        post = [token.text for token in tokens if token.start >= end]

        uncertain = _triggered(pre, UNCERTAINTY_TRIGGERS, True) or _triggered(
            post, UNCERTAINTY_TRIGGERS, False
        )
        negated = _triggered(pre, PRE_NEGATION_TRIGGERS, True) or _triggered(
            post, POST_NEGATION_TRIGGERS, False
        )
        if uncertain:
            return UNKNOWN
        if negated:
            return NEGATED
        return AFFIRMED


def section_polarity(polarities: Iterable[str]) -> str:
    """Roll mention polarities up to a section: affirmed if any is, else negated."""
    values = set(polarities)
    if AFFIRMED in values:
        return AFFIRMED
    if NEGATED in values:
        return NEGATED
    return UNKNOWN
