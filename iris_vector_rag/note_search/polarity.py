"""Mention polarity for note search (spec 088, contracts/query-semantics.md section 5).

``ReferenceNegExDetector`` is a reference detector. It is not validated on
emergency-department text and no accuracy is claimed for it.
"""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache

from .tokens import Token, tokenize_cached

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
# Single words that negate a mention they closely follow ("pregnancy test negative"):
# within POST_WINDOW tokens, with no comma, colon or semicolon between, so that
# "chest pain, negative troponin" does not negate the chest pain.
NEAR_POST_NEGATION_TRIGGERS: frozenset[str] = frozenset({"negative", "absent"})
POST_WINDOW = 3
_CLAUSE_BREAKS = frozenset(",;:")
UNCERTAINTY_TRIGGERS: tuple[tuple[str, ...], ...] = (
    ("possible",),
    ("possibly",),
    ("rule", "out"),
    ("r/o",),
    ("questionable",),
)
TERMINATORS: frozenset[str] = frozenset({"but", "however", "although"})


@dataclass(frozen=True)
class NegationLexicon:
    """The words one language uses to negate or hedge a mention.

    Every entry is lower case, matched against the tokens of ``tokens.tokenize``. A
    terminator ends the reach of a trigger ("no fever but cough": the cough is affirmed).
    """

    pre_negation: tuple[tuple[str, ...], ...]
    post_negation: tuple[tuple[str, ...], ...]
    near_post_negation: frozenset[str]
    uncertainty: tuple[tuple[str, ...], ...]
    terminators: frozenset[str]


ENGLISH = NegationLexicon(
    pre_negation=PRE_NEGATION_TRIGGERS,
    post_negation=POST_NEGATION_TRIGGERS,
    near_post_negation=NEAR_POST_NEGATION_TRIGGERS,
    uncertainty=UNCERTAINTY_TRIGGERS,
    terminators=TERMINATORS,
)

#: The lexicons shipped, by language code. Another language is a caller's ``lexicon=``.
NEGATION_LEXICONS: dict[str, NegationLexicon] = {"en": ENGLISH}


def negation_lexicon(language: str) -> NegationLexicon:
    """The shipped lexicon for ``language``.

    Raises:
        ValueError: none is shipped. English triggers on another language's text would
            give confident wrong answers, so there is no fallback.
    """
    try:
        return NEGATION_LEXICONS[language.lower()]
    except KeyError:
        raise ValueError(
            f"no negation lexicon for language {language!r}; shipped: "
            f"{sorted(NEGATION_LEXICONS)}. Pass lexicon=NegationLexicon(...) to "
            "supply one."
        ) from None


_SENTENCE_PUNCTUATION = ".?!;"


def _is_boundary(text: str, index: int) -> bool:
    """True if the character at index ends a sentence."""
    char = text[index]
    if char == "\n":
        return True
    if char in _SENTENCE_PUNCTUATION:
        return index + 1 == len(text) or text[index + 1].isspace()
    return False


_BOUNDARY = re.compile(r"\n|[.?!;](?=\s|\Z)")


@lru_cache(maxsize=16)
def _boundaries(text: str) -> tuple[int, ...]:
    """The index of every sentence-ending character in text, found once per text."""
    return tuple(match.start() for match in _BOUNDARY.finditer(text))


@lru_cache(maxsize=16)
def _token_starts(text: str) -> tuple[int, ...]:
    return tuple(token.start for token in tokenize_cached(text))


def _sentence_bounds(text: str, pos: int) -> tuple[int, int]:
    """Return the (start, end) character range of the sentence containing pos."""
    boundaries = _boundaries(text)
    at = bisect_left(boundaries, pos)
    start = boundaries[at - 1] + 1 if at else 0
    end = boundaries[at] if at < len(boundaries) else len(text)
    return start, end


def _sentence_tokens(text: str, pos: int) -> list[Token]:
    """The tokens inside the sentence that contains pos, in order."""
    sent_start, sent_end = _sentence_bounds(text, pos)
    tokens = tokenize_cached(text)
    found: list[Token] = []
    for token in tokens[bisect_left(_token_starts(text), sent_start) :]:
        if token.end > sent_end:
            break
        found.append(token)
    return found


@lru_cache(maxsize=32)
def _by_first_word(
    triggers: tuple[tuple[str, ...], ...],
) -> dict[str, tuple[tuple[str, ...], ...]]:
    """The triggers grouped by their first word, so one pass over the words finds them."""
    grouped: dict[str, list[tuple[str, ...]]] = {}
    for trigger in triggers:
        grouped.setdefault(trigger[0], []).append(trigger)
    return {word: tuple(found) for word, found in grouped.items()}


def _triggered(
    words: list[str],
    triggers: tuple[tuple[str, ...], ...],
    before_mention: bool,
    terminators: frozenset[str] = TERMINATORS,
) -> bool:
    """True if a trigger occurs with no terminator between it and the mention.

    ``words`` is ordered outward from the mention: for a pre-mention list it
    ends at the mention, for a post-mention list it starts at the mention.
    """
    candidates = _by_first_word(triggers)
    for index, word in enumerate(words):
        for trigger in candidates.get(word, ()):
            size = len(trigger)
            if tuple(words[index : index + size]) != trigger:
                continue
            reach = words[index + size :] if before_mention else words[:index]
            if not terminators.intersection(reach):
                return True
    return False


def _near_post_negation(
    text: str, mention_end: int, post: list, lexicon: NegationLexicon = ENGLISH
) -> bool:
    """True if a near post-negation word closely follows the mention in its clause."""
    for token in post[:POST_WINDOW]:
        if token.text in lexicon.terminators:
            return False
        if (
            token.text in lexicon.near_post_negation
            and not _CLAUSE_BREAKS.intersection(text[mention_end : token.start])
        ):
            return True
    return False


class ReferenceNegExDetector:
    """Reference polarity detector following section 5 of the query contract.

    ``language`` picks a shipped lexicon (``NEGATION_LEXICONS``); ``lexicon`` supplies
    one and wins over ``language``.
    """

    def __init__(
        self, language: str = "en", lexicon: NegationLexicon | None = None
    ) -> None:
        self.lexicon = lexicon or negation_lexicon(language)

    def __call__(self, text: str, span: tuple[int, int]) -> str:
        """Return the polarity of the mention at span within text."""
        start, end = span
        tokens = _sentence_tokens(text, start)
        # Tokens inside the mention are ignored; a mention is judged in the
        # sentence where it starts.
        pre = [token.text for token in tokens if token.end <= start]
        after = [token for token in tokens if token.start >= end]
        post = [token.text for token in after]

        lex = self.lexicon
        stops = lex.terminators
        uncertain = _triggered(pre, lex.uncertainty, True, stops) or _triggered(
            post, lex.uncertainty, False, stops
        )
        negated = (
            _triggered(pre, lex.pre_negation, True, stops)
            or _triggered(post, lex.post_negation, False, stops)
            or _near_post_negation(text, end, after, lex)
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
