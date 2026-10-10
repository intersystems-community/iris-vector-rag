"""Concept tagging and per-section postings with polarity (spec 091).

A tagger finds the concept mentions of a text; `postings_for_section` judges the polarity
of each concept's mentions with a polarity detector and gives one posting per concept.
No model and no embedding is involved. No clinical dictionary ships here: the caller
supplies one.

Pure: standard library and relative imports only, so the module can be vendored with the
other pure note-search modules (see tests/unit/test_note_search_standalone.py).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .polarity import ReferenceNegExDetector, section_polarity
from .tokens import tokenize

__all__ = [
    "ConceptMention",
    "ConceptTagger",
    "DictionaryTagger",
    "postings_for_section",
]


@dataclass(frozen=True)
class ConceptMention:
    """One mention of a concept: the character span [start, end) of its tokens."""

    concept_id: str
    start: int
    end: int


ConceptTagger = Callable[[str], List[ConceptMention]]

_Key = Tuple[str, ...]


class DictionaryTagger:
    """Tags a text with the concepts whose surface phrases it contains.

    Surface phrases are tokenised the way phrase search tokenises (`tokens.tokenize`),
    so matching is case-insensitive and agrees with `retrieval="phrase"`. A phrase
    matches whole tokens only ('stroke' does not match 'stroked'). At each position the
    longest phrase wins and matches do not overlap; mentions come back in text order.

    Args:
        dictionary: concept id -> iterable of surface phrases.

    Raises:
        ValueError: a surface phrase is blank, or the same surface form (after
            tokenising and lower-casing) is listed for two concepts. The message names
            both concepts.
    """

    def __init__(self, dictionary: Mapping[str, Iterable[str]]) -> None:
        self._concept_of: Dict[_Key, str] = {}
        for concept_id, phrases in dictionary.items():
            for phrase in phrases:
                key = tuple(token.text for token in tokenize(phrase))
                if not key:
                    raise ValueError(
                        f"Concept {concept_id!r} has a blank surface form {phrase!r}."
                    )
                owner = self._concept_of.get(key)
                if owner is not None and owner != concept_id:
                    raise ValueError(
                        f"The surface form {phrase!r} is listed for both {owner!r} "
                        f"and {concept_id!r}; a surface form must name one concept."
                    )
                self._concept_of[key] = concept_id
        self._longest = max((len(key) for key in self._concept_of), default=0)

    def __call__(self, text: str) -> List[ConceptMention]:
        """The concept mentions of ``text``, in text order, non-overlapping."""
        tokens = tokenize(text or "")
        words = [token.text for token in tokens]
        mentions: List[ConceptMention] = []
        i = 0
        while i < len(tokens):
            for size in range(min(self._longest, len(tokens) - i), 0, -1):
                concept_id = self._concept_of.get(tuple(words[i : i + size]))
                if concept_id is not None:
                    mentions.append(
                        ConceptMention(
                            concept_id,
                            tokens[i].start,
                            tokens[i + size - 1].end,
                        )
                    )
                    i += size
                    break
            else:
                i += 1
        return mentions


def postings_for_section(
    text: str,
    tagger: ConceptTagger,
    detector: Optional[Callable[[str, Tuple[int, int]], str]] = None,
) -> List[Dict[str, Any]]:
    """One posting per concept present in ``text``.

    Each posting is ``{"concept_id", "polarity", "mentions"}``: ``polarity`` is the
    section polarity of that concept's mentions (affirmed if any mention is affirmed,
    else negated, else unknown), judged by ``detector`` (the reference detector when
    None) with spans into ``text``; ``mentions`` counts every mention of the concept.
    Postings follow the order in which each concept first appears.
    """
    judge = detector or ReferenceNegExDetector()
    by_concept: Dict[str, List[str]] = {}
    for mention in tagger(text):
        polarity = judge(text, (mention.start, mention.end))
        by_concept.setdefault(mention.concept_id, []).append(polarity)
    return [
        {
            "concept_id": concept_id,
            "polarity": section_polarity(polarities),
            "mentions": len(polarities),
        }
        for concept_id, polarities in by_concept.items()
    ]
