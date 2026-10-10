"""Phrases: the search string for the index, and the mentions of a phrase in a text.

The index finds the sections; `find_mentions` finds where in a section the phrase is, so
polarity can be judged per mention (contracts/query-semantics.md sections 3.2 and 3.8).
"""

from __future__ import annotations

from typing import List, Tuple

from iris_vector_rag.note_search.tokens import tokenize

__all__ = ["find_mentions", "phrase_search_string", "phrase_tokens"]


def phrase_tokens(phrase: str) -> List[str]:
    """The tokens of ``phrase``, double quotes removed.

    Raises:
        ValueError: the phrase has no letter or digit, so nothing could match it. An
            empty answer would read as "no document matches".
    """
    tokens = [t.text for t in tokenize((phrase or "").replace('"', " "))]
    if not any(c.isalnum() for token in tokens for c in token):
        raise ValueError(
            f"A phrase needs at least one letter or digit, got {phrase!r}."
        )
    return tokens


def phrase_search_string(phrase: str) -> str:
    """The iFind search string for ``phrase``: its tokens as one quoted phrase.

    Operator text (`AND`, `OR`, `NOT`, `*`, parentheses) is plain text because the
    whole phrase sits inside quotes and no quote can occur inside it.
    """
    return '"' + " ".join(phrase_tokens(phrase)) + '"'


def find_mentions(text: str, phrase: str) -> List[Tuple[int, int]]:
    """Every occurrence of ``phrase`` in ``text`` as a (start, end) character span."""
    wanted = phrase_tokens(phrase)
    found = tokenize(text)
    n = len(wanted)
    spans: List[Tuple[int, int]] = []
    for i in range(len(found) - n + 1):
        if [t.text for t in found[i : i + n]] == wanted:
            spans.append((found[i].start, found[i + n - 1].end))
    return spans
