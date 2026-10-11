"""Tokens as the full-text index makes them (contracts/query-semantics.md section 3.1).

Verified against IRIS 2026.1 `%iFind.Index.Basic(LANGUAGE='en', LOWER=1)`: a token is a
whitespace-delimited word, lower-cased, with the punctuation at its start and end
removed. Punctuation inside a word stays (`chest-pain`, `r/o`, `patient's`, `3.5`), and a
word made only of punctuation (`-`) is a token of its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List

# A word is a run of non-space characters. Its token is the part from the first letter or
# digit to the last one (so edge punctuation is dropped and inner punctuation stays); a
# word with no letter or digit is its own token. `[^\W_]` is a letter or digit.
_TOKEN = re.compile(r"\S*?([^\W_](?:\S*[^\W_])?)\S*|\S+")


@dataclass(frozen=True)
class Token:
    """A lower-cased token and its character span in the text it came from."""

    text: str
    start: int
    end: int


def tokenize(text: str) -> List[Token]:
    """The tokens of ``text`` in order, with offsets into ``text``."""
    tokens: List[Token] = []
    for match in _TOKEN.finditer(text or ""):
        start, end = match.span(1) if match.lastindex else match.span()
        tokens.append(Token(match.string[start:end].lower(), start, end))
    return tokens


@lru_cache(maxsize=16)
def tokenize_cached(text: str) -> tuple[Token, ...]:
    """``tokenize`` for callers that look at the same text many times (one per mention).

    Keeps the last few texts. The result is immutable, so sharing it is safe.
    """
    return tuple(tokenize(text))
