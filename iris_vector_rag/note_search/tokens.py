"""Tokens as the full-text index makes them (contracts/query-semantics.md section 3.1).

Verified against IRIS 2026.1 `%iFind.Index.Basic(LANGUAGE='en', LOWER=1)`: a token is a
whitespace-delimited word, lower-cased, with the punctuation at its start and end
removed. Punctuation inside a word stays (`chest-pain`, `r/o`, `patient's`, `3.5`), and a
word made only of punctuation (`-`) is a token of its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Token:
    """A lower-cased token and its character span in the text it came from."""

    text: str
    start: int
    end: int


def tokenize(text: str) -> List[Token]:
    """The tokens of ``text`` in order, with offsets into ``text``."""
    tokens: List[Token] = []
    for match in _WORD.finditer(text or ""):
        word, start = match.group(), match.start()
        lead = len(word) - len(word.lstrip(_edge_punctuation(word)))
        trail = len(word) - len(word.rstrip(_edge_punctuation(word)))
        if lead + trail >= len(word):
            # Nothing but punctuation: the word is its own token.
            tokens.append(Token(word.lower(), start, match.end()))
            continue
        tokens.append(
            Token(
                word[lead : len(word) - trail].lower(),
                start + lead,
                match.end() - trail,
            )
        )
    return tokens


def _edge_punctuation(word: str) -> str:
    """The characters of ``word`` that are not letters or digits."""
    return "".join({c for c in word if not c.isalnum()})
