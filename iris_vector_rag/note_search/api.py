"""Entry points for embedded Python callers outside this package (Baystate ObjectScript).

Baystate calls these from IRIS embedded Python: text and character spans go in, plain
values come out. This module must import only the standard library and its sibling
modules (relative imports), so the pure note-search modules can be vendored as a copy.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from .phrase import find_mentions
from .polarity import (
    UNKNOWN,
    PolarityDetector,
    ReferenceNegExDetector,
    section_polarity,
)

__all__ = ["classify", "judge_section", "judge_section_json"]


def _default_detector() -> PolarityDetector:
    return ReferenceNegExDetector()


def classify(
    text: str,
    spans: Sequence[Sequence[int]],
    detector: Optional[PolarityDetector] = None,
) -> List[str]:
    """Return one polarity per (start, end) span of ``text``, in order.

    Spans are 0-based with the end exclusive, as Python slices.

    Raises:
        ValueError: a span does not satisfy 0 <= start <= end <= len(text).
    """
    judge = detector or _default_detector()
    result: List[str] = []
    for span in spans:
        start, end = span
        if not 0 <= start <= end <= len(text):
            raise ValueError(
                f"span {(start, end)} is outside the text of length {len(text)}"
            )
        result.append(judge(text, (start, end)))
    return result


def judge_section(
    text: str,
    phrase: str,
    detector: Optional[PolarityDetector] = None,
) -> Dict[str, Any]:
    """Judge every mention of ``phrase`` in ``text``.

    Returns {"polarity": section polarity (UNKNOWN when there are no mentions),
    "mentions": [{"start", "end", "polarity"}, ...]}.

    Raises:
        ValueError: the phrase has no letter or digit (see phrase.phrase_tokens).
    """
    judge = detector or _default_detector()
    mentions: List[Dict[str, Any]] = []
    for start, end in find_mentions(text, phrase):
        mentions.append(
            {"start": start, "end": end, "polarity": judge(text, (start, end))}
        )
    if not mentions:
        return {"polarity": UNKNOWN, "mentions": []}
    polarity = section_polarity(m["polarity"] for m in mentions)
    return {"polarity": polarity, "mentions": mentions}


def judge_section_json(
    text: str,
    phrase: str,
    detector: Optional[PolarityDetector] = None,
) -> str:
    """``judge_section`` as a JSON string, for callers that exchange strings only."""
    return json.dumps(judge_section(text, phrase, detector=detector))
