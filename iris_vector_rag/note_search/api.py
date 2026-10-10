"""Entry points for embedded Python callers outside this package (Baystate ObjectScript).

Baystate calls these from IRIS embedded Python: text and character spans go in, plain
values come out. This module must import only the standard library and its sibling
modules (relative imports), so the pure note-search modules can be vendored as a copy.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from .context import ContextDetector, ReferenceContextDetector, section_context
from .phrase import find_mentions
from .polarity import (
    UNKNOWN,
    PolarityDetector,
    ReferenceNegExDetector,
    section_polarity,
)

__all__ = ["classify", "classify_context", "judge_section", "judge_section_json"]


def _default_detector() -> PolarityDetector:
    return ReferenceNegExDetector()


def _default_context_detector() -> ContextDetector:
    return ReferenceContextDetector()


def _checked_spans(text: str, spans: Sequence[Sequence[int]]):
    for span in spans:
        start, end = span
        if not 0 <= start <= end <= len(text):
            raise ValueError(
                f"span {(start, end)} is outside the text of length {len(text)}"
            )
        yield start, end


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
    return [judge(text, span) for span in _checked_spans(text, spans)]


def classify_context(
    text: str,
    spans: Sequence[Sequence[int]],
    context_detector: Optional[ContextDetector] = None,
) -> List[Dict[str, str]]:
    """Return {"experiencer", "temporality"} per (start, end) span of ``text``, in order.

    Spans are checked as in ``classify``.

    Raises:
        ValueError: a span does not satisfy 0 <= start <= end <= len(text).
    """
    judge = context_detector or _default_context_detector()
    answers: List[Dict[str, str]] = []
    for span in _checked_spans(text, spans):
        context = judge(text, span)
        answers.append(
            {"experiencer": context.experiencer, "temporality": context.temporality}
        )
    return answers


def judge_section(
    text: str,
    phrase: str,
    detector: Optional[PolarityDetector] = None,
    context_detector: Optional[ContextDetector] = None,
) -> Dict[str, Any]:
    """Judge every mention of ``phrase`` in ``text``.

    Returns {"polarity": section polarity, "experiencer": section experiencer,
    "temporality": section temporality, "mentions": [{"start", "end", "polarity",
    "experiencer", "temporality"}, ...]}. With no mentions, polarity, experiencer and
    temporality are all UNKNOWN and mentions is empty.

    Raises:
        ValueError: the phrase has no letter or digit (see phrase.phrase_tokens).
    """
    judge = detector or _default_detector()
    judge_context = context_detector or _default_context_detector()
    mentions: List[Dict[str, Any]] = []
    contexts = []
    for start, end in find_mentions(text, phrase):
        context = judge_context(text, (start, end))
        contexts.append(context)
        mentions.append(
            {
                "start": start,
                "end": end,
                "polarity": judge(text, (start, end)),
                "experiencer": context.experiencer,
                "temporality": context.temporality,
            }
        )
    if not mentions:
        return {
            "polarity": UNKNOWN,
            "experiencer": UNKNOWN,
            "temporality": UNKNOWN,
            "mentions": [],
        }
    polarity = section_polarity(m["polarity"] for m in mentions)
    section = section_context(contexts)
    return {
        "polarity": polarity,
        "experiencer": section.experiencer,
        "temporality": section.temporality,
        "mentions": mentions,
    }


def judge_section_json(
    text: str,
    phrase: str,
    detector: Optional[PolarityDetector] = None,
    context_detector: Optional[ContextDetector] = None,
) -> str:
    """``judge_section`` as a JSON string, for callers that exchange strings only."""
    return json.dumps(
        judge_section(
            text, phrase, detector=detector, context_detector=context_detector
        )
    )
