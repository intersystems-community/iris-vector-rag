"""Dual-level keyword extractor for global/mix retrieval modes (Feature 081)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_LIGHTRAG_PROMPT = """\
Given the following question, identify both high-level thematic keywords and \
low-level specific entity keywords.

Return ONLY valid JSON in this exact format (no explanation, no markdown):
{{"high_level_keywords": [...], "low_level_keywords": [...]}}

Language: {language}
Question: {query}
"""


def parse_keywords(raw: str) -> Tuple[List[str], List[str]]:
    """Parse LightRAG-style keyword JSON, stripping markdown fences.

    Returns (high_level_keywords, low_level_keywords) or ([], []) on error.
    """
    text = raw.strip()
    match = _FENCE_RE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        data = json.loads(text)
        high = data.get("high_level_keywords", [])
        low = data.get("low_level_keywords", [])
        return list(high), list(low)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return [], []


class KeywordExtractor:
    """Extracts high- and low-level keywords from a query using an LLM.

    Args:
        llm_func: Callable that takes a prompt string and returns a response string.
        language: Target language hint passed to the LLM prompt (default "English").
        model_name: Optional model identifier for observability.
    """

    def __init__(
        self,
        llm_func: Callable[[str], Any],
        language: str = "English",
        model_name: Optional[str] = None,
    ) -> None:
        self._llm_func = llm_func
        self._language = language
        self.model_name = model_name

    def extract(self, query: str) -> Tuple[List[str], List[str]]:
        """Return (high_level_keywords, low_level_keywords) for *query*.

        On any error (LLM exception, bad JSON, timeout) returns ([], []).
        """
        try:
            prompt = _LIGHTRAG_PROMPT.format(language=self._language, query=query)
            raw = self._llm_func(prompt)
            if raw is None:
                return [], []
            return parse_keywords(str(raw))
        except Exception:
            logger.warning("KeywordExtractor.extract failed for query=%r", query, exc_info=True)
            return [], []
