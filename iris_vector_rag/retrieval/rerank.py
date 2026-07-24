"""Reranker resolver and process-level cache (US3).

resolve_reranker(spec) → callable(query: str, docs: list[Document]) → list[Document]

Supports:
  - True  → default cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
  - str   → named strategy (currently only "cross-encoder")
  - callable → returned directly (not cached)

Cache key: (name, model_name) — process-level dict.  Callables are not cached.
Degradation is handled by the caller (ComposableQueryMixin._maybe_rerank).
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_RERANKER_CACHE: Dict[tuple, Any] = {}
_RERANKER_LOCK = threading.Lock()


def resolve_reranker(
    spec: Union[bool, str],
    model_name: Optional[str] = None,
) -> Callable[[str, List[Any]], List[Any]]:
    """Return a reranker callable for the given spec.

    Args:
        spec: ``True`` for the default cross-encoder, or a strategy name string.
        model_name: Override the model name for cross-encoder strategies.

    Returns:
        ``callable(query: str, docs: list[Document]) -> list[Document]``

    Raises:
        ImportError: if the required dependency (sentence-transformers) is not installed.
        ValueError: if the strategy name is unrecognized.
    """
    if callable(spec):
        return spec  # type: ignore[return-value]

    strategy = "cross-encoder" if spec is True else str(spec)
    resolved_model = model_name or _DEFAULT_MODEL
    cache_key = (strategy, resolved_model)

    if cache_key in _RERANKER_CACHE:
        return _RERANKER_CACHE[cache_key]

    with _RERANKER_LOCK:
        # Double-checked locking: re-test inside the lock
        if cache_key in _RERANKER_CACHE:
            return _RERANKER_CACHE[cache_key]

        if strategy != "cross-encoder":
            raise ValueError(
                f"Unknown reranker strategy: {strategy!r}. "
                f"Supported: 'cross-encoder' or True (default)."
            )

        reranker = _build_cross_encoder_reranker(resolved_model)
        _RERANKER_CACHE[cache_key] = reranker
        return reranker


def _build_cross_encoder_reranker(model_name: str) -> Callable[[str, List[Any]], List[Any]]:
    """Build a cross-encoder reranker using sentence-transformers CrossEncoder."""
    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Cross-encoder reranking requires sentence-transformers. "
            "Install with: pip install sentence-transformers"
        ) from exc

    logger.info("Loading cross-encoder reranker model: %s", model_name)
    cross_encoder = CrossEncoder(model_name)

    def _rerank(query: str, docs: List[Any]) -> List[Any]:
        if not docs:
            return docs
        pairs = [(query, doc.page_content) for doc in docs]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked]

    return _rerank
