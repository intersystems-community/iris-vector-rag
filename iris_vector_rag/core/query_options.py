"""QueryOptions dataclass and normalize_query_params() (feature 065 — composable retrieval)."""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

logger = logging.getLogger(__name__)

_FUSION_MODES = frozenset({"hybrid", "rrf"})


@dataclass
class QueryOptions:
    """Normalized inputs to pipeline.query(). Produced by normalize_query_params()."""

    query: str
    top_k: int = 5
    generate_answer: bool = True
    include_sources: bool = True
    metadata_filter: Optional[Dict[str, Any]] = None
    similarity_threshold: float = 0.0
    retrieval: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    rerank: Optional[Union[bool, str, Callable]] = None  # type: ignore[type-arg]
    custom_prompt: Optional[str] = None


def normalize_query_params(
    query: Optional[str] = None,
    *,
    query_text: Optional[str] = None,
    top_k: int = 5,
    generate_answer: bool = True,
    include_sources: bool = True,
    metadata_filter: Optional[Dict[str, Any]] = None,
    similarity_threshold: float = 0.0,
    retrieval: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    rerank: Optional[Union[bool, str, Callable]] = None,  # type: ignore[type-arg]
    custom_prompt: Optional[str] = None,
    **_extra: Any,
) -> QueryOptions:
    """Resolve aliases, validate, and return a QueryOptions.

    Canonical param is ``query``; ``query_text`` is accepted as an alias.
    When both are given, ``query`` wins and a deprecation warning is emitted.
    All options default to ``None``/off so the pipeline's pre-existing behavior
    is preserved (Principle IV — no breaking changes).
    """
    # --- resolve query / query_text alias ---
    resolved_query: Optional[str]
    if query is not None and query_text is not None:
        warnings.warn(
            "Both 'query' and 'query_text' supplied; 'query_text' is ignored. "
            "Use 'query' going forward.",
            DeprecationWarning,
            stacklevel=3,
        )
        resolved_query = query
    elif query is not None:
        resolved_query = query
    elif query_text is not None:
        resolved_query = query_text
    else:
        raise ValueError("A query string is required (pass 'query=' or 'query_text=').")

    if not resolved_query or not resolved_query.strip():
        raise ValueError("Query must be a non-empty string.")

    # --- top_k ---
    if not (1 <= top_k <= 100):
        raise ValueError(f"top_k must be between 1 and 100, got {top_k}.")

    # --- similarity_threshold ---
    if not (0.0 <= similarity_threshold <= 1.0):
        raise ValueError(
            f"similarity_threshold must be between 0.0 and 1.0, got {similarity_threshold}."
        )

    # --- weights require a fusion mode ---
    if weights is not None and retrieval not in _FUSION_MODES:
        raise ValueError(
            f"'weights' requires a fusion retrieval mode ('hybrid' or 'rrf'), "
            f"but retrieval={retrieval!r}. Set retrieval='hybrid' or retrieval='rrf'."
        )

    return QueryOptions(
        query=resolved_query,
        top_k=top_k,
        generate_answer=generate_answer,
        include_sources=include_sources,
        metadata_filter=metadata_filter,
        similarity_threshold=similarity_threshold,
        retrieval=retrieval,
        weights=weights,
        rerank=rerank,
        custom_prompt=custom_prompt,
    )
