"""RetrievalMode registry and prerequisite checks (US4).

Defines the four standard modes: vector, text, hybrid, rrf.
Prerequisites are checked before execution — never a silent fallback (FR-012).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_REGISTRY: Dict[str, "RetrievalMode"] = {}


@dataclass
class RetrievalMode:
    """Descriptor for a named retrieval mode."""

    name: str
    sources: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    fusion: Optional[str] = None  # None | "weighted_score" | "rrf"


def _register(mode: RetrievalMode) -> RetrievalMode:
    _REGISTRY[mode.name] = mode
    return mode


# ─── standard modes ──────────────────────────────────────────────────────────

_register(RetrievalMode(name="vector", sources=["vector"], requires=[]))
_register(
    RetrievalMode(
        name="text",
        sources=["text"],
        requires=["iris_vector_graph_bm25"],
    )
)
_register(
    RetrievalMode(
        name="hybrid",
        sources=["vector", "text"],
        requires=["iris_vector_graph_bm25"],
        fusion="weighted_score",
    )
)
_register(
    RetrievalMode(
        name="rrf",
        sources=["vector", "text"],
        requires=["iris_vector_graph_bm25"],
        fusion="rrf",
    )
)

# ─── public API ──────────────────────────────────────────────────────────────


def get_mode(name: str) -> RetrievalMode:
    """Return the named mode or raise ValueError."""
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(
            f"Unknown retrieval mode: {name!r}. Available: {available}"
        )
    return _REGISTRY[name]


def list_modes() -> List[str]:
    """Return list of registered mode names."""
    return list(_REGISTRY.keys())


def _check_bm25_available(connection: Any = None) -> bool:
    """Return True if iris-vector-graph BM25 is importable and usable."""
    try:
        from iris_vector_graph.text_search import TextSearchEngine  # type: ignore[import]

        return True
    except ImportError:
        return False


def check_prerequisites(mode_name: str, connection: Any = None) -> None:
    """Verify mode prerequisites are met; raise a named error if not (FR-012).

    Args:
        mode_name: The retrieval mode to check.
        connection: Optional IRIS connection for runtime checks.

    Raises:
        RetrievalPrerequisiteError: with a clear message naming the unmet prerequisite.
    """
    mode = get_mode(mode_name)
    for req in mode.requires:
        if req == "iris_vector_graph_bm25":
            if not _check_bm25_available(connection):
                raise RetrievalPrerequisiteError(
                    f"Retrieval mode {mode_name!r} requires iris-vector-graph with BM25 "
                    f"text search support (iris_vector_graph.text_search.TextSearchEngine). "
                    f"Install with: pip install iris-vector-graph"
                )
        else:
            logger.warning("Unknown prerequisite %r for mode %r — skipping check", req, mode_name)


class RetrievalPrerequisiteError(RuntimeError):
    """Raised when a retrieval mode's prerequisites are not met (FR-012)."""
