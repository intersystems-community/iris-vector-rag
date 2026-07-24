"""ComposableQueryMixin: normalize → run_retrieval → maybe_rerank delegation seam.

Pipelines mix this in to gain composable query-time options without changing
their external interface (Principle IV — zero breaking changes).
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class ComposableQueryMixin:
    """Mixin providing the composable query delegation seam for RAG pipelines.

    Pipelines that mix this in get three hooks:
        _normalize_query(**kwargs) -> QueryOptions
        _run_retrieval(opts) -> list[Document]
        _maybe_rerank(docs, opts) -> list[Document]

    Concrete pipelines implement ``_do_retrieval(opts) -> list[Document]`` to
    plug in their retrieval logic.  Default ``supported_retrieval_modes`` is
    ``["vector"]``; pipelines that support additional modes override it.
    """

    supported_retrieval_modes: List[str] = ["vector"]

    # ------------------------------------------------------------------
    # Public delegation hooks
    # ------------------------------------------------------------------

    def _normalize_query(self, **kwargs: Any):
        """Delegate to normalize_query_params and return a QueryOptions."""
        from iris_vector_rag.core.query_options import normalize_query_params

        return normalize_query_params(**kwargs)

    def _run_retrieval(self, opts) -> List[Any]:
        """Dispatch retrieval according to opts.retrieval.

        Raises ValueError if the requested mode is not in
        ``self.supported_retrieval_modes`` (only when opts.retrieval is
        explicitly set to a non-default value).
        """
        requested = opts.retrieval
        if requested is not None and requested not in self.supported_retrieval_modes:
            raise ValueError(
                f"Retrieval mode {requested!r} is not supported by this pipeline. "
                f"Supported modes: {self.supported_retrieval_modes}"
            )
        return self._do_retrieval(opts)

    def _maybe_rerank(self, docs: List[Any], opts) -> List[Any]:
        """Apply reranking if opts.rerank is set; degrade gracefully on failure.

        Supports:
          - ``None`` / ``False`` — pass-through (no reranking)
          - ``callable`` — called as ``rerank_fn(opts.query, docs)``
          - ``True`` / ``str`` — resolved via the reranker registry (US3, not yet wired)
        """
        rerank_spec = opts.rerank
        if not rerank_spec:
            return docs

        if callable(rerank_spec):
            try:
                return rerank_spec(opts.query, docs)
            except Exception:
                logger.warning(
                    "Reranker raised an exception; falling back to original ordering.",
                    exc_info=True,
                )
                return docs

        # bool True or str — reranker registry (US3 implementation wires this up)
        try:
            from iris_vector_rag.retrieval.rerank import resolve_reranker

            reranker = resolve_reranker(rerank_spec)
            return reranker(opts.query, docs)
        except Exception:
            logger.warning(
                "Reranker resolution/execution failed; falling back to original ordering.",
                exc_info=True,
            )
            return docs

    # ------------------------------------------------------------------
    # Abstract hook — concrete pipeline overrides this
    # ------------------------------------------------------------------

    def _do_retrieval(self, opts) -> List[Any]:  # pragma: no cover
        """Execute retrieval using the pipeline's native mechanism.

        Concrete pipelines override this to plug their retrieval logic into
        the composable seam.  The default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _do_retrieval(opts)."
        )
