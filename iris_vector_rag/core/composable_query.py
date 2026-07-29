"""ComposableQueryMixin: normalize → run_retrieval → maybe_rerank delegation seam.

Pipelines mix this in to gain composable query-time options without changing
their external interface (Principle IV — zero breaking changes).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _unpack_rerank_result(raw: List[Any]) -> List[Any]:
    """Normalize reranker output: handle both plain docs and (doc, score) tuples.

    If the reranker returns ``list[tuple[doc, float]]``, extract the score into
    ``doc.metadata["rerank_score"]`` and return the plain doc list (FR-008).
    """
    if not raw:
        return raw
    first = raw[0]
    if isinstance(first, tuple) and len(first) == 2:
        result = []
        for item, score in raw:
            item.metadata["rerank_score"] = float(score)
            result.append(item)
        return result
    return list(raw)


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

    # Optional KeywordExtractor override. When None, RetrievalEngine constructs
    # one lazily. Set pipeline.keyword_extractor = KeywordExtractor(cheap_llm)
    # to control the model used for global/mix mode keyword extraction.
    keyword_extractor: Optional[Any] = None

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

    def _maybe_rerank(self, docs: List[Any], opts) -> tuple:
        """Apply reranking if opts.rerank is set; degrade gracefully on failure.

        Returns:
            (reranked_docs, degraded) where degraded=True means reranking failed
            and original order was preserved.

        Supports:
          - ``None`` / ``False`` — pass-through (no reranking)
          - ``callable`` — called as ``rerank_fn(opts.query, docs)``; may return
            plain ``list[Document]`` or ``list[tuple[Document, float]]``
          - ``True`` / ``str`` — resolved via the reranker registry
        """
        rerank_spec = opts.rerank
        if not rerank_spec:
            return docs, False

        def _apply(rerank_fn):
            raw = rerank_fn(opts.query, docs)
            return _unpack_rerank_result(raw)

        if callable(rerank_spec):
            try:
                return _apply(rerank_spec), False
            except Exception:
                logger.warning(
                    "Reranker raised an exception; falling back to original ordering.",
                    exc_info=True,
                )
                return docs, True

        # bool True or str — reranker registry
        try:
            from iris_vector_rag.retrieval.rerank import resolve_reranker

            reranker = resolve_reranker(rerank_spec)
            return _apply(reranker), False
        except Exception:
            logger.warning(
                "Reranker resolution/execution failed; falling back to original ordering.",
                exc_info=True,
            )
            return docs, True

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
