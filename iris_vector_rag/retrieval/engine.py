"""Retrieval engine: mode dispatch (US4).

Dispatches query execution to the appropriate retrieval strategy based on
QueryOptions.retrieval. Supports vector mode natively; text/hybrid/rrf
modes require iris-vector-graph BM25 support (FR-010, FR-012).
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Dispatches retrieval based on mode in QueryOptions.

    Args:
        vector_store: A VectorStore instance with similarity_search().
        connection: Optional IRIS connection for text/hybrid/rrf modes.
    """

    def __init__(self, vector_store: Any, connection: Any = None) -> None:
        self.vector_store = vector_store
        self.connection = connection

    def retrieve(self, opts: Any) -> List[Any]:
        """Execute retrieval for the given options.

        Args:
            opts: QueryOptions instance with query, retrieval, top_k, etc.

        Returns:
            List of Document objects.

        Raises:
            ValueError: if the mode is unknown (from modes registry).
            RetrievalPrerequisiteError: if mode prerequisites are not met.
            NotImplementedError: if a known mode has no implementation yet.
        """
        from iris_vector_rag.retrieval.modes import check_prerequisites, get_mode

        mode_name = opts.retrieval or "vector"
        get_mode(mode_name)  # raises ValueError for unknown modes
        check_prerequisites(mode_name, connection=self.connection)

        if mode_name == "vector":
            return self._retrieve_vector(opts)
        elif mode_name == "text":
            return self._retrieve_text(opts)
        elif mode_name in ("hybrid", "rrf"):
            return self._retrieve_fusion(opts, mode_name)
        else:
            raise NotImplementedError(
                f"Mode {mode_name!r} is registered but not implemented."
            )

    def _retrieve_vector(self, opts: Any) -> List[Any]:
        return self.vector_store.search_by_text(opts.query, top_k=opts.top_k)

    def _text_search_to_docs(self, results: List[Any]) -> List[Any]:
        """Convert (doc_id, score) tuples from TextSearchEngine to Document objects."""
        from iris_vector_rag.core.models import Document

        docs = []
        for item in results:
            if isinstance(item, tuple) and len(item) == 2:
                doc_id, score = item
                if hasattr(doc_id, "page_content"):
                    # Already a Document (mock path)
                    docs.append(doc_id)
                else:
                    doc = Document(
                        id=str(doc_id),
                        page_content="",
                        metadata={"text_score": float(score)},
                    )
                    docs.append(doc)
            else:
                docs.append(item)
        return docs

    def _retrieve_text(self, opts: Any) -> List[Any]:
        from iris_vector_graph.text_search import TextSearchEngine  # type: ignore[import]

        engine = TextSearchEngine(connection=self.connection)
        raw = engine.search_documents(opts.query, k=opts.top_k)
        return self._text_search_to_docs(raw)

    def _retrieve_fusion(self, opts: Any, mode_name: str) -> List[Any]:
        from iris_vector_graph.text_search import TextSearchEngine  # type: ignore[import]

        vector_docs = self.vector_store.search_by_text(opts.query, top_k=opts.top_k)
        text_engine = TextSearchEngine(connection=self.connection)
        raw_text = text_engine.search_documents(opts.query, k=opts.top_k)
        text_docs = self._text_search_to_docs(raw_text)

        if mode_name == "rrf":
            return _reciprocal_rank_fusion([vector_docs, text_docs], top_k=opts.top_k)

        # hybrid: weighted score fusion
        weights = opts.weights or {"vector": 0.7, "text": 0.3}
        return _weighted_score_fusion(
            [vector_docs, text_docs],
            [weights.get("vector", 0.7), weights.get("text", 0.3)],
            top_k=opts.top_k,
        )


def _reciprocal_rank_fusion(
    result_lists: List[List[Any]], top_k: int = 10, k: int = 60
) -> List[Any]:
    """Merge ranked lists via RRF scoring; record per-source scores in doc.metadata (FR-011)."""
    source_names = ["vector", "text"]
    scores: dict = {}
    per_source: dict = {}
    seen: dict = {}
    for source, ranked in zip(source_names, result_lists):
        for rank, doc in enumerate(ranked, start=1):
            doc_id = _doc_id(doc)
            contrib = 1.0 / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + contrib
            per_source.setdefault(doc_id, {})[f"{source}_score"] = contrib
            seen[doc_id] = doc
    ordered = sorted(seen.keys(), key=lambda did: scores[did], reverse=True)
    result = []
    for did in ordered[:top_k]:
        doc = seen[did]
        doc.metadata.update(per_source[did])
        doc.metadata["fusion_score"] = scores[did]
        result.append(doc)
    return result


def _weighted_score_fusion(
    result_lists: List[List[Any]], weights: List[float], top_k: int = 10
) -> List[Any]:
    """Merge ranked lists via weighted reciprocal-rank proxy scores; record per-source scores (FR-011)."""
    source_names = ["vector", "text"]
    scores: dict = {}
    per_source: dict = {}
    seen: dict = {}
    for source, (ranked, w) in zip(source_names, zip(result_lists, weights)):
        n = max(len(ranked), 1)
        for rank, doc in enumerate(ranked, start=1):
            doc_id = _doc_id(doc)
            contrib = w * (1.0 - rank / n)
            scores[doc_id] = scores.get(doc_id, 0.0) + contrib
            per_source.setdefault(doc_id, {})[f"{source}_score"] = contrib
            seen[doc_id] = doc
    ordered = sorted(seen.keys(), key=lambda did: scores[did], reverse=True)
    result = []
    for did in ordered[:top_k]:
        doc = seen[did]
        doc.metadata.update(per_source[did])
        doc.metadata["fusion_score"] = scores[did]
        result.append(doc)
    return result


def _doc_id(doc: Any) -> str:
    """Extract a stable identity key from a document."""
    if hasattr(doc, "id") and doc.id is not None:
        return str(doc.id)
    if hasattr(doc, "page_content"):
        return doc.page_content[:64]
    return str(id(doc))
