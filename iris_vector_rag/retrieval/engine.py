"""Retrieval engine: mode dispatch (US4).

Dispatches query execution to the appropriate retrieval strategy based on
QueryOptions.retrieval. Supports vector mode natively; text/hybrid/rrf
modes require iris-vector-graph BM25 support (FR-010, FR-012).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Imported at module level so tests can patch iris_vector_rag.retrieval.engine.RelationEmbeddingStore
try:
    from iris_vector_rag.storage.relation_embedding_store import RelationEmbeddingStore
except ImportError:
    RelationEmbeddingStore = None  # type: ignore[assignment,misc]


class RetrievalEngine:
    """Dispatches retrieval based on mode in QueryOptions.

    Args:
        vector_store: A VectorStore instance with similarity_search().
        connection: Optional IRIS connection for text/hybrid/rrf modes.
        config_manager: Optional ConfigurationManager for RelationEmbeddingStore.
        keyword_extractor: Optional KeywordExtractor override (lazy-init if None).
    """

    def __init__(
        self,
        vector_store: Any,
        connection: Any = None,
        config_manager: Any = None,
        keyword_extractor: Any = None,
    ) -> None:
        self.vector_store = vector_store
        self.connection = connection
        self._config_manager = config_manager
        self.keyword_extractor = keyword_extractor

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
        elif mode_name == "global":
            return self._retrieve_global(opts)
        elif mode_name == "mix":
            return self._retrieve_mix(opts)
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


    # ------------------------------------------------------------------
    # Global / Mix helpers
    # ------------------------------------------------------------------

    def _get_keyword_extractor(self) -> Any:
        """Return the extractor, constructing a default one if not set."""
        if self.keyword_extractor is not None:
            return self.keyword_extractor
        # Lazy default using the pipeline's LLM func (best-effort)
        from iris_vector_rag.retrieval.keyword_extractor import KeywordExtractor

        llm_func = getattr(self, "llm_func", None) or (lambda p: "")
        self.keyword_extractor = KeywordExtractor(llm_func=llm_func)
        return self.keyword_extractor

    def _get_or_extract_keywords(self, opts: Any):
        """Return (high_kws, low_kws) from opts if pre-supplied, else extract."""
        high = opts.high_level_keywords
        low = opts.low_level_keywords
        if high is not None and low is not None:
            return list(high), list(low)
        extractor = self._get_keyword_extractor()
        return extractor.extract(opts.query)

    def _get_relation_store(self) -> Any:
        conn_mgr = getattr(self, "_conn_mgr", None) or getattr(self, "connection", None)
        cfg_mgr = getattr(self, "_config_manager", None)
        return RelationEmbeddingStore(conn_mgr, cfg_mgr)

    def _embed_text_for_search(self, text: str) -> List[float]:
        """Embed a query text for relation-embedding search."""
        try:
            cfg_mgr = getattr(self, "_config_manager", None)
            if cfg_mgr is not None:
                from iris_vector_rag.embeddings.manager import EmbeddingManager

                return EmbeddingManager(cfg_mgr).embed_text(text)
        except Exception:
            pass
        # Fallback: zero vector (results in score=0 for all items, effectively random)
        return [0.0] * 384

    def _retrieve_global(self, opts: Any) -> Dict[str, Any]:
        """Theme-level retrieval via relation embeddings (US1).

        Extracts high-level keywords, embeds them, searches relation embeddings.
        Falls back gracefully when index empty (FR-009); hard error when KG absent
        handled by check_prerequisites in retrieve().
        """
        high_kws, low_kws = self._get_or_extract_keywords(opts)
        degraded = False
        degradation_reason = ""
        docs: List[Any] = []

        rel_store = self._get_relation_store()
        count = rel_store.count_embedded()

        if not high_kws:
            degraded = True
            degradation_reason = "high_level_keywords empty — no theme-level signal"
        elif count == 0:
            degraded = True
            degradation_reason = "relation embedding index is empty; populate with embed_and_store()"
        else:
            query_text = " ".join(high_kws)
            query_emb = self._embed_text_for_search(query_text)
            raw = rel_store.search(query_emb, top_k=opts.top_k)
            for item in raw:
                rid = item.get("relationship_id", "")
                src = item.get("source_entity_id", "")
                tgt = item.get("target_entity_id", "")
                rtype = item.get("relationship_type", "")
                score = item.get("score", 0.0)
                content = f"{rtype}: {src} → {tgt}"
                doc = _make_doc(
                    content=content,
                    doc_id=rid,
                    score=score,
                    source="high_level",
                    extra_meta={"level_score": score, "retrieval_mode": "global"},
                )
                docs.append(doc)

        threshold = getattr(opts, "similarity_threshold", None)
        if threshold is not None:
            docs = [d for d in docs if d.metadata.get("score", 0.0) >= threshold]

        extraction_model = None
        extractor = getattr(self, "keyword_extractor", None)
        if extractor is not None:
            extraction_model = getattr(extractor, "model_name", None)

        return {
            "retrieved_documents": docs,
            "metadata": {
                "high_level_keywords": high_kws,
                "low_level_keywords": [],
                "degraded": degraded,
                "degradation_reason": degradation_reason,
                "retrieval_mode": "global",
                "extraction_model": extraction_model,
            },
        }

    def _retrieve_mix(self, opts: Any) -> Dict[str, Any]:
        """Comprehensive RRF-fused retrieval across three sources (US2).

        Fuses: low-level entity search, high-level relation embedding search,
        and naive chunk vector search. Optional weights override RRF.
        """
        high_kws, low_kws = self._get_or_extract_keywords(opts)
        rel_store = self._get_relation_store()

        # 1. Naive chunk vector search (existing pipeline mechanism)
        naive_docs: List[Any] = []
        try:
            naive_raw = self.vector_store.search_by_text(opts.query, top_k=opts.top_k)
            for doc in (naive_raw or []):
                doc.metadata["retrieval_source"] = "naive"
                naive_docs.append(doc)
        except Exception:
            logger.warning("Mix mode: naive vector search failed", exc_info=True)

        # 2. High-level relation embedding search
        high_docs: List[Any] = []
        if high_kws and rel_store.count_embedded() > 0:
            query_text = " ".join(high_kws)
            query_emb = self._embed_text_for_search(query_text)
            raw = rel_store.search(query_emb, top_k=opts.top_k)
            for item in raw:
                rid = item.get("relationship_id", "")
                src = item.get("source_entity_id", "")
                tgt = item.get("target_entity_id", "")
                rtype = item.get("relationship_type", "")
                score = item.get("score", 0.0)
                doc = _make_doc(
                    content=f"{rtype}: {src} → {tgt}",
                    doc_id=rid,
                    score=score,
                    source="high_level",
                )
                high_docs.append(doc)

        # 3. Low-level entity vector search
        low_docs: List[Any] = []
        if low_kws:
            try:
                low_query = " ".join(low_kws)
                low_raw = self.vector_store.search_by_text(low_query, top_k=opts.top_k)
                for doc in (low_raw or []):
                    doc.metadata["retrieval_source"] = "low_level"
                    low_docs.append(doc)
            except Exception:
                logger.warning("Mix mode: low-level vector search failed", exc_info=True)

        # Fusion
        weights = getattr(opts, "weights", None)
        if weights:
            w_high = weights.get("relation", 0.5)
            w_low = weights.get("low_level", 0.3)
            w_naive = weights.get("vector", 0.2)
            fused = _weighted_score_fusion(
                [high_docs, low_docs, naive_docs],
                [w_high, w_low, w_naive],
                top_k=opts.top_k,
            )
            fusion_method = "weighted_score"
        else:
            fused = _reciprocal_rank_fusion(
                [high_docs, low_docs, naive_docs],
                source_names=["high_level", "low_level", "naive"],
                top_k=opts.top_k,
            )
            fusion_method = "rrf"

        # Ensure retrieval_source is set on all fused docs
        for doc in fused:
            if "retrieval_source" not in doc.metadata:
                doc.metadata["retrieval_source"] = "naive"

        extraction_model = None
        extractor = getattr(self, "keyword_extractor", None)
        if extractor is not None:
            extraction_model = getattr(extractor, "model_name", None)

        return {
            "retrieved_documents": fused,
            "metadata": {
                "fusion_method": fusion_method,
                "low_level_count": len(low_docs),
                "high_level_count": len(high_docs),
                "naive_count": len(naive_docs),
                "high_level_keywords": high_kws,
                "low_level_keywords": low_kws,
                "degraded": (not high_kws and not low_kws),
                "retrieval_mode": "mix",
                "extraction_model": extraction_model,
            },
        }


def _make_doc(
    content: str,
    doc_id: str,
    score: float = 0.0,
    source: str = "unknown",
    extra_meta: Optional[dict] = None,
) -> Any:
    """Construct a Document with retrieval metadata."""
    from iris_vector_rag.core.models import Document

    meta = {"retrieval_source": source, "score": score}
    if extra_meta:
        meta.update(extra_meta)
    return Document(id=doc_id, page_content=content, metadata=meta)


def _reciprocal_rank_fusion(
    result_lists: List[List[Any]],
    top_k: int = 10,
    k: int = 60,
    source_names: Optional[List[str]] = None,
) -> List[Any]:
    """Merge ranked lists via RRF scoring; record per-source scores in doc.metadata (FR-011)."""
    if source_names is None:
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
    result_lists: List[List[Any]],
    weights: List[float],
    top_k: int = 10,
    source_names: Optional[List[str]] = None,
) -> List[Any]:
    """Merge ranked lists via weighted reciprocal-rank proxy scores; record per-source scores (FR-011)."""
    if source_names is None:
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
