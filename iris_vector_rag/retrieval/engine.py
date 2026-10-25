"""Retrieval engine: mode dispatch (US4).

Dispatches query execution to the appropriate retrieval strategy based on
QueryOptions.retrieval. The text leg of text/hybrid/rrf searches the document table
through its full-text index (`SchemaManager.ensure_text_index`); FR-010, FR-012.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from iris_vector_rag.exceptions import EmbeddingError

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
        polarity_detector: Optional polarity detector for ``retrieval="phrase"``; the
            reference detector is used when None.
        concept_index: A `ConceptIndex` for ``retrieval="concept"``.
        concept_tagger: The tagger that finds concepts in the query for
            ``retrieval="concept"``, e.g. a `DictionaryTagger`.
    """

    def __init__(
        self,
        vector_store: Any,
        connection: Any = None,
        config_manager: Any = None,
        keyword_extractor: Any = None,
        polarity_detector: Any = None,
        concept_index: Any = None,
        concept_tagger: Any = None,
    ) -> None:
        self.vector_store = vector_store
        self.connection = connection
        self._config_manager = config_manager
        self.keyword_extractor = keyword_extractor
        self.polarity_detector = polarity_detector
        self.concept_index = concept_index
        self.concept_tagger = concept_tagger

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
        elif mode_name == "phrase":
            return self._retrieve_phrase(opts)
        elif mode_name in ("hybrid", "rrf"):
            return self._retrieve_fusion(opts, mode_name)
        elif mode_name == "concept":
            return self._retrieve_concept(opts)
        elif mode_name == "global":
            return self._retrieve_global(opts)
        elif mode_name == "mix":
            return self._retrieve_mix(opts)
        else:
            raise NotImplementedError(
                f"Mode {mode_name!r} is registered but not implemented."
            )

    def _vector_leg(self, query: str, opts: Any) -> List[Any]:
        """Vector hits for ``query`` under the options' filter and threshold.

        ``metadata_filter`` is pushed into the store. With ``similarity_threshold`` above
        zero the scored search is used so each hit carries ``metadata["score"]`` and hits
        below the threshold are dropped; ``search_by_text`` returns documents without
        their scores.
        """
        threshold = getattr(opts, "similarity_threshold", 0.0) or 0.0
        metadata_filter = getattr(opts, "metadata_filter", None)
        scope = _where(opts)
        if threshold > 0.0:
            scored = self.vector_store.similarity_search_with_score(
                query, k=opts.top_k, filter=metadata_filter, **scope
            )
            docs = []
            for doc, score in scored:
                doc.metadata["score"] = float(score)
                if float(score) >= threshold:
                    docs.append(doc)
            return docs
        return list(
            self.vector_store.search_by_text(
                query, top_k=opts.top_k, metadata_filter=metadata_filter, **scope
            )
        )

    def _retrieve_vector(self, opts: Any) -> List[Any]:
        return self._vector_leg(opts.query, opts)

    def _text_leg(self, opts: Any) -> List[Any]:
        """Text hits: full documents, best match first, scored in ``text_score``.

        The store searches its own table through its full-text index and applies
        ``metadata_filter`` the way the vector leg does. A table without the index
        raises `TextIndexMissingError` rather than returning nothing.
        """
        if not hasattr(self.vector_store, "text_search"):
            raise NotImplementedError(
                f"{type(self.vector_store).__name__} has no text_search(), so the "
                f"text leg of this retrieval mode cannot run on it."
            )
        return list(
            self.vector_store.text_search(
                opts.query,
                top_k=opts.top_k,
                metadata_filter=getattr(opts, "metadata_filter", None),
                **_where(opts),
            )
        )

    def _retrieve_text(self, opts: Any) -> List[Any]:
        if (getattr(opts, "similarity_threshold", 0.0) or 0.0) > 0.0:
            raise ValueError(
                "similarity_threshold does not apply to retrieval='text': text "
                "scores are index ranks, not similarities. Use 'hybrid' or 'rrf', "
                "where the threshold applies to the vector leg."
            )
        return self._text_leg(opts)

    def _retrieve_phrase(self, opts: Any) -> List[Any]:
        """Exact-phrase hits rolled up to parent documents, each with its polarity."""
        if (getattr(opts, "similarity_threshold", 0.0) or 0.0) > 0.0:
            raise ValueError(
                "similarity_threshold does not apply to retrieval='phrase': scores are "
                "index ranks, not similarities."
            )
        if getattr(opts, "metadata_filter", None):
            raise ValueError(
                "metadata_filter does not apply to retrieval='phrase'; narrow a phrase "
                "search with section=."
            )
        _refuse_where(opts, "phrase")
        if not hasattr(self.vector_store, "phrase_search"):
            raise NotImplementedError(
                f"{type(self.vector_store).__name__} has no phrase_search(), so "
                f"retrieval='phrase' cannot run on it."
            )
        return list(
            self.vector_store.phrase_search(
                opts.query,
                top_k=opts.top_k,
                section=getattr(opts, "section", None),
                detector=self.polarity_detector,
            )
        )

    def _retrieve_concept(self, opts: Any) -> List[Any]:
        """Parent documents that have every concept the query names (spec 091).

        The tagger finds the concepts in ``opts.query``; the index returns the parents
        that have postings for all of them, filtered by ``opts.polarity`` and
        ``opts.section``.
        """
        if (getattr(opts, "similarity_threshold", 0.0) or 0.0) > 0.0:
            raise ValueError(
                "similarity_threshold does not apply to retrieval='concept': concept "
                "hits are postings, not similarities."
            )
        if getattr(opts, "metadata_filter", None):
            raise ValueError(
                "metadata_filter does not apply to retrieval='concept'; narrow a concept "
                "search with section=."
            )
        _refuse_where(opts, "concept")
        if self.concept_index is None:
            raise NotImplementedError(
                "retrieval='concept' needs a concept_index: build a ConceptIndex over "
                "the vector store and pass it to RetrievalEngine(concept_index=...)."
            )
        if self.concept_tagger is None:
            raise NotImplementedError(
                "retrieval='concept' needs a concept_tagger to find the concepts in "
                "the query: pass RetrievalEngine(concept_tagger=...)."
            )
        concept_ids = list(
            dict.fromkeys(m.concept_id for m in self.concept_tagger(opts.query))
        )
        if not concept_ids:
            raise ValueError(
                f"The query {opts.query!r} names no known concept, so there is nothing "
                f"to search for with retrieval='concept'."
            )
        return list(
            self.concept_index.search(
                concept_ids,
                polarity=getattr(opts, "polarity", None),
                section=getattr(opts, "section", None),
                top_k=opts.top_k,
            )
        )

    def _retrieve_fusion(self, opts: Any, mode_name: str) -> List[Any]:
        vector_docs = self._vector_leg(opts.query, opts)
        text_docs = self._text_leg(opts)

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
        """Embed a query text for relation-embedding search.

        Raises `EmbeddingError` when it cannot: a zero vector would rank every relation
        the same and return them as if they matched.
        """
        cfg_mgr = getattr(self, "_config_manager", None)
        if cfg_mgr is None:
            raise EmbeddingError(
                "Cannot embed the query for relation search: the engine has no "
                "configuration manager to build an embedding manager from."
            )
        from iris_vector_rag.embeddings.manager import EmbeddingManager

        return EmbeddingManager(cfg_mgr).embed_text(text)

    def _retrieve_global(self, opts: Any) -> Dict[str, Any]:
        """Theme-level retrieval via relation embeddings (US1).

        Extracts high-level keywords, embeds them, searches relation embeddings.
        Falls back gracefully when index empty (FR-009); hard error when KG absent
        handled by check_prerequisites in retrieve().
        """
        _refuse_where(opts, "global")
        if getattr(opts, "metadata_filter", None):
            raise ValueError(
                "metadata_filter cannot be applied to retrieval='global': relation "
                "hits carry no document metadata, so the filter would be ignored and "
                "out-of-scope relations returned. Use 'mix' (which scopes its "
                "document legs) or 'vector'."
            )
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
            degradation_reason = (
                "relation embedding index is empty; populate with embed_and_store()"
            )
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

        logger.debug(
            "global retrieval complete",
            extra={
                "retrieval_mode": "global",
                "high_level_keywords_count": len(high_kws),
                "low_level_keywords_count": 0,
                "degraded": degraded,
                "result_count": len(docs),
            },
        )
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
        _refuse_where(opts, "mix")
        high_kws, low_kws = self._get_or_extract_keywords(opts)
        rel_store = self._get_relation_store()

        # 1. Naive chunk vector search (existing pipeline mechanism)
        naive_docs: List[Any] = []
        try:
            naive_raw = self._vector_leg(opts.query, opts)
            for doc in naive_raw or []:
                doc.metadata["retrieval_source"] = "naive"
                naive_docs.append(doc)
        except Exception:
            logger.warning("Mix mode: naive vector search failed", exc_info=True)

        # 2. High-level relation embedding search
        high_docs: List[Any] = []
        metadata_filter = getattr(opts, "metadata_filter", None)
        relation_skip_reason = ""
        if metadata_filter:
            # Relation hits carry no document metadata, so the filter cannot scope
            # them. Including them would leak out-of-scope content; say so instead.
            relation_skip_reason = (
                "metadata_filter supplied: relation hits carry no document metadata, "
                "so the relation leg was skipped"
            )
        elif high_kws and rel_store.count_embedded() > 0:
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
            threshold = getattr(opts, "similarity_threshold", 0.0) or 0.0
            if threshold > 0.0:
                high_docs = [
                    d for d in high_docs if d.metadata.get("score", 0.0) >= threshold
                ]

        # 3. Low-level entity vector search
        low_docs: List[Any] = []
        if low_kws:
            try:
                low_query = " ".join(low_kws)
                low_raw = self._vector_leg(low_query, opts)
                for doc in low_raw or []:
                    doc.metadata["retrieval_source"] = "low_level"
                    low_docs.append(doc)
            except Exception:
                logger.warning(
                    "Mix mode: low-level vector search failed", exc_info=True
                )

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
                source_names=["high_level", "low_level", "naive"],
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

        degraded_mix = (not high_kws and not low_kws) or bool(relation_skip_reason)
        logger.debug(
            "mix retrieval complete",
            extra={
                "retrieval_mode": "mix",
                "fusion_method": fusion_method,
                "high_level_keywords_count": len(high_kws),
                "low_level_keywords_count": len(low_kws),
                "degraded": degraded_mix,
                "result_count": len(fused),
            },
        )
        return {
            "retrieved_documents": fused,
            "metadata": {
                "fusion_method": fusion_method,
                "low_level_count": len(low_docs),
                "high_level_count": len(high_docs),
                "naive_count": len(naive_docs),
                "high_level_keywords": high_kws,
                "low_level_keywords": low_kws,
                "degraded": degraded_mix,
                "degradation_reason": relation_skip_reason,
                "retrieval_mode": "mix",
                "extraction_model": extraction_model,
            },
        }


def _where(opts: Any) -> Dict[str, Any]:
    """The options' real-column filters as keyword arguments (`where_kwargs`)."""
    from iris_vector_rag.storage.attached import where_kwargs

    return where_kwargs(getattr(opts, "where", None))


def _refuse_where(opts: Any, mode: str) -> None:
    """Real-column filters scope document rows; ``mode`` does not search them."""
    if getattr(opts, "where", None):
        raise ValueError(
            f"where does not apply to retrieval={mode!r}: it filters the rows of the "
            "document table, which this mode does not search. Use 'vector', 'text', "
            "'hybrid' or 'rrf'."
        )


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


def _check_fusion_inputs(
    result_lists: List[List[Any]],
    source_names: List[str],
    weights: Optional[List[float]] = None,
) -> None:
    """Refuse lists, names and weights that do not pair up; `zip` would drop the extras."""
    if len(source_names) != len(result_lists):
        raise ValueError(
            f"Fusion got {len(result_lists)} result lists but {len(source_names)} "
            f"source_names ({source_names}); pass one name per list."
        )
    if weights is not None and len(weights) != len(result_lists):
        raise ValueError(
            f"Fusion got {len(result_lists)} result lists but {len(weights)} "
            f"weights; pass one weight per list."
        )


def _merge_hit(seen: dict, doc_id: str, doc: Any) -> None:
    """Keep one document per id: the one with content, carrying both hits' metadata."""
    current = seen.get(doc_id)
    if current is None:
        seen[doc_id] = doc
        return
    if not getattr(current, "page_content", "") and getattr(doc, "page_content", ""):
        current, doc = doc, current
        seen[doc_id] = current
    # `current` is the keeper; the other hit only adds the keys it lacks.
    for key, value in doc.metadata.items():
        current.metadata.setdefault(key, value)


def _finish(ordered: List[str], seen: dict, per_source: dict, scores: dict, top_k: int):
    result = []
    for did in ordered[:top_k]:
        doc = seen[did]
        doc.metadata.update(per_source[did])
        doc.metadata["fusion_score"] = scores[did]
        result.append(doc)
    return result


def _reciprocal_rank_fusion(
    result_lists: List[List[Any]],
    top_k: int = 10,
    k: int = 60,
    source_names: Optional[List[str]] = None,
) -> List[Any]:
    """Merge ranked lists via RRF scoring; record per-source scores in doc.metadata (FR-011).

    A hit at 1-based rank r earns ``1 / (k + r)`` from its list; a document's score is
    the sum over the lists it appears in. One document per id survives, with content.
    """
    if source_names is None:
        source_names = ["vector", "text"][: len(result_lists)]
    _check_fusion_inputs(result_lists, source_names)
    scores: dict = {}
    per_source: dict = {}
    seen: dict = {}
    for source, ranked in zip(source_names, result_lists):
        for rank, doc in enumerate(ranked, start=1):
            doc_id = _doc_id(doc)
            contrib = 1.0 / (k + rank)
            scores[doc_id] = scores.get(doc_id, 0.0) + contrib
            per_source.setdefault(doc_id, {})[f"{source}_score"] = contrib
            _merge_hit(seen, doc_id, doc)
    ordered = sorted(seen.keys(), key=lambda did: scores[did], reverse=True)
    return _finish(ordered, seen, per_source, scores, top_k)


def _weighted_score_fusion(
    result_lists: List[List[Any]],
    weights: List[float],
    top_k: int = 10,
    source_names: Optional[List[str]] = None,
) -> List[Any]:
    """Merge ranked lists via weighted rank scores; record per-source scores (FR-011).

    A hit at 1-based rank r in a list of n hits earns ``weight * (n - r + 1) / n``: the
    top hit earns the whole weight, the last earns ``weight / n``, and the only hit in
    a list earns its whole weight. (Before 0.15 the formula was ``weight * (1 - r / n)``,
    which gave the last hit of every list, and a lone hit, nothing.)
    """
    if source_names is None:
        source_names = ["vector", "text"][: len(result_lists)]
    _check_fusion_inputs(result_lists, source_names, weights)
    scores: dict = {}
    per_source: dict = {}
    seen: dict = {}
    for source, (ranked, w) in zip(source_names, zip(result_lists, weights)):
        n = len(ranked)
        for rank, doc in enumerate(ranked, start=1):
            doc_id = _doc_id(doc)
            contrib = w * (n - rank + 1) / n
            scores[doc_id] = scores.get(doc_id, 0.0) + contrib
            per_source.setdefault(doc_id, {})[f"{source}_score"] = contrib
            _merge_hit(seen, doc_id, doc)
    ordered = sorted(seen.keys(), key=lambda did: scores[did], reverse=True)
    return _finish(ordered, seen, per_source, scores, top_k)


def _doc_id(doc: Any) -> str:
    """Extract a stable identity key from a document."""
    if hasattr(doc, "id") and doc.id is not None:
        return str(doc.id)
    if hasattr(doc, "page_content"):
        return doc.page_content[:64]
    return str(id(doc))
