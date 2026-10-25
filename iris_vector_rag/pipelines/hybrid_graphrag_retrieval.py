"""
Hybrid GraphRAG Retrieval Methods

Contains the core retrieval methods for HybridGraphRAG pipeline using
iris_vector_graph integration.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from ..core.models import Document
from ..exceptions import FilterError
from ..storage.attached import FilterSpec, compile_where, where_kwargs
from ._hybrid_utils import (
    convert_fusion_results_to_documents,
    convert_rrf_results_to_documents,
    convert_text_results_to_documents,
    convert_vector_results_to_documents,
)

logger = logging.getLogger(__name__)


# The values of ``result["metadata"]["retrieval_method"]`` from
# ``HybridGraphRAGPipeline.query``. A label says which path produced the documents,
# not which ``method=`` the caller asked for: a path that finds nothing (or fails)
# reports ``vector_fallback``.
#
# Through iris_vector_graph (``method`` hybrid / rrf / text / vector):
#   hybrid_fusion, rrf_fusion, enhanced_text, hnsw_vector
# Any of those paths falling back to the vector store, and ``method="kg"`` finding
# no matching entity:
#   vector_fallback
# Without the graph engine, or ``method="kg"``:
#   knowledge_graph, knowledge_graph_primary, kg_vector_hybrid, balanced_hybrid,
#   vector_primary, entity_guided_text, text_vector_fallback,
#   vector_entity_enhanced, vector_only, vector_search_failed
RETRIEVAL_METHODS = frozenset(
    {
        "hybrid_fusion",
        "rrf_fusion",
        "enhanced_text",
        "hnsw_vector",
        "vector_fallback",
        "knowledge_graph",
        "knowledge_graph_primary",
        "kg_vector_hybrid",
        "balanced_hybrid",
        "vector_primary",
        "entity_guided_text",
        "text_vector_fallback",
        "vector_entity_enhanced",
        "vector_only",
        "vector_search_failed",
    }
)


def _ivg_scope(kwargs: Dict[str, Any], *names: str) -> Dict[str, Any]:
    """The ivg 4.x scoping keywords the caller set; unset ones are not sent.

    `graph` selects a named graph (ivg's default graph is ''), `model_key` an
    embedding route. Omitting them keeps the 4.x default and any engine that
    predates the keyword working.
    """
    return {name: kwargs[name] for name in names if kwargs.get(name) is not None}


class ScopedRetrievalMixin:
    """What `HybridGraphRAGPipeline` needs to read documents inside a ``where``.

    Feature 092, FR-021: a leg that reads the document table filters in SQL, and a
    document a graph leg names is read under the same predicate, so none outside the
    scope reaches a result. The pipeline supplies the vector store, the embedding
    manager and the GraphRAG entity helpers these call.
    """

    __slots__ = ()

    #: A scoped graph leg asks the engine for this many times ``top_k`` hits: the
    #: engine ranks every entity, `where` removes the ones outside the scope, and the
    #: leg should still fill the page.
    SCOPED_OVERFETCH = 4

    def _where_scope(self, where: Optional[Dict[str, Any]]) -> Tuple[str, List[Any]]:
        """``(predicate, params)`` for ``where`` on the document table (feature 092).

        ``("", [])`` when there is nothing to apply, which leaves every query as it
        was. A required filter left out, or a key the table cannot filter on, raises:
        no leg is allowed to drop what it cannot apply (FR-021).
        """
        store = self.vector_store
        if store is None:
            if where:
                raise FilterError(
                    "where cannot be applied: this pipeline has no vector store",
                    rejected_keys=list(where),
                )
            return "", []
        spec = getattr(store, "filter_spec", None) or FilterSpec()
        if not where and not spec.required_filters:
            return "", []
        return compile_where(where, store.shape(), spec)

    def _content_getter(
        self, where: Optional[Dict[str, Any]]
    ) -> Callable[[str], Optional[str]]:
        """Entity id -> document text, for documents inside ``where`` only."""
        predicate, params = self._where_scope(where)
        if not predicate:
            return self._get_document_content_for_entity
        return lambda entity_id: self._get_document_content_for_entity(
            entity_id, predicate, params
        )

    def _graph_leg(
        self, leg: str, query_text: str, top_k: int, **kwargs
    ) -> tuple[List[Document], str]:
        """One iris_vector_graph leg, its documents read under ``where``.

        The engine names entities, never documents; each one is turned into text by
        `_get_document_content_for_entity`, which is where the scope applies. When it
        leaves nothing, or the leg fails, the vector store answers in the same scope.
        """
        where = kwargs.get("where")
        scoped = bool(self._where_scope(where)[0])
        fetch_k = top_k * self.SCOPED_OVERFETCH if scoped else top_k
        try:
            documents, method = getattr(self.retrieval_methods, f"retrieve_via_{leg}")(
                query_text, fetch_k, self._content_getter(where), **kwargs
            )
            if documents:
                return documents[:top_k], method
            logger.warning("%s returned 0 results. Falling back to vector search.", leg)
        except FilterError:
            raise
        except Exception as e:
            logger.error(f"{leg} retrieval failed: {e}")
            logger.info("Falling back to IRISVectorStore vector search")
        return self._fallback_to_vector_search(query_text, top_k, where), (
            "vector_fallback"
        )

    def _fallback_to_vector_search(
        self,
        query_text: str,
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """The vector store's nearest chunks, inside ``where`` when one is given."""
        if not self.vector_store:
            return []
        query_embedding = self.embedding_manager.embed_text(query_text)
        results = self.vector_store.search_by_vector(
            query_embedding, top_k=top_k, **where_kwargs(where)
        )
        return [r[0] for r in results]

    def _source_doc_ids(self, entity_ids: Set[str]) -> List[str]:
        """The documents the entities were extracted from, in id order."""
        schema = self.vector_store.shape().schema
        ordered = sorted(entity_ids)
        size = self.NEIGHBORHOOD_CHUNK_SIZE
        found: Set[str] = set()
        for offset in range(0, len(ordered), size):
            chunk = ordered[offset : offset + size]
            placeholders = ",".join("?" for _ in chunk)
            for row in self._execute_sql(
                f"SELECT DISTINCT source_doc_id FROM {schema}.Entities "
                f"WHERE entity_id IN ({placeholders})",
                chunk,
            ):
                doc_id = row.get("source_doc_id") or row.get("SOURCE_DOC_ID")
                if doc_id:
                    found.add(str(doc_id))
        return sorted(found)

    def _entity_documents(
        self,
        entity_ids: Set[str],
        top_k: int,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Documents holding the entities, only those inside ``where``.

        Fetched through the store under the compiled predicate, so a document the
        scope excludes is never read. Without a scope this is GraphRAG's own join.
        """
        predicate, params = self._where_scope(where)
        if not predicate:
            return self._get_documents_from_entities(entity_ids, top_k)
        doc_ids = self._source_doc_ids(entity_ids)
        size = self.NEIGHBORHOOD_CHUNK_SIZE
        documents: List[Document] = []
        for offset in range(0, len(doc_ids), size):
            documents.extend(
                self.vector_store.fetch_documents_by_ids(
                    doc_ids[offset : offset + size], predicate, params
                )
            )
        documents.sort(key=lambda doc: str(doc.id))
        return [
            Document(
                id=doc.id,
                page_content=doc.page_content,
                metadata={**doc.metadata, "retrieval_method": "knowledge_graph"},
            )
            for doc in documents[:top_k]
        ]

    def _kg_documents(
        self, query_text: str, top_k: int, where: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Document], str]:
        """`_retrieve_via_kg`, with the documents read inside ``where``."""
        if not self._where_scope(where)[0]:
            return self._retrieve_via_kg(query_text, top_k)
        seeds = self._find_seed_entities(query_text)
        if not seeds:
            return [], "no_matching_entities"
        entities = self._expand_neighborhood(set(seeds), depth=self.max_depth)
        return self._entity_documents(entities, top_k, where), "knowledge_graph"


class HybridRetrievalMethods:
    """Core retrieval methods for HybridGraphRAG pipeline."""

    def __init__(
        self,
        iris_engine,
        fusion_engine,
        text_engine,
        vector_optimizer,
        embedding_manager,
    ):
        self.iris_engine = iris_engine
        self.fusion_engine = fusion_engine
        self.text_engine = text_engine
        self.vector_optimizer = vector_optimizer
        self.embedding_manager = embedding_manager

    def retrieve_via_hybrid_fusion(
        self, query_text: str, top_k: int, get_content_func: Callable, **kwargs
    ) -> Tuple[List[Document], str]:
        """
        Retrieve documents using multi-modal hybrid search fusion.

        Combines vector similarity, text relevance, and graph expansion.
        """
        try:
            # Generate query vector if not provided
            query_vector = kwargs.get("vector_query")
            if not query_vector and self.embedding_manager:
                # A failure propagates: searching on text alone would answer a vector
                # question with a different kind of result and say nothing about it.
                embedding = self.embedding_manager.get_embeddings([query_text])[0]
                query_vector = (
                    str(embedding.tolist())
                    if hasattr(embedding, "tolist")
                    else str(list(embedding))
                )

            # Use fusion engine for multi-modal search
            fusion_results = self.fusion_engine.multi_modal_search(
                query_vector=query_vector,
                query_text=query_text,
                k=top_k,
                fusion_method="rrf",
                weights=kwargs.get("fusion_weights"),
            )

            # Convert fusion results to documents
            documents = convert_fusion_results_to_documents(
                fusion_results, get_content_func
            )

            return documents, "hybrid_fusion"

        except Exception as e:
            logger.error(f"Hybrid fusion retrieval failed: {e}")
            raise

    def retrieve_via_rrf(
        self, query_text: str, top_k: int, get_content_func: Callable, **kwargs
    ) -> Tuple[List[Document], str]:
        """
        Retrieve documents using RRF (Reciprocal Rank Fusion).

        Combines vector and text search results using RRF algorithm.
        """
        try:
            # Generate query vector if not provided
            query_vector = kwargs.get("vector_query")
            if not query_vector and self.embedding_manager:
                embedding = self.embedding_manager.get_embeddings([query_text])[0]
                query_vector = (
                    str(embedding.tolist())
                    if hasattr(embedding, "tolist")
                    else str(list(embedding))
                )

            # Use RRF fusion
            k1 = kwargs.get("vector_k", top_k * 2)  # Get more vector candidates
            k2 = kwargs.get("text_k", top_k * 2)  # Get more text candidates
            c = kwargs.get("rrf_c", 60)  # RRF parameter

            rrf_results = self.iris_engine.kg_RRF_FUSE(
                k=top_k,
                k1=k1,
                k2=k2,
                c=c,
                query_vector=query_vector,
                query_text=query_text,
                **_ivg_scope(kwargs, "graph"),
            )

            # Convert RRF results to documents
            documents = convert_rrf_results_to_documents(rrf_results, get_content_func)

            return documents, "rrf_fusion"

        except Exception as e:
            logger.error(f"RRF retrieval failed: {e}")
            raise

    def retrieve_via_enhanced_text(
        self, query_text: str, top_k: int, get_content_func: Callable, **kwargs
    ) -> Tuple[List[Document], str]:
        """
        Retrieve documents using enhanced text search.

        Uses self.text_engine.search() when text_engine has a search() method
        (e.g. IVGTextSearchBackend with BM25), otherwise falls back to
        self.iris_engine.kg_TXT() (iFind stored procedure).
        """
        try:
            min_confidence = kwargs.get("min_confidence", 0)

            if self.text_engine is not None and callable(
                getattr(self.text_engine, "search", None)
            ):
                text_results = self.text_engine.search(
                    query_text, k=top_k, min_confidence=min_confidence
                )
            else:
                text_results = self.iris_engine.kg_TXT(
                    query_text=query_text,
                    k=top_k,
                    min_confidence=min_confidence,
                    **_ivg_scope(kwargs, "graph"),
                )

            documents = convert_text_results_to_documents(
                text_results, get_content_func
            )

            return documents, "enhanced_text"

        except Exception as e:
            logger.error(f"Enhanced text retrieval failed: {e}")
            raise

    def retrieve_via_hnsw_vector(
        self, query_text: str, top_k: int, get_content_func: Callable, **kwargs
    ) -> Tuple[List[Document], str]:
        """
        Retrieve documents using HNSW-optimized vector search.
        """
        try:
            # Generate query vector
            query_vector = kwargs.get("vector_query")
            if not query_vector and self.embedding_manager:
                embedding = self.embedding_manager.get_embeddings([query_text])[0]
                query_vector = (
                    str(embedding.tolist())
                    if hasattr(embedding, "tolist")
                    else str(list(embedding))
                )

            if not query_vector:
                raise ValueError("No query vector available for HNSW search")

            # Use HNSW-optimized vector search
            label_filter = kwargs.get("label_filter")
            vector_results = self.iris_engine.kg_KNN_VEC(
                query_vector=query_vector,
                k=top_k,
                label_filter=label_filter,
                **_ivg_scope(kwargs, "graph", "model_key"),
            )

            # Convert vector results to documents
            documents = convert_vector_results_to_documents(
                vector_results, get_content_func
            )

            return documents, "hnsw_vector"

        except Exception as e:
            logger.error(f"HNSW vector retrieval failed: {e}")
            raise

    def check_hnsw_optimization(self):
        """Check if HNSW-optimized vector search is available"""
        if self.vector_optimizer:
            try:
                status = self.vector_optimizer.check_hnsw_availability()
                if status["available"]:
                    logger.info(
                        f"HNSW optimization available: {status['record_count']} vectors, "
                        f"{status['query_time_ms']:.1f}ms query time, "
                        f"performance tier: {status['performance_tier']}"
                    )
                else:
                    logger.warning(
                        f"HNSW optimization not available: {status['reason']}"
                    )
            except Exception as e:
                logger.warning(f"Could not check HNSW availability: {e}")

    def get_performance_statistics(self, connection_manager) -> Dict[str, Any]:
        """Get performance statistics for hybrid search components"""
        stats = {}

        if self.vector_optimizer:
            try:
                stats["hnsw_status"] = self.vector_optimizer.check_hnsw_availability()
                stats["vector_stats"] = self.vector_optimizer.get_vector_statistics()
            except Exception as e:
                logger.warning(f"Could not get vector statistics: {e}")

        # Add standard GraphRAG stats. The connection belongs to the session (it is
        # shared, not ours to close); the cursor is ours and is closed on every path.
        cursor = None
        try:
            connection = connection_manager.get_connection()
            cursor = connection.cursor()

            cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
            stats["entity_count"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM RAG.EntityRelationships")
            stats["relationship_count"] = cursor.fetchone()[0]
        except Exception as e:
            logger.warning(f"Could not get GraphRAG statistics: {e}")
        finally:
            if cursor is not None:
                cursor.close()

        return stats

    def benchmark_search_methods(
        self, query_text: str, query_func: Callable, iterations: int = 5
    ) -> Dict[str, Any]:
        """Benchmark different search methods for performance comparison"""
        if not self.iris_engine:
            logger.warning("iris_vector_graph not available for benchmarking")
            return {}

        methods = ["hybrid", "rrf", "text", "vector", "kg"]
        results = {}

        for method in methods:
            method_times = []
            for i in range(iterations):
                try:
                    start_time = time.perf_counter()
                    query_func(
                        query_text=query_text,
                        method=method,
                        generate_answer=False,
                        top_k=10,
                    )
                    elapsed_ms = (time.perf_counter() - start_time) * 1000
                    method_times.append(elapsed_ms)
                except Exception as e:
                    logger.warning(f"Benchmark failed for method {method}: {e}")
                    continue

            if method_times:
                results[method] = {
                    "avg_time_ms": sum(method_times) / len(method_times),
                    "min_time_ms": min(method_times),
                    "max_time_ms": max(method_times),
                    "iterations": len(method_times),
                }

        return results
