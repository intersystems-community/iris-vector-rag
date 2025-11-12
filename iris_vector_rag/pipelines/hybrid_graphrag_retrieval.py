"""
Hybrid GraphRAG Retrieval Methods

Contains the core retrieval methods for HybridGraphRAG pipeline using
iris_vector_graph integration.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.models import Document
from ._hybrid_utils import (
    convert_fusion_results_to_documents,
    convert_rrf_results_to_documents,
    convert_text_results_to_documents,
    convert_vector_results_to_documents,
)

logger = logging.getLogger(__name__)


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
                try:
                    # Generate embedding for the query
                    embedding = self.embedding_manager.get_embeddings([query_text])[0]
                    query_vector = (
                        str(embedding.tolist())
                        if hasattr(embedding, "tolist")
                        else str(list(embedding))
                    )
                except Exception as e:
                    logger.warning(f"Could not generate query embedding: {e}")
                    query_vector = None

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
                try:
                    embedding = self.embedding_manager.get_embeddings([query_text])[0]
                    query_vector = (
                        str(embedding.tolist())
                        if hasattr(embedding, "tolist")
                        else str(list(embedding))
                    )
                except Exception as e:
                    logger.warning(f"Could not generate query embedding: {e}")
                    raise ValueError("No query vector available for RRF search")

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
        Retrieve documents using enhanced IRIS iFind text search.
        """
        try:
            # Use enhanced text search with entity context
            min_confidence = kwargs.get("min_confidence", 0)

            text_results = self.iris_engine.kg_TXT(
                query_text=query_text, k=top_k, min_confidence=min_confidence
            )

            # Convert text results to documents
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
                query_vector=query_vector, k=top_k, label_filter=label_filter
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

        # Add standard GraphRAG stats
        try:
            connection = connection_manager.get_connection()
            cursor = connection.cursor()

            cursor.execute("SELECT COUNT(*) FROM RAG.Entities")
            stats["entity_count"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM RAG.EntityRelationships")
            stats["relationship_count"] = cursor.fetchone()[0]

            cursor.close()
        except Exception as e:
            logger.warning(f"Could not get GraphRAG statistics: {e}")

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
                    response = query_func(
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
