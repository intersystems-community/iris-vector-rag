"""
Basic RAG Pipeline implementation with ReRanking step after the initial vector search.

This pipeline extends BasicRAGPipeline to add reranking functionality while
eliminating code duplication through proper inheritance.
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..core.models import Document
from .basic import BasicRAGPipeline

logger = logging.getLogger(__name__)


def hf_reranker(query: str, docs: List[Document]) -> List[Tuple[Document, float]]:
    """
    Default HuggingFace cross-encoder reranker function.

    Uses lazy loading to avoid import-time model loading.

    Args:
        query: The query text
        docs: List of documents to rerank

    Returns:
        List of (document, score) tuples
    """
    # Lazy import to avoid module-level loading
    from sentence_transformers import CrossEncoder

    # Create cross-encoder instance (could be cached in future)
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)
    return list(zip(docs, scores))


class BasicRAGRerankingPipeline(BasicRAGPipeline):
    """
    Basic RAG pipeline with reranking support.

    This pipeline extends the standard BasicRAGPipeline by adding a reranking
    step after initial vector retrieval. The reranking uses cross-encoder models
    to improve the relevance ordering of retrieved documents.

    Key differences from BasicRAGPipeline:
    1. Retrieves more documents initially (rerank_factor * top_k)
    2. Applies reranking to reorder documents by relevance
    3. Returns top_k documents after reranking

    The pipeline supports:
    - Custom reranker functions
    - Configurable rerank factor
    - Fallback to no reranking if reranker fails
    """

    def __init__(
        self,
        connection_manager=None,
        config_manager=None,
        reranker_func: Optional[
            Callable[[str, List[Document]], List[Tuple[Document, float]]]
        ] = None,
        embedding_config: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the Basic RAG Reranking Pipeline.

        Args:
            connection_manager: Optional manager for database connections (defaults to new instance)
            config_manager: Optional manager for configuration settings (defaults to new instance)
            reranker_func: Optional custom reranker function. If None, uses default HuggingFace reranker.
            embedding_config: Optional IRIS EMBEDDING config name for auto-vectorization (Feature 051)
            **kwargs: Additional arguments passed to parent BasicRAGPipeline
        """
        # Initialize parent pipeline with all standard functionality (including embedding_config)
        super().__init__(connection_manager, config_manager, embedding_config=embedding_config, **kwargs)

        # Set up reranking-specific configuration
        # Use dedicated reranking config section with fallback to basic config
        self.reranking_config = self.config_manager.get(
            "pipelines:basic_reranking", self.config_manager.get("pipelines:basic", {})
        )

        # Reranking parameters
        self.rerank_factor = self.reranking_config.get("rerank_factor", 2)
        self.reranker_model = self.reranking_config.get(
            "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        # Set reranker function (default to HuggingFace if none provided)
        self.reranker_func = reranker_func or hf_reranker

        logger.info(
            f"Initialized BasicRAGRerankingPipeline with rerank_factor={self.rerank_factor}"
        )

    def query(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Execute RAG query with reranking - THE single method for reranking RAG operations.

        This method overrides the parent to add reranking:
        1. Retrieves rerank_factor * top_k documents using parent method
        2. Applies reranking to improve document ordering
        3. Returns top_k best documents after reranking
        4. Maintains full compatibility with parent response format

        Args:
            query: The query text
            top_k: Number of documents to return after reranking (must be between 1 and 100)
            **kwargs: Additional arguments including:
                - include_sources: Whether to include source information (default: True)
                - custom_prompt: Custom prompt template
                - generate_answer: Whether to generate LLM answer (default: True)
                - All other parent query arguments

        Returns:
            Dictionary with complete RAG response including reranked documents
        """
        # Validate query (parent will do this, but we want early validation)
        if not query or query.strip() == "":
            raise ValueError(
                "Error: Query parameter is required and cannot be empty\n"
                "Context: BasicRerankRAG pipeline query operation\n"
                "Expected: Non-empty query string\n"
                "Actual: Empty or whitespace-only string\n"
                "Fix: Provide a valid query string, e.g., query='What is diabetes?'"
            )

        # Validate top_k (parent will do this, but we want early validation before rerank_factor calculation)
        if top_k < 1 or top_k > 100:
            raise ValueError(
                f"Error: top_k parameter out of valid range\n"
                f"Context: BasicRerankRAG pipeline query operation\n"
                f"Expected: Integer between 1 and 100 (inclusive)\n"
                f"Actual: {top_k}\n"
                f"Fix: Set top_k to a value between 1 and 100, e.g., top_k=5"
            )

        # Calculate how many documents to retrieve for reranking pool
        initial_k = min(top_k * self.rerank_factor, 100)  # Cap at 100 for performance

        # Get initial candidates using parent pipeline's query method
        # Set generate_answer=False initially to avoid duplicate LLM calls
        parent_kwargs = kwargs.copy()
        parent_kwargs["generate_answer"] = (
            False  # We'll generate answer after reranking
        )

        parent_result = super().query(query, top_k=initial_k, **parent_kwargs)
        candidate_documents = parent_result.get("retrieved_documents", [])

        # Always rerank if we have multiple candidates and a reranker (fixes the logic issue!)
        if len(candidate_documents) > 1 and self.reranker_func:
            try:
                final_documents = self._rerank_documents(
                    query, candidate_documents, top_k
                )
                logger.debug(
                    f"Reranked {len(candidate_documents)} documents, returning top {len(final_documents)}"
                )
                reranked = True
            except Exception as e:
                logger.warning(f"Reranking failed, falling back to original order: {e}")
                final_documents = candidate_documents[:top_k]
                reranked = False
        else:
            # Single document or no reranker - just return what we have
            final_documents = candidate_documents[:top_k]
            reranked = False
            if len(candidate_documents) <= 1:
                logger.debug(
                    f"Only {len(candidate_documents)} candidates found, no reranking needed"
                )
            else:
                logger.debug(f"No reranker available, returning top {top_k} documents")

        # Now generate answer if requested (using reranked documents)
        generate_answer = kwargs.get("generate_answer", True)
        if generate_answer and self.llm_func and final_documents:
            try:
                custom_prompt = kwargs.get("custom_prompt")
                answer = self._generate_answer(
                    query, final_documents, custom_prompt
                )
            except Exception as e:
                logger.warning(f"Answer generation failed: {e}")
                answer = "Error generating answer"
        elif not generate_answer:
            answer = None
        elif not final_documents:
            answer = "No relevant documents found to answer the query."
        else:
            answer = "No LLM function provided. Retrieved documents only."

        # Build complete response (matching parent format exactly)
        contexts_list = [doc.page_content for doc in final_documents]
        sources = self._extract_sources(final_documents) if kwargs.get("include_sources", True) else []
        retrieval_method = kwargs.get("method", "rerank")

        response = {
            "query": query,
            "answer": answer,
            "retrieved_documents": final_documents,
            "contexts": contexts_list,
            "execution_time": parent_result.get("execution_time", 0.0),
            "metadata": {
                "num_retrieved": len(final_documents),
                "processing_time": parent_result.get("execution_time", 0.0),
                "pipeline_type": "basic_rag_reranking",
                "reranked": reranked,
                "initial_candidates": len(candidate_documents),
                "rerank_factor": self.rerank_factor,
                "generated_answer": generate_answer and answer is not None,
                "retrieval_method": retrieval_method,  # FR-003: Include retrieval method
                "context_count": len(contexts_list),  # FR-003: Include context count
                "sources": sources,  # FR-003: Include sources in metadata
            },
        }

        # Add sources to top level if requested
        if kwargs.get("include_sources", True):
            response["sources"] = sources

        logger.info(
            f"Reranking RAG query completed - {len(final_documents)} docs returned (reranked: {reranked})"
        )
        return response

    def _rerank_documents(
        self, query_text: str, documents: List[Document], top_k: int = 5
    ) -> List[Document]:
        """
        Apply reranking function to reorder retrieved documents.

        Args:
            query_text: The query text
            documents: Initial retrieved documents
            top_k: Number of top documents to return

        Returns:
            Reranked list of top-k documents
        """
        try:
            logger.debug(
                f"Reranking {len(documents)} documents for query: {query_text[:50]}..."
            )

            # Apply reranker function
            reranked_results = self.reranker_func(query_text, documents)

            # Sort by score (descending)
            reranked_results = sorted(
                reranked_results, key=lambda x: x[1], reverse=True
            )

            # Log reranking results
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Post-reranking document order:")
                for i, (doc, score) in enumerate(reranked_results[:top_k]):
                    source = doc.metadata.get("source", "Unknown")
                    logger.debug(f"  [{i}] {source} (score: {score:.4f})")

            # Return top_k documents
            return [doc for doc, score in reranked_results[:top_k]]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback to original order
            return documents[:top_k]

    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about this pipeline's configuration.

        Returns:
            Dictionary with pipeline information
        """
        info = (
            super().get_pipeline_info() if hasattr(super(), "get_pipeline_info") else {}
        )

        info.update(
            {
                "pipeline_type": "basic_rag_reranking",
                "rerank_factor": self.rerank_factor,
                "reranker_model": self.reranker_model,
                "has_reranker": self.reranker_func is not None,
            }
        )

        return info
