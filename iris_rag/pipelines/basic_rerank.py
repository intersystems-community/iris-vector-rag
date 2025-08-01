"""
Basic RAG Pipeline implementation with ReRanking step after the initial vector search.

This pipeline extends BasicRAGPipeline to add reranking functionality while
eliminating code duplication through proper inheritance.
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from .basic import BasicRAGPipeline
from ..core.models import Document

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
    
    def __init__(self, connection_manager, config_manager,
                 reranker_func: Optional[Callable[[str, List[Document]], List[Tuple[Document, float]]]] = None, 
                 **kwargs):
        """
        Initialize the Basic RAG Reranking Pipeline.
        
        Args:
            connection_manager: Manager for database connections
            config_manager: Manager for configuration settings
            reranker_func: Optional custom reranker function. If None, uses default HuggingFace reranker.
            **kwargs: Additional arguments passed to parent BasicRAGPipeline
        """
        # Initialize parent pipeline with all standard functionality
        super().__init__(connection_manager, config_manager, **kwargs)
        
        # Set up reranking-specific configuration
        # Use dedicated reranking config section with fallback to basic config
        self.reranking_config = self.config_manager.get("pipelines:basic_reranking", 
                                                       self.config_manager.get("pipelines:basic", {}))
        
        # Reranking parameters
        self.rerank_factor = self.reranking_config.get("rerank_factor", 2)
        self.reranker_model = self.reranking_config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        # Set reranker function (default to HuggingFace if none provided)
        self.reranker_func = reranker_func or hf_reranker
        
        logger.info(f"Initialized BasicRAGRerankingPipeline with rerank_factor={self.rerank_factor}")
    
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Retrieve and rerank relevant documents for a query.
        
        This method overrides the parent query method to add reranking:
        1. Retrieves rerank_factor * top_k documents using parent method
        2. Applies reranking to improve document ordering  
        3. Returns top_k best documents after reranking
        
        Args:
            query_text: The query text
            top_k: Number of documents to return after reranking
            **kwargs: Additional arguments passed to parent query method
                
        Returns:
            Dictionary with reranked documents in standard format
        """
        # Calculate how many documents to retrieve for reranking pool
        initial_k = min(top_k * self.rerank_factor, 100)  # Cap at 100 for performance
        
        # Get initial candidates using parent pipeline's query method
        parent_result = super().query(query_text, top_k=initial_k, **kwargs)
        candidate_documents = parent_result.get("retrieved_documents", [])
        
        # If we have fewer candidates than requested, just return them
        if len(candidate_documents) <= top_k:
            logger.debug(f"Only {len(candidate_documents)} candidates found, skipping reranking")
            final_documents = candidate_documents
        else:
            # Apply reranking if we have a reranker function and multiple candidates
            if self.reranker_func and len(candidate_documents) > 1:
                try:
                    final_documents = self._rerank_documents(query_text, candidate_documents, top_k)
                    logger.debug(f"Reranked {len(candidate_documents)} documents, returning top {len(final_documents)}")
                except Exception as e:
                    logger.warning(f"Reranking failed, falling back to original order: {e}")
                    final_documents = candidate_documents[:top_k]
            else:
                # No reranking available, return top candidates
                logger.debug(f"No reranking applied, returning top {top_k} documents")
                final_documents = candidate_documents[:top_k]
        
        # Return in same format as parent, but with reranked documents
        return {
            "query": query_text,
            "retrieved_documents": final_documents,
            "answer": None,
            "metadata": {
                "num_retrieved": len(final_documents),
                "pipeline_type": "basic_rag_reranking",
                "reranked": self.reranker_func is not None and len(candidate_documents) > top_k
            }
        }
    
    def _rerank_documents(self, query_text: str, documents: List[Document], top_k: int = 5) -> List[Document]:
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
            logger.debug(f"Reranking {len(documents)} documents for query: {query_text[:50]}...")
            
            # Apply reranker function
            reranked_results = self.reranker_func(query_text, documents)
            
            # Sort by score (descending)
            reranked_results = sorted(reranked_results, key=lambda x: x[1], reverse=True)
            
            # Log reranking results
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Post-reranking document order:")
                for i, (doc, score) in enumerate(reranked_results[:top_k]):
                    source = doc.metadata.get('source', 'Unknown')
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
        info = super().get_pipeline_info() if hasattr(super(), 'get_pipeline_info') else {}
        
        info.update({
            "pipeline_type": "basic_rag_reranking",
            "rerank_factor": self.rerank_factor,
            "reranker_model": self.reranker_model,
            "has_reranker": self.reranker_func is not None
        })
        
        return info