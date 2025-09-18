"""
PyLate ColBERT Pipeline with Consistent Configuration

Simple PyLate-based ColBERT implementation that follows the same configuration
patterns as BasicRAGReranking for consistency across the evaluation framework.
"""

import logging
import tempfile
from typing import List, Dict, Any, Optional, Callable, Tuple
from ..basic import BasicRAGPipeline
from ...core.models import Document

logger = logging.getLogger(__name__)


class PyLateColBERTPipeline(BasicRAGPipeline):
    """
    PyLate-based ColBERT pipeline with native reranking.
    
    Maintains configuration consistency with BasicRAGReranking while using
    PyLate's native rank.rerank method for ColBERT-style late interaction.
    """
    
    def __init__(
        self,
        connection_manager,
        config_manager,
        **kwargs,
    ):
        """Initialize PyLate ColBERT pipeline with consistent configuration."""
        # Initialize parent pipeline 
        super().__init__(connection_manager, config_manager, **kwargs)
        
        # Use same config pattern as BasicRAGReranking for consistency
        self.colbert_config = self.config_manager.get(
            "pipelines:colbert_pylate", 
            self.config_manager.get("pipelines:basic_reranking", {})
        )
        
        # Configuration parameters (consistent naming with BasicRAGReranking)
        self.rerank_factor = self.colbert_config.get("rerank_factor", 2)
        self.model_name = self.colbert_config.get("model_name", "lightonai/GTE-ModernColBERT-v1")
        self.batch_size = self.colbert_config.get("batch_size", 32)
        
        # PyLate-specific parameters
        self.use_native_reranking = self.colbert_config.get("use_native_reranking", True)
        self.cache_embeddings = self.colbert_config.get("cache_embeddings", True)
        self.max_doc_length = self.colbert_config.get("max_doc_length", 4096)
        
        # Initialize components
        self.model = None
        self.is_initialized = False
        self._document_store = {}
        self._embedding_cache = {}
        self.index_folder = None
        
        # Statistics
        self.stats = {
            'queries_processed': 0,
            'documents_indexed': 0,
            'reranking_operations': 0
        }
        
        self._initialize_components()
        
        logger.info(f"Initialized PyLateColBERT with rerank_factor={self.rerank_factor}, model={self.model_name}")
    
    def _initialize_components(self):
        """Initialize PyLate components - fail hard if not available."""
        self._import_pylate()
        self._setup_index_folder()
        self._initialize_model()
        self.is_initialized = True
        logger.info("PyLate ColBERT pipeline initialized successfully")
    
    def _import_pylate(self):
        """Import PyLate components."""
        global models, rank
        from pylate import models, rank
        logger.debug("PyLate library imported successfully")
    
    def _setup_index_folder(self):
        """Setup temporary index folder."""
        self.index_folder = tempfile.mkdtemp(prefix="pylate_index_")
        logger.debug(f"Created index folder: {self.index_folder}")
    
    def _initialize_model(self):
        """Initialize PyLate ColBERT model."""
        self.model = models.ColBERT(model_name_or_path=self.model_name)
        logger.info(f"PyLate model '{self.model_name}' loaded")
    
    def load_documents(self, documents, **kwargs) -> Dict[str, Any]:
        """Load documents and prepare for retrieval."""
        # Handle both file paths and Document objects
        if isinstance(documents, str):
            # File path - delegate to parent
            result = super().load_documents(documents, **kwargs)
            if 'documents' in result:
                docs = result['documents']
                # Store documents for PyLate reranking
                for i, doc in enumerate(docs):
                    self._document_store[str(i)] = doc
                self.stats['documents_indexed'] = len(docs)
        else:
            # List of Document objects
            # Store documents for PyLate reranking
            for i, doc in enumerate(documents):
                self._document_store[str(i)] = doc
            
            # Call parent to handle vector store indexing
            result = super().load_documents(documents, **kwargs)
            self.stats['documents_indexed'] = len(documents)
        
        logger.info(f"Loaded {self.stats['documents_indexed']} documents for PyLate ColBERT")
        return result
    
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Execute ColBERT query with PyLate native reranking.
        
        Follows the same pattern as BasicRAGReranking:
        1. Retrieve initial candidates (rerank_factor * top_k)
        2. Apply PyLate native reranking
        3. Return top_k documents with consistent response format
        """
        # Calculate initial retrieval size
        initial_k = min(top_k * self.rerank_factor, 100)
        
        # Get initial candidates using parent pipeline
        parent_kwargs = kwargs.copy()
        parent_kwargs["generate_answer"] = False  # Generate after reranking
        
        parent_result = super().query(query_text, top_k=initial_k, **parent_kwargs)
        candidate_documents = parent_result.get("retrieved_documents", [])
        
        # Apply PyLate native reranking if available and beneficial
        if len(candidate_documents) > 1 and self.use_native_reranking and self.is_initialized:
            final_documents = self._pylate_rerank(query_text, candidate_documents, top_k)
            reranked = True
            self.stats['reranking_operations'] += 1
            logger.debug(f"PyLate reranked {len(candidate_documents)} → {len(final_documents)} documents")
        else:
            final_documents = candidate_documents[:top_k]
            reranked = False
            logger.debug(f"No reranking applied, returning {len(final_documents)} documents")
        
        # Generate answer if requested (same as BasicRAGReranking)
        generate_answer = kwargs.get("generate_answer", True)
        if generate_answer and self.llm_func and final_documents:
            try:
                custom_prompt = kwargs.get("custom_prompt")
                answer = self._generate_answer(query_text, final_documents, custom_prompt)
            except Exception as e:
                logger.warning(f"Answer generation failed: {e}")
                answer = "Error generating answer"
        elif not generate_answer:
            answer = None
        elif not final_documents:
            answer = "No relevant documents found to answer the query."
        else:
            answer = "No LLM function provided. Retrieved documents only."
        
        # Build response with consistent format
        response = {
            "query": query_text,
            "answer": answer,
            "retrieved_documents": final_documents,
            "contexts": [doc.page_content for doc in final_documents],
            "execution_time": parent_result.get("execution_time", 0.0),
            "metadata": {
                "num_retrieved": len(final_documents),
                "processing_time": parent_result.get("execution_time", 0.0),
                "pipeline_type": "colbert_pylate",
                "reranked": reranked,
                "initial_candidates": len(candidate_documents),
                "rerank_factor": self.rerank_factor,
                "generated_answer": generate_answer and answer is not None,
                "model_name": self.model_name,
                "native_reranking": self.use_native_reranking,
            },
        }
        
        # Add sources if requested
        include_sources = kwargs.get("include_sources", True)
        if include_sources:
            response["sources"] = self._extract_sources(final_documents)
        
        self.stats['queries_processed'] += 1
        logger.info(f"PyLate ColBERT query completed - {len(final_documents)} docs returned (reranked: {reranked})")
        return response
    
    def _pylate_rerank(self, query_text: str, documents: List[Document], top_k: int) -> List[Document]:
        """Apply PyLate native reranking using rank.rerank method."""
        # Prepare documents for PyLate
        doc_texts = [doc.page_content for doc in documents]
        doc_ids = list(range(len(documents)))
        
        # Generate embeddings
        query_embeddings = self.model.encode([query_text], is_query=True)
        doc_embeddings = self.model.encode(doc_texts, is_query=False)
        
        # Apply PyLate native reranking
        reranked_results = rank.rerank(
            documents_ids=[doc_ids],  # Nested list as required by PyLate
            queries_embeddings=query_embeddings,
            documents_embeddings=doc_embeddings,
        )
        
        # Extract reranked document order (first query results)
        if reranked_results and len(reranked_results) > 0:
            reranked_ids = reranked_results[0][:top_k]  # Top k from first query
            return [documents[doc_id] for doc_id in reranked_ids]
        else:
            raise RuntimeError("PyLate reranking returned empty results")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information with consistent format."""
        info = super().get_pipeline_info() if hasattr(super(), "get_pipeline_info") else {}
        
        info.update({
            "pipeline_type": "colbert_pylate",
            "rerank_factor": self.rerank_factor,
            "model_name": self.model_name,
            "use_native_reranking": self.use_native_reranking,
            "batch_size": self.batch_size,
            "is_initialized": self.is_initialized,
            "stats": self.stats.copy(),
        })
        
        return info

