"""
Hybrid Vector-Text RAG Pipeline - Single Table Implementation.

This pipeline demonstrates the single table approach for hybrid search,
using the main SourceDocuments table with vector search and text fallback.
Created following the Pipeline Development Guide patterns.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable
from ..pipelines.basic import BasicRAGPipeline
from ..core.models import Document
from ..core.connection import ConnectionManager
from ..config.manager import ConfigurationManager

logger = logging.getLogger(__name__)


class HybridVectorTextPipeline(BasicRAGPipeline):
    """
    Hybrid Vector-Text RAG Pipeline - Single Table Implementation.
    
    This pipeline extends BasicRAGPipeline to add text search capabilities
    while using the main SourceDocuments table (single table approach).
    
    Features:
    - Vector similarity search (primary)
    - Text search fallback (when text fields support it)
    - Reciprocal Rank Fusion for result combination
    - Config-driven table and parameter management
    - Schema manager integration
    """
    
    def __init__(self, connection_manager: ConnectionManager, config_manager: ConfigurationManager,
                 vector_store=None, llm_func: Optional[Callable[[str], str]] = None):
        """
        Initialize the Hybrid Vector-Text RAG Pipeline.
        
        Args:
            connection_manager: Manager for database connections
            config_manager: Manager for configuration settings
            vector_store: Optional VectorStore instance
            llm_func: Optional LLM function for answer generation
        """
        # Initialize parent BasicRAGPipeline
        super().__init__(connection_manager, config_manager, llm_func, vector_store)
        
        # Get pipeline-specific configuration
        self.pipeline_config = self.config_manager.get("pipelines:hybrid_vector_text", {})
        self.vector_weight = self.pipeline_config.get("vector_weight", 0.7)
        self.text_weight = self.pipeline_config.get("text_weight", 0.3)
        self.enable_text_search = self.pipeline_config.get("enable_text_search", True)
        self.min_text_score = self.pipeline_config.get("min_text_score", 0.1)
        
        # Use schema manager to get the correct table name
        self.table_name = self.pipeline_config.get("table_name", "RAG.SourceDocuments")
        
        logger.info(f"Initialized HybridVectorTextPipeline with vector_weight={self.vector_weight}")
        logger.info(f"Using table: {self.table_name}, text search enabled: {self.enable_text_search}")
    
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Execute hybrid vector + text search query.
        
        This method overrides BasicRAGPipeline.query() to add text search
        and fusion capabilities while maintaining the unified API.
        
        Args:
            query_text: The query string
            top_k: Number of top documents to retrieve
            **kwargs: Additional arguments (passed to parent)
            
        Returns:
            Dictionary with complete RAG response in standard format
        """
        start_time = time.time()
        logger.info(f"Processing Hybrid Vector-Text query: {query_text}")
        
        try:
            # Step 1: Perform vector search using parent class
            vector_documents = self._vector_search(query_text, top_k)
            
            # Step 2: Perform text search (if enabled and supported)
            text_documents = []
            if self.enable_text_search:
                text_documents = self._text_search(query_text, top_k)
            
            # Step 3: Fuse results using reciprocal rank fusion
            if text_documents:
                fused_documents = self._fuse_results(vector_documents, text_documents, top_k)
                search_method = "hybrid"
            else:
                fused_documents = vector_documents[:top_k]
                search_method = "vector_only"
            
            # Step 4: Generate answer using parent method if LLM available
            generate_answer = kwargs.get("generate_answer", True)
            if generate_answer and self.llm_func and fused_documents:
                answer = self._generate_answer(query_text, fused_documents, kwargs.get("custom_prompt"))
            elif not generate_answer:
                answer = None
            elif not fused_documents:
                answer = "No relevant documents found to answer the query."
            else:
                answer = "No LLM function provided. Retrieved documents only."
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Step 5: Return complete response in standard format
            response = {
                "query": query_text,
                "answer": answer,
                "retrieved_documents": fused_documents,
                "contexts": [doc.page_content for doc in fused_documents],
                "execution_time": execution_time,
                "metadata": {
                    "num_retrieved": len(fused_documents),
                    "processing_time": execution_time,
                    "pipeline_type": "hybrid_vector_text",
                    "search_method": search_method,
                    "vector_results": len(vector_documents),
                    "text_results": len(text_documents),
                    "generated_answer": generate_answer and answer is not None
                }
            }
            
            # Add sources if requested
            if kwargs.get("include_sources", True):
                response["sources"] = self._extract_sources(fused_documents)
            
            logger.info(f"Hybrid Vector-Text query completed in {execution_time:.2f}s - {search_method}")
            return response
            
        except Exception as e:
            logger.error(f"Hybrid Vector-Text query failed: {e}")
            return {
                "query": query_text,
                "answer": None,
                "retrieved_documents": [],
                "contexts": [],
                "execution_time": time.time() - start_time,
                "error": str(e),
                "metadata": {
                    "pipeline_type": "hybrid_vector_text",
                    "search_method": "failed"
                }
            }
    
    def _vector_search(self, query_text: str, top_k: int) -> List[Document]:
        """Perform vector similarity search using parent class vector store."""
        try:
            if hasattr(self, 'vector_store') and self.vector_store:
                retrieved_documents = self.vector_store.similarity_search(query_text, k=top_k)
                # Add vector search metadata
                for doc in retrieved_documents:
                    doc.metadata.update({
                        "search_type": "vector",
                        "pipeline_source": "hybrid_vector_text"
                    })
                return retrieved_documents
            else:
                logger.warning("No vector store available for vector search")
                return []
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _text_search(self, query_text: str, top_k: int) -> List[Document]:
        """Perform text search with graceful fallback for different field types."""
        if not self.enable_text_search:
            return []
        
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Try LIKE search (simpler and more compatible)
            like_sql = f"""
            SELECT TOP {top_k}
                doc_id, title, text_content, 1.0 as text_score
            FROM {self.table_name}
            WHERE text_content LIKE ?
            ORDER BY doc_id
            """
            
            like_params = [f"%{query_text}%"]
            cursor.execute(like_sql, like_params)
            results = cursor.fetchall()
            
            logger.debug(f"Text search returned {len(results)} results")
            
            documents = []
            for row in results:
                # Handle potential stream objects
                title = str(row[1]) if row[1] else ""
                content = str(row[2]) if row[2] else ""
                
                doc = Document(
                    id=row[0],
                    page_content=content,
                    metadata={
                        "title": title,
                        "search_type": "text",
                        "text_score": float(row[3]),
                        "pipeline_source": "hybrid_vector_text"
                    }
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.warning(f"Text search failed: {e}. Continuing with vector-only search.")
            return []
        finally:
            cursor.close()
    
    def _fuse_results(self, vector_docs: List[Document], text_docs: List[Document], top_k: int) -> List[Document]:
        """Fuse vector and text results using reciprocal rank fusion."""
        # Create a dictionary to combine results by doc_id
        doc_scores = {}
        
        # Add vector results with rank-based scoring
        for rank, doc in enumerate(vector_docs):
            doc_id = getattr(doc, 'id', f'vec_{rank}')
            vector_rank_score = 1.0 / (rank + 1)  # Reciprocal rank fusion
            doc_scores[doc_id] = {
                "document": doc,
                "vector_rank_score": vector_rank_score,
                "text_rank_score": 0.0,
                "has_vector": True,
                "has_text": False
            }
            # Update metadata
            doc.metadata.update({
                "vector_rank": rank + 1,
                "vector_rank_score": vector_rank_score
            })
        
        # Add text results with rank-based scoring
        for rank, doc in enumerate(text_docs):
            doc_id = getattr(doc, 'id', f'text_{rank}')
            text_rank_score = 1.0 / (rank + 1)  # Reciprocal rank fusion
            
            if doc_id in doc_scores:
                # Document found in both searches - combine scores
                doc_scores[doc_id]["text_rank_score"] = text_rank_score
                doc_scores[doc_id]["has_text"] = True
                # Update existing document metadata
                existing_doc = doc_scores[doc_id]["document"]
                existing_doc.metadata.update({
                    "search_type": "hybrid",
                    "text_rank": rank + 1,
                    "text_rank_score": text_rank_score,
                    "has_text": True
                })
            else:
                # Document only found in text search
                doc_scores[doc_id] = {
                    "document": doc,
                    "vector_rank_score": 0.0,
                    "text_rank_score": text_rank_score,
                    "has_vector": False,
                    "has_text": True
                }
                # Update metadata
                doc.metadata.update({
                    "text_rank": rank + 1,
                    "text_rank_score": text_rank_score
                })
        
        # Calculate hybrid scores and sort
        for doc_id, doc_data in doc_scores.items():
            # Combine rank scores with weights
            hybrid_score = (self.vector_weight * doc_data["vector_rank_score"] + 
                          self.text_weight * doc_data["text_rank_score"])
            
            # Update document metadata with final scores
            doc_data["document"].metadata.update({
                "hybrid_score": hybrid_score,
                "vector_weight": self.vector_weight,
                "text_weight": self.text_weight
            })
        
        # Sort by hybrid score and return top_k documents
        sorted_docs = sorted(doc_scores.values(), 
                           key=lambda x: (self.vector_weight * x["vector_rank_score"] + 
                                        self.text_weight * x["text_rank_score"]), 
                           reverse=True)
        
        return [item["document"] for item in sorted_docs[:top_k]]