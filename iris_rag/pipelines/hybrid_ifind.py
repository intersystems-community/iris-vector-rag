"""
Hybrid IFind RAG Pipeline implementation.

This module provides a RAG implementation that combines vector search with IRIS IFind text search.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable
from ..core.base import RAGPipeline
from ..core.models import Document
from ..core.connection import ConnectionManager
from ..config.manager import ConfigurationManager
from ..storage.iris import IRISStorage
from ..embeddings.manager import EmbeddingManager

logger = logging.getLogger(__name__)


class HybridIFindRAGPipeline(RAGPipeline):
    """
    Hybrid IFind RAG pipeline implementation.
    
    This pipeline combines:
    1. Vector similarity search for semantic matching
    2. IRIS IFind text search for keyword matching
    3. Hybrid ranking and result fusion
    """
    
    def __init__(self, connection_manager: ConnectionManager, config_manager: ConfigurationManager,
                 vector_store=None, llm_func: Optional[Callable[[str], str]] = None):
        """
        Initialize the Hybrid IFind RAG Pipeline.
        
        Args:
            connection_manager: Manager for database connections
            config_manager: Manager for configuration settings
            vector_store: Optional VectorStore instance
            llm_func: Optional LLM function for answer generation
        """
        super().__init__(connection_manager, config_manager, vector_store)
        self.llm_func = llm_func
        
        # Initialize components
        self.storage = IRISStorage(connection_manager, config_manager)
        self.embedding_manager = EmbeddingManager(config_manager)
        
        # Get pipeline configuration
        self.pipeline_config = self.config_manager.get("pipelines:hybrid_ifind", {})
        self.top_k = self.pipeline_config.get("top_k", 5)
        self.vector_weight = self.pipeline_config.get("vector_weight", 0.6)
        self.ifind_weight = self.pipeline_config.get("ifind_weight", 0.4)
        self.min_ifind_score = self.pipeline_config.get("min_ifind_score", 0.1)
        
        logger.info(f"Initialized HybridIFindRAGPipeline with vector_weight={self.vector_weight}")
    
    def execute(self, query_text: str, **kwargs) -> dict:
        """
        Execute the Hybrid IFind pipeline (required abstract method).
        
        Args:
            query_text: The input query string
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing query, answer, and retrieved documents
        """
        top_k = kwargs.get("top_k", 5)
        return self.query(query_text, top_k)
    
    def load_documents(self, documents_path: str, **kwargs) -> None:
        """
        Load documents into the knowledge base (required abstract method).
        
        Args:
            documents_path: Path to documents or directory
            **kwargs: Additional keyword arguments including:
                - documents: List of Document objects (if providing directly)
                - chunk_documents: Whether to chunk documents (default: True)
                - generate_embeddings: Whether to generate embeddings (default: True)
        """
        # Handle direct document input
        if "documents" in kwargs:
            documents = kwargs["documents"]
            if not isinstance(documents, list):
                raise ValueError("Documents must be provided as a list")
        else:
            # Load documents from path - basic implementation
            import os
            documents = []
            
            if os.path.isfile(documents_path):
                with open(documents_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"source": documents_path}
                )
                documents.append(doc)
            elif os.path.isdir(documents_path):
                for filename in os.listdir(documents_path):
                    file_path = os.path.join(documents_path, filename)
                    if os.path.isfile(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            doc = Document(
                                page_content=content,
                                metadata={"source": file_path, "filename": filename}
                            )
                            documents.append(doc)
                        except Exception as e:
                            logger.warning(f"Failed to load file {file_path}: {e}")
        
        # Use the ingest_documents method
        result = self.ingest_documents(documents)
        logger.info(f"Hybrid IFind: Loaded {len(documents)} documents - {result}")
    
    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Ingest documents with IFind indexing support.
        
        Args:
            documents: List of documents to ingest
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        logger.info(f"Starting Hybrid IFind ingestion of {len(documents)} documents")
        
        try:
            # Store documents (this will also create IFind indexes if configured)
            result = self.storage.store_documents(documents)
            
            # Ensure IFind indexes are built
            self._ensure_ifind_indexes()
            
            end_time = time.time()
            result.update({
                "processing_time": end_time - start_time,
                "pipeline_type": "hybrid_ifind_rag"
            })
            
            logger.info(f"Hybrid IFind ingestion completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Hybrid IFind ingestion failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "pipeline_type": "hybrid_ifind_rag"
            }
    
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute a query using hybrid vector + IFind search.
        
        Args:
            query_text: The query string
            top_k: Number of top documents to retrieve
            
        Returns:
            Dictionary with query results
        """
        start_time = time.time()
        logger.info(f"Processing Hybrid IFind query: {query_text}")
        
        try:
            # Perform vector search
            query_embedding = self.embedding_manager.embed_text(query_text)
            vector_results = self._vector_search(query_embedding, top_k * 2)  # Get more for fusion
            
            # Perform IFind text search
            ifind_results = self._ifind_search(query_text, top_k * 2)  # Get more for fusion
            
            # Fuse results using hybrid ranking
            fused_results = self._fuse_results(vector_results, ifind_results, top_k)
            
            # Convert to Document objects for standardized return format
            retrieved_documents = []
            for doc_dict in fused_results:
                doc = Document(
                    id=doc_dict["doc_id"],
                    page_content=str(doc_dict["content"]),  # VectorStore guarantees string content
                    metadata={
                        "title": str(doc_dict.get("title", "")),
                        "hybrid_score": doc_dict.get("hybrid_score", 0.0),
                        "vector_score": doc_dict.get("vector_score", 0.0),
                        "ifind_score": doc_dict.get("ifind_score", 0.0),
                        "has_vector": doc_dict.get("has_vector", False),
                        "has_ifind": doc_dict.get("has_ifind", False),
                        "search_type": doc_dict.get("search_type", "hybrid")
                    }
                )
                retrieved_documents.append(doc)
            
            # Generate answer if LLM function is available
            answer = None
            if self.llm_func and retrieved_documents:
                context = self._build_context_from_documents(retrieved_documents)
                prompt = self._build_prompt(query_text, context)
                answer = self.llm_func(prompt)
            
            end_time = time.time()
            
            result = {
                "query": query_text,
                "answer": answer,
                "retrieved_documents": retrieved_documents,
                "vector_results_count": len(vector_results),
                "ifind_results_count": len(ifind_results),
                "num_documents_retrieved": len(retrieved_documents),
                "processing_time": end_time - start_time,
                "pipeline_type": "hybrid_ifind_rag"
            }
            
            logger.info(f"Hybrid IFind query completed in {end_time - start_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Hybrid IFind query failed: {e}")
            return {
                "query": query_text,
                "answer": None,
                "error": str(e),
                "pipeline_type": "hybrid_ifind_rag"
            }
    
    def _ensure_ifind_indexes(self):
        """Ensure IFind indexes are created for text search."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Check if IFind index exists
            check_sql = """
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.INDEXES 
            WHERE TABLE_NAME = 'SourceDocuments' 
            AND TABLE_SCHEMA = 'RAG' 
            AND INDEX_NAME LIKE '%IFIND%'
            """
            
            cursor.execute(check_sql)
            result = cursor.fetchone()
            
            if result[0] == 0:
                logger.info("Creating IFind indexes for text search")
                
                # Create IFind index on content
                create_index_sql = """
                CREATE INDEX IF NOT EXISTS idx_sourcedocs_ifind_content
                ON RAG.SourceDocuments (text_content)
                WITH (TYPE = 'IFIND')
                """
                
                cursor.execute(create_index_sql)
                connection.commit()
                logger.info("IFind index created successfully")
            
        except Exception as e:
            logger.warning(f"Could not create IFind index: {e}")
            # Continue without IFind - will fallback to regular text search
        finally:
            cursor.close()
    
    def _vector_search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            search_sql = f"""
            SELECT TOP {top_k}
                doc_id, title, text_content,
                VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?)) as vector_score
            FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
            ORDER BY vector_score DESC
            """
            
            cursor.execute(search_sql, [str(query_embedding)])
            results = cursor.fetchall()
            
            documents = []
            for row in results:
                documents.append({
                    "doc_id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "vector_score": float(row[3]),
                    "search_type": "vector"
                })
            
            return documents
            
        finally:
            cursor.close()
    
    def _ifind_search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform IFind text search."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Try IFind search first
            ifind_sql = f"""
            SELECT TOP {top_k}
                doc_id, title, text_content,
                $SCORE(text_content) as ifind_score
            FROM RAG.SourceDocuments
            WHERE $FIND(text_content, ?)
            ORDER BY ifind_score DESC
            """
            
            try:
                cursor.execute(ifind_sql, [query_text])
                results = cursor.fetchall()
                
                documents = []
                for row in results:
                    ifind_score = float(row[3]) if row[3] is not None else 0.0
                    if ifind_score >= self.min_ifind_score:
                        documents.append({
                            "doc_id": row[0],
                            "title": row[1],
                            "content": row[2],
                            "ifind_score": ifind_score,
                            "search_type": "ifind"
                        })
                
                return documents
                
            except Exception as ifind_error:
                logger.warning(f"IFind search failed, falling back to LIKE search: {ifind_error}")
                
                # Fallback to simple text search
                like_sql = f"""
                SELECT TOP {top_k}
                    doc_id, title, text_content,
                    1.0 as text_score
                FROM RAG.SourceDocuments
                WHERE text_content LIKE ?
                """
                
                cursor.execute(like_sql, [f"%{query_text}%"])
                results = cursor.fetchall()
                
                documents = []
                for row in results:
                    documents.append({
                        "doc_id": row[0],
                        "title": row[1],
                        "content": row[2],
                        "ifind_score": 1.0,
                        "search_type": "text_fallback"
                    })
                
                return documents
            
        finally:
            cursor.close()
    
    def _fuse_results(self, vector_results: List[Dict[str, Any]], 
                     ifind_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Fuse vector and IFind results using hybrid ranking."""
        
        # Normalize scores
        vector_results = self._normalize_scores(vector_results, "vector_score")
        ifind_results = self._normalize_scores(ifind_results, "ifind_score")
        
        # Create combined results dictionary
        combined_docs = {}
        
        # Add vector results
        for doc in vector_results:
            doc_id = doc["doc_id"]
            combined_docs[doc_id] = doc.copy()
            combined_docs[doc_id]["hybrid_score"] = self.vector_weight * doc["vector_score"]
            combined_docs[doc_id]["has_vector"] = True
            combined_docs[doc_id]["has_ifind"] = False
        
        # Add/merge IFind results
        for doc in ifind_results:
            doc_id = doc["doc_id"]
            if doc_id in combined_docs:
                # Merge scores
                combined_docs[doc_id]["hybrid_score"] += self.ifind_weight * doc["ifind_score"]
                combined_docs[doc_id]["has_ifind"] = True
                combined_docs[doc_id]["ifind_score"] = doc["ifind_score"]
            else:
                # New document from IFind
                combined_docs[doc_id] = doc.copy()
                combined_docs[doc_id]["hybrid_score"] = self.ifind_weight * doc["ifind_score"]
                combined_docs[doc_id]["has_vector"] = False
                combined_docs[doc_id]["has_ifind"] = True
                combined_docs[doc_id]["vector_score"] = 0.0
        
        # Sort by hybrid score and return top_k
        sorted_docs = sorted(combined_docs.values(), 
                           key=lambda x: x["hybrid_score"], 
                           reverse=True)
        
        return sorted_docs[:top_k]
    
    def _normalize_scores(self, results: List[Dict[str, Any]], score_field: str) -> List[Dict[str, Any]]:
        """Normalize scores to 0-1 range."""
        if not results:
            return results
        
        scores = [doc[score_field] for doc in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same
            for doc in results:
                doc[score_field] = 1.0
        else:
            # Normalize to 0-1
            for doc in results:
                doc[score_field] = (doc[score_field] - min_score) / (max_score - min_score)
        
        return results
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents (legacy method)."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Untitled')
            content = doc.get('content', '')
            score = doc.get('hybrid_score', 0.0)
            context_parts.append(f"[Document {i}: {title} (Score: {score:.3f})]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _build_context_from_documents(self, documents: List[Document]) -> str:
        """Build context string from Document objects."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get('title', 'Untitled')
            content = doc.page_content
            score = doc.metadata.get('hybrid_score', 0.0)
            context_parts.append(f"[Document {i}: {title} (Score: {score:.3f})]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM generation."""
        return f"""Based on the following retrieved documents (ranked by hybrid vector + text search), please answer the question.

Context:
{context}

Question: {query}

Answer:"""