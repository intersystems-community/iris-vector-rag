"""
HyDE (Hypothetical Document Embeddings) RAG Pipeline implementation.

This module provides a RAG implementation using the HyDE technique for improved retrieval.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable
from ..core.base import RAGPipeline
from ..core.models import Document
from ..core.connection import ConnectionManager
from ..config.manager import ConfigurationManager
from ..embeddings.manager import EmbeddingManager

logger = logging.getLogger(__name__)


class HyDERAGPipeline(RAGPipeline):
    """
    HyDE RAG pipeline implementation.
    
    This pipeline implements the HyDE (Hypothetical Document Embeddings) approach:
    1. Generate hypothetical document for the query
    2. Use hypothetical document embedding for retrieval
    3. Context augmentation and LLM generation
    """
    
    def __init__(self, connection_manager: Optional[ConnectionManager] = None,
                 config_manager: Optional[ConfigurationManager] = None,
                 llm_func: Optional[Callable[[str], str]] = None, vector_store=None):
        """
        Initialize the HyDE RAG Pipeline.
        
        Args:
            connection_manager: Optional manager for database connections (defaults to new instance)
            config_manager: Optional manager for configuration settings (defaults to new instance)
            llm_func: Optional LLM function for answer generation
            vector_store: Optional VectorStore instance
        """
        # Create default instances if not provided
        if connection_manager is None:
            try:
                connection_manager = ConnectionManager()
            except Exception as e:
                logger.warning(f"Failed to create default ConnectionManager: {e}")
                connection_manager = None
        
        if config_manager is None:
            try:
                config_manager = ConfigurationManager()
            except Exception as e:
                logger.warning(f"Failed to create default ConfigurationManager: {e}")
                config_manager = ConfigurationManager()  # Always need config manager
        
        super().__init__(connection_manager, config_manager, vector_store)
        self.llm_func = llm_func
        
        # Initialize components
        self.embedding_manager = EmbeddingManager(config_manager)
        
        # Get pipeline configuration
        self.pipeline_config = self.config_manager.get("pipelines:hyde", {})
        self.top_k = self.pipeline_config.get("top_k", 5)
        self.use_hypothetical_doc = self.pipeline_config.get("use_hypothetical_doc", True)
        
        logger.info(f"Initialized HyDERAGPipeline with top_k={self.top_k}")
    
    def execute(self, query_text: str, **kwargs) -> dict:
        """
        Execute the HyDE pipeline (required abstract method).
        
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
        logger.info(f"HyDE: Loaded {len(documents)} documents - {result}")
    
    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Ingest documents using standard storage.
        
        Args:
            documents: List of documents to ingest
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        logger.info(f"Starting HyDE ingestion of {len(documents)} documents")
        
        try:
            # Use vector store for document storage
            document_ids = self._store_documents(documents)
            
            end_time = time.time()
            result = {
                "status": "success",
                "documents_stored": len(document_ids),
                "document_ids": document_ids,
                "processing_time": end_time - start_time,
                "pipeline_type": "hyde_rag"
            }
            
            logger.info(f"HyDE ingestion completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"HyDE ingestion failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "pipeline_type": "hyde_rag"
            }
    
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute a query using HyDE technique.
        
        Args:
            query_text: The query string
            top_k: Number of top documents to retrieve
            
        Returns:
            Dictionary with query results
        """
        start_time = time.time()
        logger.info(f"Processing HyDE query: {query_text}")
        
        try:
            # Generate hypothetical document if LLM is available
            hypothetical_doc = None
            search_text = query_text
            
            if self.use_hypothetical_doc and self.llm_func:
                hypothetical_doc = self._generate_hypothetical_document(query_text)
                search_text = hypothetical_doc
                logger.info(f"Generated hypothetical document: {hypothetical_doc[:100]}...")
            
            # Generate embedding for search text (query or hypothetical doc)
            search_embedding = self.embedding_manager.embed_text(search_text)
            
            # Retrieve relevant documents
            relevant_docs = self._retrieve_documents(search_embedding, top_k)
            
            # Generate answer if LLM function is available
            answer = None
            if self.llm_func and relevant_docs:
                context = self._build_context(relevant_docs)
                prompt = self._build_prompt(query_text, context)
                answer = self.llm_func(prompt)
            
            # Provide fallback message if answer is still None
            if answer is None:
                if not self.llm_func:
                    answer = "No LLM function available for answer generation. Please configure an LLM function to generate answers."
                elif not relevant_docs:
                    answer = "No relevant documents found for the query. Unable to generate an answer without context."
                else:
                    answer = "LLM function failed to generate an answer. Please check the LLM configuration."
            
            end_time = time.time()
            
            result = {
                "query": query_text,
                "hypothetical_document": hypothetical_doc,
                "answer": answer,
                "retrieved_documents": relevant_docs,
                "num_documents_retrieved": len(relevant_docs),
                "processing_time": end_time - start_time,
                "pipeline_type": "hyde_rag"
            }
            
            logger.info(f"HyDE query completed in {end_time - start_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"HyDE query failed: {e}")
            return {
                "query": query_text,
                "answer": None,
                "error": str(e),
                "pipeline_type": "hyde_rag"
            }
    
    def _generate_hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        hyde_prompt = f"""Please write a hypothetical document that would contain the answer to this question: {query}

Write a detailed, informative passage that directly addresses the question. Focus on providing factual information that would be found in a relevant document.

Hypothetical document:"""
        
        try:
            hypothetical_doc = self.llm_func(hyde_prompt)
            return hypothetical_doc.strip()
        except Exception as e:
            logger.warning(f"Failed to generate hypothetical document: {e}")
            return query  # Fallback to original query
    
    def _retrieve_documents(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using vector similarity."""
        # Use base class helper method for vector search
        results = self._retrieve_documents_by_vector(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # Convert Document objects to dictionary format for backward compatibility
        documents = []
        for doc, score in results:
            documents.append({
                "doc_id": doc.id,
                "title": doc.metadata.get("title", ""),
                "content": doc.page_content,
                "similarity_score": float(score)
            })
        
        return documents
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Untitled')
            content = doc.get('content', '')
            context_parts.append(f"[Document {i}: {title}]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM generation."""
        return f"""Based on the following retrieved documents, please answer the question.

Context:
{context}

Question: {query}

Answer:"""