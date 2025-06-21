"""
Base pipeline class for all RAG techniques
Provides common functionality and connection management
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable
from abc import ABC, abstractmethod

from .connection_manager import get_connection_manager, ConnectionManager # Changed to relative import
from .utils import Document # Changed to relative import

logger = logging.getLogger(__name__)

class BaseRAGPipeline(ABC):
    """Base class for all RAG pipeline implementations"""
    
    def __init__(
        self, 
        connection_manager: Optional[ConnectionManager] = None,
        embedding_func: Optional[Callable] = None,
        llm_func: Optional[Callable] = None
    ):
        """
        Initialize base pipeline
        
        Args:
            connection_manager: Optional connection manager instance
            embedding_func: Function to generate embeddings
            llm_func: Function to generate LLM responses
        """
        self.connection_manager = connection_manager or get_connection_manager()
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        
        # Validate connection
        try:
            result = self.connection_manager.execute("SELECT 1")
            logger.info(f"Pipeline initialized with {self.connection_manager.connection_type.upper()} connection")
        except Exception as e:
            logger.error(f"Failed to validate connection: {e}")
            raise
    
    @abstractmethod
    def retrieve_documents(
        self, 
        query: str, 
        top_k: int = 5, 
        **kwargs
    ) -> List[Document]:
        """
        Retrieve relevant documents for the query
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            **kwargs: Additional technique-specific parameters
            
        Returns:
            List of retrieved documents
        """
        pass
    
    @abstractmethod
    def generate_answer(
        self, 
        query: str, 
        documents: List[Document],
        **kwargs
    ) -> str:
        """
        Generate answer based on retrieved documents
        
        Args:
            query: User query
            documents: Retrieved documents
            **kwargs: Additional technique-specific parameters
            
        Returns:
            Generated answer
        """
        pass
    
    def run(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Run the complete RAG pipeline
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            **kwargs: Additional technique-specific parameters
            
        Returns:
            Dictionary with query, answer, documents, and metadata
        """
        start_time = time.time()
        
        # Retrieve documents
        retrieval_start = time.time()
        documents = self.retrieve_documents(query, top_k, **kwargs)
        retrieval_time = time.time() - retrieval_start
        
        # Generate answer
        generation_start = time.time()
        answer = self.generate_answer(query, documents, **kwargs)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": documents,
            "metadata": {
                "retrieval_time": retrieval_time,
                "generation_time": generation_time,
                "total_time": total_time,
                "num_documents": len(documents),
                "connection_type": self.connection_manager.connection_type.upper(),
                "pipeline_type": self.__class__.__name__
            }
        }
    
    def execute_query(self, query: str, params: Optional[List[Any]] = None) -> List[Any]:
        """
        Execute a database query using the connection manager
        
        Args:
            query: SQL query
            params: Optional query parameters
            
        Returns:
            Query results
        """
        return self.connection_manager.execute(query, params)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not self.embedding_func:
            raise ValueError("Embedding function not provided")
        return self.embedding_func([text])[0]
    
    def format_embedding_for_iris(self, embedding: List[float]) -> str:
        """
        Format embedding for IRIS vector operations
        
        Args:
            embedding: Embedding vector
            
        Returns:
            Comma-separated string representation
        """
        # Use fixed decimal format to avoid scientific notation
        return ','.join([f'{x:.10f}' for x in embedding])
    
    def create_prompt(self, query: str, documents: List[Document], template: Optional[str] = None) -> str:
        """
        Create prompt for LLM
        
        Args:
            query: User query
            documents: Retrieved documents
            template: Optional prompt template
            
        Returns:
            Formatted prompt
        """
        if template:
            # Use custom template
            context = self._format_document_context(documents)
            return template.format(query=query, context=context)
        
        # Default template
        context_parts = []
        for i, doc in enumerate(documents[:3]):  # Use top 3 documents
            score = getattr(doc, 'score', 0.0)
            context_parts.append(f"Document {i+1} (Score: {score:.3f}):")
            if hasattr(doc, 'title') and doc.title:
                context_parts.append(f"Title: {doc.title}")
            context_parts.append(f"Content: {doc.content[:500]}...")
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        return f"""Based on the following documents, answer the question.

Context:
{context}

Question: {query}

Answer:"""
    
    def _format_document_context(self, documents: List[Document]) -> str:
        """Format documents into context string"""
        context_parts = []
        for doc in documents:
            if hasattr(doc, 'content'):
                context_parts.append(doc.content)
            elif hasattr(doc, 'text'):
                context_parts.append(doc.text)
        return "\n\n".join(context_parts)