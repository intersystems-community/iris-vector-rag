"""
HyDE (Hypothetical Document Embeddings) RAG Pipeline implementation.

This module provides a RAG implementation using the HyDE technique for improved retrieval.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable
from ..core.base import RAGPipeline
from ..core.models import Document
from common.iris_connection_manager import get_iris_connection
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
    
    def __init__(self, config_manager: ConfigurationManager,
                 llm_func: Optional[Callable[[str], str]] = None, vector_store=None):
        """
        Initialize the HyDE RAG Pipeline.
        
        Args:
            config_manager: Manager for configuration settings
            llm_func: Optional LLM function for answer generation
            vector_store: Optional VectorStore instance
        """
        super().__init__(config_manager, vector_store)
        self.connection = get_iris_connection()
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
        Load and process documents into the pipeline's knowledge base with embeddings.
        
        Args:
            documents_path: Path to documents or directory of documents
            **kwargs: Additional arguments including:
                - documents: List of Document objects (if providing directly)
                - auto_chunk: Whether to enable automatic chunking (default: True)
                - chunking_strategy: Strategy to use for chunking (default: from config)
                - generate_embeddings: Whether to generate embeddings (default: True)
        """
        start_time = time.time()
        
        # Get documents from path or kwargs
        documents = self._get_documents(documents_path, **kwargs)
        
        # Get chunking configuration
        pipeline_name = self._get_pipeline_name()
        chunking_config = self._get_chunking_config(pipeline_name)
        
        auto_chunk = kwargs.get("auto_chunk", chunking_config.get("enabled", True))
        chunking_strategy = kwargs.get("chunking_strategy", chunking_config.get("strategy", "fixed_size"))
        
        # Generate embeddings for documents (like BasicRAG does)
        self._generate_and_store_embeddings(documents, auto_chunk, chunking_strategy)
        
        processing_time = time.time() - start_time
        logger.info(f"HyDE loaded documents in {processing_time:.2f} seconds")
    
    def _generate_and_store_embeddings(self, documents: List[Document], auto_chunk: bool = True, chunking_strategy: Optional[str] = None) -> None:
        """
        Generate embeddings for documents and store them using automatic chunking.
        
        Args:
            documents: List of documents to process
            auto_chunk: Whether to enable automatic chunking
            chunking_strategy: Strategy to use for chunking (optional)
        """
        # Extract text content
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings in batches
        batch_size = self.pipeline_config.get("embedding_batch_size", 32)
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.embedding_manager.embed_texts(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        # Store documents with embeddings using vector store's automatic chunking
        self.vector_store.add_documents(
            documents,
            embeddings=all_embeddings,
            auto_chunk=auto_chunk,
            chunking_strategy=chunking_strategy
        )
        logger.info(f"HyDE generated and stored embeddings for {len(documents)} documents with auto_chunk={auto_chunk}")

    def ingest_documents(self, documents: List[Document], **kwargs) -> Dict[str, Any]:
        """
        Ingest documents using standard storage with chunking support.
        
        Args:
            documents: List of documents to ingest
            **kwargs: Additional parameters including:
                - auto_chunk: Whether to enable automatic chunking
                - chunking_strategy: Strategy to use ('fixed_size', 'semantic', 'hybrid')
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        logger.info(f"Starting HyDE ingestion of {len(documents)} documents")
        
        try:
            # Extract chunking parameters
            auto_chunk = kwargs.get('auto_chunk', True)
            chunking_strategy = kwargs.get('chunking_strategy', 'fixed_size')
            
            # Use the helper method for consistency
            self._generate_and_store_embeddings(documents, auto_chunk, chunking_strategy)
            document_ids = []  # Vector store doesn't return IDs in this context
            
            end_time = time.time()
            result = {
                "status": "success",
                "processing_time": end_time - start_time,
                "pipeline_type": "hyde_rag",
                "document_ids": document_ids,
                "documents_processed": len(documents)
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
    
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> List[Document]:
        """
        Retrieve relevant documents for a query using HyDE technique.
        
        Args:
            query_text: The query text
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            List of retrieved documents
        """
        # Generate hypothetical document if LLM is available
        search_text = query_text
        
        if self.use_hypothetical_doc and self.llm_func:
            hypothetical_doc = self._generate_hypothetical_document(query_text)
            search_text = hypothetical_doc
            logger.debug(f"Generated hypothetical document: {hypothetical_doc[:100]}...")
        
        # Generate embedding for search text (query or hypothetical doc)
        search_embedding = self.embedding_manager.embed_text(search_text)
        
        # Retrieve relevant documents using the base class helper
        relevant_docs_tuples = self._retrieve_documents_by_vector(
            query_embedding=search_embedding,
            top_k=top_k
        )
        relevant_docs = [doc for doc, score in relevant_docs_tuples]
        
        logger.debug(f"HyDE retrieved {len(relevant_docs)} documents for query: {query_text[:50]}...")
        return relevant_docs

    def execute(self, query_text: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the full HyDE pipeline for a query.
        
        Args:
            query_text: The input query
            **kwargs: Additional arguments including:
                - top_k: Number of documents to retrieve
                - include_sources: Whether to include source information
                - custom_prompt: Custom prompt template
                
        Returns:
            Dictionary with query, answer, retrieved documents, contexts, and execution_time
        """
        start_time = time.time()
        logger.info(f"Processing HyDE query: {query_text}")
        
        try:
            # Get parameters
            top_k = kwargs.get("top_k", self.top_k)
            include_sources = kwargs.get("include_sources", True)
            custom_prompt = kwargs.get("custom_prompt")
            
            # Step 1: Retrieve relevant documents using HyDE technique
            # Remove top_k from kwargs to avoid duplicate parameter error
            query_kwargs = {k: v for k, v in kwargs.items() if k != 'top_k'}
            retrieved_documents = self.query(query_text, top_k=top_k, **query_kwargs)
            
            # Store hypothetical document for response
            hypothetical_doc = None
            if self.use_hypothetical_doc and self.llm_func:
                hypothetical_doc = self._generate_hypothetical_document(query_text)
            
            # Step 2: Generate answer using LLM
            if self.llm_func:
                answer = self._generate_answer(query_text, retrieved_documents, custom_prompt)
            else:
                answer = "No LLM function provided. Retrieved documents only."
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Step 3: Prepare response
            response = {
                "query": query_text,
                "hypothetical_document": hypothetical_doc,
                "answer": answer,
                "retrieved_documents": retrieved_documents,
                "contexts": [doc.page_content for doc in retrieved_documents],  # String contexts for RAGAS
                "execution_time": execution_time  # Required for RAGAS debug harness
            }
            
            if include_sources:
                response["sources"] = self._extract_sources(retrieved_documents)
            
            # Add metadata
            response["metadata"] = {
                "num_retrieved": len(retrieved_documents),
                "processing_time": execution_time,
                "pipeline_type": "hyde_rag"
            }
            
            logger.info(f"HyDE pipeline executed in {execution_time:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"HyDE query failed: {e}")
            return {
                "query": query_text,
                "answer": f"HyDE pipeline error: {e}",
                "retrieved_documents": [],
                "contexts": [],  # Required for RAGAS
                "execution_time": time.time() - start_time,
                "error": str(e),
                "pipeline_type": "hyde_rag",
                "metadata": {
                    "num_retrieved": 0,
                    "processing_time": time.time() - start_time,
                    "pipeline_type": "hyde_rag"
                }
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
    
    
    def _build_context(self, documents: List[Document]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get('title', 'Untitled')
            content = doc.page_content
            context_parts.append(f"[Document {i}: {title}]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM generation."""
        return f"""Based on the following retrieved documents, please answer the question.

Context:
{context}

Question: {query}

Answer:"""
    def _generate_answer(self, query: str, documents: List[Document], custom_prompt: Optional[str] = None) -> str:
        """
        Generate an answer using the LLM and retrieved documents.
        
        Args:
            query: The original query
            documents: Retrieved documents for context
            custom_prompt: Optional custom prompt template
            
        Returns:
            Generated answer
        """
        if not documents:
            return "No relevant documents found to answer the query."
        
        # Prepare context from documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            context_parts.append(f"Document {i} (Source: {source}):\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # Use custom prompt or default
        if custom_prompt:
            prompt = custom_prompt.format(query=query, context=context)
        else:
            prompt = self._build_prompt(query, context)
        
        # Generate answer using LLM
        try:
            answer = self.llm_func(prompt)
            return answer
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating answer: {e}"

    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Extract source information from documents.
        
        Args:
            documents: List of documents
            
        Returns:
            List of source information dictionaries
        """
        sources = []
        for doc in documents:
            source_info = {
                "document_id": doc.id,
                "source": doc.metadata.get("source", "Unknown"),
                "filename": doc.metadata.get("filename", "Unknown")
            }
            
            # Add chunk information if available
            if "chunk_index" in doc.metadata:
                source_info["chunk_index"] = doc.metadata["chunk_index"]
            
            sources.append(source_info)
        
        return sources

    def get_document_count(self) -> int:
        """
        Get the total number of documents in the knowledge base.
        
        Returns:
            Document count
        """
        return self.vector_store.get_document_count()
    
    def clear_knowledge_base(self) -> None:
        """
        Clear all documents from the knowledge base.
        
        Warning: This operation is irreversible.
        """
        self.vector_store.clear_documents()
        logger.info("Knowledge base cleared")