"""
Basic RAG Pipeline implementation.

This module provides a straightforward implementation of the RAG (Retrieval Augmented Generation)
pipeline using vector similarity search and LLM generation.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable
from ..core.base import RAGPipeline
from ..core.models import Document
from ..config.manager import ConfigurationManager
from ..embeddings.manager import EmbeddingManager

logger = logging.getLogger(__name__)


class BasicRAGPipeline(RAGPipeline):
    """
    Basic RAG pipeline implementation.
    
    This pipeline implements the standard RAG approach:
    1. Document ingestion with automatic chunking via IRISVectorStore
    2. Vector similarity search for retrieval
    3. Context augmentation and LLM generation
    
    Note: Chunking is now handled automatically by the IRISVectorStore layer,
    making this pipeline chunking-agnostic and configuration-driven.
    """
    
    def __init__(self, config_manager: ConfigurationManager,
                 llm_func: Optional[Callable[[str], str]] = None, vector_store=None):
        """
        Initialize the Basic RAG Pipeline.
        
        Args:
            config_manager: Manager for configuration settings
            llm_func: Optional LLM function for answer generation
            vector_store: Optional VectorStore instance
        """
        super().__init__(config_manager, vector_store)
        self.llm_func = llm_func
        
        # Initialize components
        self.embedding_manager = EmbeddingManager(config_manager)
        
        # Get pipeline configuration
        self.pipeline_config = self.config_manager.get("pipelines:basic", {})
        self.default_top_k = self.pipeline_config.get("default_top_k", 5)
    
    def load_documents(self, documents_path: str, **kwargs) -> None:
        """
        Load and process documents into the pipeline's knowledge base.
        
        Args:
            documents_path: Path to documents or directory of documents
            **kwargs: Additional arguments including:
                - documents: List of Document objects (if providing directly)
                - auto_chunk: Whether to enable automatic chunking (default: True)
                - chunking_strategy: Strategy to use for chunking (default: from config)
                - generate_embeddings: Whether to generate embeddings (default: True)
        """
        start_time = time.time()
        
        # Check if we need custom embedding generation
        generate_embeddings = kwargs.get("generate_embeddings", True)
        
        if generate_embeddings:
            # Custom logic for BasicRAG with embedding generation
            documents = self._get_documents(documents_path, **kwargs)
            
            # Get chunking configuration
            pipeline_name = self._get_pipeline_name()
            chunking_config = self._get_chunking_config(pipeline_name)
            
            auto_chunk = kwargs.get("auto_chunk", chunking_config.get("enabled", True))
            chunking_strategy = kwargs.get("chunking_strategy", chunking_config.get("strategy", "fixed_size"))
            
            self._generate_and_store_embeddings(documents, auto_chunk, chunking_strategy)
        else:
            # Use default implementation for non-embedding case
            super().load_documents(documents_path, **kwargs)
        
        processing_time = time.time() - start_time
        logger.info(f"Loaded documents in {processing_time:.2f} seconds")
    
    def _load_documents_from_path(self, documents_path: str) -> List[Document]:
        """
        Load documents from a file or directory path.
        
        Args:
            documents_path: Path to load documents from
            
        Returns:
            List of Document objects
        """
        import os
        
        documents = []
        
        if os.path.isfile(documents_path):
            # Single file
            documents.append(self._load_single_file(documents_path))
        elif os.path.isdir(documents_path):
            # Directory of files
            for filename in os.listdir(documents_path):
                file_path = os.path.join(documents_path, filename)
                if os.path.isfile(file_path):
                    try:
                        doc = self._load_single_file(file_path)
                        documents.append(doc)
                    except Exception as e:
                        logger.warning(f"Failed to load file {file_path}: {e}")
        else:
            raise ValueError(f"Path does not exist: {documents_path}")
        
        return documents
    
    def _load_single_file(self, file_path: str) -> Document:
        """
        Load a single file as a Document.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document object
        """
        import os
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path)
        }
        
        return Document(
            page_content=content,
            metadata=metadata
        )
    
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
        logger.info(f"Generated and stored embeddings for {len(documents)} documents with auto_chunk={auto_chunk}")
    
    def _store_documents_with_chunking(self, documents: List[Document], auto_chunk: bool = True, chunking_strategy: Optional[str] = None) -> List[str]:
        """
        Store documents using vector store's automatic chunking without pre-computed embeddings.
        
        Args:
            documents: List of Document objects to store
            auto_chunk: Whether to enable automatic chunking
            chunking_strategy: Strategy to use for chunking (optional)
            
        Returns:
            List of document IDs that were stored
        """
        return self.vector_store.add_documents(
            documents,
            auto_chunk=auto_chunk,
            chunking_strategy=chunking_strategy
        )
    
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query_text: The query text
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments including:
                - metadata_filter: Optional metadata filters
                - similarity_threshold: Minimum similarity score
                
        Returns:
            List of retrieved documents
        """
        # Generate query embedding
        query_embedding = self.embedding_manager.embed_text(query_text)
        
        # Get optional parameters
        metadata_filter = kwargs.get("metadata_filter")
        similarity_threshold = kwargs.get("similarity_threshold", 0.0)
        
        # Perform vector search using base class helper
        results = self._retrieve_documents_by_vector(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter
        )
        
        # Filter by similarity threshold if specified
        if similarity_threshold > 0.0:
            results = [(doc, score) for doc, score in results if score >= similarity_threshold]
        
        # Return just the documents
        documents = [doc for doc, score in results]
        
        logger.debug(f"Retrieved {len(documents)} documents for query: {query_text[:50]}...")
        return documents
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Run the full RAG pipeline for a query (main API method).
        
        Args:
            query: The input query
            **kwargs: Additional arguments including:
                - top_k: Number of documents to retrieve
                - include_sources: Whether to include source information
                - custom_prompt: Custom prompt template
                
        Returns:
            Dictionary with query, answer, and retrieved documents
        """
        return self.execute(query, **kwargs)
    
    def execute(self, query_text: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline for a query.
        
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
        
        # Get parameters
        top_k = kwargs.get("top_k", self.default_top_k)
        include_sources = kwargs.get("include_sources", True)
        custom_prompt = kwargs.get("custom_prompt")
        
        # Step 1: Retrieve relevant documents
        # Remove top_k from kwargs to avoid duplicate parameter error
        query_kwargs = {k: v for k, v in kwargs.items() if k != 'top_k'}
        retrieved_documents = self.query(query_text, top_k=top_k, **query_kwargs)
        
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
            "pipeline_type": "basic_rag"
        }
        
        logger.info(f"RAG pipeline executed in {execution_time:.2f} seconds")
        return response
    
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
            prompt = self._create_default_prompt(query, context)
        
        # Generate answer using LLM
        try:
            answer = self.llm_func(prompt)
            return answer
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return f"Error generating answer: {e}"
    
    def _create_default_prompt(self, query: str, context: str) -> str:
        """
        Create a default prompt for answer generation.
        
        Args:
            query: The user query
            context: Retrieved document context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Based on the following context documents, please answer the question.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the information in the context documents. If the context doesn't contain enough information to fully answer the question, please indicate what information is missing.

Answer:"""
        
        return prompt
    
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