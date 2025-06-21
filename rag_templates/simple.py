"""
Simple API for RAG Templates Library Consumption Framework.

This module provides a zero-configuration Simple API that enables immediate RAG usage
with sensible defaults. The RAG class implements lazy initialization and provides
a clean, simple interface for document addition and querying.
"""

import logging
from typing import List, Union, Optional, Dict, Any
from .core.config_manager import ConfigurationManager
from .core.errors import RAGFrameworkError, InitializationError, ConfigurationError

logger = logging.getLogger(__name__)


class RAG:
    """
    Zero-configuration Simple API for RAG operations.
    
    This class provides immediate RAG functionality with sensible defaults,
    implementing lazy initialization to defer expensive operations until needed.
    
    Example usage:
        # Zero-config initialization
        rag = RAG()
        
        # Add documents
        rag.add_documents(["Document 1 text", "Document 2 text"])
        
        # Query for answers
        answer = rag.query("What is machine learning?")
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize the Simple RAG API.
        
        Args:
            config_path: Optional path to configuration file
            **kwargs: Additional configuration overrides
        """
        self._config_manager: Optional[ConfigurationManager] = None
        self._pipeline = None
        self._initialized = False
        self._config_path = config_path
        self._config_overrides = kwargs
        
        # Initialize configuration manager immediately (lightweight)
        try:
            self._config_manager = ConfigurationManager(config_path)
            
            # Apply any configuration overrides
            for key, value in kwargs.items():
                if ':' in key:  # Support dot notation in kwargs
                    self._config_manager.set(key, value)
            
            logger.info("Simple RAG API initialized with zero-configuration defaults")
            
        except Exception as e:
            raise InitializationError(
                "Failed to initialize Simple RAG API configuration",
                component="ConfigurationManager",
                details={"error": str(e)}
            ) from e
    
    def add_documents(self, documents: Union[List[str], List[Dict[str, Any]]], 
                     **kwargs) -> None:
        """
        Add documents to the RAG knowledge base.
        
        Args:
            documents: List of document texts or document dictionaries
            **kwargs: Additional options for document processing
                - chunk_documents: Whether to chunk documents (default: True)
                - generate_embeddings: Whether to generate embeddings (default: True)
        """
        try:
            # Ensure pipeline is initialized
            pipeline = self._get_pipeline()
            
            # Convert string documents to proper format
            processed_docs = self._process_documents(documents)
            
            # Use the pipeline's load_documents method
            pipeline.load_documents(
                documents_path="",  # Not used when documents are provided directly
                documents=processed_docs,
                **kwargs
            )
            
            logger.info(f"Added {len(documents)} documents to knowledge base")
            
        except Exception as e:
            raise RAGFrameworkError(
                f"Failed to add documents: {str(e)}",
                details={"document_count": len(documents), "error": str(e)}
            ) from e
    
    def query(self, query_text: str, **kwargs) -> str:
        """
        Query the RAG system and return a simple string answer.
        
        Args:
            query_text: The question or query text
            **kwargs: Additional query options
                - top_k: Number of documents to retrieve (default: 5)
                - include_sources: Whether to include source information
                
        Returns:
            String answer to the query
        """
        try:
            # Ensure pipeline is initialized
            pipeline = self._get_pipeline()
            
            # Execute the query
            result = pipeline.execute(query_text, **kwargs)
            
            # Extract the answer string
            answer = result.get("answer", "No answer generated")
            
            logger.debug(f"Query processed: {query_text[:50]}...")
            return answer
            
        except Exception as e:
            error_msg = f"Failed to process query: {str(e)}"
            logger.error(error_msg)
            
            # Return a helpful error message instead of raising
            return f"Error: {error_msg}. Please check your configuration and try again."
    
    def _get_pipeline(self):
        """
        Get or initialize the RAG pipeline using lazy initialization.
        
        Returns:
            Initialized RAG pipeline instance
        """
        if not self._initialized:
            self._initialize_pipeline()
        
        return self._pipeline
    
    def _initialize_pipeline(self) -> None:
        """
        Initialize the RAG pipeline with lazy loading.
        
        This method defers expensive operations like database connections
        and model loading until actually needed.
        """
        try:
            # Import here to avoid circular imports and defer heavy imports
            from iris_rag.core.connection import ConnectionManager
            from iris_rag.pipelines.basic import BasicRAGPipeline
            from common.utils import get_embedding_func, get_llm_func
            
            # Initialize connection manager
            connection_manager = ConnectionManager(self._config_manager)
            
            # Get embedding and LLM functions
            embedding_func = get_embedding_func()
            llm_func = get_llm_func()
            
            # Initialize the basic RAG pipeline
            self._pipeline = BasicRAGPipeline(
                connection_manager=connection_manager,
                config_manager=self._config_manager,
                llm_func=llm_func
            )
            
            self._initialized = True
            logger.info("RAG pipeline initialized successfully")
            
        except ImportError as e:
            raise InitializationError(
                "Failed to import required RAG components. Ensure iris_rag is properly installed.",
                component="RAGPipeline",
                details={"import_error": str(e)}
            ) from e
        except Exception as e:
            raise InitializationError(
                f"Failed to initialize RAG pipeline: {str(e)}",
                component="RAGPipeline",
                details={"error": str(e)}
            ) from e
    
    def _process_documents(self, documents: Union[List[str], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Process input documents into the format expected by the pipeline.
        
        Args:
            documents: List of document texts or document dictionaries
            
        Returns:
            List of processed document dictionaries
        """
        processed = []
        
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                # Convert string to document format
                processed_doc = {
                    "page_content": doc,
                    "metadata": {
                        "source": f"simple_api_doc_{i}",
                        "document_id": f"doc_{i}",
                        "added_via": "simple_api"
                    }
                }
            elif isinstance(doc, dict):
                # Ensure required fields exist
                if "page_content" not in doc:
                    raise ValueError(f"Document {i} missing 'page_content' field")
                
                processed_doc = doc.copy()
                if "metadata" not in processed_doc:
                    processed_doc["metadata"] = {}
                
                # Add default metadata
                processed_doc["metadata"].update({
                    "document_id": processed_doc["metadata"].get("document_id", f"doc_{i}"),
                    "added_via": "simple_api"
                })
            else:
                raise ValueError(f"Document {i} must be string or dictionary, got {type(doc)}")
            
            processed.append(processed_doc)
        
        return processed
    
    def get_document_count(self) -> int:
        """
        Get the number of documents in the knowledge base.
        
        Returns:
            Number of documents stored
        """
        try:
            if not self._initialized:
                return 0
            
            return self._pipeline.get_document_count()
        except Exception as e:
            logger.warning(f"Failed to get document count: {e}")
            return 0
    
    def clear_knowledge_base(self) -> None:
        """
        Clear all documents from the knowledge base.
        
        Warning: This operation is irreversible.
        """
        try:
            if self._initialized and self._pipeline:
                self._pipeline.clear_knowledge_base()
                logger.info("Knowledge base cleared")
        except Exception as e:
            raise RAGFrameworkError(
                f"Failed to clear knowledge base: {str(e)}",
                details={"error": str(e)}
            ) from e
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key in dot notation (e.g., "database:iris:host")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        if not self._config_manager:
            return default
        
        return self._config_manager.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        if not self._config_manager:
            raise ConfigurationError("Configuration manager not initialized")
        
        self._config_manager.set(key, value)
        
        # If pipeline is already initialized, warn that restart may be needed
        if self._initialized:
            logger.warning(f"Configuration changed after initialization: {key}. "
                          "Some changes may require restarting the application.")
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If validation fails
        """
        if not self._config_manager:
            raise ConfigurationError("Configuration manager not initialized")
        
        try:
            self._config_manager.validate()
            return True
        except Exception as e:
            raise ConfigurationError(
                f"Configuration validation failed: {str(e)}",
                details={"error": str(e)}
            ) from e
    
    def __repr__(self) -> str:
        """Return string representation of the RAG instance."""
        status = "initialized" if self._initialized else "not initialized"
        doc_count = self.get_document_count() if self._initialized else 0
        return f"RAG(status={status}, documents={doc_count})"