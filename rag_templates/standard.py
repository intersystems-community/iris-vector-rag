"""
Standard API for RAG Templates Library Consumption Framework.

This module provides an advanced Standard API that enables configurable RAG usage
with technique selection, advanced configuration, and dependency injection while
maintaining backward compatibility with the Simple API.
"""

import logging
from typing import Dict, Any, Optional, Union, List
from .core.config_manager import ConfigurationManager
from .core.pipeline_factory import PipelineFactory
from .core.technique_registry import TechniqueRegistry
from .core.errors import RAGFrameworkError, ConfigurationError, InitializationError
from iris_rag.core.models import Document

logger = logging.getLogger(__name__)


class ConfigurableRAG:
    """
    Advanced Standard API for configurable RAG operations.
    
    This class provides advanced RAG functionality with technique selection,
    complex configuration support, and dependency injection while maintaining
    the simplicity of the Simple API for basic use cases.
    
    Example usage:
        # Basic technique selection
        rag = ConfigurableRAG({"technique": "colbert"})
        
        # Advanced configuration
        rag = ConfigurableRAG({
            "technique": "colbert",
            "llm_provider": "anthropic",
            "llm_config": {
                "model": "claude-3-sonnet",
                "temperature": 0.1
            },
            "technique_config": {
                "max_query_length": 512,
                "top_k": 15
            }
        })
        
        # Query with advanced options
        result = rag.query("What is machine learning?", {
            "include_sources": True,
            "min_similarity": 0.8,
            "max_results": 10
        })
    """
    
    def __init__(self, config: Dict[str, Any], config_path: Optional[str] = None):
        """
        Initialize the Standard RAG API.
        
        Args:
            config: Configuration dictionary with technique and options
            config_path: Optional path to configuration file
        """
        self._config = config.copy()
        self._config_path = config_path
        self._technique = config.get("technique", "basic").lower()
        self._pipeline = None
        self._initialized = False
        
        # Initialize core components
        try:
            # Create enhanced configuration manager
            self._config_manager = ConfigurationManager(config_path)
            
            # Apply configuration overrides from the config dict
            self._apply_config_overrides()
            
            # Initialize technique registry and pipeline factory
            self._technique_registry = TechniqueRegistry()
            self._pipeline_factory = PipelineFactory(
                self._config_manager, 
                self._technique_registry
            )
            
            logger.info(f"Standard RAG API initialized with technique: {self._technique}")
            
        except Exception as e:
            raise InitializationError(
                "Failed to initialize Standard RAG API",
                component="ConfigurableRAG",
                details={"error": str(e), "technique": self._technique}
            ) from e
    
    def _apply_config_overrides(self) -> None:
        """Apply configuration overrides from the config dictionary."""
        # Map config keys to configuration manager paths
        config_mappings = {
            "llm_provider": "llm:provider",
            "llm_model": "llm:model",
            "llm_api_key": "llm:api_key",
            "llm_temperature": "llm:temperature",
            "llm_max_tokens": "llm:max_tokens",
            "embedding_model": "embeddings:model",
            "embedding_dimension": "embeddings:dimension",
            "embedding_provider": "embeddings:provider",
            "max_results": "pipelines:basic:default_top_k",
            "chunk_size": "pipelines:basic:chunk_size",
            "chunk_overlap": "pipelines:basic:chunk_overlap"
        }
        
        # Apply direct mappings
        for config_key, manager_path in config_mappings.items():
            if config_key in self._config:
                self._config_manager.set(manager_path, self._config[config_key])
        
        # Apply nested configurations
        if "llm_config" in self._config:
            llm_config = self._config["llm_config"]
            for key, value in llm_config.items():
                self._config_manager.set(f"llm:{key}", value)
        
        if "embedding_config" in self._config:
            embedding_config = self._config["embedding_config"]
            for key, value in embedding_config.items():
                self._config_manager.set(f"embeddings:{key}", value)
        
        if "technique_config" in self._config:
            technique_config = self._config["technique_config"]
            technique_name = self._technique
            for key, value in technique_config.items():
                self._config_manager.set(f"pipelines:{technique_name}:{key}", value)
        
        if "vector_index" in self._config:
            vector_config = self._config["vector_index"]
            for key, value in vector_config.items():
                self._config_manager.set(f"vector_index:{key}", value)
    
    def query(self, query_text: str, options: Optional[Dict[str, Any]] = None) -> Union[str, Dict[str, Any]]:
        """
        Query the RAG system with advanced options.
        
        Args:
            query_text: The question or query text
            options: Advanced query options including:
                - include_sources: Return sources and metadata (default: False)
                - min_similarity: Minimum similarity threshold
                - max_results: Maximum number of results to return
                - source_filter: Filter results by source
                - return_dict: Return full result dictionary (default: False)
                
        Returns:
            String answer (default) or full result dictionary if include_sources=True
            or return_dict=True
        """
        try:
            # Ensure pipeline is initialized
            pipeline = self._get_pipeline()
            
            # Prepare query options
            query_options = options.copy() if options else {}
            
            # Map standard options to pipeline parameters
            if "max_results" in query_options:
                query_options["top_k"] = query_options.pop("max_results")
            
            # Execute the query
            result = pipeline.execute(query_text, **query_options)
            
            # Determine return format
            include_sources = options.get("include_sources", False) if options else False
            return_dict = options.get("return_dict", False) if options else False
            
            if include_sources or return_dict:
                # Return enhanced result dictionary
                enhanced_result = {
                    "answer": result.get("answer", "No answer generated"),
                    "query": query_text,
                    "sources": self._extract_sources(result),
                    "metadata": self._extract_metadata(result),
                    "technique": self._technique
                }
                
                # Add retrieved documents if available
                if "retrieved_documents" in result:
                    enhanced_result["retrieved_documents"] = result["retrieved_documents"]
                
                return enhanced_result
            else:
                # Return simple string answer for backward compatibility
                return result.get("answer", "No answer generated")
                
        except Exception as e:
            error_msg = f"Failed to process query with technique '{self._technique}': {str(e)}"
            logger.error(error_msg)
            
            # Return error in appropriate format
            if options and (options.get("include_sources") or options.get("return_dict")):
                return {
                    "answer": f"Error: {error_msg}",
                    "query": query_text,
                    "error": str(e),
                    "technique": self._technique
                }
            else:
                return f"Error: {error_msg}"
    
    def add_documents(self, documents: Union[List[str], List[Dict[str, Any]]], 
                     **kwargs) -> None:
        """
        Add documents to the RAG knowledge base.
        
        Args:
            documents: List of document texts or document dictionaries
            **kwargs: Additional options for document processing
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
            
            logger.info(f"Added {len(documents)} documents to knowledge base using {self._technique}")
            
        except Exception as e:
            raise RAGFrameworkError(
                f"Failed to add documents with technique '{self._technique}': {str(e)}",
                details={"document_count": len(documents), "technique": self._technique, "error": str(e)}
            ) from e
    
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
        Initialize the RAG pipeline with the selected technique.
        """
        try:
            # Validate technique is available
            if not self._technique_registry.is_technique_available(self._technique):
                available = self._technique_registry.get_enabled_techniques()
                raise ConfigurationError(
                    f"Technique '{self._technique}' is not available. "
                    f"Available techniques: {', '.join(available)}",
                    details={"technique": self._technique, "available": available}
                )
            
            # Create pipeline using factory
            technique_config = self._config.get("technique_config", {})
            self._pipeline = self._pipeline_factory.create_pipeline(
                self._technique, 
                technique_config
            )
            
            self._initialized = True
            logger.info(f"Pipeline initialized successfully for technique: {self._technique}")
            
        except Exception as e:
            raise InitializationError(
                f"Failed to initialize pipeline for technique '{self._technique}': {str(e)}",
                component="ConfigurableRAG",
                details={"technique": self._technique, "error": str(e)}
            ) from e
    
    def _process_documents(self, documents: Union[List[str], List[Dict[str, Any]]]) -> List[Document]:
        """
        Process input documents into the format expected by the pipeline.
        
        Args:
            documents: List of document texts or document dictionaries
            
        Returns:
            List of processed Document objects
        """
        processed = []
        
        for i, doc in enumerate(documents):
            if isinstance(doc, str):
                # Convert string to document format
                metadata = {
                    "source": f"standard_api_doc_{i}",
                    "document_id": f"doc_{i}",
                    "added_via": "standard_api",
                    "technique": self._technique
                }
                processed_doc = Document(page_content=doc, metadata=metadata)
            elif isinstance(doc, dict):
                # Ensure required fields exist
                if "page_content" not in doc:
                    raise ValueError(f"Document {i} missing 'page_content' field")
                
                metadata = doc.get("metadata", {})
                metadata.update({
                    "document_id": metadata.get("document_id", f"doc_{i}"),
                    "added_via": "standard_api",
                    "technique": self._technique
                })
                
                processed_doc = Document(
                    page_content=doc["page_content"],
                    metadata=metadata
                )
            else:
                raise ValueError(f"Document {i} must be string or dictionary, got {type(doc)}")
            
            processed.append(processed_doc)
        
        return processed
    
    def _extract_sources(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract source information from pipeline result."""
        sources = []
        
        # Extract from retrieved documents
        retrieved_docs = result.get("retrieved_documents", [])
        for doc in retrieved_docs:
            if isinstance(doc, dict):
                source_info = {
                    "content": doc.get("page_content", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": doc.get("score", None)
                }
                sources.append(source_info)
        
        return sources
    
    def _extract_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from pipeline result."""
        metadata = {
            "technique": self._technique,
            "num_retrieved": len(result.get("retrieved_documents", [])),
            "execution_time": result.get("execution_time"),
            "similarity_scores": []
        }
        
        # Extract similarity scores if available
        retrieved_docs = result.get("retrieved_documents", [])
        for doc in retrieved_docs:
            if isinstance(doc, dict) and "score" in doc:
                metadata["similarity_scores"].append(doc["score"])
        
        return metadata
    
    def get_available_techniques(self) -> List[str]:
        """
        Get list of available RAG techniques.
        
        Returns:
            List of available technique names
        """
        return self._technique_registry.get_enabled_techniques()
    
    def get_technique_info(self, technique_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a technique.
        
        Args:
            technique_name: Name of technique (uses current if None)
            
        Returns:
            Technique information dictionary
        """
        name = technique_name or self._technique
        return self._technique_registry.get_technique_info(name) or {}
    
    def switch_technique(self, new_technique: str, 
                        technique_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Switch to a different RAG technique.
        
        Args:
            new_technique: Name of the new technique
            technique_config: Optional configuration for the new technique
        """
        if not self._technique_registry.is_technique_available(new_technique):
            available = self._technique_registry.get_enabled_techniques()
            raise ConfigurationError(
                f"Technique '{new_technique}' is not available. "
                f"Available techniques: {', '.join(available)}",
                details={"technique": new_technique, "available": available}
            )
        
        # Update configuration
        self._technique = new_technique.lower()
        self._config["technique"] = self._technique
        
        if technique_config:
            self._config["technique_config"] = technique_config
            self._apply_config_overrides()
        
        # Reset pipeline to force reinitialization
        self._pipeline = None
        self._initialized = False
        
        logger.info(f"Switched to technique: {self._technique}")
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        # Check local config first
        if key in self._config:
            return self._config[key]
        
        # Check configuration manager
        return self._config_manager.get(key, default)
    
    def __repr__(self) -> str:
        """Return string representation of the ConfigurableRAG instance."""
        status = "initialized" if self._initialized else "not initialized"
        return f"ConfigurableRAG(technique={self._technique}, status={status})"