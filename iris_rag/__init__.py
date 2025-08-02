# iris_rag package
# This file makes the iris_rag directory a Python package.

from typing import Dict, Any, Optional, Callable, List
from .core.base import RAGPipeline
from .core.connection import ConnectionManager
from .config.manager import ConfigurationManager
from .pipelines.basic import BasicRAGPipeline
from .pipelines.colbert import ColBERTRAGPipeline
from .pipelines.crag import CRAGPipeline

# Import validation components
from .validation.factory import ValidatedPipelineFactory, create_validated_pipeline
from .validation.validator import PreConditionValidator
from .validation.orchestrator import SetupOrchestrator
from .validation.requirements import get_pipeline_requirements
from common.utils import get_llm_func  # Import get_llm_func

# Package version
__version__ = "0.1.0"
__author__ = "InterSystems IRIS RAG Templates Project"
__description__ = "A comprehensive, production-ready framework for implementing Retrieval Augmented Generation (RAG) pipelines using InterSystems IRIS as the vector database backend."

def create_pipeline(pipeline_type: str, config_path: Optional[str] = None,
                   llm_func: Optional[Callable[[str], str]] = None,
                   embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
                   external_connection=None,
                   validate_requirements: bool = True,
                   auto_setup: bool = False,
                   **kwargs) -> RAGPipeline:
    """
    Factory function to create RAG pipeline instances with validation.

    Args:
        pipeline_type: The type of pipeline to create (e.g., "basic").
        config_path: Optional path to configuration file.
        llm_func: Optional LLM function for answer generation.
        embedding_func: Optional embedding function for vector generation.
        external_connection: Optional existing database connection to use.
        validate_requirements: Whether to validate pipeline requirements before creation.
        auto_setup: Whether to automatically set up missing requirements.
        **kwargs: Additional configuration parameters.

    Returns:
        An instance of a RAGPipeline.

    Raises:
        ValueError: If the pipeline_type is unknown.
        PipelineValidationError: If validation fails and auto_setup is False.
    """
    # Initialize configuration manager
    config_manager = ConfigurationManager(config_path)
    
    # Initialize connection manager
    if external_connection:
        # Create a wrapper connection manager that uses the external connection
        connection_manager = ExternalConnectionWrapper(external_connection, config_manager)
    else:
        connection_manager = ConnectionManager(config_manager)
    
    # Use validated factory if validation is requested
    if validate_requirements:
        factory = ValidatedPipelineFactory(connection_manager, config_manager)
        return factory.create_pipeline(
            pipeline_type=pipeline_type,
            llm_func=llm_func,
            auto_setup=auto_setup,
            validate_requirements=True,
            **kwargs
        )
    
    # Legacy creation without validation
    # Pass embedding_func through kwargs for pipelines that need it
    if embedding_func:
        kwargs['embedding_func'] = embedding_func

    # If no llm_func is provided, get one using the utility function
    effective_llm_func = llm_func
    if effective_llm_func is None:
        # Use 'stub' provider for testing if not specified otherwise in config or kwargs
        llm_provider = config_manager.get("llm.provider", "stub")
        llm_model_name = config_manager.get("llm.model_name", "stub-model")
        effective_llm_func = get_llm_func(provider=llm_provider, model_name=llm_model_name, **kwargs)


    return _create_pipeline_legacy(pipeline_type, connection_manager, config_manager, effective_llm_func, **kwargs)


def _create_pipeline_legacy(pipeline_type: str, connection_manager: ConnectionManager,
                           config_manager: ConfigurationManager,
                           llm_func: Optional[Callable[[str], str]], **kwargs) -> RAGPipeline:
    """Legacy pipeline creation without validation."""
    if pipeline_type == "basic":
        return BasicRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
            llm_func=llm_func
        )
    elif pipeline_type == "basic_rerank":
        from .pipelines.basic_rerank import BasicRAGRerankingPipeline
        return BasicRAGRerankingPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
            llm_func=llm_func
        )
    elif pipeline_type == "colbert":
        return ColBERTRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
            llm_func=llm_func,
            **kwargs
        )
    elif pipeline_type == "crag":
        # Extract embedding_func from kwargs if provided
        embedding_func = kwargs.get('embedding_func')
        return CRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
    elif pipeline_type == "hyde":
        from .pipelines.hyde import HyDERAGPipeline
        return HyDERAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
            llm_func=llm_func
        )
    elif pipeline_type == "graphrag":
        from .pipelines.graphrag import GraphRAGPipeline
        return GraphRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
            llm_func=llm_func
        )
    elif pipeline_type == "hybrid_ifind":
        from .pipelines.hybrid_ifind import HybridIFindRAGPipeline
        return HybridIFindRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
            llm_func=llm_func
        )
    elif pipeline_type == "noderag":
        from .pipelines.noderag import NodeRAGPipeline
        return NodeRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
            llm_func=llm_func
        )
    elif pipeline_type == "sql_rag":
        from .pipelines.sql_rag import SQLRAGPipeline
        return SQLRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
            llm_func=llm_func
        )
    else:
        available_types = ["basic", "basic_rerank", "colbert", "crag", "hyde", "graphrag", "hybrid_ifind", "noderag", "sql_rag"]
        raise ValueError(f"Unknown pipeline type: {pipeline_type}. Available: {available_types}")


def validate_pipeline(pipeline_type: str, config_path: Optional[str] = None,
                     external_connection=None) -> Dict[str, Any]:
    """
    Validate pipeline requirements without creating an instance.
    
    Args:
        pipeline_type: Type of pipeline to validate
        config_path: Optional path to configuration file
        external_connection: Optional existing database connection
        
    Returns:
        Validation results dictionary
    """
    config_manager = ConfigurationManager(config_path)
    
    if external_connection:
        connection_manager = ExternalConnectionWrapper(external_connection, config_manager)
    else:
        connection_manager = ConnectionManager(config_manager)
    
    factory = ValidatedPipelineFactory(connection_manager, config_manager)
    return factory.validate_pipeline_type(pipeline_type)


def setup_pipeline(pipeline_type: str, config_path: Optional[str] = None,
                  external_connection=None) -> Dict[str, Any]:
    """
    Set up all requirements for a pipeline type.
    
    Args:
        pipeline_type: Type of pipeline to set up
        config_path: Optional path to configuration file
        external_connection: Optional existing database connection
        
    Returns:
        Setup results dictionary
    """
    config_manager = ConfigurationManager(config_path)
    
    if external_connection:
        connection_manager = ExternalConnectionWrapper(external_connection, config_manager)
    else:
        connection_manager = ConnectionManager(config_manager)
    
    factory = ValidatedPipelineFactory(connection_manager, config_manager)
    return factory.setup_pipeline_requirements(pipeline_type)


def get_pipeline_status(pipeline_type: str, config_path: Optional[str] = None,
                       external_connection=None) -> Dict[str, Any]:
    """
    Get detailed status information for a pipeline type.
    
    Args:
        pipeline_type: Type of pipeline to check
        config_path: Optional path to configuration file
        external_connection: Optional existing database connection
        
    Returns:
        Detailed status information
    """
    config_manager = ConfigurationManager(config_path)
    
    if external_connection:
        connection_manager = ExternalConnectionWrapper(external_connection, config_manager)
    else:
        connection_manager = ConnectionManager(config_manager)
    
    factory = ValidatedPipelineFactory(connection_manager, config_manager)
    return factory.get_pipeline_status(pipeline_type)


class ExternalConnectionWrapper:
    """Wrapper to make external connections work with iris_rag ConnectionManager interface."""
    
    def __init__(self, external_connection, config_manager):
        self.external_connection = external_connection
        self.config_manager = config_manager
        
    def get_connection(self, backend_name: str = "iris"):
        """Return the external connection for any backend."""
        return self.external_connection

__all__ = [
    "create_pipeline", "validate_pipeline", "setup_pipeline", "get_pipeline_status",
    "create_validated_pipeline", "RAGPipeline", "ConnectionManager", "ConfigurationManager",
    "BasicRAGPipeline", "ColBERTRAGPipeline", "CRAGPipeline",
    "ValidatedPipelineFactory", "PreConditionValidator", "SetupOrchestrator",
    "get_pipeline_requirements"
]