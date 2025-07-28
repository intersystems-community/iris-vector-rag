# iris_rag package
# This file makes the iris_rag directory a Python package.

from typing import Dict, Any, Optional, Callable, List
from .core.base import RAGPipeline
from .config.manager import ConfigurationManager
from .pipelines.basic import BasicRAGPipeline
from .pipelines.crag import CRAGPipeline
from .pipelines.colbert.pipeline import ColBERTRAGPipeline

# Import validation components
from .validation.factory import ValidatedPipelineFactory, create_validated_pipeline
from .validation.validator import PreConditionValidator
from .validation.orchestrator import SetupOrchestrator
from .validation.requirements import get_pipeline_requirements
from common.utils import get_llm_func  # Import get_llm_func
from common.iris_connection_manager import IRISConnectionManager as ConnectionManager

# Package version
__version__ = "0.1.0"
__author__ = "InterSystems IRIS RAG Templates Project"
__description__ = "A comprehensive, production-ready framework for implementing Retrieval Augmented Generation (RAG) pipelines using InterSystems IRIS as the vector database backend."

def create_pipeline(pipeline_type: str, config_path: Optional[str] = None,
                   llm_func: Optional[Callable[[str], str]] = None,
                   embedding_func: Optional[Callable[[List[str]], List[List[float]]]] = None,
                   external_connection=None,
                   connection_manager=None,
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
        external_connection: Optional existing database connection to use (deprecated).
        connection_manager: Optional connection manager instance for dependency injection.
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
    
    # Use validated factory if validation is requested
    if validate_requirements:
        factory = ValidatedPipelineFactory(config_manager, connection_manager=connection_manager)
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


    return _create_pipeline_legacy(pipeline_type, config_manager, effective_llm_func, connection_manager=connection_manager, **kwargs)


def _create_pipeline_legacy(pipeline_type: str,
                           config_manager: ConfigurationManager,
                           llm_func: Optional[Callable[[str], str]],
                           connection_manager=None, **kwargs) -> RAGPipeline:
    """Legacy pipeline creation without validation."""
    if pipeline_type == "basic":
        return BasicRAGPipeline(
            config_manager=config_manager,
            llm_func=llm_func,
            connection_manager=connection_manager
        )
    elif pipeline_type == "crag":
        # Extract embedding_func from kwargs if provided
        embedding_func = kwargs.get('embedding_func')
        return CRAGPipeline(
            config_manager=config_manager,
            embedding_func=embedding_func,
            llm_func=llm_func,
            connection_manager=connection_manager
        )
    elif pipeline_type == "hyde":
        from .pipelines.hyde import HyDERAGPipeline
        return HyDERAGPipeline(
            config_manager=config_manager,
            llm_func=llm_func,
            connection_manager=connection_manager
        )
    elif pipeline_type == "graphrag":
        from .pipelines.graphrag import GraphRAGPipeline
        return GraphRAGPipeline(
            config_manager=config_manager,
            llm_func=llm_func,
            connection_manager=connection_manager
        )
    elif pipeline_type == "hybrid_ifind":
        from .pipelines.hybrid_ifind import HybridIFindRAGPipeline
        return HybridIFindRAGPipeline(
            config_manager=config_manager,
            llm_func=llm_func,
            connection_manager=connection_manager
        )
    elif pipeline_type == "noderag":
        from .pipelines.noderag import NodeRAGPipeline
        return NodeRAGPipeline(
            config_manager=config_manager,
            llm_func=llm_func,
            connection_manager=connection_manager
        )
    elif pipeline_type == "colbert":
        return ColBERTRAGPipeline(
            config_manager=config_manager,
            llm_func=llm_func,
            connection_manager=connection_manager
        )
    else:
        available_types = ["basic", "crag", "hyde", "graphrag", "hybrid_ifind", "noderag", "colbert"]
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
    
    factory = ValidatedPipelineFactory(config_manager, connection_manager)
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
    
    factory = ValidatedPipelineFactory(config_manager, connection_manager)
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
    
    factory = ValidatedPipelineFactory(config_manager, connection_manager)
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
    "create_validated_pipeline", "RAGPipeline", "ConfigurationManager", "ConnectionManager",
    "BasicRAGPipeline", "CRAGPipeline", "ColBERTRAGPipeline",
    "ValidatedPipelineFactory", "PreConditionValidator", "SetupOrchestrator",
    "get_pipeline_requirements"
]