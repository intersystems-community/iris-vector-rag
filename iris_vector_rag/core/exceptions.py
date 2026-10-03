"""
Core exceptions for the IRIS RAG system.

This module defines the pipeline configuration and loading exceptions.

`RAGException` and the vector store errors are NOT defined here: they live in
`iris_vector_rag/exceptions.py`, which is the module the storage layer raises
them from. They were once declared in both places, which made
`iris_vector_rag.core.VectorStoreConnectionError` a different class from the one
`storage/vector_store_iris.py` raises, so `except` on either one silently missed
half the failures. They are re-exported below so both import paths keep working
and name the same class.
"""

from ..exceptions import (
    RAGException,
    VectorStoreCLOBError,
    VectorStoreConnectionError,
    VectorStoreDataError,
    VectorStoreError,
)

__all__ = [
    "RAGException",
    "PipelineConfigurationError",
    "PipelineNotFoundError",
    "PipelineCreationError",
    "ModuleLoadingError",
    "VectorStoreError",
    "VectorStoreConnectionError",
    "VectorStoreCLOBError",
    "VectorStoreDataError",
]


class PipelineConfigurationError(Exception):
    """
    Raised when pipeline configuration is invalid or malformed.

    This exception is raised when:
    - Configuration file cannot be parsed
    - Required fields are missing from pipeline definitions
    - Configuration schema validation fails
    - Invalid parameter values are provided
    """


class PipelineNotFoundError(Exception):
    """
    Raised when a requested pipeline cannot be found.

    This exception is raised when:
    - Pipeline name is not defined in configuration
    - Pipeline is disabled in configuration
    - Pipeline definition is malformed
    """


class PipelineCreationError(Exception):
    """
    Raised when pipeline instantiation fails.

    This exception is raised when:
    - Pipeline class cannot be imported
    - Pipeline constructor fails
    - Required dependencies are missing
    - Invalid parameters are passed to constructor
    """


class ModuleLoadingError(Exception):
    """Raised when dynamic module loading fails."""
