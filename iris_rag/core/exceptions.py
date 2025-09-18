"""
Core exceptions for the IRIS RAG system.

This module defines custom exceptions used throughout the IRIS RAG framework,
particularly for pipeline configuration and loading operations.
"""


class RAGException(Exception):
    """
    Base exception for all IRIS RAG framework exceptions.

    This is the base class for all custom exceptions in the IRIS RAG system.
    It provides a common interface for error handling and allows for
    centralized exception management.
    """


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
    """
    Raised when dynamic module loading fails.

    This exception is raised when:
    - Module cannot be imported
    - Module does not exist
    - Module import raises an exception
    - Class is not found in module
    """
