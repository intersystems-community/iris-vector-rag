"""
Exception classes for vector store operations.

This module defines the exception hierarchy for vector store implementations,
providing specific error types for different failure scenarios.
"""


class VectorStoreError(Exception):
    """
    Base exception for all vector store related errors.
    
    This is the parent class for all vector store specific exceptions,
    allowing for broad exception handling when needed.
    """
    pass


class VectorStoreConnectionError(VectorStoreError):
    """
    Exception raised when there are connection issues with the vector store.
    
    This includes database connection failures, network timeouts,
    authentication failures, and other connection-related problems.
    """
    pass


class VectorStoreDataError(VectorStoreError):
    """
    Exception raised when there are data-related issues in the vector store.
    
    This includes malformed documents, invalid embeddings, schema violations,
    and other data integrity problems.
    """
    pass


class VectorStoreCLOBError(VectorStoreDataError):
    """
    Exception raised when there are issues processing CLOB data.
    
    This is a specific type of data error that occurs when converting
    CLOB (Character Large Object) data to strings, such as when reading
    from databases that store large text fields as CLOBs.
    """
    pass


class VectorStoreConfigurationError(VectorStoreError):
    """
    Exception raised when there are configuration-related issues in the vector store.
    
    This includes invalid table names, unsupported configuration values,
    and other configuration validation failures.
    """
    pass