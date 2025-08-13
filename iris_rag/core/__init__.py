# iris_rag.core sub-package
# This file makes the iris_rag/core directory a Python sub-package.

# We can expose key classes from this sub-package here for easier imports.
from .base import RAGPipeline
from .connection import ConnectionManager
from .models import Document
from .vector_store import VectorStore
from .vector_store_exceptions import (
    VectorStoreError,
    VectorStoreConnectionError,
    VectorStoreDataError,
    VectorStoreCLOBError,
)

__all__ = [
    "RAGPipeline",
    "ConnectionManager",
    "Document",
    "VectorStore",
    "VectorStoreError",
    "VectorStoreConnectionError",
    "VectorStoreDataError",
    "VectorStoreCLOBError",
]
