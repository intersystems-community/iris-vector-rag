"""
Storage layer module for RAG templates.

This module provides storage abstractions and implementations for different
database backends, with a focus on InterSystems IRIS.
"""

from .enterprise_storage import IRISStorage
from .vector_store_iris import IRISVectorStore
from .clob_handler import (
    convert_clob_to_string,
    process_document_row,
    ensure_string_content,
)

__all__ = [
    "IRISStorage",
    "IRISVectorStore",
    "convert_clob_to_string",
    "process_document_row",
    "ensure_string_content",
]
