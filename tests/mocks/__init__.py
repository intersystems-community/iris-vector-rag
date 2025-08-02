"""
Mock classes for testing RAG templates.

This module provides standardized mock implementations for database connections,
embedding functions, and other external dependencies used in tests.
"""

from .db import MockIRISConnector, MockIRISCursor
from .models import (
    mock_embedding_func,
    mock_llm_func,
    mock_colbert_doc_encoder,
    mock_colbert_query_encoder
)

__all__ = [
    'MockIRISConnector',
    'MockIRISCursor',
    'mock_embedding_func',
    'mock_llm_func',
    'mock_colbert_doc_encoder',
    'mock_colbert_query_encoder'
]