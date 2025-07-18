"""
Standardized mock implementations for RAG template components.
Centralizes mock objects to reduce boilerplate and improve maintainability.
"""

from tests.mocks.db import MockIRISConnector, MockIRISCursor
from tests.mocks.models import (
    mock_embedding_func,
    mock_llm_func,
    mock_colbert_doc_encoder, 
    mock_colbert_query_encoder,
)

__all__ = [
    'MockIRISConnector',
    'MockIRISCursor',
    'mock_embedding_func',
    'mock_llm_func',
    'mock_colbert_doc_encoder',
    'mock_colbert_query_encoder',
]
