"""
Embeddings module for RAG templates.

This module provides embedding functionality with support for multiple backends
and graceful fallback mechanisms.
"""

from .manager import EmbeddingManager

__all__ = ["EmbeddingManager"]