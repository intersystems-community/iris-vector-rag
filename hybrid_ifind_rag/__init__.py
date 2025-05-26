"""
Hybrid iFind+Graph+Vector RAG Pipeline

This module implements a sophisticated hybrid RAG pipeline that combines:
- IRIS iFind keyword search for exact term matching
- Graph-based retrieval for relationship discovery  
- Vector similarity search for semantic matching
- SQL reciprocal rank fusion to balance results

The pipeline leverages IRIS's unique capabilities for multi-modal retrieval
and represents the 7th RAG technique in the enterprise RAG templates collection.
"""

__version__ = "1.0.0"
__author__ = "RAG Templates Team"

from .pipeline import HybridiFindRAGPipeline

__all__ = ["HybridiFindRAGPipeline"]