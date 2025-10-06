"""
RAG Pipeline implementations module.

This module contains concrete implementations of the RAGPipeline abstract base class.
Each pipeline represents a different RAG technique or approach.
"""

from .basic import BasicRAGPipeline
from .basic_rerank import BasicRAGRerankingPipeline
from .colbert_pylate.pylate_pipeline import PyLateColBERTPipeline
from .crag import CRAGPipeline
from .graphrag import GraphRAGPipeline

# Optional imports with graceful fallback
try:
    from .hybrid_graphrag import HybridGraphRAGPipeline

    HYBRID_GRAPHRAG_AVAILABLE = True
except ImportError:
    HybridGraphRAGPipeline = None
    HYBRID_GRAPHRAG_AVAILABLE = False

try:
    from .iris_global_graphrag import IRISGlobalGraphRAGPipeline

    IRIS_GLOBAL_GRAPHRAG_AVAILABLE = True
except ImportError:
    IRISGlobalGraphRAGPipeline = None
    IRIS_GLOBAL_GRAPHRAG_AVAILABLE = False

__all__ = [
    "BasicRAGPipeline",
    "CRAGPipeline",
    "BasicRAGRerankingPipeline",
    "GraphRAGPipeline",
    "PyLateColBERTPipeline",
]

if HYBRID_GRAPHRAG_AVAILABLE:
    __all__.append("HybridGraphRAGPipeline")

if IRIS_GLOBAL_GRAPHRAG_AVAILABLE:
    __all__.append("IRISGlobalGraphRAGPipeline")
