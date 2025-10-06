#!/usr/bin/env python3
"""
RAG Memory Components

Generic, reusable memory management patterns for RAG applications.
These components demonstrate how to add memory capabilities to any RAG pipeline.
"""

from .knowledge_extractor import (
    Entity,
    KnowledgePattern,
    KnowledgePatternExtractor,
    Relationship,
)
from .models import GenericMemoryItem, MemoryConfig
from .rag_integration import EnrichedRAGResponse, MemoryEnabledRAGPipeline
from .temporal_manager import MemoryItem, TemporalMemoryManager, TemporalWindow

# Incremental manager disabled (requires external kg_memory package)
_HAS_INCREMENTAL_MANAGER = False

__all__ = [
    # Knowledge extraction
    "KnowledgePatternExtractor",
    "KnowledgePattern",
    "Entity",
    "Relationship",
    # Temporal memory
    "TemporalMemoryManager",
    "TemporalWindow",
    "MemoryItem",
    # RAG integration
    "MemoryEnabledRAGPipeline",
    "EnrichedRAGResponse",
    # Models
    "GenericMemoryItem",
    "MemoryConfig",
]
