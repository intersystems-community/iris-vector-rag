#!/usr/bin/env python3
"""
RAG Memory Components

Generic, reusable memory management patterns for RAG applications.
These components demonstrate how to add memory capabilities to any RAG pipeline.
"""

from .knowledge_extractor import KnowledgePatternExtractor, KnowledgePattern, Entity, Relationship
from .temporal_manager import TemporalMemoryManager, TemporalWindow, MemoryItem
from .incremental_manager import IncrementalLearningManager, LearningResult
from .rag_integration import MemoryEnabledRAGPipeline, EnrichedRAGResponse
from .models import GenericMemoryItem, MemoryConfig

__all__ = [
    # Knowledge extraction
    'KnowledgePatternExtractor',
    'KnowledgePattern', 
    'Entity',
    'Relationship',
    
    # Temporal memory
    'TemporalMemoryManager',
    'TemporalWindow',
    'MemoryItem',
    
    # Incremental learning
    'IncrementalLearningManager',
    'LearningResult',
    
    # RAG integration
    'MemoryEnabledRAGPipeline',
    'EnrichedRAGResponse',
    
    # Models
    'GenericMemoryItem',
    'MemoryConfig'
]