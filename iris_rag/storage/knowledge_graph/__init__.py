"""
Knowledge Graph Infrastructure for GraphRAG.

This module provides comprehensive knowledge graph capabilities including:
- Entity extraction and management
- Relationship mapping and storage
- Graph traversal and querying
- Performance optimization with IRIS globals
"""

from .interfaces import (
    IKnowledgeGraphManager,
    IEntityExtractor,
    IRelationshipExtractor,
    IGraphQueryEngine
)

from .models import (
    Entity,
    Relationship,
    EntityMention,
    RelationshipEvidence,
    SubGraph,
    GraphBuildResult,
    TraversalConfig
)

from .manager import KnowledgeGraphManager
from .entity_extractor import EntityExtractionService
from .relationship_extractor import RelationshipExtractionService
from .query_engine import GraphQueryEngine

__all__ = [
    # Interfaces
    'IKnowledgeGraphManager',
    'IEntityExtractor', 
    'IRelationshipExtractor',
    'IGraphQueryEngine',
    
    # Models
    'Entity',
    'Relationship',
    'EntityMention',
    'RelationshipEvidence',
    'SubGraph',
    'GraphBuildResult',
    'TraversalConfig',
    
    # Implementations
    'KnowledgeGraphManager',
    'EntityExtractionService',
    'RelationshipExtractionService',
    'GraphQueryEngine'
]