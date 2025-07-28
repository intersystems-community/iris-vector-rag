"""
Hierarchical NodeRAG package initialization extending GraphRAG infrastructure.

This module provides the complete hierarchical NodeRAG infrastructure including
models, interfaces, schema management, and service components that extend the
existing GraphRAG foundation.
"""

# Import base GraphRAG components
from .interfaces import (
    IKnowledgeGraphManager,
    IEntityExtractor,
    IRelationshipExtractor,
    IGraphQueryEngine,
    IKnowledgeGraphStorage
)

from .models import (
    Entity,
    Relationship,
    EntityMention,
    RelationshipEvidence,
    SubGraph,
    GraphBuildResult,
    TraversalConfig,
    EntityContext,
    GraphStatistics,
    DomainConfig
)

# Import hierarchical extensions
from .hierarchical_models import (
    # Core hierarchical models
    HierarchicalNode,
    DocumentNode,
    SectionNode,
    ParagraphNode,
    SentenceNode,
    HierarchicalRelationship,
    
    # Supporting models
    NodeContext,
    HierarchicalSubGraph,
    DocumentStructure,
    SectionInfo,
    
    # Enums and types
    NodeType,
    ContextStrategy,
    HierarchicalRelationType,
    
    # Type aliases
    HierarchicalNodeType,
    NodeList,
    NodeDict,
    RelationshipList
)

from .hierarchical_interfaces import (
    # Core hierarchical interfaces
    INodeHierarchyManager,
    IDocumentStructureAnalyzer,
    IHierarchicalRetriever,
    IHierarchicalStorage,
    
    # Extended interfaces
    IHierarchicalEntityExtractor,
    IHierarchicalRelationshipExtractor,
    
    # Type aliases
    HierarchicalManagerType,
    StructureAnalyzerType,
    HierarchicalRetrieverType,
    HierarchicalStorageType
)

from .hierarchical_schema import (
    HierarchicalSchemaManager,
    create_hierarchical_schema
)

# Service implementations (to be created)
try:
    from .hierarchical_manager import NodeHierarchyManager
    from .structure_analyzer import DocumentStructureAnalyzer
    from .hierarchical_retriever import HierarchicalRetriever
    from .hierarchical_storage import HierarchicalStorage
except ImportError:
    # Services not yet implemented
    NodeHierarchyManager = None
    DocumentStructureAnalyzer = None
    HierarchicalRetriever = None
    HierarchicalStorage = None

__all__ = [
    # Base GraphRAG interfaces
    'IKnowledgeGraphManager',
    'IEntityExtractor',
    'IRelationshipExtractor',
    'IGraphQueryEngine',
    'IKnowledgeGraphStorage',
    
    # Base GraphRAG models
    'Entity',
    'Relationship',
    'EntityMention',
    'RelationshipEvidence',
    'SubGraph',
    'GraphBuildResult',
    'TraversalConfig',
    'EntityContext',
    'GraphStatistics',
    'DomainConfig',
    
    # Hierarchical interfaces
    'INodeHierarchyManager',
    'IDocumentStructureAnalyzer',
    'IHierarchicalRetriever',
    'IHierarchicalStorage',
    'IHierarchicalEntityExtractor',
    'IHierarchicalRelationshipExtractor',
    
    # Hierarchical models
    'HierarchicalNode',
    'DocumentNode',
    'SectionNode',
    'ParagraphNode',
    'SentenceNode',
    'HierarchicalRelationship',
    'NodeContext',
    'HierarchicalSubGraph',
    'DocumentStructure',
    'SectionInfo',
    
    # Enums and types
    'NodeType',
    'ContextStrategy',
    'HierarchicalRelationType',
    'HierarchicalNodeType',
    'NodeList',
    'NodeDict',
    'RelationshipList',
    
    # Type aliases
    'HierarchicalManagerType',
    'StructureAnalyzerType',
    'HierarchicalRetrieverType',
    'HierarchicalStorageType',
    
    # Schema management
    'HierarchicalSchemaManager',
    'create_hierarchical_schema',
    
    # Service implementations (when available)
    'NodeHierarchyManager',
    'DocumentStructureAnalyzer',
    'HierarchicalRetriever',
    'HierarchicalStorage'
]

# Version information
__version__ = "1.0.0"
__author__ = "NodeRAG Architecture Team"
__description__ = "Hierarchical NodeRAG infrastructure extending GraphRAG"

# Compatibility information
GRAPHRAG_COMPATIBLE = True
IRIS_OPTIMIZED = True
SUPPORTS_HIERARCHICAL_RETRIEVAL = True
SUPPORTS_MULTI_LEVEL_CONTEXT = True

def get_hierarchical_info():
    """
    Get information about the hierarchical NodeRAG infrastructure.
    
    Returns:
        Dictionary with infrastructure information
    """
    return {
        "version": __version__,
        "description": __description__,
        "graphrag_compatible": GRAPHRAG_COMPATIBLE,
        "iris_optimized": IRIS_OPTIMIZED,
        "features": {
            "hierarchical_retrieval": SUPPORTS_HIERARCHICAL_RETRIEVAL,
            "multi_level_context": SUPPORTS_MULTI_LEVEL_CONTEXT,
            "document_structure_analysis": True,
            "context_expansion": True,
            "performance_optimization": True
        },
        "node_types": [nt.value for nt in NodeType],
        "context_strategies": [cs.value for cs in ContextStrategy],
        "relationship_types": [rt.value for rt in HierarchicalRelationType]
    }

def create_hierarchical_pipeline_config():
    """
    Create default configuration for hierarchical NodeRAG pipeline.
    
    Returns:
        Dictionary with default hierarchical configuration
    """
    return {
        "hierarchy": {
            "max_depth": 3,
            "auto_detect_structure": True,
            "preserve_document_structure": True
        },
        "retrieval": {
            "context_strategy": ContextStrategy.SMART_EXPANSION.value,
            "max_context_nodes": 50,
            "similarity_threshold": 0.7,
            "multi_level_weights": {
                NodeType.DOCUMENT.value: 0.2,
                NodeType.SECTION.value: 0.3,
                NodeType.PARAGRAPH.value: 0.4,
                NodeType.SENTENCE.value: 0.1
            }
        },
        "performance": {
            "use_context_cache": True,
            "cache_expiry_hours": 24,
            "enable_hierarchy_optimization": True,
            "batch_size": 100
        },
        "analysis": {
            "detect_headings": True,
            "split_paragraphs": True,
            "split_sentences": True,
            "extract_metadata": True,
            "language": "en"
        }
    }