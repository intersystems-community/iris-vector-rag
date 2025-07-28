"""
Hierarchical data models for NodeRAG extending GraphRAG infrastructure.

This module defines hierarchical node structures that extend the existing GraphRAG
Entity and Relationship models to support document-level hierarchical representation
with parent-child relationships and multi-level retrieval capabilities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Union
from enum import Enum
from abc import ABC, abstractmethod

# Import base GraphRAG models to extend
from .models import (
    Entity, Relationship, EntityMention, RelationshipEvidence,
    ENTITY_TYPE_REGISTRY, RELATIONSHIP_TYPE_REGISTRY
)


class NodeType(Enum):
    """Enumeration of hierarchical node types."""
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    CHUNK = "chunk"  # For compatibility with existing chunking systems


class HierarchicalRelationType(Enum):
    """Enumeration of hierarchical relationship types."""
    PARENT_CHILD = "parent_child"
    SIBLING = "sibling"
    ANCESTOR_DESCENDANT = "ancestor_descendant"
    CONTAINS = "contains"
    PART_OF = "part_of"
    FOLLOWS = "follows"
    PRECEDES = "precedes"


class ContextStrategy(Enum):
    """Strategies for hierarchical context expansion."""
    EXPAND_UP = "expand_up"  # Include parent nodes
    EXPAND_DOWN = "expand_down"  # Include child nodes
    EXPAND_UP_DOWN = "expand_up_down"  # Include both parents and children
    EXPAND_SIBLINGS = "expand_siblings"  # Include sibling nodes
    SMART_EXPANSION = "smart_expansion"  # Intelligent context expansion
    NO_EXPANSION = "no_expansion"  # Use nodes as-is


@dataclass
class DocumentStructure:
    """Represents the hierarchical structure of a document."""
    document_id: str
    root_node_id: str
    total_nodes: int = 0
    max_depth: int = 0
    node_type_counts: Dict[str, int] = field(default_factory=dict)
    structure_metadata: Dict[str, Any] = field(default_factory=dict)
    analyzed_at: datetime = field(default_factory=datetime.now)
    
    def get_node_count_by_type(self, node_type: NodeType) -> int:
        """Get count of nodes by type."""
        return self.node_type_counts.get(node_type.value, 0)
    
    def add_node_count(self, node_type: NodeType, count: int = 1) -> None:
        """Add to node count for a specific type."""
        current = self.node_type_counts.get(node_type.value, 0)
        self.node_type_counts[node_type.value] = current + count


@dataclass
class SectionInfo:
    """Information about a document section."""
    title: str
    content: str
    start_offset: int
    end_offset: int
    heading_level: int = 1
    section_type: str = "section"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HierarchicalNode(Entity):
    """
    Base class for hierarchical document nodes extending GraphRAG Entity.
    
    This class extends the existing Entity model to add hierarchical capabilities
    while maintaining compatibility with the existing GraphRAG infrastructure.
    """
    # Hierarchical-specific fields
    node_type: NodeType
    parent_id: Optional[str] = None
    child_ids: List[str] = field(default_factory=list)
    sibling_order: int = 0
    depth_level: int = 0
    content: str = ""
    node_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Override Entity fields with hierarchical defaults
    entity_type: str = "HIERARCHICAL_NODE"
    
    def __post_init__(self):
        """Initialize hierarchical node after dataclass creation."""
        super().__post_init__()
        
        # Register hierarchical entity type if not already registered
        if not ENTITY_TYPE_REGISTRY.is_valid_type("HIERARCHICAL_NODE"):
            ENTITY_TYPE_REGISTRY.register_type("HIERARCHICAL_NODE", "Hierarchical document node")
        
        # Set entity_name from content if not provided
        if not self.entity_name and self.content:
            self.entity_name = self.content[:100] + "..." if len(self.content) > 100 else self.content
        
        # Set canonical_name if not provided
        if not self.canonical_name:
            self.canonical_name = f"{self.node_type.value}_{self.entity_id}"
    
    def add_child(self, child_id: str) -> None:
        """Add a child node ID."""
        if child_id not in self.child_ids:
            self.child_ids.append(child_id)
            self.updated_at = datetime.now()
    
    def remove_child(self, child_id: str) -> None:
        """Remove a child node ID."""
        if child_id in self.child_ids:
            self.child_ids.remove(child_id)
            self.updated_at = datetime.now()
    
    def is_root_node(self) -> bool:
        """Check if this is a root node (no parent)."""
        return self.parent_id is None
    
    def is_leaf_node(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.child_ids) == 0
    
    def get_node_path(self) -> str:
        """Get a string representation of the node's hierarchical path."""
        return f"{self.node_type.value}[{self.depth_level}]:{self.entity_id}"
    
    def get_content_summary(self, max_length: int = 200) -> str:
        """Get a summary of the node's content."""
        if not self.content:
            return f"[{self.node_type.value} node]"
        
        if len(self.content) <= max_length:
            return self.content
        
        return self.content[:max_length] + "..."


@dataclass
class DocumentNode(HierarchicalNode):
    """Top-level document representation in the hierarchy."""
    document_id: str = ""
    title: str = ""
    source_path: str = ""
    total_sections: int = 0
    document_type: str = "document"
    language: str = "en"
    
    def __post_init__(self):
        """Initialize document node."""
        self.node_type = NodeType.DOCUMENT
        self.depth_level = 0
        self.entity_type = "DOCUMENT_NODE"
        
        # Set document_id from entity_id if not provided
        if not self.document_id:
            self.document_id = self.entity_id
        
        # Set entity_name from title if available
        if self.title and not self.entity_name:
            self.entity_name = self.title
        
        super().__post_init__()
        
        # Register document node type
        if not ENTITY_TYPE_REGISTRY.is_valid_type("DOCUMENT_NODE"):
            ENTITY_TYPE_REGISTRY.register_type("DOCUMENT_NODE", "Document-level hierarchical node")


@dataclass
class SectionNode(HierarchicalNode):
    """Document section representation (chapters, headings, etc.)."""
    section_title: str = ""
    section_type: str = "section"  # heading, chapter, abstract, introduction, etc.
    heading_level: int = 1
    section_index: int = 0
    
    def __post_init__(self):
        """Initialize section node."""
        self.node_type = NodeType.SECTION
        self.depth_level = 1
        self.entity_type = "SECTION_NODE"
        
        # Set entity_name from section_title if available
        if self.section_title and not self.entity_name:
            self.entity_name = self.section_title
        
        super().__post_init__()
        
        # Register section node type
        if not ENTITY_TYPE_REGISTRY.is_valid_type("SECTION_NODE"):
            ENTITY_TYPE_REGISTRY.register_type("SECTION_NODE", "Section-level hierarchical node")
    
    def is_heading_section(self) -> bool:
        """Check if this is a heading section."""
        return self.section_type in ["heading", "title", "subtitle"]
    
    def get_heading_hierarchy(self) -> str:
        """Get heading hierarchy representation."""
        return f"H{self.heading_level}: {self.section_title}"


@dataclass
class ParagraphNode(HierarchicalNode):
    """Paragraph-level content representation."""
    paragraph_index: int = 0
    sentence_count: int = 0
    word_count: int = 0
    
    def __post_init__(self):
        """Initialize paragraph node."""
        self.node_type = NodeType.PARAGRAPH
        self.depth_level = 2
        self.entity_type = "PARAGRAPH_NODE"
        
        # Calculate word count if content is available
        if self.content and self.word_count == 0:
            self.word_count = len(self.content.split())
        
        super().__post_init__()
        
        # Register paragraph node type
        if not ENTITY_TYPE_REGISTRY.is_valid_type("PARAGRAPH_NODE"):
            ENTITY_TYPE_REGISTRY.register_type("PARAGRAPH_NODE", "Paragraph-level hierarchical node")
    
    def is_short_paragraph(self, threshold: int = 50) -> bool:
        """Check if this is a short paragraph."""
        return self.word_count < threshold
    
    def get_paragraph_summary(self) -> str:
        """Get a summary of the paragraph."""
        return f"Paragraph {self.paragraph_index}: {self.word_count} words, {self.sentence_count} sentences"


@dataclass
class SentenceNode(HierarchicalNode):
    """Sentence-level granular representation."""
    sentence_index: int = 0
    word_count: int = 0
    sentence_type: str = "declarative"  # declarative, interrogative, imperative, exclamatory
    
    def __post_init__(self):
        """Initialize sentence node."""
        self.node_type = NodeType.SENTENCE
        self.depth_level = 3
        self.entity_type = "SENTENCE_NODE"
        
        # Calculate word count if content is available
        if self.content and self.word_count == 0:
            self.word_count = len(self.content.split())
        
        # Detect sentence type if not provided
        if self.sentence_type == "declarative" and self.content:
            self.sentence_type = self._detect_sentence_type(self.content)
        
        super().__post_init__()
        
        # Register sentence node type
        if not ENTITY_TYPE_REGISTRY.is_valid_type("SENTENCE_NODE"):
            ENTITY_TYPE_REGISTRY.register_type("SENTENCE_NODE", "Sentence-level hierarchical node")
    
    def _detect_sentence_type(self, content: str) -> str:
        """Detect sentence type from content."""
        content = content.strip()
        if content.endswith('?'):
            return "interrogative"
        elif content.endswith('!'):
            return "exclamatory"
        elif content.startswith(('Please', 'Do', 'Don\'t', 'Let', 'Make')):
            return "imperative"
        else:
            return "declarative"
    
    def is_question(self) -> bool:
        """Check if this sentence is a question."""
        return self.sentence_type == "interrogative"
    
    def is_short_sentence(self, threshold: int = 5) -> bool:
        """Check if this is a short sentence."""
        return self.word_count < threshold


@dataclass
class HierarchicalRelationship(Relationship):
    """
    Hierarchical relationship extending GraphRAG Relationship.
    
    Represents relationships between hierarchical nodes with additional
    context about the hierarchical structure.
    """
    relationship_type: HierarchicalRelationType
    depth_difference: int = 0
    path: List[str] = field(default_factory=list)  # Path from source to target
    hierarchy_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize hierarchical relationship."""
        super().__post_init__()
        
        # Register hierarchical relationship types
        for rel_type in HierarchicalRelationType:
            if not RELATIONSHIP_TYPE_REGISTRY.is_valid_type(rel_type.value):
                RELATIONSHIP_TYPE_REGISTRY.register_type(
                    rel_type.value, 
                    f"Hierarchical {rel_type.value.replace('_', ' ')} relationship"
                )
    
    def is_parent_child(self) -> bool:
        """Check if this is a parent-child relationship."""
        return self.relationship_type == HierarchicalRelationType.PARENT_CHILD
    
    def is_sibling(self) -> bool:
        """Check if this is a sibling relationship."""
        return self.relationship_type == HierarchicalRelationType.SIBLING
    
    def is_hierarchical(self) -> bool:
        """Check if this is a hierarchical relationship."""
        return self.relationship_type in [
            HierarchicalRelationType.PARENT_CHILD,
            HierarchicalRelationType.ANCESTOR_DESCENDANT,
            HierarchicalRelationType.CONTAINS,
            HierarchicalRelationType.PART_OF
        ]
    
    def get_relationship_direction(self) -> str:
        """Get the direction of the relationship."""
        if self.relationship_type in [HierarchicalRelationType.PARENT_CHILD, HierarchicalRelationType.CONTAINS]:
            return "downward"  # From parent to child
        elif self.relationship_type == HierarchicalRelationType.PART_OF:
            return "upward"  # From child to parent
        elif self.relationship_type == HierarchicalRelationType.SIBLING:
            return "lateral"  # Same level
        else:
            return "bidirectional"


@dataclass
class NodeContext:
    """Context information for a hierarchical node."""
    node: HierarchicalNode
    parent_nodes: List[HierarchicalNode] = field(default_factory=list)
    child_nodes: List[HierarchicalNode] = field(default_factory=list)
    sibling_nodes: List[HierarchicalNode] = field(default_factory=list)
    ancestor_nodes: List[HierarchicalNode] = field(default_factory=list)
    descendant_nodes: List[HierarchicalNode] = field(default_factory=list)
    relationships: List[HierarchicalRelationship] = field(default_factory=list)
    context_strategy: ContextStrategy = ContextStrategy.NO_EXPANSION
    relevance_score: float = 0.0
    
    def get_all_context_nodes(self) -> List[HierarchicalNode]:
        """Get all nodes in the context."""
        all_nodes = [self.node]
        all_nodes.extend(self.parent_nodes)
        all_nodes.extend(self.child_nodes)
        all_nodes.extend(self.sibling_nodes)
        all_nodes.extend(self.ancestor_nodes)
        all_nodes.extend(self.descendant_nodes)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_nodes = []
        for node in all_nodes:
            if node.entity_id not in seen:
                seen.add(node.entity_id)
                unique_nodes.append(node)
        
        return unique_nodes
    
    def get_context_summary(self) -> Dict[str, int]:
        """Get a summary of context node counts."""
        return {
            "total_nodes": len(self.get_all_context_nodes()),
            "parent_nodes": len(self.parent_nodes),
            "child_nodes": len(self.child_nodes),
            "sibling_nodes": len(self.sibling_nodes),
            "ancestor_nodes": len(self.ancestor_nodes),
            "descendant_nodes": len(self.descendant_nodes),
            "relationships": len(self.relationships)
        }


@dataclass
class HierarchicalSubGraph:
    """Represents a hierarchical subgraph with structured node relationships."""
    root_nodes: List[HierarchicalNode]
    all_nodes: List[HierarchicalNode]
    relationships: List[HierarchicalRelationship]
    depth: int
    node_contexts: Dict[str, NodeContext] = field(default_factory=dict)
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[HierarchicalNode]:
        """Get all nodes of a specific type."""
        return [node for node in self.all_nodes if node.node_type == node_type]
    
    def get_nodes_by_depth(self, depth: int) -> List[HierarchicalNode]:
        """Get all nodes at a specific depth level."""
        return [node for node in self.all_nodes if node.depth_level == depth]
    
    def get_node_count_by_type(self) -> Dict[str, int]:
        """Get count of nodes by type."""
        counts = {}
        for node in self.all_nodes:
            node_type = node.node_type.value
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts
    
    def get_max_depth(self) -> int:
        """Get the maximum depth in the subgraph."""
        if not self.all_nodes:
            return 0
        return max(node.depth_level for node in self.all_nodes)
    
    def add_node_context(self, node_id: str, context: NodeContext) -> None:
        """Add context information for a node."""
        self.node_contexts[node_id] = context
    
    def get_node_context(self, node_id: str) -> Optional[NodeContext]:
        """Get context information for a node."""
        return self.node_contexts.get(node_id)


# Type aliases for convenience
HierarchicalNodeType = Union[DocumentNode, SectionNode, ParagraphNode, SentenceNode]
NodeList = List[HierarchicalNode]
NodeDict = Dict[str, HierarchicalNode]
RelationshipList = List[HierarchicalRelationship]