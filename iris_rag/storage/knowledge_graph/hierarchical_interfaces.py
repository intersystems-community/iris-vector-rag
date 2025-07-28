"""
Hierarchical interfaces for NodeRAG extending GraphRAG infrastructure.

This module defines interfaces for hierarchical node management, document structure
analysis, and hierarchical retrieval that extend the existing GraphRAG interfaces
while maintaining compatibility and clean separation of concerns.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Union, Tuple
from ..core.models import Document
from .interfaces import (
    IKnowledgeGraphManager, IGraphQueryEngine, IKnowledgeGraphStorage,
    IEntityExtractor, IRelationshipExtractor
)
from .models import TraversalConfig, SubGraph
from .hierarchical_models import (
    HierarchicalNode, DocumentNode, SectionNode, ParagraphNode, SentenceNode,
    HierarchicalRelationship, NodeContext, HierarchicalSubGraph,
    DocumentStructure, SectionInfo, NodeType, ContextStrategy,
    HierarchicalRelationType
)


class INodeHierarchyManager(IKnowledgeGraphManager):
    """
    Interface for hierarchical node management extending GraphRAG manager.
    
    This interface extends the existing IKnowledgeGraphManager to add
    hierarchical document processing and node relationship management.
    """
    
    @abstractmethod
    def create_document_hierarchy(self, document: Document, 
                                max_depth: int = 3) -> DocumentNode:
        """
        Create hierarchical node structure from a document.
        
        Args:
            document: Source document to process
            max_depth: Maximum hierarchy depth to create
            
        Returns:
            Root document node with complete hierarchy
        """
        pass
    
    @abstractmethod
    def get_node_children(self, node_id: str, 
                         node_type: Optional[NodeType] = None) -> List[HierarchicalNode]:
        """
        Get direct children of a hierarchical node.
        
        Args:
            node_id: ID of the parent node
            node_type: Optional filter by child node type
            
        Returns:
            List of child nodes
        """
        pass
    
    @abstractmethod
    def get_node_parent(self, node_id: str) -> Optional[HierarchicalNode]:
        """
        Get parent of a hierarchical node.
        
        Args:
            node_id: ID of the child node
            
        Returns:
            Parent node if exists, None otherwise
        """
        pass
    
    @abstractmethod
    def get_node_ancestors(self, node_id: str, 
                          max_depth: Optional[int] = None) -> List[HierarchicalNode]:
        """
        Get all ancestors of a hierarchical node.
        
        Args:
            node_id: ID of the descendant node
            max_depth: Maximum depth to traverse upward
            
        Returns:
            List of ancestor nodes ordered from immediate parent to root
        """
        pass
    
    @abstractmethod
    def get_node_descendants(self, node_id: str, 
                           max_depth: Optional[int] = None,
                           node_type: Optional[NodeType] = None) -> List[HierarchicalNode]:
        """
        Get all descendants of a hierarchical node.
        
        Args:
            node_id: ID of the ancestor node
            max_depth: Maximum depth to traverse downward
            node_type: Optional filter by descendant node type
            
        Returns:
            List of descendant nodes
        """
        pass
    
    @abstractmethod
    def get_node_siblings(self, node_id: str, 
                         include_self: bool = False) -> List[HierarchicalNode]:
        """
        Get sibling nodes at the same hierarchy level.
        
        Args:
            node_id: ID of the reference node
            include_self: Whether to include the reference node in results
            
        Returns:
            List of sibling nodes
        """
        pass
    
    @abstractmethod
    def get_node_by_id(self, node_id: str) -> Optional[HierarchicalNode]:
        """
        Get a hierarchical node by its ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Hierarchical node if found, None otherwise
        """
        pass
    
    @abstractmethod
    def update_node_hierarchy(self, node_id: str, 
                            new_parent_id: Optional[str] = None,
                            new_sibling_order: Optional[int] = None) -> bool:
        """
        Update hierarchical relationships for a node.
        
        Args:
            node_id: ID of the node to update
            new_parent_id: New parent node ID (None to make root)
            new_sibling_order: New position among siblings
            
        Returns:
            True if update successful, False otherwise
        """
        pass
    
    @abstractmethod
    def delete_node_hierarchy(self, node_id: str, 
                            cascade: bool = True) -> bool:
        """
        Delete a node and optionally its hierarchy.
        
        Args:
            node_id: ID of the node to delete
            cascade: Whether to delete child nodes recursively
            
        Returns:
            True if deletion successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_document_structure(self, document_id: str) -> Optional[DocumentStructure]:
        """
        Get the hierarchical structure of a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Document structure if found, None otherwise
        """
        pass


class IDocumentStructureAnalyzer(ABC):
    """
    Interface for analyzing and creating document hierarchical structure.
    
    This interface provides methods for automatic document structure detection
    and hierarchical node creation from various document formats.
    """
    
    @abstractmethod
    def analyze_document_structure(self, document: Document) -> DocumentStructure:
        """
        Analyze document and identify hierarchical structure.
        
        Args:
            document: Document to analyze
            
        Returns:
            Detected document structure
        """
        pass
    
    @abstractmethod
    def create_hierarchical_nodes(self, document: Document, 
                                structure: DocumentStructure) -> List[HierarchicalNode]:
        """
        Create hierarchical nodes from document structure.
        
        Args:
            document: Source document
            structure: Analyzed document structure
            
        Returns:
            List of created hierarchical nodes
        """
        pass
    
    @abstractmethod
    def detect_sections(self, content: str, 
                       document_type: str = "text") -> List[SectionInfo]:
        """
        Detect sections in document content.
        
        Args:
            content: Document content to analyze
            document_type: Type of document (text, html, markdown, etc.)
            
        Returns:
            List of detected sections
        """
        pass
    
    @abstractmethod
    def split_into_paragraphs(self, content: str, 
                            preserve_formatting: bool = True) -> List[str]:
        """
        Split content into paragraphs.
        
        Args:
            content: Content to split
            preserve_formatting: Whether to preserve original formatting
            
        Returns:
            List of paragraph strings
        """
        pass
    
    @abstractmethod
    def split_into_sentences(self, content: str, 
                           language: str = "en") -> List[str]:
        """
        Split content into sentences.
        
        Args:
            content: Content to split
            language: Language code for sentence detection
            
        Returns:
            List of sentence strings
        """
        pass
    
    @abstractmethod
    def detect_heading_hierarchy(self, content: str) -> Dict[int, List[str]]:
        """
        Detect heading hierarchy in content.
        
        Args:
            content: Content to analyze
            
        Returns:
            Dictionary mapping heading levels to heading texts
        """
        pass
    
    @abstractmethod
    def extract_metadata(self, document: Document) -> Dict[str, Any]:
        """
        Extract metadata relevant to hierarchical structure.
        
        Args:
            document: Document to extract metadata from
            
        Returns:
            Dictionary of extracted metadata
        """
        pass


class IHierarchicalRetriever(IGraphQueryEngine):
    """
    Interface for hierarchical retrieval extending GraphRAG query engine.
    
    This interface extends the existing IGraphQueryEngine to add hierarchical
    context expansion and multi-level retrieval capabilities.
    """
    
    @abstractmethod
    def retrieve_with_context(self, query: str, 
                            node_ids: List[str],
                            context_strategy: ContextStrategy = ContextStrategy.SMART_EXPANSION,
                            max_context_nodes: int = 50) -> List[HierarchicalNode]:
        """
        Retrieve nodes with hierarchical context expansion.
        
        Args:
            query: Search query for relevance scoring
            node_ids: Initial node IDs to expand context around
            context_strategy: Strategy for context expansion
            max_context_nodes: Maximum number of context nodes to include
            
        Returns:
            List of nodes with expanded context
        """
        pass
    
    @abstractmethod
    def expand_context_up(self, node_ids: List[str], 
                         levels: int = 1,
                         include_siblings: bool = False) -> List[HierarchicalNode]:
        """
        Expand context by traversing up the hierarchy.
        
        Args:
            node_ids: Starting node IDs
            levels: Number of levels to traverse upward
            include_siblings: Whether to include sibling nodes
            
        Returns:
            List of parent/ancestor nodes
        """
        pass
    
    @abstractmethod
    def expand_context_down(self, node_ids: List[str], 
                          levels: int = 1,
                          node_type_filter: Optional[NodeType] = None) -> List[HierarchicalNode]:
        """
        Expand context by traversing down the hierarchy.
        
        Args:
            node_ids: Starting node IDs
            levels: Number of levels to traverse downward
            node_type_filter: Optional filter for descendant node types
            
        Returns:
            List of child/descendant nodes
        """
        pass
    
    @abstractmethod
    def retrieve_at_level(self, query: str, 
                         node_type: NodeType,
                         top_k: int = 10,
                         similarity_threshold: float = 0.7) -> List[HierarchicalNode]:
        """
        Retrieve nodes at a specific hierarchy level.
        
        Args:
            query: Search query
            node_type: Target node type/level
            top_k: Maximum number of nodes to retrieve
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of relevant nodes at the specified level
        """
        pass
    
    @abstractmethod
    def retrieve_multi_level(self, query: str,
                           level_weights: Dict[NodeType, float],
                           top_k: int = 10) -> List[Tuple[HierarchicalNode, float]]:
        """
        Retrieve nodes across multiple hierarchy levels with weighted scoring.
        
        Args:
            query: Search query
            level_weights: Weights for different node types/levels
            top_k: Maximum number of nodes to retrieve
            
        Returns:
            List of (node, score) tuples sorted by relevance
        """
        pass
    
    @abstractmethod
    def find_hierarchical_subgraph(self, seed_node_ids: List[str],
                                  max_depth: int = 2,
                                  max_nodes: int = 50,
                                  context_strategy: ContextStrategy = ContextStrategy.SMART_EXPANSION) -> HierarchicalSubGraph:
        """
        Find a hierarchical subgraph around seed nodes.
        
        Args:
            seed_node_ids: Starting node IDs
            max_depth: Maximum traversal depth
            max_nodes: Maximum number of nodes in subgraph
            context_strategy: Strategy for context expansion
            
        Returns:
            Hierarchical subgraph with structured relationships
        """
        pass
    
    @abstractmethod
    def get_node_context(self, node_id: str,
                        context_strategy: ContextStrategy = ContextStrategy.EXPAND_UP_DOWN,
                        max_context_size: int = 20) -> NodeContext:
        """
        Get rich context information for a hierarchical node.
        
        Args:
            node_id: ID of the target node
            context_strategy: Strategy for gathering context
            max_context_size: Maximum number of context nodes
            
        Returns:
            Rich context information for the node
        """
        pass


class IHierarchicalStorage(IKnowledgeGraphStorage):
    """
    Interface for hierarchical node storage extending GraphRAG storage.
    
    This interface extends the existing IKnowledgeGraphStorage to add
    hierarchical-specific storage and retrieval operations.
    """
    
    @abstractmethod
    def store_hierarchical_node(self, node: HierarchicalNode) -> str:
        """
        Store a hierarchical node with its relationships.
        
        Args:
            node: Hierarchical node to store
            
        Returns:
            Stored node ID
        """
        pass
    
    @abstractmethod
    def store_hierarchical_relationship(self, relationship: HierarchicalRelationship) -> str:
        """
        Store a hierarchical relationship.
        
        Args:
            relationship: Hierarchical relationship to store
            
        Returns:
            Stored relationship ID
        """
        pass
    
    @abstractmethod
    def get_hierarchical_node(self, node_id: str) -> Optional[HierarchicalNode]:
        """
        Retrieve a hierarchical node by ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Hierarchical node if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_nodes_by_parent(self, parent_id: str) -> List[HierarchicalNode]:
        """
        Get all child nodes of a parent.
        
        Args:
            parent_id: ID of the parent node
            
        Returns:
            List of child nodes
        """
        pass
    
    @abstractmethod
    def get_nodes_by_type(self, node_type: NodeType,
                         limit: int = 100) -> List[HierarchicalNode]:
        """
        Get nodes of a specific type.
        
        Args:
            node_type: Type of nodes to retrieve
            limit: Maximum number of nodes to return
            
        Returns:
            List of nodes of the specified type
        """
        pass
    
    @abstractmethod
    def get_hierarchical_relationships(self, node_id: str,
                                     relationship_types: Optional[List[HierarchicalRelationType]] = None) -> List[HierarchicalRelationship]:
        """
        Get hierarchical relationships for a node.
        
        Args:
            node_id: ID of the node
            relationship_types: Optional filter for relationship types
            
        Returns:
            List of hierarchical relationships
        """
        pass
    
    @abstractmethod
    def update_node_hierarchy_path(self, node_id: str) -> bool:
        """
        Update the hierarchy path optimization table for a node.
        
        Args:
            node_id: ID of the node to update paths for
            
        Returns:
            True if update successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_ancestor_path(self, node_id: str) -> List[str]:
        """
        Get the path from root to the specified node.
        
        Args:
            node_id: ID of the target node
            
        Returns:
            List of node IDs from root to target
        """
        pass
    
    @abstractmethod
    def bulk_store_hierarchy(self, nodes: List[HierarchicalNode],
                           relationships: List[HierarchicalRelationship]) -> Dict[str, str]:
        """
        Efficiently store multiple nodes and relationships.
        
        Args:
            nodes: List of hierarchical nodes to store
            relationships: List of hierarchical relationships to store
            
        Returns:
            Dictionary mapping original IDs to stored IDs
        """
        pass


class IHierarchicalEntityExtractor(IEntityExtractor):
    """
    Interface for hierarchical entity extraction.
    
    This interface extends entity extraction to work with hierarchical nodes
    and extract entities at different granularity levels.
    """
    
    @abstractmethod
    def extract_entities_from_hierarchy(self, nodes: List[HierarchicalNode]) -> Dict[str, List[Any]]:
        """
        Extract entities from hierarchical nodes.
        
        Args:
            nodes: List of hierarchical nodes to process
            
        Returns:
            Dictionary mapping node IDs to extracted entities
        """
        pass
    
    @abstractmethod
    def extract_level_specific_entities(self, node: HierarchicalNode) -> List[Any]:
        """
        Extract entities appropriate for the node's hierarchy level.
        
        Args:
            node: Hierarchical node to extract entities from
            
        Returns:
            List of extracted entities
        """
        pass


class IHierarchicalRelationshipExtractor(IRelationshipExtractor):
    """
    Interface for hierarchical relationship extraction.
    
    This interface extends relationship extraction to identify both
    content-based and structural hierarchical relationships.
    """
    
    @abstractmethod
    def extract_hierarchical_relationships(self, nodes: List[HierarchicalNode]) -> List[HierarchicalRelationship]:
        """
        Extract hierarchical relationships between nodes.
        
        Args:
            nodes: List of hierarchical nodes to analyze
            
        Returns:
            List of extracted hierarchical relationships
        """
        pass
    
    @abstractmethod
    def extract_structural_relationships(self, parent_node: HierarchicalNode,
                                       child_nodes: List[HierarchicalNode]) -> List[HierarchicalRelationship]:
        """
        Extract structural parent-child relationships.
        
        Args:
            parent_node: Parent node in the hierarchy
            child_nodes: Child nodes in the hierarchy
            
        Returns:
            List of structural relationships
        """
        pass
    
    @abstractmethod
    def extract_content_relationships(self, nodes: List[HierarchicalNode]) -> List[HierarchicalRelationship]:
        """
        Extract content-based relationships between nodes.
        
        Args:
            nodes: List of hierarchical nodes to analyze
            
        Returns:
            List of content-based relationships
        """
        pass


# Type aliases for convenience
HierarchicalManagerType = INodeHierarchyManager
StructureAnalyzerType = IDocumentStructureAnalyzer
HierarchicalRetrieverType = IHierarchicalRetriever
HierarchicalStorageType = IHierarchicalStorage