"""
Abstract interfaces for knowledge graph components.

This module defines the core interfaces that all knowledge graph implementations
must adhere to, ensuring consistent behavior and enabling dependency injection.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from ..core.models import Document
from .models import (
    Entity, Relationship, SubGraph, GraphBuildResult, 
    TraversalConfig, EntityContext, GraphStatistics, DomainConfig
)


class IEntityExtractor(ABC):
    """Interface for entity extraction services."""
    
    @abstractmethod
    def extract_entities(self, document: Document) -> List[Entity]:
        """
        Extract entities from a document.
        
        Args:
            document: The document to extract entities from
            
        Returns:
            List of extracted entities
        """
        pass
    
    @abstractmethod
    def set_domain_config(self, domain_config: DomainConfig) -> None:
        """
        Configure the extractor for a specific domain.
        
        Args:
            domain_config: Domain-specific configuration
        """
        pass
    
    @abstractmethod
    def get_supported_entity_types(self) -> List[str]:
        """
        Get the entity types this extractor can identify.
        
        Returns:
            List of supported entity type codes
        """
        pass


class IRelationshipExtractor(ABC):
    """Interface for relationship extraction services."""
    
    @abstractmethod
    def extract_relationships(self, document: Document, 
                            entities: List[Entity]) -> List[Relationship]:
        """
        Extract relationships between entities in a document.
        
        Args:
            document: The document containing the entities
            entities: List of entities found in the document
            
        Returns:
            List of extracted relationships
        """
        pass
    
    @abstractmethod
    def set_domain_config(self, domain_config: DomainConfig) -> None:
        """
        Configure the extractor for a specific domain.
        
        Args:
            domain_config: Domain-specific configuration
        """
        pass
    
    @abstractmethod
    def get_supported_relationship_types(self) -> List[str]:
        """
        Get the relationship types this extractor can identify.
        
        Returns:
            List of supported relationship type codes
        """
        pass


class IEntityLinker(ABC):
    """Interface for entity linking and disambiguation services."""
    
    @abstractmethod
    def link_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Link and disambiguate entities.
        
        Args:
            entities: List of entities to link
            
        Returns:
            List of linked entities with updated canonical names
        """
        pass
    
    @abstractmethod
    def find_similar_entities(self, entity: Entity, 
                            threshold: float = 0.8) -> List[Entity]:
        """
        Find entities similar to the given entity.
        
        Args:
            entity: The entity to find similarities for
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            List of similar entities
        """
        pass


class IGraphQueryEngine(ABC):
    """Interface for graph querying and traversal operations."""
    
    @abstractmethod
    def find_subgraph(self, seed_entities: List[str], 
                     config: TraversalConfig) -> SubGraph:
        """
        Extract a subgraph around seed entities.
        
        Args:
            seed_entities: List of entity IDs to start traversal from
            config: Configuration for traversal behavior
            
        Returns:
            Extracted subgraph
        """
        pass
    
    @abstractmethod
    def traverse_graph(self, start_entity: str, 
                      config: TraversalConfig) -> List[str]:
        """
        Traverse the graph from a starting entity.
        
        Args:
            start_entity: Entity ID to start traversal from
            config: Configuration for traversal behavior
            
        Returns:
            List of entity IDs found during traversal
        """
        pass
    
    @abstractmethod
    def get_entity_context(self, entity_id: str, 
                          context_depth: int = 2) -> EntityContext:
        """
        Get rich context information for an entity.
        
        Args:
            entity_id: The entity to get context for
            context_depth: Depth of context to retrieve
            
        Returns:
            Rich context information
        """
        pass
    
    @abstractmethod
    def find_shortest_path(self, source_entity: str, 
                          target_entity: str) -> Optional[List[str]]:
        """
        Find the shortest path between two entities.
        
        Args:
            source_entity: Source entity ID
            target_entity: Target entity ID
            
        Returns:
            List of entity IDs forming the shortest path, or None if no path exists
        """
        pass


class IKnowledgeGraphStorage(ABC):
    """Interface for knowledge graph storage operations."""
    
    @abstractmethod
    def store_entity(self, entity: Entity) -> str:
        """
        Store an entity in the graph.
        
        Args:
            entity: The entity to store
            
        Returns:
            The stored entity ID
        """
        pass
    
    @abstractmethod
    def store_relationship(self, relationship: Relationship) -> str:
        """
        Store a relationship in the graph.
        
        Args:
            relationship: The relationship to store
            
        Returns:
            The stored relationship ID
        """
        pass
    
    @abstractmethod
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """
        Retrieve an entity by ID.
        
        Args:
            entity_id: The entity ID to retrieve
            
        Returns:
            The entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_relationships(self, entity_id: str, 
                         relationship_types: Optional[List[str]] = None) -> List[Relationship]:
        """
        Get relationships for an entity.
        
        Args:
            entity_id: The entity ID
            relationship_types: Optional filter for relationship types
            
        Returns:
            List of relationships involving the entity
        """
        pass
    
    @abstractmethod
    def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity and its relationships.
        
        Args:
            entity_id: The entity ID to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_graph_statistics(self) -> GraphStatistics:
        """
        Get statistics about the knowledge graph.
        
        Returns:
            Graph statistics
        """
        pass


class IKnowledgeGraphManager(ABC):
    """Main interface for knowledge graph management operations."""
    
    @abstractmethod
    def build_graph(self, documents: List[Document], 
                   domain_config: Optional[DomainConfig] = None) -> GraphBuildResult:
        """
        Build a knowledge graph from documents.
        
        Args:
            documents: List of documents to process
            domain_config: Optional domain-specific configuration
            
        Returns:
            Result of the graph building process
        """
        pass
    
    @abstractmethod
    def add_entity(self, entity: Entity) -> str:
        """
        Add an entity to the knowledge graph.
        
        Args:
            entity: The entity to add
            
        Returns:
            The entity ID
        """
        pass
    
    @abstractmethod
    def link_entities(self, entity_id1: str, entity_id2: str, 
                     relationship: Relationship) -> str:
        """
        Create a relationship between two entities.
        
        Args:
            entity_id1: First entity ID
            entity_id2: Second entity ID
            relationship: The relationship to create
            
        Returns:
            The relationship ID
        """
        pass
    
    @abstractmethod
    def query_graph(self, query_entities: List[str], 
                   config: TraversalConfig) -> SubGraph:
        """
        Query the knowledge graph for relevant information.
        
        Args:
            query_entities: List of entity IDs to query around
            config: Configuration for the query
            
        Returns:
            Relevant subgraph
        """
        pass
    
    @abstractmethod
    def update_entity(self, entity: Entity) -> bool:
        """
        Update an existing entity.
        
        Args:
            entity: The updated entity
            
        Returns:
            True if updated successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_entity_by_name(self, entity_name: str, 
                          entity_type: Optional[str] = None) -> Optional[Entity]:
        """
        Find an entity by name and optionally type.
        
        Args:
            entity_name: The entity name to search for
            entity_type: Optional entity type filter
            
        Returns:
            The entity if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_entities_by_type(self, entity_type: str, 
                           limit: int = 100) -> List[Entity]:
        """
        Get entities of a specific type.
        
        Args:
            entity_type: The entity type to filter by
            limit: Maximum number of entities to return
            
        Returns:
            List of entities of the specified type
        """
        pass
    
    @abstractmethod
    def rebuild_graph(self, documents: List[Document], 
                     domain_config: Optional[DomainConfig] = None) -> GraphBuildResult:
        """
        Rebuild the entire knowledge graph from scratch.
        
        Args:
            documents: List of documents to process
            domain_config: Optional domain-specific configuration
            
        Returns:
            Result of the graph rebuilding process
        """
        pass
    
    @abstractmethod
    def export_graph(self, format: str = "json") -> Dict[str, Any]:
        """
        Export the knowledge graph in the specified format.
        
        Args:
            format: Export format ("json", "rdf", "cypher")
            
        Returns:
            Exported graph data
        """
        pass
    
    @abstractmethod
    def import_graph(self, graph_data: Dict[str, Any], 
                    format: str = "json") -> bool:
        """
        Import knowledge graph data.
        
        Args:
            graph_data: The graph data to import
            format: Import format ("json", "rdf", "cypher")
            
        Returns:
            True if imported successfully, False otherwise
        """
        pass


class IGraphOptimizer(ABC):
    """Interface for graph performance optimization."""
    
    @abstractmethod
    def optimize_traversal(self, config: TraversalConfig) -> TraversalConfig:
        """
        Optimize traversal configuration for performance.
        
        Args:
            config: Original traversal configuration
            
        Returns:
            Optimized configuration
        """
        pass
    
    @abstractmethod
    def should_use_globals_optimization(self, query_params: Dict[str, Any]) -> bool:
        """
        Determine if IRIS globals optimization should be used.
        
        Args:
            query_params: Parameters of the query to optimize
            
        Returns:
            True if globals optimization should be used
        """
        pass
    
    @abstractmethod
    def create_performance_indexes(self) -> bool:
        """
        Create performance indexes for graph operations.
        
        Returns:
            True if indexes created successfully
        """
        pass


class IGraphCache(ABC):
    """Interface for graph query caching."""
    
    @abstractmethod
    def get_cached_subgraph(self, cache_key: str) -> Optional[SubGraph]:
        """
        Get a cached subgraph.
        
        Args:
            cache_key: The cache key
            
        Returns:
            Cached subgraph if found, None otherwise
        """
        pass
    
    @abstractmethod
    def cache_subgraph(self, cache_key: str, subgraph: SubGraph) -> None:
        """
        Cache a subgraph.
        
        Args:
            cache_key: The cache key
            subgraph: The subgraph to cache
        """
        pass
    
    @abstractmethod
    def invalidate_cache(self, pattern: Optional[str] = None) -> None:
        """
        Invalidate cached entries.
        
        Args:
            pattern: Optional pattern to match cache keys for selective invalidation
        """
        pass
    
    @abstractmethod
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Cache statistics
        """
        pass