"""
Data models for knowledge graph components.

This module defines the core data structures used throughout the knowledge graph
infrastructure with extensible type systems for domain-specific customization.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Union
from abc import ABC, abstractmethod


class EntityTypeRegistry:
    """Registry for managing domain-specific entity types."""
    
    def __init__(self):
        self._types = {}
        self._load_default_types()
    
    def _load_default_types(self):
        """Load default entity types."""
        # General types
        self.register_type("PERSON", "Person or individual")
        self.register_type("ORGANIZATION", "Organization or institution")
        self.register_type("LOCATION", "Geographic location")
        self.register_type("CONCEPT", "Abstract concept or idea")
        self.register_type("PRODUCT", "Product or service")
        self.register_type("EVENT", "Event or occurrence")
        
        # Biomedical types
        self.register_type("GENE", "Gene or genetic element")
        self.register_type("PROTEIN", "Protein or enzyme")
        self.register_type("DISEASE", "Disease or medical condition")
        self.register_type("DRUG", "Drug or pharmaceutical compound")
        self.register_type("CHEMICAL", "Chemical compound or substance")
        self.register_type("CELL_TYPE", "Cell type or cellular component")
        self.register_type("TISSUE", "Tissue or organ")
        self.register_type("PATHWAY", "Biological pathway or process")
        self.register_type("PHENOTYPE", "Observable characteristic or trait")
        self.register_type("MUTATION", "Genetic mutation or variant")
        
        # Technical types
        self.register_type("ALGORITHM", "Algorithm or computational method")
        self.register_type("SOFTWARE", "Software or tool")
        self.register_type("DATASET", "Dataset or data collection")
        self.register_type("METRIC", "Measurement or metric")
        
        # Default fallback
        self.register_type("UNKNOWN", "Unknown or unclassified entity")
    
    def register_type(self, type_code: str, description: str) -> None:
        """Register a new entity type."""
        self._types[type_code] = description
    
    def get_types(self) -> Dict[str, str]:
        """Get all registered entity types."""
        return self._types.copy()
    
    def is_valid_type(self, type_code: str) -> bool:
        """Check if an entity type is valid."""
        return type_code in self._types
    
    def get_description(self, type_code: str) -> Optional[str]:
        """Get description for an entity type."""
        return self._types.get(type_code)


class RelationshipTypeRegistry:
    """Registry for managing domain-specific relationship types."""
    
    def __init__(self):
        self._types = {}
        self._load_default_types()
    
    def _load_default_types(self):
        """Load default relationship types."""
        # General relationships
        self.register_type("CO_OCCURS", "Entities co-occur in text")
        self.register_type("RELATED_TO", "General relationship")
        self.register_type("PART_OF", "Part-whole relationship")
        self.register_type("CAUSES", "Causal relationship")
        self.register_type("LOCATED_IN", "Location relationship")
        self.register_type("WORKS_FOR", "Employment relationship")
        
        # Biomedical relationships
        self.register_type("INTERACTS_WITH", "Molecular interaction")
        self.register_type("REGULATES", "Regulatory relationship")
        self.register_type("INHIBITS", "Inhibitory relationship")
        self.register_type("ACTIVATES", "Activation relationship")
        self.register_type("BINDS_TO", "Binding relationship")
        self.register_type("METABOLIZES", "Metabolic relationship")
        self.register_type("TREATS", "Treatment relationship")
        self.register_type("ASSOCIATED_WITH", "Disease association")
        self.register_type("EXPRESSED_IN", "Expression relationship")
        self.register_type("MUTATED_IN", "Mutation relationship")
        
        # Hierarchical relationships
        self.register_type("IS_A", "Type-of relationship")
        self.register_type("SUBCLASS_OF", "Subclass relationship")
        self.register_type("INSTANCE_OF", "Instance relationship")
        
        # Default fallback
        self.register_type("UNKNOWN", "Unknown relationship type")
    
    def register_type(self, type_code: str, description: str) -> None:
        """Register a new relationship type."""
        self._types[type_code] = description
    
    def get_types(self) -> Dict[str, str]:
        """Get all registered relationship types."""
        return self._types.copy()
    
    def is_valid_type(self, type_code: str) -> bool:
        """Check if a relationship type is valid."""
        return type_code in self._types
    
    def get_description(self, type_code: str) -> Optional[str]:
        """Get description for a relationship type."""
        return self._types.get(type_code)


# Global registries (can be customized per application)
ENTITY_TYPE_REGISTRY = EntityTypeRegistry()
RELATIONSHIP_TYPE_REGISTRY = RelationshipTypeRegistry()


@dataclass
class EntityMention:
    """Represents a mention of an entity in a document."""
    mention_id: str
    entity_id: str
    document_id: str
    text: str
    start_offset: int
    end_offset: int
    context: str
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Entity:
    """Core entity model for knowledge graph."""
    entity_id: str
    entity_name: str
    entity_type: str  # Now uses string for flexibility
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    confidence_score: float = 1.0
    source_documents: List[str] = field(default_factory=list)
    mentions: List[EntityMention] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate entity type after initialization."""
        if not ENTITY_TYPE_REGISTRY.is_valid_type(self.entity_type):
            # Auto-register unknown types as UNKNOWN
            if self.entity_type not in ENTITY_TYPE_REGISTRY.get_types():
                self.entity_type = "UNKNOWN"
    
    def add_mention(self, mention: EntityMention) -> None:
        """Add a mention to this entity."""
        self.mentions.append(mention)
        if mention.document_id not in self.source_documents:
            self.source_documents.append(mention.document_id)
        self.updated_at = datetime.now()
    
    def get_primary_name(self) -> str:
        """Get the primary name for this entity."""
        return self.canonical_name or self.entity_name
    
    def is_biomedical_entity(self) -> bool:
        """Check if this is a biomedical entity."""
        biomedical_types = {
            "GENE", "PROTEIN", "DISEASE", "DRUG", "CHEMICAL", 
            "CELL_TYPE", "TISSUE", "PATHWAY", "PHENOTYPE", "MUTATION"
        }
        return self.entity_type in biomedical_types


@dataclass
class RelationshipEvidence:
    """Evidence supporting a relationship."""
    evidence_id: str
    document_id: str
    text_span: str
    context: str
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Relationship:
    """Core relationship model for knowledge graph."""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str  # Now uses string for flexibility
    description: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    strength: float = 1.0
    confidence: float = 1.0
    evidence: List[RelationshipEvidence] = field(default_factory=list)
    source_documents: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate relationship type after initialization."""
        if not RELATIONSHIP_TYPE_REGISTRY.is_valid_type(self.relationship_type):
            # Auto-register unknown types as UNKNOWN
            if self.relationship_type not in RELATIONSHIP_TYPE_REGISTRY.get_types():
                self.relationship_type = "UNKNOWN"
    
    def add_evidence(self, evidence: RelationshipEvidence) -> None:
        """Add evidence to this relationship."""
        self.evidence.append(evidence)
        if evidence.document_id not in self.source_documents:
            self.source_documents.append(evidence.document_id)
        self.updated_at = datetime.now()
    
    def is_biomedical_relationship(self) -> bool:
        """Check if this is a biomedical relationship."""
        biomedical_types = {
            "INTERACTS_WITH", "REGULATES", "INHIBITS", "ACTIVATES",
            "BINDS_TO", "METABOLIZES", "TREATS", "ASSOCIATED_WITH",
            "EXPRESSED_IN", "MUTATED_IN"
        }
        return self.relationship_type in biomedical_types


@dataclass
class SubGraph:
    """Represents a subgraph extracted from the knowledge graph."""
    entities: List[Entity]
    relationships: List[Relationship]
    center_entities: List[str]  # The entities this subgraph was built around
    depth: int
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_entity_ids(self) -> Set[str]:
        """Get all entity IDs in this subgraph."""
        return {entity.entity_id for entity in self.entities}
    
    def get_relationship_count(self) -> int:
        """Get the number of relationships in this subgraph."""
        return len(self.relationships)
    
    def get_biomedical_entities(self) -> List[Entity]:
        """Get only biomedical entities from this subgraph."""
        return [entity for entity in self.entities if entity.is_biomedical_entity()]


@dataclass
class TraversalConfig:
    """Configuration for graph traversal operations."""
    max_depth: int = 2
    max_entities: int = 50
    relationship_types: Optional[List[str]] = None
    min_relationship_strength: float = 0.1
    include_entity_types: Optional[List[str]] = None
    exclude_entity_types: Optional[List[str]] = None
    use_globals_optimization: bool = False
    domain_focus: Optional[str] = None  # e.g., "biomedical", "technical"
    
    def should_include_entity(self, entity: Entity) -> bool:
        """Check if entity should be included based on config."""
        if self.include_entity_types and entity.entity_type not in self.include_entity_types:
            return False
        if self.exclude_entity_types and entity.entity_type in self.exclude_entity_types:
            return False
        
        # Domain-specific filtering
        if self.domain_focus == "biomedical" and not entity.is_biomedical_entity():
            return False
            
        return True
    
    def should_include_relationship(self, relationship: Relationship) -> bool:
        """Check if relationship should be included based on config."""
        if self.relationship_types and relationship.relationship_type not in self.relationship_types:
            return False
        if relationship.strength < self.min_relationship_strength:
            return False
        
        # Domain-specific filtering
        if self.domain_focus == "biomedical" and not relationship.is_biomedical_relationship():
            return False
            
        return True


@dataclass
class DomainConfig:
    """Configuration for domain-specific knowledge graph behavior."""
    domain_name: str
    entity_types: Dict[str, str]  # type_code -> description
    relationship_types: Dict[str, str]  # type_code -> description
    extraction_patterns: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    def apply_to_registries(self) -> None:
        """Apply this domain config to the global registries."""
        for type_code, description in self.entity_types.items():
            ENTITY_TYPE_REGISTRY.register_type(type_code, description)
        
        for type_code, description in self.relationship_types.items():
            RELATIONSHIP_TYPE_REGISTRY.register_type(type_code, description)


# Predefined domain configurations
BIOMEDICAL_DOMAIN_CONFIG = DomainConfig(
    domain_name="biomedical",
    entity_types={
        "GENE": "Gene or genetic element",
        "PROTEIN": "Protein or enzyme", 
        "DISEASE": "Disease or medical condition",
        "DRUG": "Drug or pharmaceutical compound",
        "CHEMICAL": "Chemical compound or substance",
        "CELL_TYPE": "Cell type or cellular component",
        "TISSUE": "Tissue or organ",
        "PATHWAY": "Biological pathway or process",
        "PHENOTYPE": "Observable characteristic or trait",
        "MUTATION": "Genetic mutation or variant",
        "BIOMARKER": "Biological marker",
        "TREATMENT": "Medical treatment or therapy"
    },
    relationship_types={
        "INTERACTS_WITH": "Molecular interaction",
        "REGULATES": "Regulatory relationship",
        "INHIBITS": "Inhibitory relationship", 
        "ACTIVATES": "Activation relationship",
        "BINDS_TO": "Binding relationship",
        "METABOLIZES": "Metabolic relationship",
        "TREATS": "Treatment relationship",
        "ASSOCIATED_WITH": "Disease association",
        "EXPRESSED_IN": "Expression relationship",
        "MUTATED_IN": "Mutation relationship",
        "TARGETS": "Drug target relationship",
        "BIOMARKER_FOR": "Biomarker relationship"
    }
)


@dataclass
class GraphBuildResult:
    """Result of building a knowledge graph from documents."""
    entities_created: int
    relationships_created: int
    entities_linked: int
    processing_time: float
    documents_processed: int
    domain_config: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        """Check if the graph build was successful."""
        return len(self.errors) == 0
    
    def add_error(self, error: str) -> None:
        """Add an error to the result."""
        self.errors.append(error)
    
    def add_warning(self, warning: str) -> None:
        """Add a warning to the result."""
        self.warnings.append(warning)


@dataclass
class EntityContext:
    """Rich context information for an entity."""
    entity: Entity
    related_entities: List[Entity]
    relationships: List[Relationship]
    document_contexts: List[str]
    relevance_score: float = 0.0
    
    def get_relationship_summary(self) -> Dict[str, int]:
        """Get a summary of relationship types."""
        summary = {}
        for rel in self.relationships:
            rel_type = rel.relationship_type
            summary[rel_type] = summary.get(rel_type, 0) + 1
        return summary


@dataclass
class GraphStatistics:
    """Statistics about the knowledge graph."""
    total_entities: int
    total_relationships: int
    entity_type_distribution: Dict[str, int]
    relationship_type_distribution: Dict[str, int]
    average_entity_degree: float
    max_entity_degree: int
    connected_components: int
    graph_density: float
    domain_focus: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)