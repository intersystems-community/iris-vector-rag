"""
Ontology Data Models for IRIS RAG

This module defines the core data structures for representing ontological knowledge
including concepts, relationships, hierarchies, and semantic mappings.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

logger = logging.getLogger(__name__)


class ConceptType(Enum):
    """Types of ontological concepts."""

    CLASS = "class"
    INSTANCE = "instance"
    PROPERTY = "property"
    DATATYPE = "datatype"


class RelationType(Enum):
    """Types of relationships between concepts."""

    IS_A = "is_a"  # Subsumption/inheritance
    PART_OF = "part_of"  # Partitive relationship
    RELATED_TO = "related_to"  # General association
    EQUIVALENT_TO = "equivalent_to"  # Equivalence
    DISJOINT_WITH = "disjoint_with"  # Disjoint classes
    SAME_AS = "same_as"  # Identity
    CAUSES = "causes"  # Causal relationship
    TREATS = "treats"  # Treatment relationship
    MANAGES = "manages"  # Management relationship
    IMPLEMENTS = "implements"  # Implementation relationship
    USES = "uses"  # Usage relationship


@dataclass
class Concept:
    """Represents an ontological concept with metadata and relationships."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    uri: Optional[str] = None
    label: str = ""
    description: Optional[str] = None
    concept_type: ConceptType = ConceptType.CLASS
    domain: Optional[str] = None

    # Synonyms and alternative labels
    synonyms: Set[str] = field(default_factory=set)
    alternative_labels: Set[str] = field(default_factory=set)

    # Hierarchical information
    parent_concepts: Set[str] = field(default_factory=set)
    child_concepts: Set[str] = field(default_factory=set)

    # Domain-specific metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # External identifiers (UMLS CUI, SNOMED codes, etc.)
    external_ids: Dict[str, str] = field(default_factory=dict)

    # Confidence and provenance
    confidence: float = 1.0
    source: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

    def add_synonym(self, synonym: str) -> None:
        """Add a synonym for this concept."""
        self.synonyms.add(synonym.lower().strip())

    def add_parent(self, parent_id: str) -> None:
        """Add a parent concept."""
        self.parent_concepts.add(parent_id)

    def add_child(self, child_id: str) -> None:
        """Add a child concept."""
        self.child_concepts.add(child_id)

    def has_ancestor(
        self, concept_hierarchy: "ConceptHierarchy", ancestor_id: str
    ) -> bool:
        """Check if this concept has a specific ancestor."""
        return concept_hierarchy.is_ancestor(ancestor_id, self.id)

    def get_all_synonyms(self) -> Set[str]:
        """Get all synonyms including label and alternative labels."""
        all_synonyms = set(self.synonyms)
        all_synonyms.add(self.label.lower())
        all_synonyms.update(label.lower() for label in self.alternative_labels)
        return all_synonyms

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get a metadata value by key."""
        return self.metadata.get(key, default)


@dataclass
class OntologyRelationship:
    """Represents a relationship between two concepts."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_concept_id: str = ""
    target_concept_id: str = ""
    relation_type: RelationType = RelationType.RELATED_TO

    # Relationship properties
    label: Optional[str] = None
    description: Optional[str] = None
    inverse_relation: Optional[str] = None

    # Metadata and provenance
    confidence: float = 1.0
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def is_hierarchical(self) -> bool:
        """Check if this is a hierarchical relationship."""
        return self.relation_type in {RelationType.IS_A, RelationType.PART_OF}

    def is_symmetric(self) -> bool:
        """Check if this relationship is symmetric."""
        return self.relation_type in {
            RelationType.EQUIVALENT_TO,
            RelationType.DISJOINT_WITH,
            RelationType.SAME_AS,
        }


@dataclass
class ConceptHierarchy:
    """Manages hierarchical relationships between concepts."""

    concepts: Dict[str, Concept] = field(default_factory=dict)
    relationships: Dict[str, OntologyRelationship] = field(default_factory=dict)
    root_concepts: Set[str] = field(default_factory=set)

    def add_concept(self, concept: Concept) -> None:
        """Add a concept to the hierarchy."""
        self.concepts[concept.id] = concept

        # Update root concepts
        if not concept.parent_concepts:
            self.root_concepts.add(concept.id)
        else:
            self.root_concepts.discard(concept.id)

    def add_relationship(
        self,
        relationship: Union[OntologyRelationship, str],
        target: Optional[str] = None,
        relation: Union[RelationType, str] = RelationType.RELATED_TO,
    ) -> None:
        """Add a relationship to the hierarchy.
        Supports either an OntologyRelationship instance or (source_id, target_id, relation) parameters.
        """
        if isinstance(relationship, OntologyRelationship):
            rel = relationship
        else:
            source_id = str(relationship)
            target_id = str(target) if target is not None else ""
            if isinstance(relation, str):
                try:
                    relation_type = RelationType(relation)
                except Exception:
                    relation_type = RelationType.RELATED_TO
            else:
                relation_type = relation
            rel = OntologyRelationship(
                source_concept_id=source_id,
                target_concept_id=target_id,
                relation_type=relation_type,
            )
        self.relationships[rel.id] = rel

        # Update concept parent/child relationships for hierarchical relations
        if rel.is_hierarchical():
            source_concept = self.concepts.get(rel.source_concept_id)
            target_concept = self.concepts.get(rel.target_concept_id)
            if source_concept and target_concept:
                if rel.relation_type == RelationType.IS_A:
                    source_concept.add_parent(rel.target_concept_id)
                    target_concept.add_child(rel.source_concept_id)
                    self.root_concepts.discard(rel.source_concept_id)

    def get_ancestors(self, concept_id: str, max_depth: int = 10) -> Set[str]:
        """Get all ancestors of a concept up to max_depth."""
        ancestors = set()
        current_level = {concept_id}

        for _ in range(max_depth):
            next_level = set()
            for cid in current_level:
                concept = self.concepts.get(cid)
                if concept:
                    next_level.update(concept.parent_concepts)

            if not next_level or next_level.issubset(ancestors):
                break

            ancestors.update(next_level)
            current_level = next_level

        return ancestors

    def get_descendants(self, concept_id: str, max_depth: int = 10) -> Set[str]:
        """Get all descendants of a concept up to max_depth."""
        descendants = set()
        current_level = {concept_id}

        for _ in range(max_depth):
            next_level = set()
            for cid in current_level:
                concept = self.concepts.get(cid)
                if concept:
                    next_level.update(concept.child_concepts)

            if not next_level or next_level.issubset(descendants):
                break

            descendants.update(next_level)
            current_level = next_level

        return descendants

    def is_ancestor(self, ancestor_id: str, descendant_id: str) -> bool:
        """Check if ancestor_id is an ancestor of descendant_id."""
        return ancestor_id in self.get_ancestors(descendant_id)

    def get_common_ancestors(self, concept_ids: List[str]) -> Set[str]:
        """Find common ancestors of multiple concepts."""
        if not concept_ids:
            return set()

        common_ancestors = self.get_ancestors(concept_ids[0])
        for concept_id in concept_ids[1:]:
            ancestors = self.get_ancestors(concept_id)
            common_ancestors &= ancestors

        return common_ancestors


@dataclass
class SemanticMapping:
    """Maps ontology concepts to entity types and provides semantic enrichment."""

    entity_type: str = ""
    concept_ids: Set[str] = field(default_factory=set)
    confidence_threshold: float = 0.7
    expansion_rules: Dict[str, Any] = field(default_factory=dict)

    def maps_to_concept(self, concept_id: str) -> bool:
        """Check if this mapping includes a specific concept."""
        return concept_id in self.concept_ids

    def add_concept_mapping(self, concept_id: str) -> None:
        """Add a concept mapping for this entity type."""
        self.concept_ids.add(concept_id)


@dataclass
class InferenceRule:
    """Represents a rule for ontological inference."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: Optional[str] = None

    # Rule conditions and conclusions
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    conclusions: List[Dict[str, Any]] = field(default_factory=list)

    # Rule metadata
    confidence: float = 1.0
    domain: Optional[str] = None
    enabled: bool = True
    priority: int = 0

    def applies_to_domain(self, domain: str) -> bool:
        """Check if this rule applies to a specific domain."""
        return self.domain is None or self.domain == domain

    def is_applicable(self, facts: Dict[str, Any]) -> bool:
        """Check if this rule is applicable given current facts."""
        if not self.enabled:
            return False

        # Simple condition matching - can be extended with more complex logic
        for condition in self.conditions:
            if not self._matches_condition(condition, facts):
                return False

        return True

    def _matches_condition(
        self, condition: Dict[str, Any], facts: Dict[str, Any]
    ) -> bool:
        """Check if a condition matches the current facts."""
        # This is a simplified implementation
        # Real implementation would include more sophisticated pattern matching
        for key, value in condition.items():
            if key not in facts or facts[key] != value:
                return False
        return True


@dataclass
class OntologyMetadata:
    """Metadata about an ontology."""

    name: str = ""
    version: str = ""
    description: Optional[str] = None
    domain: Optional[str] = None
    language: str = "en"

    # Provenance information
    source_url: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    license: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

    # Statistics
    concept_count: int = 0
    relationship_count: int = 0
    max_depth: int = 0

    def update_statistics(self, hierarchy: ConceptHierarchy) -> None:
        """Update statistics based on concept hierarchy."""
        self.concept_count = len(hierarchy.concepts)
        self.relationship_count = len(hierarchy.relationships)

        # Calculate max depth
        max_depth = 0
        for root_id in hierarchy.root_concepts:
            depth = self._calculate_depth(hierarchy, root_id)
            max_depth = max(max_depth, depth)
        self.max_depth = max_depth
        self.modified_at = datetime.now()

    def _calculate_depth(
        self, hierarchy: ConceptHierarchy, concept_id: str, visited: Set[str] = None
    ) -> int:
        """Calculate the depth of a concept hierarchy."""
        if visited is None:
            visited = set()

        if concept_id in visited:
            return 0  # Avoid cycles

        visited.add(concept_id)
        concept = hierarchy.concepts.get(concept_id)
        if not concept:
            return 0

        max_child_depth = 0
        for child_id in concept.child_concepts:
            child_depth = self._calculate_depth(hierarchy, child_id, visited.copy())
            max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth + 1
