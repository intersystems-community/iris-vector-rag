"""
Ontology Reasoning Engine for IRIS RAG

This module provides reasoning capabilities for ontology-based inference,
query expansion, and semantic enrichment. Incorporates modern approaches
from semantic web standards and GraphRAG research.

Key Features:
- Subsumption reasoning (class hierarchy inference)
- Property reasoning (transitivity, symmetry, functionality)
- Query expansion with synonyms and related concepts
- Semantic similarity computation
- Rule-based inference
- Hybrid symbolic-neural reasoning support
"""

import logging
from typing import Dict, List, Set, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re
from collections import defaultdict, deque
import math

from .models import (
    Concept, OntologyRelationship, ConceptHierarchy, 
    SemanticMapping, InferenceRule, RelationType
)

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Result of an inference operation."""
    new_relationships: List[OntologyRelationship] = field(default_factory=list)
    expanded_concepts: Set[str] = field(default_factory=set)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    reasoning_trace: List[str] = field(default_factory=list)


@dataclass
class QueryExpansionResult:
    """Result of query expansion."""
    original_query: str = ""
    expanded_query: str = ""
    expansion_terms: List[str] = field(default_factory=list)
    semantic_concepts: List[str] = field(default_factory=list)
    confidence: float = 1.0


class ReasoningStrategy(ABC):
    """Abstract base class for reasoning strategies."""
    
    @abstractmethod
    def infer(self, hierarchy: ConceptHierarchy, facts: Dict[str, Any]) -> InferenceResult:
        """Perform inference on the concept hierarchy."""
        pass


class SubsumptionReasoner(ReasoningStrategy):
    """Implements subsumption (is-a) reasoning for class hierarchies."""
    
    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
    
    def infer(self, hierarchy: ConceptHierarchy, facts: Dict[str, Any]) -> InferenceResult:
        """Infer new subsumption relationships."""
        result = InferenceResult()
        
        # Transitive closure of is-a relationships
        for concept_id in hierarchy.concepts:
            ancestors = hierarchy.get_ancestors(concept_id, self.max_depth)
            
            # Add any missing direct ancestor relationships
            for ancestor_id in ancestors:
                if not self._has_direct_path(hierarchy, concept_id, ancestor_id):
                    relationship = OntologyRelationship(
                        source_concept_id=concept_id,
                        target_concept_id=ancestor_id,
                        relation_type=RelationType.IS_A,
                        confidence=0.9,  # Inferred with high confidence
                        metadata={"inferred": True, "method": "subsumption"}
                    )
                    result.new_relationships.append(relationship)
                    result.reasoning_trace.append(
                        f"Inferred {concept_id} is_a {ancestor_id} via transitive closure"
                    )
        
        return result
    
    def _has_direct_path(self, hierarchy: ConceptHierarchy, source: str, target: str) -> bool:
        """Check if there's already a direct relationship path."""
        for rel in hierarchy.relationships.values():
            if (rel.source_concept_id == source and 
                rel.target_concept_id == target and 
                rel.relation_type == RelationType.IS_A):
                return True
        return False


class PropertyReasoner(ReasoningStrategy):
    """Implements property-based reasoning (transitivity, symmetry, etc.)."""
    
    def infer(self, hierarchy: ConceptHierarchy, facts: Dict[str, Any]) -> InferenceResult:
        """Infer new relationships based on property characteristics."""
        result = InferenceResult()
        
        # Group relationships by type
        rel_by_type = defaultdict(list)
        for rel in hierarchy.relationships.values():
            rel_by_type[rel.relation_type].append(rel)
        
        # Apply transitivity rules
        self._apply_transitivity(rel_by_type, result)
        
        # Apply symmetry rules
        self._apply_symmetry(rel_by_type, result)
        
        return result
    
    def _apply_transitivity(self, rel_by_type: Dict, result: InferenceResult) -> None:
        """Apply transitivity rules for appropriate relationship types."""
        transitive_types = {RelationType.IS_A, RelationType.PART_OF}
        
        for rel_type in transitive_types:
            relationships = rel_by_type[rel_type]
            
            # Build adjacency map
            adj_map = defaultdict(set)
            for rel in relationships:
                adj_map[rel.source_concept_id].add(rel.target_concept_id)
            
            # Find transitive closures
            for start_concept in adj_map:
                visited = set()
                queue = deque([(start_concept, 0)])
                
                while queue:
                    concept, depth = queue.popleft()
                    if concept in visited or depth > 5:  # Limit depth
                        continue
                    
                    visited.add(concept)
                    
                    for next_concept in adj_map[concept]:
                        if next_concept not in visited and depth > 0:
                            # Found transitive relationship
                            confidence = max(0.5, 0.9 - (depth * 0.1))
                            relationship = OntologyRelationship(
                                source_concept_id=start_concept,
                                target_concept_id=next_concept,
                                relation_type=rel_type,
                                confidence=confidence,
                                metadata={"inferred": True, "method": "transitivity", "depth": depth}
                            )
                            result.new_relationships.append(relationship)
                        
                        queue.append((next_concept, depth + 1))
    
    def _apply_symmetry(self, rel_by_type: Dict, result: InferenceResult) -> None:
        """Apply symmetry rules for symmetric relationship types."""
        symmetric_types = {RelationType.RELATED_TO, RelationType.EQUIVALENT_TO}
        
        for rel_type in symmetric_types:
            relationships = rel_by_type[rel_type]
            
            for rel in relationships:
                # Check if inverse relationship exists
                inverse_exists = any(
                    r.source_concept_id == rel.target_concept_id and
                    r.target_concept_id == rel.source_concept_id and
                    r.relation_type == rel_type
                    for r in relationships
                )
                
                if not inverse_exists:
                    inverse_rel = OntologyRelationship(
                        source_concept_id=rel.target_concept_id,
                        target_concept_id=rel.source_concept_id,
                        relation_type=rel_type,
                        confidence=rel.confidence * 0.9,
                        metadata={"inferred": True, "method": "symmetry"}
                    )
                    result.new_relationships.append(inverse_rel)


class RuleBasedReasoner(ReasoningStrategy):
    """Implements rule-based inference using custom inference rules."""
    
    def __init__(self, rules: List[InferenceRule]):
        self.rules = sorted(rules, key=lambda r: r.priority, reverse=True)
    
    def infer(self, hierarchy: ConceptHierarchy, facts: Dict[str, Any]) -> InferenceResult:
        """Apply inference rules to derive new knowledge."""
        result = InferenceResult()
        
        # Apply each rule
        for rule in self.rules:
            if rule.is_applicable(facts):
                rule_results = self._apply_rule(rule, hierarchy, facts)
                result.new_relationships.extend(rule_results.new_relationships)
                result.expanded_concepts.update(rule_results.expanded_concepts)
                result.reasoning_trace.extend(rule_results.reasoning_trace)
        
        return result
    
    def _apply_rule(self, rule: InferenceRule, hierarchy: ConceptHierarchy, 
                   facts: Dict[str, Any]) -> InferenceResult:
        """Apply a specific inference rule."""
        result = InferenceResult()
        
        # This is a simplified rule application
        # Real implementation would include more sophisticated pattern matching
        for conclusion in rule.conclusions:
            if conclusion.get('type') == 'relationship':
                relationship = OntologyRelationship(
                    source_concept_id=conclusion.get('source', ''),
                    target_concept_id=conclusion.get('target', ''),
                    relation_type=RelationType(conclusion.get('relation_type', 'related_to')),
                    confidence=rule.confidence,
                    metadata={"inferred": True, "rule": rule.name}
                )
                result.new_relationships.append(relationship)
                result.reasoning_trace.append(f"Applied rule: {rule.name}")
        
        return result


class QueryExpander:
    """Expands queries using ontological knowledge for enhanced retrieval."""
    
    def __init__(self, hierarchy: ConceptHierarchy, max_expansions: int = 10):
        self.hierarchy = hierarchy
        self.max_expansions = max_expansions
        self.stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def expand_query(self, query: str, expansion_strategy: str = "synonyms") -> QueryExpansionResult:
        """Expand query using ontology concepts."""
        result = QueryExpansionResult(original_query=query)
        
        # Extract potential concepts from query
        query_concepts = self._extract_concepts_from_text(query)
        
        # Expand based on strategy
        if expansion_strategy == "synonyms":
            result = self._expand_with_synonyms(query, query_concepts)
        elif expansion_strategy == "hierarchical":
            result = self._expand_hierarchically(query, query_concepts)
        elif expansion_strategy == "semantic":
            result = self._expand_semantically(query, query_concepts)
        else:
            result.expanded_query = query
        
        return result
    
    def _extract_concepts_from_text(self, text: str) -> List[Concept]:
        """Extract ontology concepts mentioned in text."""
        concepts = []
        text_lower = text.lower()
        
        # Simple approach: check if any concept labels/synonyms appear in text
        for concept in self.hierarchy.concepts.values():
            if concept.label.lower() in text_lower:
                concepts.append(concept)
                continue
            
            # Check synonyms
            for synonym in concept.get_all_synonyms():
                if synonym in text_lower:
                    concepts.append(concept)
                    break
        
        return concepts
    
    def _expand_with_synonyms(self, query: str, concepts: List[Concept]) -> QueryExpansionResult:
        """Expand query with synonyms and alternative labels."""
        result = QueryExpansionResult(original_query=query)
        expansion_terms = []
        
        for concept in concepts[:self.max_expansions]:
            # Add synonyms
            synonyms = list(concept.get_all_synonyms())[:3]  # Limit synonyms
            expansion_terms.extend(synonyms)
            
            # Add alternative labels
            alt_labels = list(concept.alternative_labels)[:2]
            expansion_terms.extend(alt_labels)
        
        # Remove duplicates and create expanded query
        unique_terms = list(set(expansion_terms))
        if unique_terms:
            result.expanded_query = f"{query} ({' OR '.join(unique_terms)})"
            result.expansion_terms = unique_terms
            result.confidence = 0.8
        else:
            result.expanded_query = query
            result.confidence = 1.0
        
        return result
    
    def _expand_hierarchically(self, query: str, concepts: List[Concept]) -> QueryExpansionResult:
        """Expand query with hierarchically related concepts."""
        result = QueryExpansionResult(original_query=query)
        expansion_terms = []
        
        for concept in concepts[:self.max_expansions]:
            # Add parent concepts (generalizations)
            for parent_id in list(concept.parent_concepts)[:2]:
                parent = self.hierarchy.concepts.get(parent_id)
                if parent:
                    expansion_terms.append(parent.label)
            
            # Add child concepts (specializations)
            for child_id in list(concept.child_concepts)[:3]:
                child = self.hierarchy.concepts.get(child_id)
                if child:
                    expansion_terms.append(child.label)
        
        if expansion_terms:
            result.expanded_query = f"{query} (related: {' OR '.join(expansion_terms)})"
            result.expansion_terms = expansion_terms
            result.confidence = 0.7
        else:
            result.expanded_query = query
            result.confidence = 1.0
        
        return result
    
    def _expand_semantically(self, query: str, concepts: List[Concept]) -> QueryExpansionResult:
        """Expand query with semantically related concepts."""
        result = QueryExpansionResult(original_query=query)
        expansion_terms = []
        
        # Find semantically related concepts through relationships
        for concept in concepts:
            related_concepts = self._find_related_concepts(concept.id)
            for related_id in list(related_concepts)[:3]:
                related = self.hierarchy.concepts.get(related_id)
                if related:
                    expansion_terms.append(related.label)
        
        if expansion_terms:
            result.expanded_query = f"{query} (semantic: {' OR '.join(expansion_terms)})"
            result.expansion_terms = expansion_terms
            result.confidence = 0.6
        else:
            result.expanded_query = query
            result.confidence = 1.0
        
        return result
    
    def _find_related_concepts(self, concept_id: str) -> Set[str]:
        """Find concepts related through various relationship types."""
        related = set()
        
        for rel in self.hierarchy.relationships.values():
            if rel.source_concept_id == concept_id:
                if rel.relation_type in {RelationType.RELATED_TO, RelationType.USES, RelationType.TREATS}:
                    related.add(rel.target_concept_id)
            elif rel.target_concept_id == concept_id:
                if rel.relation_type in {RelationType.RELATED_TO, RelationType.CAUSES}:
                    related.add(rel.source_concept_id)
        
        return related


class OntologyReasoner:
    """
    Main reasoning engine that coordinates different reasoning strategies
    and provides a unified interface for ontology-based inference.
    """
    
    def __init__(self, hierarchy: ConceptHierarchy):
        self.hierarchy = hierarchy
        self.strategies: List[ReasoningStrategy] = []
        self.query_expander = QueryExpander(hierarchy)
        self.inference_cache: Dict[str, InferenceResult] = {}
        
        # Initialize default strategies
        self._init_default_strategies()
    
    def _init_default_strategies(self) -> None:
        """Initialize default reasoning strategies."""
        self.strategies = [
            SubsumptionReasoner(),
            PropertyReasoner(),
        ]
    
    def add_reasoning_strategy(self, strategy: ReasoningStrategy) -> None:
        """Add a custom reasoning strategy."""
        self.strategies.append(strategy)
    
    def add_inference_rules(self, rules: List[InferenceRule]) -> None:
        """Add custom inference rules."""
        rule_reasoner = RuleBasedReasoner(rules)
        self.strategies.append(rule_reasoner)
    
    def infer_relationships(self, entities: List[Any]) -> List[OntologyRelationship]:
        """Infer new relationships using ontology."""
        all_results = []
        facts = {"entities": entities}
        
        # Apply all reasoning strategies
        for strategy in self.strategies:
            try:
                result = strategy.infer(self.hierarchy, facts)
                all_results.extend(result.new_relationships)
                logger.debug(f"Strategy {type(strategy).__name__} inferred {len(result.new_relationships)} relationships")
            except Exception as e:
                logger.error(f"Reasoning strategy {type(strategy).__name__} failed: {e}")
        
        # Remove duplicates and filter by confidence
        unique_results = self._deduplicate_relationships(all_results)
        return [rel for rel in unique_results if rel.confidence >= 0.5]
    
    def expand_query(self, query: str, strategy: str = "synonyms") -> str:
        """Expand query using ontology concepts."""
        result = self.query_expander.expand_query(query, strategy)
        return result.expanded_query
    
    def subsumption_reasoning(self, concept_a: str, concept_b: str) -> bool:
        """Check if concept_a subsumes concept_b (concept_b is_a concept_a)."""
        return self.hierarchy.is_ancestor(concept_a, concept_b)
    
    def find_common_concepts(self, entities: List[str]) -> List[str]:
        """Find common ontology concepts for a list of entities."""
        common_concepts = []
        
        # Find concepts that match multiple entities
        entity_concepts = {}
        for entity in entities:
            concepts = self._find_concepts_for_entity(entity)
            entity_concepts[entity] = concepts
        
        # Find intersections
        if entity_concepts:
            all_concepts = set(entity_concepts[entities[0]])
            for entity in entities[1:]:
                all_concepts &= set(entity_concepts[entity])
            common_concepts = list(all_concepts)
        
        return common_concepts
    
    def semantic_similarity(self, concept1_id: str, concept2_id: str) -> float:
        """Calculate semantic similarity between two concepts."""
        if concept1_id == concept2_id:
            return 1.0
        
        # Use hierarchy depth and common ancestors
        ancestors1 = self.hierarchy.get_ancestors(concept1_id)
        ancestors2 = self.hierarchy.get_ancestors(concept2_id)
        
        common_ancestors = ancestors1 & ancestors2
        
        if not common_ancestors:
            return 0.0
        
        # Simple similarity based on common ancestors
        total_ancestors = len(ancestors1 | ancestors2)
        if total_ancestors == 0:
            return 0.0
        
        similarity = len(common_ancestors) / total_ancestors
        return min(1.0, similarity)
    
    def _find_concepts_for_entity(self, entity: str) -> List[str]:
        """Find ontology concepts that match an entity."""
        matching_concepts = []
        entity_lower = entity.lower()
        
        for concept in self.hierarchy.concepts.values():
            if entity_lower in concept.label.lower():
                matching_concepts.append(concept.id)
            elif any(entity_lower in syn for syn in concept.get_all_synonyms()):
                matching_concepts.append(concept.id)
        
        return matching_concepts
    
    def _deduplicate_relationships(self, relationships: List[OntologyRelationship]) -> List[OntologyRelationship]:
        """Remove duplicate relationships, keeping highest confidence."""
        seen = {}
        
        for rel in relationships:
            key = (rel.source_concept_id, rel.target_concept_id, rel.relation_type)
            if key not in seen or rel.confidence > seen[key].confidence:
                seen[key] = rel
        
        return list(seen.values())


class InferenceEngine:
    """
    High-level inference engine that orchestrates reasoning operations
    and maintains inference state.
    """
    
    def __init__(self, reasoner: OntologyReasoner):
        self.reasoner = reasoner
        self.inference_history: List[InferenceResult] = []
        self.confidence_threshold = 0.6
    
    def perform_inference(self, entities: List[Any], context: Dict[str, Any] = None) -> InferenceResult:
        """Perform comprehensive inference on entities."""
        if context is None:
            context = {}
        
        # Infer new relationships
        new_relationships = self.reasoner.infer_relationships(entities)
        
        # Find expanded concepts
        expanded_concepts = set()
        for entity in entities:
            if hasattr(entity, 'text'):
                concepts = self.reasoner._find_concepts_for_entity(entity.text)
                expanded_concepts.update(concepts)
        
        # Calculate confidence scores
        confidence_scores = {}
        for rel in new_relationships:
            confidence_scores[rel.id] = rel.confidence
        
        result = InferenceResult(
            new_relationships=new_relationships,
            expanded_concepts=expanded_concepts,
            confidence_scores=confidence_scores,
            reasoning_trace=[f"Inferred {len(new_relationships)} relationships"]
        )
        
        self.inference_history.append(result)
        return result
    
    def get_inference_summary(self) -> Dict[str, Any]:
        """Get summary of inference operations."""
        total_relationships = sum(len(r.new_relationships) for r in self.inference_history)
        total_concepts = len(set().union(*[r.expanded_concepts for r in self.inference_history]))
        
        return {
            "total_inferences": len(self.inference_history),
            "total_relationships": total_relationships,
            "total_concepts": total_concepts,
            "average_confidence": sum(
                sum(r.confidence_scores.values()) / max(1, len(r.confidence_scores))
                for r in self.inference_history
            ) / max(1, len(self.inference_history))
        }