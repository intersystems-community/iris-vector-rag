"""
Ontology Loader Module for IRIS RAG

This module provides base classes and implementations for loading ontologies
from various formats including OWL, RDF, and SKOS.
"""

import logging
import json
import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Optional, Any, Union, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
import urllib.parse
import uuid
from datetime import datetime

from .models import (
    Concept, OntologyRelationship, ConceptHierarchy, 
    SemanticMapping, OntologyMetadata, ConceptType, RelationType
)

logger = logging.getLogger(__name__)


class OntologyLoadError(Exception):
    """Exception raised when ontology loading fails."""
    pass


class OntologyLoader(ABC):
    """
    Base class for loading and processing ontologies.
    
    Provides common functionality and defines the interface for
    format-specific loaders.
    """
    
    def __init__(self, ontology_path: str):
        """Initialize the ontology loader."""
        self.ontology_path = ontology_path
        self.concepts: Dict[str, Concept] = {}
        self.relationships: Dict[str, OntologyRelationship] = {}
        self.hierarchy = ConceptHierarchy()
        self.metadata = OntologyMetadata()
        
        # Configuration
        self.max_concepts = 10000  # Prevent memory issues
        self.default_confidence = 0.8
        self.supported_languages = {'en', 'es', 'fr', 'de'}
        
    @abstractmethod
    def load(self, filepath: str) -> ConceptHierarchy:
        """Load ontology from file and return concept hierarchy."""
        pass
    
    @abstractmethod
    def _parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse the ontology file into a structured format."""
        pass
    
    def extract_concepts(self) -> List[Concept]:
        """Extract concepts from loaded ontology."""
        return list(self.concepts.values())
    
    def extract_hierarchies(self) -> Dict[str, List[str]]:
        """Extract concept hierarchies (is-a relationships)."""
        hierarchies = {}
        
        for concept in self.concepts.values():
            if concept.parent_concepts:
                hierarchies[concept.id] = list(concept.parent_concepts)
        
        return hierarchies
    
    def map_to_entities(self, entity_types: List[str]) -> Dict[str, SemanticMapping]:
        """Map ontology concepts to GraphRAG entity types."""
        mappings = {}
        
        for entity_type in entity_types:
            mapping = SemanticMapping(entity_type=entity_type)
            
            # Find concepts that match the entity type
            for concept in self.concepts.values():
                if self._matches_entity_type(concept, entity_type):
                    mapping.add_concept_mapping(concept.id)
            
            mappings[entity_type] = mapping
        
        return mappings
    
    def _matches_entity_type(self, concept: Concept, entity_type: str) -> bool:
        """Check if a concept matches an entity type."""
        entity_lower = entity_type.lower()
        
        # Check label and synonyms
        if entity_lower in concept.label.lower():
            return True
        
        for synonym in concept.get_all_synonyms():
            if entity_lower in synonym.lower():
                return True
        
        # Check metadata for type hints
        concept_types = concept.metadata.get('types', [])
        if isinstance(concept_types, list):
            for ctype in concept_types:
                if entity_lower in str(ctype).lower():
                    return True
        
        return False
    
    def get_concept_by_label(self, label: str) -> Optional[Concept]:
        """Find concept by label or synonym."""
        label_lower = label.lower().strip()
        
        for concept in self.concepts.values():
            if concept.label.lower() == label_lower:
                return concept
            
            if label_lower in concept.get_all_synonyms():
                return concept
        
        return None
    
    def _generate_concept_id(self, uri: Optional[str] = None, label: str = "") -> str:
        """Generate a unique concept ID."""
        if uri:
            # Use URI fragment or last part as base
            parsed = urllib.parse.urlparse(uri)
            if parsed.fragment:
                base_id = parsed.fragment
            else:
                base_id = parsed.path.split('/')[-1] if parsed.path else str(uuid.uuid4())
        else:
            # Use label-based ID
            base_id = label.lower().replace(' ', '_').replace('-', '_')
            if not base_id:
                base_id = str(uuid.uuid4())
        
        # Ensure uniqueness
        concept_id = base_id
        counter = 1
        while concept_id in self.concepts:
            concept_id = f"{base_id}_{counter}"
            counter += 1
        
        return concept_id
    
    def _validate_loaded_data(self) -> bool:
        """Validate the loaded ontology data."""
        if not self.concepts:
            logger.warning("No concepts loaded")
            return False
        
        if len(self.concepts) > self.max_concepts:
            logger.warning(f"Too many concepts loaded: {len(self.concepts)} > {self.max_concepts}")
            return False
        
        # Check for cycles in hierarchy
        cycles = self._detect_cycles()
        if cycles:
            logger.warning(f"Detected cycles in hierarchy: {cycles}")
            return False
        
        return True
    
    def _detect_cycles(self) -> List[List[str]]:
        """Detect cycles in the concept hierarchy."""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def dfs(concept_id: str, path: List[str]) -> bool:
            if concept_id in rec_stack:
                # Found cycle
                cycle_start = path.index(concept_id)
                cycles.append(path[cycle_start:] + [concept_id])
                return True
            
            if concept_id in visited:
                return False
            
            visited.add(concept_id)
            rec_stack.add(concept_id)
            
            concept = self.concepts.get(concept_id)
            if concept:
                for parent_id in concept.parent_concepts:
                    if dfs(parent_id, path + [concept_id]):
                        return True
            
            rec_stack.remove(concept_id)
            return False
        
        for concept_id in self.concepts:
            if concept_id not in visited:
                dfs(concept_id, [])
        
        return cycles


class OWLLoader(OntologyLoader):
    """Loader for OWL (Web Ontology Language) files."""
    
    def __init__(self, ontology_path: str):
        super().__init__(ontology_path)
        self.owl_namespaces = {
            'owl': 'http://www.w3.org/2002/07/owl#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'xsd': 'http://www.w3.org/2001/XMLSchema#'
        }
    
    def load(self, filepath: str) -> ConceptHierarchy:
        """Load OWL ontology file."""
        try:
            logger.info(f"Loading OWL ontology from {filepath}")
            
            if not Path(filepath).exists():
                raise OntologyLoadError(f"File not found: {filepath}")
            
            # Parse OWL file
            owl_data = self._parse_file(filepath)
            
            # Extract concepts and relationships
            self._extract_owl_classes(owl_data)
            self._extract_owl_properties(owl_data)
            self._extract_owl_relationships(owl_data)
            
            # Build hierarchy
            self._build_hierarchy()
            
            # Validate
            if not self._validate_loaded_data():
                raise OntologyLoadError("Validation failed for loaded OWL data")
            
            # Update metadata
            self.metadata.name = owl_data.get('ontology_name', Path(filepath).stem)
            self.metadata.source_url = filepath
            self.metadata.update_statistics(self.hierarchy)
            
            logger.info(f"Successfully loaded {len(self.concepts)} concepts from OWL")
            return self.hierarchy
            
        except Exception as e:
            logger.error(f"Failed to load OWL ontology: {e}")
            raise OntologyLoadError(f"OWL loading failed: {e}")
    
    def _parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse OWL XML file."""
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Extract basic ontology information
            owl_data = {
                'root': root,
                'classes': [],
                'properties': [],
                'individuals': [],
                'ontology_name': root.get('ontologyIRI', Path(filepath).stem)
            }
            
            # Find all classes
            for cls in root.findall('.//owl:Class', self.owl_namespaces):
                owl_data['classes'].append(cls)
            
            # Find all properties
            for prop in root.findall('.//owl:ObjectProperty', self.owl_namespaces):
                owl_data['properties'].append(prop)
            
            for prop in root.findall('.//owl:DatatypeProperty', self.owl_namespaces):
                owl_data['properties'].append(prop)
            
            # Find all individuals
            for ind in root.findall('.//owl:NamedIndividual', self.owl_namespaces):
                owl_data['individuals'].append(ind)
            
            return owl_data
            
        except ET.ParseError as e:
            raise OntologyLoadError(f"XML parsing error: {e}")
    
    def _extract_owl_classes(self, owl_data: Dict[str, Any]) -> None:
        """Extract classes from OWL data."""
        for cls_elem in owl_data['classes']:
            concept = self._parse_owl_class(cls_elem)
            if concept:
                self.concepts[concept.id] = concept
    
    def _extract_owl_properties(self, owl_data: Dict[str, Any]) -> None:
        """Extract properties from OWL data."""
        for prop_elem in owl_data['properties']:
            concept = self._parse_owl_property(prop_elem)
            if concept:
                self.concepts[concept.id] = concept
    
    def _extract_owl_relationships(self, owl_data: Dict[str, Any]) -> None:
        """Extract relationships from OWL data."""
        # Extract subClassOf relationships
        for cls_elem in owl_data['classes']:
            self._extract_subclass_relationships(cls_elem)
    
    def _parse_owl_class(self, cls_elem) -> Optional[Concept]:
        """Parse an OWL class element into a Concept."""
        uri = cls_elem.get(f'{{{self.owl_namespaces["rdf"]}}}about')
        if not uri:
            return None
        
        # Extract label
        label_elem = cls_elem.find('.//rdfs:label', self.owl_namespaces)
        label = label_elem.text if label_elem is not None else uri.split('#')[-1]
        
        # Extract description/comment
        comment_elem = cls_elem.find('.//rdfs:comment', self.owl_namespaces)
        description = comment_elem.text if comment_elem is not None else None
        
        concept_id = self._generate_concept_id(uri, label)
        
        concept = Concept(
            id=concept_id,
            uri=uri,
            label=label,
            description=description,
            concept_type=ConceptType.CLASS,
            confidence=self.default_confidence,
            source=self.ontology_path
        )
        
        # Extract additional labels and synonyms
        for alt_label in cls_elem.findall('.//rdfs:altLabel', self.owl_namespaces):
            concept.alternative_labels.add(alt_label.text)
        
        return concept
    
    def _parse_owl_property(self, prop_elem) -> Optional[Concept]:
        """Parse an OWL property element into a Concept."""
        uri = prop_elem.get(f'{{{self.owl_namespaces["rdf"]}}}about')
        if not uri:
            return None
        
        label_elem = prop_elem.find('.//rdfs:label', self.owl_namespaces)
        label = label_elem.text if label_elem is not None else uri.split('#')[-1]
        
        concept_id = self._generate_concept_id(uri, label)
        
        concept = Concept(
            id=concept_id,
            uri=uri,
            label=label,
            concept_type=ConceptType.PROPERTY,
            confidence=self.default_confidence,
            source=self.ontology_path
        )
        
        return concept
    
    def _extract_subclass_relationships(self, cls_elem) -> None:
        """Extract subClassOf relationships from a class element."""
        uri = cls_elem.get(f'{{{self.owl_namespaces["rdf"]}}}about')
        if not uri:
            return
        
        source_concept_id = self._uri_to_concept_id(uri)
        if not source_concept_id:
            return
        
        # Find subClassOf elements
        for subclass_elem in cls_elem.findall('.//rdfs:subClassOf', self.owl_namespaces):
            target_uri = subclass_elem.get(f'{{{self.owl_namespaces["rdf"]}}}resource')
            if target_uri:
                target_concept_id = self._uri_to_concept_id(target_uri)
                if target_concept_id:
                    relationship = OntologyRelationship(
                        source_concept_id=source_concept_id,
                        target_concept_id=target_concept_id,
                        relation_type=RelationType.IS_A,
                        confidence=self.default_confidence,
                        source=self.ontology_path
                    )
                    self.relationships[relationship.id] = relationship
    
    def _uri_to_concept_id(self, uri: str) -> Optional[str]:
        """Convert URI to concept ID if concept exists."""
        for concept in self.concepts.values():
            if concept.uri == uri:
                return concept.id
        return None
    
    def _build_hierarchy(self) -> None:
        """Build the concept hierarchy from loaded concepts and relationships."""
        # Add all concepts to hierarchy
        for concept in self.concepts.values():
            self.hierarchy.add_concept(concept)
        
        # Add all relationships
        for relationship in self.relationships.values():
            self.hierarchy.add_relationship(relationship)


class RDFLoader(OntologyLoader):
    """Loader for RDF (Resource Description Framework) files."""
    
    def load(self, filepath: str) -> ConceptHierarchy:
        """Load RDF ontology file."""
        try:
            logger.info(f"Loading RDF ontology from {filepath}")
            
            # For now, delegate to OWL loader since RDF and OWL share similar structure
            # In production, this could use libraries like rdflib for proper RDF parsing
            owl_loader = OWLLoader(self.ontology_path)
            self.hierarchy = owl_loader.load(filepath)
            self.concepts = owl_loader.concepts
            self.relationships = owl_loader.relationships
            self.metadata = owl_loader.metadata
            self.metadata.name = f"RDF_{Path(filepath).stem}"
            
            logger.info(f"Successfully loaded {len(self.concepts)} concepts from RDF")
            return self.hierarchy
            
        except Exception as e:
            logger.error(f"Failed to load RDF ontology: {e}")
            raise OntologyLoadError(f"RDF loading failed: {e}")
    
    def _parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse RDF file - simplified implementation."""
        # This is a simplified implementation
        # Production version should use proper RDF parsing libraries
        return {"concepts": [], "relationships": []}


class SKOSLoader(OntologyLoader):
    """Loader for SKOS (Simple Knowledge Organization System) vocabularies."""
    
    def __init__(self, ontology_path: str):
        super().__init__(ontology_path)
        self.skos_namespaces = {
            'skos': 'http://www.w3.org/2004/02/skos/core#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#'
        }
    
    def load(self, filepath: str) -> ConceptHierarchy:
        """Load SKOS vocabulary."""
        try:
            logger.info(f"Loading SKOS vocabulary from {filepath}")
            
            if not Path(filepath).exists():
                raise OntologyLoadError(f"File not found: {filepath}")
            
            skos_data = self._parse_file(filepath)
            
            # Extract SKOS concepts
            self._extract_skos_concepts(skos_data)
            self._extract_skos_relationships(skos_data)
            
            # Build hierarchy
            self._build_hierarchy()
            
            if not self._validate_loaded_data():
                raise OntologyLoadError("Validation failed for loaded SKOS data")
            
            self.metadata.name = f"SKOS_{Path(filepath).stem}"
            self.metadata.source_url = filepath
            self.metadata.update_statistics(self.hierarchy)
            
            logger.info(f"Successfully loaded {len(self.concepts)} concepts from SKOS")
            return self.hierarchy
            
        except Exception as e:
            logger.error(f"Failed to load SKOS vocabulary: {e}")
            raise OntologyLoadError(f"SKOS loading failed: {e}")
    
    def _parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse SKOS XML file."""
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            skos_data = {
                'root': root,
                'concepts': [],
                'concept_schemes': []
            }
            
            # Find all SKOS concepts
            for concept in root.findall('.//skos:Concept', self.skos_namespaces):
                skos_data['concepts'].append(concept)
            
            # Find concept schemes
            for scheme in root.findall('.//skos:ConceptScheme', self.skos_namespaces):
                skos_data['concept_schemes'].append(scheme)
            
            return skos_data
            
        except ET.ParseError as e:
            raise OntologyLoadError(f"SKOS XML parsing error: {e}")
    
    def _extract_skos_concepts(self, skos_data: Dict[str, Any]) -> None:
        """Extract concepts from SKOS data."""
        for concept_elem in skos_data['concepts']:
            concept = self._parse_skos_concept(concept_elem)
            if concept:
                self.concepts[concept.id] = concept
    
    def _extract_skos_relationships(self, skos_data: Dict[str, Any]) -> None:
        """Extract relationships from SKOS data."""
        for concept_elem in skos_data['concepts']:
            self._extract_skos_concept_relationships(concept_elem)
    
    def _parse_skos_concept(self, concept_elem) -> Optional[Concept]:
        """Parse a SKOS concept element."""
        uri = concept_elem.get(f'{{{self.skos_namespaces["rdf"]}}}about')
        if not uri:
            return None
        
        # Extract preferred label
        pref_label_elem = concept_elem.find('.//skos:prefLabel', self.skos_namespaces)
        label = pref_label_elem.text if pref_label_elem is not None else uri.split('#')[-1]
        
        # Extract definition
        definition_elem = concept_elem.find('.//skos:definition', self.skos_namespaces)
        description = definition_elem.text if definition_elem is not None else None
        
        concept_id = self._generate_concept_id(uri, label)
        
        concept = Concept(
            id=concept_id,
            uri=uri,
            label=label,
            description=description,
            concept_type=ConceptType.CLASS,
            confidence=self.default_confidence,
            source=self.ontology_path
        )
        
        # Extract alternative labels
        for alt_label in concept_elem.findall('.//skos:altLabel', self.skos_namespaces):
            concept.alternative_labels.add(alt_label.text)
        
        # Extract hidden labels (synonyms)
        for hidden_label in concept_elem.findall('.//skos:hiddenLabel', self.skos_namespaces):
            concept.add_synonym(hidden_label.text)
        
        return concept
    
    def _extract_skos_concept_relationships(self, concept_elem) -> None:
        """Extract relationships for a SKOS concept."""
        uri = concept_elem.get(f'{{{self.skos_namespaces["rdf"]}}}about')
        if not uri:
            return
        
        source_concept_id = self._uri_to_concept_id(uri)
        if not source_concept_id:
            return
        
        # Extract broader relationships
        for broader_elem in concept_elem.findall('.//skos:broader', self.skos_namespaces):
            target_uri = broader_elem.get(f'{{{self.skos_namespaces["rdf"]}}}resource')
            if target_uri:
                target_concept_id = self._uri_to_concept_id(target_uri)
                if target_concept_id:
                    relationship = OntologyRelationship(
                        source_concept_id=source_concept_id,
                        target_concept_id=target_concept_id,
                        relation_type=RelationType.IS_A,
                        confidence=self.default_confidence,
                        source=self.ontology_path
                    )
                    self.relationships[relationship.id] = relationship
        
        # Extract related relationships
        for related_elem in concept_elem.findall('.//skos:related', self.skos_namespaces):
            target_uri = related_elem.get(f'{{{self.skos_namespaces["rdf"]}}}resource')
            if target_uri:
                target_concept_id = self._uri_to_concept_id(target_uri)
                if target_concept_id:
                    relationship = OntologyRelationship(
                        source_concept_id=source_concept_id,
                        target_concept_id=target_concept_id,
                        relation_type=RelationType.RELATED_TO,
                        confidence=self.default_confidence,
                        source=self.ontology_path
                    )
                    self.relationships[relationship.id] = relationship
    
    def _uri_to_concept_id(self, uri: str) -> Optional[str]:
        """Convert URI to concept ID if concept exists."""
        for concept in self.concepts.values():
            if concept.uri == uri:
                return concept.id
        return None
    
    def _build_hierarchy(self) -> None:
        """Build the concept hierarchy from loaded concepts and relationships."""
        # Add all concepts to hierarchy
        for concept in self.concepts.values():
            self.hierarchy.add_concept(concept)
        
        # Add all relationships
        for relationship in self.relationships.values():
            self.hierarchy.add_relationship(relationship)