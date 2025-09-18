"""
IRIS RAG Ontology Support Module

This module provides advanced ontology support for GraphRAG to enable domain-specific 
knowledge representation and reasoning. It supports loading ontologies in various formats
(OWL, RDF, SKOS) and provides reasoning capabilities for enhanced entity extraction.

Key Features:
- Multi-format ontology loading (OWL, RDF, SKOS)
- Domain-specific plugins (medical, IT, software development, support)
- Hierarchical reasoning and concept expansion
- Ontology-aware entity extraction and mapping
- Semantic relationship inference
- Query expansion with synonyms and related concepts

Components:
- models: Core ontology data models (Concept, Relationship, etc.)
- loader: Base ontology loader with format-specific implementations
- reasoner: Reasoning engine for ontology-based inference
- plugins: Domain-specific ontology implementations
"""

from .models import (
    Concept,
    OntologyRelationship,
    ConceptHierarchy,
    SemanticMapping,
    InferenceRule
)

from .loader import (
    OntologyLoader,
    OWLLoader,
    RDFLoader,
    SKOSLoader
)

from .reasoner import (
    OntologyReasoner,
    InferenceEngine,
    QueryExpander
)

__version__ = "1.0.0"
__all__ = [
    # Models
    "Concept",
    "OntologyRelationship", 
    "ConceptHierarchy",
    "SemanticMapping",
    "InferenceRule",
    
    # Loaders
    "OntologyLoader",
    "OWLLoader", 
    "RDFLoader",
    "SKOSLoader",
    
    # Reasoning
    "OntologyReasoner",
    "InferenceEngine",
    "QueryExpander"
]