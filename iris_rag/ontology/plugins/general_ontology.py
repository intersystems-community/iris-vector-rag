"""
General-Purpose Ontology Plugin for IRIS RAG

This plugin provides domain-agnostic ontology support that can work with ANY domain
without hardcoded assumptions. It automatically detects domains, generates entity
mappings, and supports custom domain definitions.

Key Features:
- Auto-detection of domain from ontology metadata
- Dynamic entity mapping generation from ontology concepts
- Support for custom domain definitions via configuration
- Compatible with OWL, RDF, and SKOS formats
- No hardcoded domain-specific assumptions
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..loader import OntologyLoader, OWLLoader, RDFLoader, SKOSLoader
from ..models import (
    Concept,
    ConceptHierarchy,
    ConceptType,
    OntologyRelationship,
    RelationType,
    SemanticMapping,
)

logger = logging.getLogger(__name__)


@dataclass
class DomainConfiguration:
    """Configuration for a custom domain."""

    domain_name: str
    description: Optional[str] = None
    entity_types: Dict[str, List[str]] = None  # Custom entity type mappings
    extraction_patterns: Dict[str, List[str]] = None  # Custom extraction patterns
    synonyms: Dict[str, List[str]] = None  # Custom synonym mappings
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.entity_types is None:
            self.entity_types = {}
        if self.extraction_patterns is None:
            self.extraction_patterns = {}
        if self.synonyms is None:
            self.synonyms = {}
        if self.metadata is None:
            self.metadata = {}


class GeneralOntologyPlugin(OntologyLoader):
    """
    General-purpose ontology plugin for any domain.

    This plugin can work with ontologies from any domain without hardcoded
    assumptions. It automatically detects the domain, generates entity mappings,
    and supports custom domain configurations.
    """

    def __init__(
        self,
        ontology_path: str = "",
        domain_config: Optional[DomainConfiguration] = None,
    ):
        super().__init__(ontology_path)
        self.domain = "general"  # Default domain
        self.domain_config = domain_config

        # Dynamic entity mappings loaded from ontology or config
        self.entity_mappings: Dict[str, List[str]] = {}
        self.extraction_patterns: Dict[str, List[str]] = {}

        # Auto-detection settings
        self.auto_detect_domain_enabled = True
        self.confidence_threshold = 0.7

        # Initialize with custom domain if provided
        if self.domain_config:
            self._load_custom_domain(self.domain_config)

    def auto_detect_domain(self, ontology_data: Optional[Dict] = None) -> str:
        """
        Auto-detect domain from ontology metadata or content.

        Args:
            ontology_data: Optional ontology data to analyze

        Returns:
            Detected domain name
        """
        if not self.auto_detect_domain_enabled:
            return self.domain

        domain_indicators = {}

        # Analyze ontology metadata
        if self.metadata and self.metadata.name:
            domain_indicators.update(self._analyze_metadata_for_domain())

        # Analyze concept labels and descriptions
        if self.concepts:
            domain_indicators.update(self._analyze_concepts_for_domain())

        # Analyze external ontology data if provided
        if ontology_data:
            domain_indicators.update(
                self._analyze_ontology_data_for_domain(ontology_data)
            )

        # Determine most likely domain
        if domain_indicators:
            detected_domain = max(domain_indicators, key=domain_indicators.get)
            confidence = domain_indicators[detected_domain]

            if confidence >= self.confidence_threshold:
                logger.info(
                    f"Auto-detected domain: {detected_domain} (confidence: {confidence:.2f})"
                )
                self.domain = detected_domain
                return detected_domain

        logger.info(f"No clear domain detected, using default: {self.domain}")
        return self.domain

    def _analyze_metadata_for_domain(self) -> Dict[str, float]:
        """Analyze ontology metadata for domain indicators."""
        domain_keywords = {
            "medical": [
                "medical",
                "clinical",
                "health",
                "disease",
                "umls",
                "snomed",
                "icd",
            ],
            "legal": ["legal", "law", "court", "legislation", "statute", "regulation"],
            "financial": [
                "financial",
                "banking",
                "investment",
                "accounting",
                "finance",
            ],
            "scientific": ["scientific", "research", "experiment", "study", "journal"],
            "technical": ["technical", "engineering", "software", "system", "computer"],
            "business": [
                "business",
                "management",
                "corporate",
                "organization",
                "company",
            ],
            "educational": [
                "educational",
                "academic",
                "university",
                "learning",
                "course",
            ],
            "geographic": ["geographic", "location", "place", "region", "country"],
        }

        indicators = {}
        metadata_text = (
            f"{self.metadata.name} {self.metadata.description or ''} "
            f"{' '.join(self.metadata.authors)} {self.metadata.source_url or ''}"
        ).lower()

        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in metadata_text)
            if score > 0:
                indicators[domain] = score / len(keywords)

        return indicators

    def _analyze_concepts_for_domain(self) -> Dict[str, float]:
        """Analyze concept labels and descriptions for domain indicators."""
        domain_patterns = {
            "medical": [
                r"\b(?:disease|disorder|syndrome|symptom|treatment|therapy|drug|medication|clinical|patient|diagnosis)\b",
                r"\b(?:medical|health|hospital|doctor|physician|nurse|surgery|procedure)\b",
            ],
            "legal": [
                r"\b(?:law|legal|court|judge|attorney|lawyer|statute|regulation|contract|agreement)\b",
                r"\b(?:legislation|jurisdiction|litigation|verdict|ruling|evidence)\b",
            ],
            "financial": [
                r"\b(?:financial|finance|bank|investment|loan|credit|debt|asset|liability)\b",
                r"\b(?:accounting|audit|budget|revenue|profit|loss|market|stock)\b",
            ],
            "technical": [
                r"\b(?:software|hardware|system|computer|network|database|server|application)\b",
                r"\b(?:algorithm|programming|development|engineering|technology|digital)\b",
            ],
        }

        indicators = {}
        concept_text = " ".join(
            [
                f"{concept.label} {concept.description or ''} {' '.join(concept.get_all_synonyms())}"
                for concept in self.concepts.values()
            ]
        ).lower()

        for domain, patterns in domain_patterns.items():
            total_matches = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, concept_text))
                total_matches += matches

            if total_matches > 0:
                # Normalize by number of concepts and patterns
                indicators[domain] = total_matches / (
                    len(self.concepts) + len(patterns)
                )

        return indicators

    def _analyze_ontology_data_for_domain(
        self, ontology_data: Dict
    ) -> Dict[str, float]:
        """Analyze raw ontology data for domain indicators."""
        indicators = {}

        # Check for well-known ontology URIs
        known_ontologies = {
            "medical": ["umls", "snomed", "icd", "mesh", "rxnorm", "loinc"],
            "legal": ["lkif", "legal", "statute", "regulation"],
            "technical": ["swo", "edam", "schema.org"],
            "business": ["fibo", "org", "business", "enterprise"],
        }

        ontology_uri = ontology_data.get("ontology_uri", "").lower()
        for domain, uri_patterns in known_ontologies.items():
            matches = sum(1 for pattern in uri_patterns if pattern in ontology_uri)
            if matches > 0:
                indicators[domain] = matches / len(uri_patterns)

        return indicators

    def auto_generate_mappings(
        self, concepts: Optional[List[Concept]] = None
    ) -> Dict[str, List[str]]:
        """
        Automatically generate entity mappings from ontology concepts.

        Args:
            concepts: Optional list of concepts to analyze

        Returns:
            Generated entity type mappings
        """
        if concepts is None:
            concepts = list(self.concepts.values())

        mappings = {}

        # Group concepts by semantic similarity and type
        concept_groups = self._group_concepts_by_semantics(concepts)

        # Generate entity types from concept groups
        for group_name, group_concepts in concept_groups.items():
            entity_type = self._generate_entity_type_name(group_name, group_concepts)

            # Generate synonyms from concept labels and synonyms
            synonyms = set()
            for concept in group_concepts:
                synonyms.add(concept.label.lower())
                synonyms.update(concept.get_all_synonyms())

            mappings[entity_type] = list(synonyms)

        self.entity_mappings.update(mappings)
        logger.info(f"Auto-generated {len(mappings)} entity type mappings")

        return mappings

    def _group_concepts_by_semantics(
        self, concepts: List[Concept]
    ) -> Dict[str, List[Concept]]:
        """Group concepts by semantic similarity."""
        groups = {}

        # Simple grouping by concept type and hierarchical position
        for concept in concepts:
            # Use concept type as primary grouping
            group_key = concept.concept_type.value

            # Refine grouping by analyzing labels for common patterns
            label_tokens = set(concept.label.lower().split())

            # Look for semantic categories in labels
            semantic_categories = {
                "entity": ["person", "organization", "location", "place"],
                "concept": ["concept", "idea", "notion", "principle"],
                "process": ["process", "procedure", "method", "workflow"],
                "resource": ["resource", "material", "substance", "component"],
                "event": ["event", "occurrence", "incident", "activity"],
                "quality": ["quality", "attribute", "property", "characteristic"],
            }

            for category, keywords in semantic_categories.items():
                if any(keyword in label_tokens for keyword in keywords):
                    group_key = f"{group_key}_{category}"
                    break

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(concept)

        return groups

    def _generate_entity_type_name(
        self, group_name: str, concepts: List[Concept]
    ) -> str:
        """Generate a meaningful entity type name from concept group."""
        # Extract common words from concept labels
        all_words = []
        for concept in concepts:
            all_words.extend(concept.label.split())

        # Find most common meaningful words
        word_counts = {}
        for word in all_words:
            clean_word = re.sub(r"[^a-zA-Z]", "", word).upper()
            if len(clean_word) > 2:  # Skip short words
                word_counts[clean_word] = word_counts.get(clean_word, 0) + 1

        if word_counts:
            most_common_word = max(word_counts, key=word_counts.get)
            return most_common_word

        # Fallback to group name
        return group_name.upper().replace("_", "_TYPE_")

    def load_custom_domain(self, domain_definition: Dict):
        """
        Load a custom domain definition dynamically.

        Args:
            domain_definition: Dictionary containing domain configuration
        """
        domain_config = DomainConfiguration(**domain_definition)
        self._load_custom_domain(domain_config)

    def _load_custom_domain(self, domain_config: DomainConfiguration):
        """Load configuration from DomainConfiguration object."""
        self.domain = domain_config.domain_name
        self.domain_config = domain_config

        # Load custom entity mappings
        if domain_config.entity_types:
            self.entity_mappings.update(domain_config.entity_types)

        # Load custom extraction patterns
        if domain_config.extraction_patterns:
            self.extraction_patterns.update(domain_config.extraction_patterns)

        # Update metadata
        if domain_config.metadata:
            self.metadata.metadata.update(domain_config.metadata)

        logger.info(f"Loaded custom domain configuration: {domain_config.domain_name}")

    def load_ontology_from_file(
        self, filepath: str, ontology_format: str = "auto"
    ) -> ConceptHierarchy:
        """
        Load ontology from file with format auto-detection.

        Args:
            filepath: Path to ontology file
            ontology_format: Format hint ("owl", "rdf", "skos", "auto")

        Returns:
            Loaded concept hierarchy
        """
        if ontology_format == "auto":
            ontology_format = self._detect_format(filepath)

        # Choose appropriate loader
        if ontology_format.lower() == "owl":
            loader = OWLLoader(filepath)
        elif ontology_format.lower() == "rdf":
            loader = RDFLoader(filepath)
        elif ontology_format.lower() == "skos":
            loader = SKOSLoader(filepath)
        else:
            # Default to OWL loader
            loader = OWLLoader(filepath)

        # Load ontology
        hierarchy = loader.load(filepath)

        # Copy loaded data
        self.concepts = loader.concepts
        self.relationships = loader.relationships
        self.hierarchy = hierarchy
        self.metadata = loader.metadata

        # Auto-detect domain and generate mappings
        self.auto_detect_domain()
        self.auto_generate_mappings()

        logger.info(
            f"Loaded ontology from {filepath} with {len(self.concepts)} concepts"
        )
        return hierarchy

    def get_supported_formats(self) -> List[str]:
        """Return supported ontology file formats."""
        return ["owl", "rdf", "skos", "xml", "ttl", "n3"]

    def _detect_format(self, filepath: str) -> str:
        """Detect ontology format from file extension or content."""
        path = Path(filepath)
        extension = path.suffix.lower()

        format_mapping = {
            ".owl": "owl",
            ".rdf": "rdf",
            ".skos": "skos",
            ".xml": "owl",  # Default XML to OWL
            ".ttl": "rdf",
            ".n3": "rdf",
        }

        return format_mapping.get(extension, "owl")

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities using ontology-driven approach.

        Args:
            text: Text to extract entities from

        Returns:
            List of extracted entities
        """
        entities = []
        text_lower = text.lower()

        # Phase 1: Concept-based extraction
        for concept in self.concepts.values():
            # Check direct label matches
            if concept.label.lower() in text_lower:
                entities.append(
                    self._create_entity_from_concept(concept, text, concept.label)
                )

            # Check synonym matches
            for synonym in concept.get_all_synonyms():
                if synonym in text_lower:
                    entities.append(
                        self._create_entity_from_concept(concept, text, synonym)
                    )

        # Phase 2: Pattern-based extraction (if custom patterns available)
        if self.extraction_patterns:
            pattern_entities = self._extract_using_patterns(text)
            entities.extend(pattern_entities)

        # Remove duplicates and overlaps
        entities = self._deduplicate_entities(entities)

        return entities

    def _create_entity_from_concept(
        self, concept: Concept, text: str, matched_text: str
    ) -> Dict[str, Any]:
        """Create entity dictionary from matched concept."""
        # Find entity type based on concept
        entity_type = self._determine_entity_type(concept)

        # Find position in text
        start_pos = text.lower().find(matched_text.lower())

        return {
            "text": matched_text,
            "type": entity_type,
            "start": start_pos,
            "end": start_pos + len(matched_text),
            "confidence": 0.9,
            "method": "ontology_concept",
            "concept_id": concept.id,
            "concept_uri": concept.uri,
            "domain": self.domain,
        }

    def _determine_entity_type(self, concept: Concept) -> str:
        """Determine entity type for a concept."""
        # Check if concept maps to any defined entity types
        for entity_type, synonyms in self.entity_mappings.items():
            concept_labels = {concept.label.lower()} | concept.get_all_synonyms()
            if any(synonym in concept_labels for synonym in synonyms):
                return entity_type

        # Fallback to concept type or label-based type
        if hasattr(concept, "semantic_type") and concept.semantic_type:
            return concept.semantic_type.upper().replace(" ", "_")

        return concept.concept_type.value.upper()

    def _extract_using_patterns(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using custom extraction patterns."""
        entities = []

        for entity_type, patterns in self.extraction_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = {
                        "text": match.group(0),
                        "type": entity_type,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.7,
                        "method": "pattern_extraction",
                        "domain": self.domain,
                    }
                    entities.append(entity)

        return entities

    def _deduplicate_entities(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate and overlapping entities."""
        # Sort by start position
        entities.sort(key=lambda x: x["start"])

        deduplicated = []
        for entity in entities:
            # Check for overlaps with existing entities
            overlap = False
            for existing in deduplicated:
                if (
                    entity["start"] < existing["end"]
                    and entity["end"] > existing["start"]
                ):
                    # Keep entity with higher confidence
                    if entity["confidence"] > existing["confidence"]:
                        deduplicated.remove(existing)
                    else:
                        overlap = True
                    break

            if not overlap:
                deduplicated.append(entity)

        return deduplicated

    def _load_example_concepts(self, source: Optional[Any] = None) -> None:
        """Load a small set of example concepts for testing/performance baselines."""
        if not hasattr(self, "hierarchy") or self.hierarchy is None:
            self.hierarchy = ConceptHierarchy()
        if not hasattr(self, "concepts") or self.concepts is None:
            self.concepts = {}
        disease = Concept(
            label="Disease", description="A medical condition", domain="medical"
        )
        treatment = Concept(
            label="Treatment", description="An intervention", domain="medical"
        )
        treatment.add_synonym("therapy")
        self.hierarchy.add_concept(disease)
        self.hierarchy.add_concept(treatment)
        self.concepts[disease.id] = disease
        self.concepts[treatment.id] = treatment
        self.hierarchy.add_relationship(
            disease.id, treatment.id, RelationType.RELATED_TO
        )
        self.auto_generate_mappings([disease, treatment])

    def load(self, filepath: str):
        """Load ontology from file (implements abstract method)."""
        return self.load_ontology_from_file(filepath)

    def _parse_file(self, filepath: str):
        """Parse ontology file (implements abstract method)."""
        # Delegate to format-specific loader
        format_type = self._detect_format(filepath)

        if format_type == "owl":
            loader = OWLLoader(filepath)
        elif format_type == "rdf":
            loader = RDFLoader(filepath)
        elif format_type == "skos":
            loader = SKOSLoader(filepath)
        else:
            loader = OWLLoader(filepath)

        return loader._parse_file(filepath)
