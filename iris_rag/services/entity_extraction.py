"""
Domain-Agnostic Entity Extraction Service for IRIS RAG Framework

Enhanced entity extraction service that works with ANY ontology domain without
hardcoded assumptions. Uses the general-purpose ontology plugin system for
improved accuracy and semantic understanding.

Key Features:
- Universal ontology support (medical, legal, financial, technical, etc.)
- Two-phase extraction (rule-based + ontology mapping)
- Auto-detection of domain from ontology content
- Dynamic entity type generation from ontology concepts
- Backward compatibility with existing entity extraction patterns
- Integration with NetworkX algorithms via IRIS globals (no parallel stores)
"""

import logging
import re
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

from ..core.models import Document, Entity, Relationship, EntityTypes, RelationshipTypes
from ..config.manager import ConfigurationManager
from ..core.connection import ConnectionManager
from ..embeddings.manager import EmbeddingManager
from .storage import EntityStorageAdapter

# Import general-purpose ontology components
from ..ontology.plugins import (
    get_ontology_plugin,
    create_plugin_from_config,
    GeneralOntologyPlugin,
    DomainConfiguration
)
from ..ontology.reasoner import OntologyReasoner
from ..ontology.models import Concept, OntologyRelationship

logger = logging.getLogger(__name__)


class OntologyAwareEntityExtractor:
    """
    Universal entity extraction with ontology support for ANY domain.
    
    Provides two-phase extraction (rule-based + ontology mapping) and semantic
    enrichment that works with any ontology format and domain.
    """
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        connection_manager: Optional[ConnectionManager] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        ontology_sources: Optional[List[Dict[str, Any]]] = None
    ):
        """Initialize the domain-agnostic entity extraction service."""
        self.config_manager = config_manager
        self.connection_manager = connection_manager
        self.embedding_manager = embedding_manager
        
        # Load configuration
        self.config = self.config_manager.get("entity_extraction", {})
        self.ontology_config = self.config_manager.get("ontology", {})
        
        # Extraction configuration
        self.method = self.config.get("method", "ontology_hybrid")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.enabled_types = set(self.config.get("entity_types", ["ENTITY", "CONCEPT", "PROCESS"]))
        self.max_entities_per_doc = self.config.get("max_entities", 100)
        
        # Ontology configuration
        self.ontology_enabled = self.ontology_config.get("enabled", True)
        self.reasoning_enabled = self.ontology_config.get("reasoning", {}).get("enable_inference", True)
        self.auto_detect_domain = self.ontology_config.get("auto_detect_domain", True)
        
        # Initialize ontology plugin
        self.ontology_plugin = None
        self.reasoner = None
        if self.ontology_enabled:
            self._init_ontology_plugin(ontology_sources)
        
        # Initialize storage adapter
        self.storage_adapter = None
        if self.connection_manager:
            self.storage_adapter = EntityStorageAdapter(
                self.connection_manager,
                self.config_manager._config
            )
        
        # Initialize fallback patterns for backward compatibility
        self._init_patterns()
        
        logger.info(f"OntologyAwareEntityExtractor initialized with domain-agnostic ontology support")
    
    def _init_ontology_plugin(self, ontology_sources: Optional[List[Dict[str, Any]]] = None) -> None:
        """Initialize the general-purpose ontology plugin."""
        try:
            if ontology_sources:
                # Create plugin from provided sources
                self.ontology_plugin = create_plugin_from_config({
                    "auto_detect_domain": self.auto_detect_domain,
                    "sources": ontology_sources
                })
            elif self.ontology_config.get("sources"):
                # Create plugin from configuration
                self.ontology_plugin = create_plugin_from_config(self.ontology_config)
            else:
                # Create empty plugin that can be loaded dynamically
                self.ontology_plugin = GeneralOntologyPlugin()
                self.ontology_plugin.auto_detect_domain = self.auto_detect_domain
            
            # Initialize reasoner if reasoning is enabled
            if self.reasoning_enabled and self.ontology_plugin and self.ontology_plugin.concepts:
                self.reasoner = OntologyReasoner(self.ontology_plugin.hierarchy)
                logger.info("Ontology reasoner initialized")
            
            if self.ontology_plugin:
                detected_domain = self.ontology_plugin.domain
                concept_count = len(self.ontology_plugin.concepts)
                logger.info(f"Loaded ontology plugin: domain={detected_domain}, concepts={concept_count}")
        
        except Exception as e:
            logger.error(f"Failed to initialize ontology plugin: {e}")
            self.ontology_plugin = None
    
    def load_ontology_from_file(self, filepath: str, ontology_format: str = "auto") -> bool:
        """
        Load ontology from file dynamically.
        
        Args:
            filepath: Path to ontology file
            ontology_format: Format of ontology ("owl", "rdf", "skos", "auto")
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not self.ontology_plugin:
                self.ontology_plugin = GeneralOntologyPlugin()
            
            hierarchy = self.ontology_plugin.load_ontology_from_file(filepath, ontology_format)
            
            # Update reasoner if reasoning is enabled
            if self.reasoning_enabled and hierarchy:
                self.reasoner = OntologyReasoner(hierarchy)
            
            # Update entity types based on loaded ontology
            self._update_entity_types_from_ontology()
            
            logger.info(f"Successfully loaded ontology from {filepath}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load ontology from {filepath}: {e}")
            return False
    
    def _update_entity_types_from_ontology(self):
        """Update enabled entity types based on loaded ontology."""
        if not self.ontology_plugin or not self.ontology_plugin.entity_mappings:
            return
        
        # Add ontology-derived entity types to enabled types
        ontology_types = set(self.ontology_plugin.entity_mappings.keys())
        self.enabled_types.update(ontology_types)
        
        logger.debug(f"Updated entity types from ontology: {ontology_types}")
    
    def extract_with_ontology(self, text: str, document: Optional[Document] = None) -> List[Entity]:
        """
        Extract entities using ontology-enhanced approach for any domain.
        
        Phase 1: Ontology-based concept extraction
        Phase 2: Pattern-based extraction (if available)
        Phase 3: Semantic enrichment and reasoning
        """
        if not self.ontology_enabled or not self.ontology_plugin:
            # Fallback to basic extraction
            return self.extract_entities_basic(text, document)
        
        all_entities = []
        
        # Phase 1: Extract entities using ontology concepts
        ontology_entities = self._extract_ontology_entities(text, document)
        all_entities.extend(ontology_entities)
        
        # Phase 2: Extract entities using custom patterns (if available)
        if self.ontology_plugin.extraction_patterns:
            pattern_entities = self._extract_pattern_entities(text, document)
            all_entities.extend(pattern_entities)
        
        # Phase 3: Apply reasoning if enabled
        if self.reasoning_enabled and self.reasoner:
            enriched_entities = self._apply_ontology_reasoning(all_entities, text)
            all_entities = enriched_entities
        
        # Remove duplicates and apply filtering
        merged_entities = self._merge_and_deduplicate_entities(all_entities)
        
        # Apply confidence filtering and limit results
        filtered_entities = [
            e for e in merged_entities
            if e.confidence >= self.confidence_threshold
        ][:self.max_entities_per_doc]
        
        logger.debug(f"Extracted {len(filtered_entities)} entities from {len(all_entities)} candidates")
        return filtered_entities
    
    def _extract_ontology_entities(self, text: str, document: Optional[Document] = None) -> List[Entity]:
        """Extract entities using ontology concepts."""
        entities = []
        
        if not self.ontology_plugin:
            return entities
        
        try:
            # Use the general plugin's extract_entities method
            raw_entities = self.ontology_plugin.extract_entities(text)
            
            # Convert to Entity objects
            for raw_entity in raw_entities:
                entity = self._convert_to_entity(raw_entity, document)
                if entity:
                    entities.append(entity)
        
        except Exception as e:
            logger.error(f"Error extracting ontology entities: {e}")
        
        return entities
    
    def _extract_pattern_entities(self, text: str, document: Optional[Document] = None) -> List[Entity]:
        """Extract entities using custom extraction patterns."""
        entities = []
        
        if not self.ontology_plugin or not self.ontology_plugin.extraction_patterns:
            return entities
        
        try:
            for entity_type, patterns in self.ontology_plugin.extraction_patterns.items():
                for pattern in patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        raw_entity = {
                            "text": match.group(0),
                            "type": entity_type,
                            "start": match.start(),
                            "end": match.end(),
                            "confidence": 0.7,
                            "method": "pattern_extraction"
                        }
                        
                        entity = self._convert_to_entity(raw_entity, document)
                        if entity:
                            entities.append(entity)
        
        except Exception as e:
            logger.error(f"Error extracting pattern entities: {e}")
        
        return entities
    
    def _convert_to_entity(self, raw_entity: Dict[str, Any], document: Optional[Document]) -> Optional[Entity]:
        """Convert raw entity extraction result to Entity object."""
        try:
            entity = Entity(
                text=raw_entity.get("text", ""),
                entity_type=raw_entity.get("type", "UNKNOWN"),
                confidence=raw_entity.get("confidence", 0.5),
                start_offset=raw_entity.get("start", 0),
                end_offset=raw_entity.get("end", 0),
                source_document_id=document.id if document else None,
                metadata={
                    "method": raw_entity.get("method", "ontology"),
                    "domain": getattr(self.ontology_plugin, 'domain', 'general') if self.ontology_plugin else 'general',
                    "concept_id": raw_entity.get("concept_id"),
                    "concept_uri": raw_entity.get("concept_uri"),
                    "ontology_metadata": {
                        k: v for k, v in raw_entity.items()
                        if k not in ["text", "type", "confidence", "start", "end", "method"]
                    }
                }
            )
            return entity
        except Exception as e:
            logger.error(f"Failed to convert raw entity to Entity object: {e}")
            return None
    
    def _merge_and_deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Merge entities and remove duplicates."""
        # Group overlapping entities
        grouped_entities = []
        entities_sorted = sorted(entities, key=lambda e: (e.start_offset, e.end_offset))
        
        for entity in entities_sorted:
            # Find overlapping groups
            overlapping_group = None
            for group in grouped_entities:
                for existing in group:
                    if self._entities_overlap(entity, existing):
                        overlapping_group = group
                        break
                if overlapping_group:
                    break
            
            if overlapping_group:
                overlapping_group.append(entity)
            else:
                grouped_entities.append([entity])
        
        # Select best entity from each group
        merged_entities = []
        for group in grouped_entities:
            if len(group) == 1:
                merged_entities.append(group[0])
            else:
                # Choose entity with highest confidence
                best_entity = max(group, key=lambda e: e.confidence)
                
                # Merge metadata from all entities in group
                merged_metadata = best_entity.metadata.copy()
                merged_metadata["merged_entities_count"] = len(group)
                best_entity.metadata = merged_metadata
                
                merged_entities.append(best_entity)
        
        return merged_entities
    
    def _entities_overlap(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities overlap significantly."""
        start1, end1 = entity1.start_offset, entity1.end_offset
        start2, end2 = entity2.start_offset, entity2.end_offset
        
        # Calculate overlap
        overlap_start = max(start1, start2)
        overlap_end = min(end1, end2)
        
        if overlap_end <= overlap_start:
            return False  # No overlap
        
        overlap_length = overlap_end - overlap_start
        min_length = min(end1 - start1, end2 - start2)
        
        # Consider overlapping if > 50% of shorter entity overlaps
        return overlap_length / min_length > 0.5
    
    def _apply_ontology_reasoning(self, entities: List[Entity], text: str) -> List[Entity]:
        """Apply ontology-based reasoning to enrich entities."""
        if not self.reasoner:
            return entities
        
        enriched_entities = entities.copy()
        
        try:
            # Enrich each entity with semantic information
            for entity in enriched_entities:
                # Find related concepts through reasoning
                related_concepts = self._find_related_concepts(entity)
                if related_concepts:
                    entity.metadata["inferred_relations"] = related_concepts
                
                # Enhance with semantic similarity
                semantic_info = self._get_semantic_enrichment(entity)
                if semantic_info:
                    entity.metadata["semantic_enrichment"] = semantic_info
        
        except Exception as e:
            logger.error(f"Error in ontology reasoning: {e}")
        
        return enriched_entities
    
    def _find_related_concepts(self, entity: Entity) -> List[Dict[str, Any]]:
        """Find concepts related to an entity through ontology reasoning."""
        concept_id = entity.metadata.get("concept_id")
        if not concept_id or not self.reasoner:
            return []
        
        related = []
        try:
            # Get hierarchical relationships
            ancestors = self.reasoner.hierarchy.get_ancestors(concept_id, max_depth=3)
            descendants = self.reasoner.hierarchy.get_descendants(concept_id, max_depth=2)
            
            for ancestor_id in ancestors:
                ancestor_concept = self.reasoner.hierarchy.concepts.get(ancestor_id)
                if ancestor_concept:
                    related.append({
                        "concept_id": ancestor_id,
                        "label": ancestor_concept.label,
                        "relationship": "ancestor",
                        "confidence": 0.8
                    })
            
            for descendant_id in descendants:
                descendant_concept = self.reasoner.hierarchy.concepts.get(descendant_id)
                if descendant_concept:
                    related.append({
                        "concept_id": descendant_id,
                        "label": descendant_concept.label,
                        "relationship": "descendant",
                        "confidence": 0.7
                    })
        
        except Exception as e:
            logger.error(f"Error finding related concepts: {e}")
        
        return related[:10]  # Limit to top 10 related concepts
    
    def _get_semantic_enrichment(self, entity: Entity) -> Dict[str, Any]:
        """Get semantic enrichment information for an entity."""
        concept_id = entity.metadata.get("concept_id")
        if not concept_id or not self.reasoner:
            return {}
        
        enrichment = {}
        try:
            concept = self.reasoner.hierarchy.concepts.get(concept_id)
            if concept:
                # Add synonyms and alternative labels
                enrichment["synonyms"] = list(concept.get_all_synonyms())[:5]
                enrichment["description"] = concept.description
                enrichment["domain_metadata"] = concept.metadata
                
                # Add external identifiers if available
                if concept.external_ids:
                    enrichment["external_ids"] = concept.external_ids
        
        except Exception as e:
            logger.error(f"Error getting semantic enrichment: {e}")
        
        return enrichment


class EntityExtractionService(OntologyAwareEntityExtractor):
    """
    Backward-compatible entity extraction service.
    
    Maintains compatibility with existing code while providing enhanced
    domain-agnostic ontology capabilities when enabled.
    """
    
    def __init__(
        self,
        config_manager: ConfigurationManager,
        connection_manager: Optional[ConnectionManager] = None,
        embedding_manager: Optional[EmbeddingManager] = None
    ):
        """Initialize with backward compatibility."""
        # Check if ontology features should be enabled
        ontology_config = config_manager.get("ontology", {})
        ontology_enabled = ontology_config.get("enabled", False)
        
        if ontology_enabled:
            # Use enhanced ontology-aware extraction
            super().__init__(config_manager, connection_manager, embedding_manager)
        else:
            # Use basic extraction for backward compatibility
            self.config_manager = config_manager
            self.connection_manager = connection_manager
            self.embedding_manager = embedding_manager
            
            # Load basic configuration
            self.config = self.config_manager.get("entity_extraction", {})
            self.method = self.config.get("method", "llm_basic")
            self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
            self.enabled_types = set(self.config.get("entity_types", ["PERSON", "DISEASE", "DRUG"]))
            
            # Initialize storage adapter
            self.storage_adapter = None
            if self.connection_manager:
                self.storage_adapter = EntityStorageAdapter(
                    self.connection_manager,
                    self.config_manager._config
                )
            
            # Initialize basic patterns
            self._init_patterns()
            
            logger.info(f"EntityExtractionService initialized with basic method: {self.method}")
    
    def extract_entities(self, document: Document) -> List[Entity]:
        """Extract entities with automatic ontology enhancement if enabled."""
        if hasattr(self, 'ontology_enabled') and self.ontology_enabled:
            # Use ontology-aware extraction
            return self.extract_with_ontology(document.page_content, document)
        else:
            # Use basic extraction for backward compatibility
            return self.extract_entities_basic(document.page_content, document)
    
    def extract_entities_basic(self, text: str, document: Optional[Document] = None) -> List[Entity]:
        """Basic entity extraction for backward compatibility."""
        try:
            if self.method == "llm_basic":
                return self._extract_llm(text, document)
            elif self.method == "pattern_only":
                return self._extract_patterns(text, document)
            elif self.method == "hybrid":
                llm_entities = self._extract_llm(text, document)
                pattern_entities = self._extract_patterns(text, document)
                return self._merge_entities(llm_entities, pattern_entities)
            else:
                logger.warning(f"Unknown method {self.method}, using LLM")
                return self._extract_llm(text, document)
        except Exception as e:
            logger.error(f"Basic entity extraction failed: {e}")
            return []
    
    def _init_patterns(self):
        """Initialize basic regex patterns for entity extraction."""
        self.patterns = {
            EntityTypes.DRUG: [
                r'\b[A-Z][a-z]+(?:ine|ide|ate|cin|pam|zole|pril|sartan)\b',
                r'\b(?:aspirin|ibuprofen|acetaminophen|morphine|insulin)\b'
            ],
            EntityTypes.DISEASE: [
                r'\b(?:diabetes|cancer|hypertension|pneumonia|arthritis)\b',
                r'\b[A-Z][a-z]+(?:itis|osis|emia|pathy)\b'
            ],
            EntityTypes.GENE: [
                r'\b[A-Z]{2,}[0-9]+\b',  # TP53, BRCA1
                r'\b(?:gene|protein)\s+([A-Z][A-Z0-9]+)\b'
            ]
        }
    
    def _extract_llm(self, text: str, document: Optional[Document] = None) -> List[Entity]:
        """Extract entities using LLM with simple prompt."""
        # This is a simplified LLM extraction - in production would use actual LLM
        entities = []
        try:
            prompt = self._build_prompt(text)
            response = self._call_llm(prompt)
            entities = self._parse_llm_response(response, document)
            
            # Filter by confidence and enabled types
            filtered = [
                e for e in entities
                if e.confidence >= self.confidence_threshold and e.entity_type in self.enabled_types
            ]
            return filtered[:self.max_entities_per_doc]
        
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []
    
    def _extract_patterns(self, text: str, document: Optional[Document] = None) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            if entity_type not in self.enabled_types:
                continue
            
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = Entity(
                        text=match.group(0),
                        entity_type=entity_type,
                        confidence=0.8,
                        start_offset=match.start(),
                        end_offset=match.end(),
                        source_document_id=document.id if document else None,
                        metadata={"method": "pattern"}
                    )
                    entities.append(entity)
        
        return entities
    
    def _build_prompt(self, text: str) -> str:
        """Build prompt for LLM entity extraction."""
        enabled_types_str = ", ".join(self.enabled_types)
        return f"""
        Extract entities of types: {enabled_types_str}
        
        Text: {text[:2000]}
        
        Return JSON array with entities:
        [{{"text": "entity text", "type": "ENTITY_TYPE", "confidence": 0.9}}]
        """
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM for entity extraction."""
        # Simplified mock implementation
        # In production, this would call actual LLM service
        return '[{"text": "example", "type": "ENTITY", "confidence": 0.8}]'
    
    def _parse_llm_response(self, response: str, document: Optional[Document]) -> List[Entity]:
        """Parse LLM JSON response into Entity objects."""
        entities = []
        try:
            raw_entities = json.loads(response)
            for raw_entity in raw_entities:
                entity = Entity(
                    text=raw_entity.get("text", ""),
                    entity_type=raw_entity.get("type", "UNKNOWN"),
                    confidence=raw_entity.get("confidence", 0.5),
                    start_offset=0,  # Would need text search for actual position
                    end_offset=len(raw_entity.get("text", "")),
                    source_document_id=document.id if document else None,
                    metadata={"method": "llm"}
                )
                entities.append(entity)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
        
        return entities
    
    def _merge_entities(self, llm_entities: List[Entity], pattern_entities: List[Entity]) -> List[Entity]:
        """Merge entities from different extraction methods."""
        all_entities = llm_entities + pattern_entities
        
        # Simple deduplication based on text and type
        seen = set()
        merged = []
        
        for entity in all_entities:
            key = (entity.text.lower(), entity.entity_type)
            if key not in seen:
                seen.add(key)
                merged.append(entity)
        
        return merged
    
    def extract_relationships(self, entities: List[Entity], document: Document) -> List[Relationship]:
        """Extract relationships between entities."""
        relationships = []
        
        # Simple co-occurrence based relationship extraction
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Calculate distance between entities
                distance = abs(entity1.start_offset - entity2.start_offset)
                
                # Create relationship if entities are close
                if distance < 200:  # Within 200 characters
                    relationship = Relationship(
                        source_entity_id=entity1.id,
                        target_entity_id=entity2.id,
                        relationship_type=RelationshipTypes.RELATED_TO,
                        confidence=0.6,
                        source_document_id=document.id,
                        metadata={
                            "method": "co_occurrence",
                            "distance": distance
                        }
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def store_entities_and_relationships(
        self,
        entities: List[Entity],
        relationships: List[Relationship]
    ) -> Dict[str, Any]:
        """Store entities and relationships using storage adapter."""
        if not self.storage_adapter:
            logger.warning("No storage adapter available")
            return {"stored_entities": 0, "stored_relationships": 0}
        
        try:
            # Store entities
            stored_entities = 0
            for entity in entities:
                success = self.storage_adapter.store_entity(entity)
                if success:
                    stored_entities += 1
            
            # Store relationships
            stored_relationships = 0
            for relationship in relationships:
                success = self.storage_adapter.store_relationship(relationship)
                if success:
                    stored_relationships += 1
            
            return {
                "stored_entities": stored_entities,
                "stored_relationships": stored_relationships
            }
        
        except Exception as e:
            logger.error(f"Failed to store entities/relationships: {e}")
            return {"stored_entities": 0, "stored_relationships": 0, "error": str(e)}
    
    def process_document(self, document: Document) -> Dict[str, Any]:
        """
        Complete document processing: extract entities, relationships, and store.
        
        Returns:
            Processing results including counts and any errors
        """
        results = {
            "document_id": document.id,
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "entities_stored": 0,
            "relationships_stored": 0,
            "errors": []
        }
        
        try:
            # Extract entities
            entities = self.extract_entities(document)
            results["entities_extracted"] = len(entities)
            
            # Extract relationships
            relationships = self.extract_relationships(entities, document)
            results["relationships_extracted"] = len(relationships)
            
            # Store results
            if self.storage_adapter:
                try:
                    storage_results = self.store_entities_and_relationships(entities, relationships)
                    results["entities_stored"] = storage_results.get("stored_entities", 0)
                    results["relationships_stored"] = storage_results.get("stored_relationships", 0)
                    
                    if "error" in storage_results:
                        results["errors"].append(f"Storage error: {storage_results['error']}")
                
                except Exception as e:
                    results["errors"].append(f"Storage failed: {e}")
            
            logger.info(f"Processed document {document.id}: {results}")
            
        except Exception as e:
            error_msg = f"Document processing failed: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
        
        return results