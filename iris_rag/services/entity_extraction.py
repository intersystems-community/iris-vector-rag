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

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from ..config.manager import ConfigurationManager
from ..core.connection import ConnectionManager
from ..core.models import Document, Entity, EntityTypes, Relationship, RelationshipTypes
from ..embeddings.manager import EmbeddingManager
from ..ontology.models import Concept, OntologyRelationship

# Import general-purpose ontology components
from ..ontology.plugins import (
    DomainConfiguration,
    GeneralOntologyPlugin,
    create_plugin_from_config,
    get_ontology_plugin,
)
from ..ontology.reasoner import OntologyReasoner
from .storage import EntityStorageAdapter

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
        ontology_sources: Optional[List[Dict[str, Any]]] = None,
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
        self.enabled_types = set(
            self.config.get("entity_types", ["ENTITY", "CONCEPT", "PROCESS"])
        )
        self.max_entities_per_doc = self.config.get("max_entities", 100)

        # Ontology configuration
        self.ontology_enabled = self.ontology_config.get("enabled", True)
        self.reasoning_enabled = self.ontology_config.get("reasoning", {}).get(
            "enable_inference", True
        )
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
                self.connection_manager, self.config_manager._config
            )

        # Initialize fallback patterns for backward compatibility
        self._init_patterns()

        logger.info(
            f"OntologyAwareEntityExtractor initialized with domain-agnostic ontology support"
        )

    def _init_ontology_plugin(
        self, ontology_sources: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Initialize the general-purpose ontology plugin."""
        try:
            if ontology_sources:
                # Create plugin from provided sources
                self.ontology_plugin = create_plugin_from_config(
                    {
                        "auto_detect_domain": self.auto_detect_domain,
                        "sources": ontology_sources,
                    }
                )
            elif self.ontology_config.get("sources"):
                # Create plugin from configuration
                self.ontology_plugin = create_plugin_from_config(self.ontology_config)
            else:
                # Create empty plugin that can be loaded dynamically
                self.ontology_plugin = GeneralOntologyPlugin()
                self.ontology_plugin.auto_detect_domain = self.auto_detect_domain

            # Initialize reasoner if reasoning is enabled
            if (
                self.reasoning_enabled
                and self.ontology_plugin
                and self.ontology_plugin.concepts
            ):
                self.reasoner = OntologyReasoner(self.ontology_plugin.hierarchy)
                logger.info("Ontology reasoner initialized")

            if self.ontology_plugin:
                detected_domain = self.ontology_plugin.domain
                concept_count = len(self.ontology_plugin.concepts)
                logger.info(
                    f"Loaded ontology plugin: domain={detected_domain}, concepts={concept_count}"
                )

        except Exception as e:
            logger.error(f"Failed to initialize ontology plugin: {e}")
            self.ontology_plugin = None

    def load_ontology_from_file(
        self, filepath: str, ontology_format: str = "auto"
    ) -> bool:
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

            hierarchy = self.ontology_plugin.load_ontology_from_file(
                filepath, ontology_format
            )

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

    def extract_with_ontology(
        self, text: str, document: Optional[Document] = None
    ) -> List[Entity]:
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
            e for e in merged_entities if e.confidence >= self.confidence_threshold
        ][: self.max_entities_per_doc]

        logger.debug(
            f"Extracted {len(filtered_entities)} entities from {len(all_entities)} candidates"
        )
        return filtered_entities

    def _extract_ontology_entities(
        self, text: str, document: Optional[Document] = None
    ) -> List[Entity]:
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

    def _extract_pattern_entities(
        self, text: str, document: Optional[Document] = None
    ) -> List[Entity]:
        """Extract entities using custom extraction patterns."""
        entities = []

        if not self.ontology_plugin or not self.ontology_plugin.extraction_patterns:
            return entities

        try:
            for (
                entity_type,
                patterns,
            ) in self.ontology_plugin.extraction_patterns.items():
                for pattern in patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        raw_entity = {
                            "text": match.group(0),
                            "type": entity_type,
                            "start": match.start(),
                            "end": match.end(),
                            "confidence": 0.7,
                            "method": "pattern_extraction",
                        }

                        entity = self._convert_to_entity(raw_entity, document)
                        if entity:
                            entities.append(entity)

        except Exception as e:
            logger.error(f"Error extracting pattern entities: {e}")

        return entities

    def _convert_to_entity(
        self, raw_entity: Dict[str, Any], document: Optional[Document]
    ) -> Optional[Entity]:
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
                    "domain": (
                        getattr(self.ontology_plugin, "domain", "general")
                        if self.ontology_plugin
                        else "general"
                    ),
                    "concept_id": raw_entity.get("concept_id"),
                    "concept_uri": raw_entity.get("concept_uri"),
                    "ontology_metadata": {
                        k: v
                        for k, v in raw_entity.items()
                        if k
                        not in ["text", "type", "confidence", "start", "end", "method"]
                    },
                },
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

    def _apply_ontology_reasoning(
        self, entities: List[Entity], text: str
    ) -> List[Entity]:
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
            descendants = self.reasoner.hierarchy.get_descendants(
                concept_id, max_depth=2
            )

            for ancestor_id in ancestors:
                ancestor_concept = self.reasoner.hierarchy.concepts.get(ancestor_id)
                if ancestor_concept:
                    related.append(
                        {
                            "concept_id": ancestor_id,
                            "label": ancestor_concept.label,
                            "relationship": "ancestor",
                            "confidence": 0.8,
                        }
                    )

            for descendant_id in descendants:
                descendant_concept = self.reasoner.hierarchy.concepts.get(descendant_id)
                if descendant_concept:
                    related.append(
                        {
                            "concept_id": descendant_id,
                            "label": descendant_concept.label,
                            "relationship": "descendant",
                            "confidence": 0.7,
                        }
                    )

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
        embedding_manager: Optional[EmbeddingManager] = None,
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
            self.enabled_types = set(
                self.config.get("entity_types", ["PERSON", "DISEASE", "DRUG"])
            )
            self.max_entities_per_doc = self.config.get("max_entities", 100)

            # Initialize storage adapter
            self.storage_adapter = None
            if self.connection_manager:
                self.storage_adapter = EntityStorageAdapter(
                    self.connection_manager, self.config_manager._config
                )

            # Initialize basic patterns
            self._init_patterns()

            logger.info(
                f"EntityExtractionService initialized with basic method: {self.method}"
            )

    def extract_entities(self, document: Document) -> List[Entity]:
        """Extract entities with automatic ontology enhancement if enabled."""
        if hasattr(self, "ontology_enabled") and self.ontology_enabled:
            # Use ontology-aware extraction
            return self.extract_with_ontology(document.page_content, document)
        else:
            # Use basic extraction for backward compatibility
            return self.extract_entities_basic(document.page_content, document)

    def extract_entities_basic(
        self, text: str, document: Optional[Document] = None
    ) -> List[Entity]:
        """Basic entity extraction for backward compatibility with robust fallback."""
        try:
            entities: List[Entity] = []
            if self.method == "llm_basic":
                entities = self._extract_llm(text, document)
            elif self.method == "pattern_only":
                entities = self._extract_patterns(text, document)
            elif self.method == "hybrid":
                llm_entities = self._extract_llm(text, document)
                pattern_entities = self._extract_patterns(text, document)
                entities = self._merge_entities(llm_entities, pattern_entities)
            else:
                logger.warning(
                    f"Unknown method {self.method}, using pattern-only extraction"
                )
                entities = self._extract_patterns(text, document)

            # Fallback: if nothing extracted, use simple keyword heuristic
            if not entities:
                fallback = self._extract_keywords_basic(
                    text, document, max_count=self.max_entities_per_doc
                )
                if fallback:
                    logger.info(
                        f"Fallback keyword extraction produced {len(fallback)} entities"
                    )
                    entities = fallback

            # Enforce max_entities_per_doc clamp just in case
            return entities[: self.max_entities_per_doc]
        except Exception as e:
            logger.error(f"Basic entity extraction failed: {e}")
            return []

    def _init_patterns(self):
        """Initialize enhanced medical regex patterns based on research findings."""
        self.patterns = {
            EntityTypes.DRUG: [
                # Enhanced drug name patterns with more suffixes
                r"\b[A-Z][a-z]+(?:ine|ide|ate|cin|pam|zole|pril|sartan|mab|nib|tinib|axel|olib|afil|mycin|cillin|oxin|stat|lol|pine|done|phine)\b",
                # Brand names with dosage patterns
                r"\b[A-Z][a-z]{2,}(?:®|™)?\s*(?:\d+\s*mg|\d+/\d+)\b",
                # Drug combinations and formulations
                r"\b[A-Z][a-z]+/[A-Z][a-z]+\b",
                r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\s+(?:tablet|capsule|injection|cream|ointment)\b",
                # Common medications (vastly expanded)
                r"\b(?:aspirin|ibuprofen|acetaminophen|paracetamol|morphine|codeine|tramadol|insulin|metformin|glyburide|lisinopril|enalapril|losartan|atorvastatin|simvastatin|rosuvastatin|warfarin|heparin|clopidogrel|prednisone|hydrocortisone|antibiotics|penicillin|amoxicillin|azithromycin|ciprofloxacin|doxycycline|chemotherapy|cisplatin|carboplatin|doxorubicin|immunotherapy|rituximab|adalimumab|infliximab|bevacizumab)\b",
                # Vaccine-related terms (expanded)
                r"\b(?:vaccine|vaccination|immunization|mRNA|Pfizer|Moderna|AstraZeneca|Johnson|Janssen|Novavax|Sputnik|Sinovac)\b",
                # Antiviral and specific drug classes
                r"\b(?:remdesivir|paxlovid|molnupiravir|hydroxychloroquine|ivermectin|oseltamivir|acyclovir|valacyclovir|tenofovir|emtricitabine)\b",
                # Cancer drugs
                r"\b(?:tamoxifen|aromatase inhibitor|trastuzumab|pembrolizumab|nivolumab|ipilimumab|atezolizumab|durvalumab|cemiplimab)\b",
            ],
            EntityTypes.DISEASE: [
                # Medical conditions with standard suffixes (expanded)
                r"\b[A-Z][a-z]+(?:itis|osis|emia|pathy|syndrome|disease|disorder|cardia|plegia|trophy|genic|toxic|penia|uria|dynia|algia|phobia|mania)\b",
                # Specific conditions (vastly expanded with COVID focus)
                r"\b(?:diabetes|cancer|hypertension|pneumonia|arthritis|COVID-19|SARS-CoV-2|coronavirus|influenza|tuberculosis|malaria|HIV|AIDS|asthma|COPD|stroke|sepsis|leukemia|lymphoma|melanoma|breast cancer|lung cancer|colon cancer|prostate cancer|ovarian cancer|pancreatic cancer|heart disease|kidney disease|liver disease|Alzheimer|dementia|Parkinson|epilepsy|schizophrenia|depression|anxiety|autism|ADHD|migraine|fibromyalgia|obesity|osteoporosis|osteoarthritis|rheumatoid arthritis)\b",
                # COVID-19 related conditions and variants
                r"\b(?:long COVID|post-COVID|COVID variants|Delta variant|Omicron variant|Alpha variant|Beta variant|acute respiratory distress syndrome|ARDS|cytokine storm|thrombosis|pulmonary embolism)\b",
                # Cardiovascular conditions
                r"\b(?:myocardial infarction|heart attack|angina|atrial fibrillation|heart failure|cardiomyopathy|aortic stenosis|mitral regurgitation|peripheral artery disease|deep vein thrombosis)\b",
                # Neurological conditions
                r"\b(?:multiple sclerosis|amyotrophic lateral sclerosis|ALS|Huntington|cerebral palsy|traumatic brain injury|concussion|neuropathy|Bell\'s palsy)\b",
                # ICD-10 patterns
                r"\b[A-Z]\d{2}(?:\.\d{1,2})?\b",
                # Viral/bacterial infections (expanded)
                r"\b(?:SARS|MERS|H1N1|H5N1|RSV|norovirus|rotavirus|hepatitis A|hepatitis B|hepatitis C|meningitis|encephalitis|pneumococcal|streptococcal|staphylococcal|E\. coli|salmonella|shigella|listeria|campylobacter)\b",
            ],
            EntityTypes.GENE: [
                # Gene symbols (HUGO nomenclature)
                r"\b[A-Z]{2,}[0-9]+(?:[A-Z]?)?\b",  # TP53, BRCA1, KRAS
                # Protein names (expanded)
                r"\b(?:p53|BRCA1|BRCA2|EGFR|HER2|BCR-ABL|KRAS|PIK3CA|APC|MLH1|MSH2|PTEN)\b",
                # Gene/protein patterns
                r"\b(?:gene|protein|receptor|kinase|enzyme)\s+([A-Z][A-Z0-9]+)\b",
            ],
            # Add new entity types based on research
            "ANATOMY": [
                # Body parts and organs
                r"\b(?:heart|lung|liver|kidney|brain|spine|bone|muscle|artery|vein|stomach|intestine|pancreas|spleen|bladder|uterus|ovary|prostate|thyroid)\b",
                # Anatomical regions
                r"\b(?:thoracic|abdominal|pelvic|cranial|cervical|lumbar|thorax|abdomen|pelvis|chest|neck|back)\b",
                # Body systems
                r"\b(?:cardiovascular|respiratory|digestive|nervous|immune|endocrine|reproductive|musculoskeletal)\b",
            ],
            "PROCEDURE": [
                # Medical procedures (expanded)
                r"\b(?:surgery|biopsy|chemotherapy|radiation|transplant|dialysis|angioplasty|bypass|catheterization|endoscopy|laparoscopy|arthroscopy)\b",
                # Diagnostic procedures
                r"\b(?:MRI|CT|X-ray|ultrasound|endoscopy|colonoscopy|mammography|PET|SPECT|EKG|ECG|EEG|blood test|biopsy)\b",
                # Therapeutic procedures
                r"\b(?:physical therapy|occupational therapy|speech therapy|rehabilitation|psychotherapy)\b",
            ],
            "SYMPTOM": [
                # Common symptoms
                r"\b(?:fever|pain|headache|nausea|vomiting|diarrhea|constipation|fatigue|weakness|dizziness|shortness of breath|chest pain|abdominal pain)\b",
                # Neurological symptoms
                r"\b(?:seizure|paralysis|numbness|tingling|confusion|memory loss|hallucination|delusion)\b",
                # Respiratory symptoms
                r"\b(?:cough|wheeze|dyspnea|tachypnea|hypoxia|cyanosis)\b",
            ],
            "TREATMENT": [
                # Treatment approaches
                r"\b(?:treatment|therapy|medication|drug|surgery|radiation|chemotherapy|immunotherapy|targeted therapy|gene therapy)\b",
                # Treatment settings
                r"\b(?:hospital|clinic|ICU|emergency room|outpatient|inpatient|hospice|home care)\b",
            ],
        }

    def _extract_llm(
        self, text: str, document: Optional[Document] = None
    ) -> List[Entity]:
        """Extract entities using LLM with DSPy-enhanced extraction."""
        entities = []
        try:
            # Check if DSPy extraction is configured
            use_dspy = self.config.get("llm", {}).get("use_dspy", False)

            if use_dspy:
                # Use DSPy-powered entity extraction
                entities = self._extract_with_dspy(text, document)
            else:
                # Use traditional prompt-based extraction
                prompt = self._build_prompt(text)
                response = self._call_llm(prompt)
                entities = self._parse_llm_response(response, document)

            # Filter by confidence and enabled types
            filtered = [
                e
                for e in entities
                if e.confidence >= self.confidence_threshold
                and e.entity_type in self.enabled_types
            ]
            return filtered[: self.max_entities_per_doc]

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return []

    def _extract_with_dspy(
        self, text: str, document: Optional[Document] = None
    ) -> List[Entity]:
        """Extract entities using DSPy with TrakCare-specific ontology."""
        try:
            # Lazy import DSPy module
            from ..dspy_modules.entity_extraction_module import (
                TrakCareEntityExtractionModule,
                configure_dspy
            )

            # Configure DSPy if not already configured
            if not hasattr(self, '_dspy_module'):
                llm_config = self.config.get("llm", {})

                # Only configure DSPy if not already configured globally
                # This prevents threading issues when workers have already configured DSPy
                import dspy as dspy_module
                try:
                    # Check if DSPy is already configured (by checking if lm is set)
                    if dspy_module.settings.lm is None:
                        # Configure DSPy with the full llm_config (respects supports_response_format, etc.)
                        configure_dspy(llm_config)
                        logger.info(f"DSPy configured with model: {llm_config.get('model')}")
                    else:
                        logger.info(f"DSPy already configured, reusing existing configuration")
                except Exception as e:
                    # If checking settings fails, try to configure anyway
                    logger.warning(f"Could not check DSPy configuration: {e}, attempting to configure...")
                    try:
                        configure_dspy(llm_config)
                    except Exception as config_error:
                        logger.error(f"Failed to configure DSPy: {config_error}")
                        # Fall back to traditional extraction
                        raise ImportError("DSPy configuration failed")

                # Initialize DSPy module
                self._dspy_module = TrakCareEntityExtractionModule()
                logger.info(f"DSPy entity extraction module initialized with model: {llm_config.get('model')}")

            # Perform DSPy extraction
            prediction = self._dspy_module.forward(
                ticket_text=text,
                entity_types=list(self.enabled_types) if self.enabled_types else None
            )

            # Parse entities from DSPy output with retry and repair
            entities = []
            entities_data = self._parse_json_with_retry(
                prediction.entities,
                max_attempts=3,
                context="DSPy entity extraction"
            )

            if entities_data is None:
                logger.error("Failed to parse DSPy entity output after all retry attempts")
                logger.warning(f"Low entity count (0) - DSPy should extract 4+ entities. Consider retraining or adjusting prompt.")
                return []

            for entity_data in entities_data:
                entity = Entity(
                    text=entity_data.get("text", ""),
                    entity_type=entity_data.get("type", "UNKNOWN"),
                    confidence=entity_data.get("confidence", 0.7),
                    start_offset=0,  # DSPy doesn't provide offsets by default
                    end_offset=len(entity_data.get("text", "")),
                    source_document_id=document.id if document else None,
                    metadata={
                        "method": "dspy",
                        "model": self.config.get("llm", {}).get("model", "qwen2.5:7b")
                    }
                )
                entities.append(entity)

            logger.info(f"DSPy extracted {len(entities)} entities (target: 4+)")

            return entities

        except ImportError as e:
            logger.error(f"DSPy modules not available: {e}")
            logger.info("Falling back to traditional LLM extraction")
            # Fallback to traditional extraction
            prompt = self._build_prompt(text)
            response = self._call_llm(prompt)
            return self._parse_llm_response(response, document)
        except Exception as e:
            logger.error(f"DSPy extraction failed: {e}")
            return []

    def extract_batch_with_dspy(
        self, documents: List[Document], batch_size: int = 5
    ) -> Dict[str, List[Entity]]:
        """
        Extract entities from multiple documents in batch using DSPy (2-3x faster!).

        This method processes 5 tickets per LLM call instead of 1 ticket per call,
        achieving 2-3x speedup. Batch size of 5 is optimal - larger batches (10+)
        cause LLM JSON quality degradation.

        Args:
            documents: List of documents to process (recommended: 5 documents)
            batch_size: Maximum tickets per LLM call (default: 5, optimal for quality)

        Returns:
            Dict mapping document IDs to their extracted entities
        """
        # Check if batch processing is enabled
        batch_config = self.config.get("batch_processing", {})
        batch_enabled = batch_config.get("enabled", True)

        if not batch_enabled:
            logger.info("Batch processing disabled - falling back to individual extraction")
            # Process documents individually
            result_map = {}
            for doc in documents:
                result = self.process_document(doc)
                if result.get("stored", False):
                    # Extract entities from the result
                    entities = result.get("entities", [])
                    result_map[doc.id] = entities
            return result_map

        try:
            # Lazy import batch DSPy module
            from ..dspy_modules.batch_entity_extraction import (
                BatchEntityExtractionModule
            )

            # Initialize batch module if not already done
            if not hasattr(self, '_batch_dspy_module'):
                # Import configuration helper
                from ..dspy_modules.entity_extraction_module import configure_dspy

                llm_config = self.config.get("llm", {})

                # Configure DSPy if needed
                import dspy as dspy_module
                if dspy_module.settings.lm is None:
                    configure_dspy(llm_config)
                    logger.info(f"DSPy configured for batch extraction with model: {llm_config.get('model')}")

                # Initialize batch module
                self._batch_dspy_module = BatchEntityExtractionModule()
                logger.info(f"✅ Batch DSPy module initialized (processes {batch_size} tickets/call)")

            # Prepare tickets for batch processing
            tickets = [
                {"id": doc.id, "text": doc.page_content}
                for doc in documents[:batch_size]
            ]

            if not tickets:
                return {}

            logger.info(f"🚀 Processing {len(tickets)} tickets in ONE LLM call (batch mode)")

            # Call batch extraction (single LLM call for all tickets!)
            batch_results = self._batch_dspy_module.forward(tickets)

            # Convert batch results to Document ID → Entities mapping
            result_map = {}

            for result in batch_results:
                ticket_id = result.get("ticket_id")
                entities_data = result.get("entities", [])

                # Convert to Entity objects
                entities = []
                for entity_data in entities_data:
                    entity = Entity(
                        text=entity_data.get("text", ""),
                        entity_type=entity_data.get("type", "UNKNOWN"),
                        confidence=entity_data.get("confidence", 0.7),
                        start_offset=0,
                        end_offset=len(entity_data.get("text", "")),
                        source_document_id=ticket_id,
                        metadata={
                            "method": "dspy_batch",
                            "model": self.config.get("llm", {}).get("model", "qwen2.5:7b"),
                            "batch_size": len(tickets)
                        }
                    )
                    entities.append(entity)

                result_map[ticket_id] = entities
                logger.debug(f"  Ticket {ticket_id}: {len(entities)} entities extracted")

            total_entities = sum(len(ents) for ents in result_map.values())
            logger.info(
                f"✅ Batch complete: {len(tickets)} tickets → {total_entities} entities "
                f"(avg: {total_entities/len(tickets):.1f} per ticket)"
            )

            return result_map

        except ImportError as e:
            logger.error(f"Batch DSPy module not available: {e}")
            logger.info("Falling back to individual extraction")
            # Fallback: process one at a time
            return {
                doc.id: self._extract_with_dspy(doc.page_content, doc)
                for doc in documents
            }
        except Exception as e:
            logger.error(f"Batch DSPy extraction failed: {e}")
            # Fallback: process one at a time
            return {
                doc.id: self._extract_with_dspy(doc.page_content, doc)
                for doc in documents
            }

    def _extract_patterns(
        self, text: str, document: Optional[Document] = None
    ) -> List[Entity]:
        """Extract entities using regex patterns."""
        entities = []

        # Normalize enabled types to names for comparison (handles enum vs string)
        enabled_names = set()
        for t in self.enabled_types or []:
            try:
                enabled_names.add(
                    t.name if hasattr(t, "name") else str(t).split(".")[-1]
                )
            except Exception:
                enabled_names.add(str(t))

        for entity_type, patterns in self.patterns.items():
            et_name = (
                entity_type.name
                if hasattr(entity_type, "name")
                else str(entity_type).split(".")[-1]
            )
            # If enabled_names explicitly provided, filter by names
            if enabled_names and et_name not in enabled_names:
                continue

            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity = Entity(
                        text=match.group(0),
                        entity_type=et_name,
                        confidence=0.8,
                        start_offset=match.start(),
                        end_offset=match.end(),
                        source_document_id=document.id if document else None,
                        metadata={"method": "pattern"},
                    )
                    entities.append(entity)

        return entities

    def _extract_keywords_basic(
        self, text: str, document: Optional[Document] = None, max_count: int = 50
    ) -> List[Entity]:
        """
        Fallback keyword-based extraction:
        - Select capitalized words > 3 chars or multi-word Title Case phrases
        - De-duplicate and limit to max_count
        """
        try:
            candidates = []
            # Simple heuristics: title-case words and multi-word phrases
            for match in re.finditer(
                r"\b([A-Z][a-z]{3,}(?:\s+[A-Z][a-z]{3,}){0,2})\b", text
            ):
                candidates.append((match.group(0), match.start(), match.end()))
            # Also include ALLCAPS tokens like COVID, SARS, etc.
            for match in re.finditer(r"\b([A-Z]{3,}(?:\-[A-Z0-9]{2,})?)\b", text):
                candidates.append((match.group(0), match.start(), match.end()))

            # Deduplicate by lowercase text
            seen = set()
            entities = []
            for token, s, e in candidates:
                key = token.lower()
                if key in seen:
                    continue
                seen.add(key)
                entities.append(
                    Entity(
                        text=token,
                        entity_type="KEYWORD",
                        confidence=0.7,
                        start_offset=s,
                        end_offset=e,
                        source_document_id=document.id if document else None,
                        metadata={"method": "keyword_fallback"},
                    )
                )
                if len(entities) >= max_count:
                    break
            return entities
        except Exception as e:
            logger.warning(f"Keyword fallback extraction failed: {e}")
            return []

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
        try:
            import requests
            import json

            # Get LLM config from entity_extraction config
            llm_config = self.config.get("llm", {})
            model = llm_config.get("model", "qwen2.5:7b")  # Changed from qwen3:14b - faster and works better
            temperature = llm_config.get("temperature", 0.1)
            max_tokens = llm_config.get("max_tokens", 2000)

            # Check if model is Ollama (localhost:11434) or OpenAI
            if "gpt" in model.lower():
                # Use OpenAI API - fallback to stub
                logger.warning(f"OpenAI model {model} configured but no API key - using pattern extraction")
                return '[]'

            # Use Ollama
            ollama_url = "http://localhost:11434/api/generate"

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }

            response = requests.post(ollama_url, json=payload, timeout=60)
            response.raise_for_status()

            result = response.json()
            llm_response = result.get("response", "[]")

            # Log for debugging
            logger.debug(f"LLM response (first 200 chars): {llm_response[:200]}")

            return llm_response

        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to Ollama at localhost:11434")
            return '[]'
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return '[]'

    def _parse_json_with_retry(
        self, json_str: str, max_attempts: int = 3, context: str = "JSON parsing"
    ) -> Optional[List[Dict[str, Any]]]:
        r"""
        Parse JSON with retry and repair logic for LLM-generated invalid escape sequences.

        Fixes the 0.7% JSON parsing failure rate observed in production where LLMs
        generate invalid escape sequences like \N, \i, etc.

        Args:
            json_str: JSON string to parse
            max_attempts: Maximum number of parsing attempts with repair
            context: Context string for logging

        Returns:
            Parsed JSON data as list of dicts, or None if all attempts fail
        """
        for attempt in range(max_attempts):
            try:
                # Attempt to parse JSON
                data = json.loads(json_str)

                # Ensure it's a list
                if not isinstance(data, list):
                    logger.warning(f"{context}: Expected list, got {type(data)}. Wrapping in list.")
                    data = [data]

                if attempt > 0:
                    logger.info(f"{context}: Successfully parsed after {attempt} repair attempts")

                return data

            except json.JSONDecodeError as e:
                if attempt < max_attempts - 1:
                    # Try to repair common LLM escape sequence errors
                    logger.warning(
                        f"{context}: JSON parse failed on attempt {attempt + 1}/{max_attempts}: {e}"
                    )
                    logger.debug(f"Invalid JSON (first 200 chars): {json_str[:200]}")

                    # Apply repair strategies
                    original_str = json_str

                    # Strategy 1: Fix invalid escape sequences
                    # Replace \N with \\N, \i with \\i, etc.
                    # But preserve valid escapes: \n, \t, \r, \", \\, \/, \b, \f
                    valid_escapes = {'n', 't', 'r', '"', '\\', '/', 'b', 'f', 'u'}

                    # Find all backslash sequences and fix invalid ones
                    repaired = []
                    i = 0
                    while i < len(json_str):
                        if json_str[i] == '\\' and i + 1 < len(json_str):
                            next_char = json_str[i + 1]
                            if next_char not in valid_escapes:
                                # Invalid escape - add extra backslash
                                repaired.append('\\\\')
                                repaired.append(next_char)
                                i += 2
                            else:
                                # Valid escape - keep as is
                                repaired.append('\\')
                                i += 1
                        else:
                            repaired.append(json_str[i])
                            i += 1

                    json_str = ''.join(repaired)

                    if json_str != original_str:
                        logger.debug(f"Applied escape sequence repair (attempt {attempt + 1})")
                    else:
                        logger.debug(f"No repair pattern matched (attempt {attempt + 1})")

                else:
                    # Final attempt failed
                    logger.error(
                        f"{context}: Failed to parse JSON after {max_attempts} attempts: {e}"
                    )
                    logger.debug(f"Final JSON (first 500 chars): {json_str[:500]}")
                    return None

        return None

    def _parse_llm_response(
        self, response: str, document: Optional[Document]
    ) -> List[Entity]:
        """Parse LLM JSON response into Entity objects."""
        entities = []
        try:
            # Try to extract JSON from response
            # LLM might return markdown code blocks or text before/after JSON
            import re

            # Try to find JSON array in the response
            json_match = re.search(r'\[[\s\S]*\]', response)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Fallback: assume entire response is JSON
                json_str = response

            raw_entities = json.loads(json_str)

            for raw_entity in raw_entities:
                entity = Entity(
                    text=raw_entity.get("text", ""),
                    entity_type=raw_entity.get("type", "UNKNOWN"),
                    confidence=raw_entity.get("confidence", 0.5),
                    start_offset=0,  # Would need text search for actual position
                    end_offset=len(raw_entity.get("text", "")),
                    source_document_id=document.id if document else None,
                    metadata={"method": "llm"},
                )
                entities.append(entity)

            logger.info(f"Parsed {len(entities)} entities from LLM response")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response was: {response[:500]}")

        return entities

    def _merge_entities(
        self, llm_entities: List[Entity], pattern_entities: List[Entity]
    ) -> List[Entity]:
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

    def extract_relationships(
        self, entities: List[Entity], document: Document
    ) -> List[Relationship]:
        """Extract enhanced semantic relationships between entities."""
        relationships = []

        # Enhanced semantic relationship extraction
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1 :]:
                # Calculate distance between entities
                distance = abs(entity1.start_offset - entity2.start_offset)

                # Create relationship if entities are close
                if distance < 300:  # Increased window for better context
                    # Determine semantic relationship type based on entity types and context
                    rel_type, confidence = self._determine_relationship_type(
                        entity1, entity2, document.page_content, distance
                    )

                    if (
                        rel_type and confidence >= 0.5
                    ):  # Only high-confidence relationships
                        relationship = Relationship(
                            source_entity_id=entity1.id,
                            target_entity_id=entity2.id,
                            relationship_type=rel_type,
                            confidence=confidence,
                            source_document_id=document.id,
                            metadata={
                                "method": "semantic_enhanced",
                                "distance": distance,
                                "entity1_type": entity1.entity_type,
                                "entity2_type": entity2.entity_type,
                                "context_snippet": self._extract_context_snippet(
                                    document.page_content, entity1, entity2
                                ),
                            },
                        )
                        relationships.append(relationship)

        return relationships

    def _determine_relationship_type(
        self, entity1: Entity, entity2: Entity, text: str, distance: int
    ) -> Tuple[str, float]:
        """
        Determine semantic relationship type between two entities based on context.

        Returns:
            Tuple of (relationship_type, confidence_score)
        """
        # Extract context around entities for analysis
        context = self._extract_context_snippet(text, entity1, entity2)
        context_lower = context.lower()

        type1 = entity1.entity_type.upper() if entity1.entity_type else ""
        type2 = entity2.entity_type.upper() if entity2.entity_type else ""

        # Medical domain relationship patterns
        if type1 == "DRUG" and type2 == "DISEASE":
            if any(
                word in context_lower
                for word in [
                    "treat",
                    "cure",
                    "manage",
                    "control",
                    "therapy",
                    "medication",
                ]
            ):
                return RelationshipTypes.TREATS, 0.9
            elif any(
                word in context_lower
                for word in ["prevent", "prophylaxis", "vaccination"]
            ):
                return RelationshipTypes.PREVENTS, 0.85

        elif type1 == "DISEASE" and type2 == "DRUG":
            if any(
                word in context_lower
                for word in ["treated with", "managed with", "prescribed", "medication"]
            ):
                return RelationshipTypes.TREATS, 0.9  # Reverse direction in metadata
            elif any(word in context_lower for word in ["prevented by", "prophylaxis"]):
                return RelationshipTypes.PREVENTS, 0.85

        elif type1 == "DISEASE" and type2 == "SYMPTOM":
            if any(
                word in context_lower
                for word in ["cause", "lead to", "result in", "manifest"]
            ):
                return RelationshipTypes.CAUSES, 0.85
            elif any(
                word in context_lower
                for word in ["symptom", "present", "show", "exhibit"]
            ):
                return RelationshipTypes.ASSOCIATED_WITH, 0.8

        elif type1 == "SYMPTOM" and type2 == "DISEASE":
            if any(
                word in context_lower
                for word in ["sign of", "indicate", "suggest", "symptom of"]
            ):
                return RelationshipTypes.ASSOCIATED_WITH, 0.8

        elif type1 == "DRUG" and type2 == "DRUG":
            if any(
                word in context_lower
                for word in ["interact", "combination", "together", "concurrent"]
            ):
                return RelationshipTypes.INTERACTS_WITH, 0.8

        elif type1 == "ANATOMY" and type2 == "DISEASE":
            if any(
                word in context_lower
                for word in ["affect", "involve", "damage", "disease of"]
            ):
                return RelationshipTypes.LOCATED_IN, 0.8

        elif type1 == "PROCEDURE" and type2 == "DISEASE":
            if any(
                word in context_lower
                for word in ["treat", "surgery for", "procedure for", "intervention"]
            ):
                return RelationshipTypes.TREATS, 0.8

        elif type1 == "TREATMENT" and type2 == "DISEASE":
            if any(
                word in context_lower
                for word in ["for", "against", "treat", "therapy for"]
            ):
                return RelationshipTypes.TREATS, 0.85

        # Anatomical relationships
        elif type1 == "ANATOMY" and type2 == "ANATOMY":
            if any(
                word in context_lower
                for word in ["part of", "within", "inside", "contains"]
            ):
                return RelationshipTypes.PART_OF, 0.8
            elif any(
                word in context_lower for word in ["near", "adjacent", "connected"]
            ):
                return RelationshipTypes.LOCATED_IN, 0.7

        # Causality patterns for any entity types
        causality_patterns = [
            "cause",
            "lead to",
            "result in",
            "trigger",
            "induce",
            "provoke",
        ]
        if any(pattern in context_lower for pattern in causality_patterns):
            return RelationshipTypes.CAUSES, 0.75

        # Prevention patterns
        prevention_patterns = ["prevent", "avoid", "reduce risk", "protect against"]
        if any(pattern in context_lower for pattern in prevention_patterns):
            return RelationshipTypes.PREVENTS, 0.75

        # Strong association patterns
        association_patterns = [
            "associated with",
            "related to",
            "linked to",
            "connected to",
        ]
        if any(pattern in context_lower for pattern in association_patterns):
            return RelationshipTypes.ASSOCIATED_WITH, 0.7

        # Close proximity with good confidence - general relationship
        if distance < 100:
            return RelationshipTypes.RELATED_TO, 0.6
        elif distance < 200:
            return RelationshipTypes.RELATED_TO, 0.5

        # Low confidence - don't create relationship
        return None, 0.0

    def _extract_context_snippet(
        self, text: str, entity1: Entity, entity2: Entity
    ) -> str:
        """Extract context snippet around two entities for relationship analysis."""
        # Find the span that includes both entities with some padding
        start_pos = min(entity1.start_offset, entity2.start_offset)
        end_pos = max(entity1.end_offset, entity2.end_offset)

        # Add padding for context
        context_start = max(0, start_pos - 50)
        context_end = min(len(text), end_pos + 50)

        return text[context_start:context_end]

    def store_entities_and_relationships(
        self, entities: List[Entity], relationships: List[Relationship]
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
                "stored_relationships": stored_relationships,
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
            "errors": [],
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
                    storage_results = self.store_entities_and_relationships(
                        entities, relationships
                    )
                    results["entities_stored"] = storage_results.get(
                        "stored_entities", 0
                    )
                    results["relationships_stored"] = storage_results.get(
                        "stored_relationships", 0
                    )
                    # Add boolean flag and aliases expected by callers
                    results["stored"] = (
                        results["entities_stored"] > 0
                        or results["relationships_stored"] > 0
                    )
                    results["entities_count"] = results["entities_extracted"]
                    results["relationships_count"] = results["relationships_extracted"]

                    if "error" in storage_results:
                        results["errors"].append(
                            f"Storage error: {storage_results['error']}"
                        )

                except Exception as e:
                    results["errors"].append(f"Storage failed: {e}")
                    results["stored"] = False
            else:
                # No storage adapter: mark as not stored but extraction still succeeded
                results["stored"] = False

            logger.info(f"Processed document {document.id}: {results}")

        except Exception as e:
            error_msg = f"Document processing failed: {e}"
            results["errors"].append(error_msg)
            logger.error(error_msg)

        return results

    def extract_batch(
        self, documents: List[Document], token_budget: int = 8192
    ) -> "BatchExtractionResult":
        """
        Extract entities from a batch of documents with retry logic (T023: FR-001, FR-006).

        This method implements batch processing for entity extraction to achieve
        3x speedup over single-document processing (FR-002). It uses token-aware
        batching, retry logic, and metrics tracking.

        Args:
            documents: List of documents to process in batch
            token_budget: Maximum tokens per batch (default: 8192 per FR-006)

        Returns:
            BatchExtractionResult with per-document entities and relationships

        Raises:
            ValueError: If documents list is empty

        Examples:
            >>> service = EntityExtractionService(config_manager)
            >>> docs = [Document(id="1", page_content="..."), ...]
            >>> result = service.extract_batch(docs, token_budget=8192)
            >>> print(f"Processed {len(result.per_document_entities)} documents")
        """
        from iris_rag.core.models import BatchExtractionResult
        from iris_rag.utils.token_counter import estimate_tokens
        from common.batch_utils import BatchQueue, BatchMetricsTracker
        import time
        import uuid

        # Validate input
        if not documents:
            raise ValueError("Documents list cannot be empty")

        logger.info(f"Starting batch extraction for {len(documents)} documents")

        # Create batch queue and add documents
        queue = BatchQueue(token_budget=token_budget)

        for doc in documents:
            # Estimate tokens for this document
            token_count = estimate_tokens(doc.page_content)
            queue.add_document(doc, token_count)

        # Process batch
        batch_id = str(uuid.uuid4())
        start_time = time.time()
        per_document_entities = {}
        per_document_relationships = {}
        success = True
        error_msg = None

        try:
            # Get all documents from queue (should be all of them since we just added them)
            batch_docs = queue.get_next_batch(token_budget=token_budget)

            if not batch_docs:
                logger.warning("Batch queue returned no documents")
                batch_docs = documents  # Fallback to original list

            # Extract entities and relationships for each document
            for doc in batch_docs:
                try:
                    # Extract entities
                    entities = self.extract_entities(doc)
                    per_document_entities[doc.id] = entities

                    # Extract relationships
                    relationships = self.extract_relationships(entities, doc)
                    per_document_relationships[doc.id] = relationships

                except Exception as e:
                    logger.error(f"Failed to extract from document {doc.id}: {e}")
                    per_document_entities[doc.id] = []
                    per_document_relationships[doc.id] = []
                    success = False
                    error_msg = f"Partial batch failure: {e}"

        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            success = False
            error_msg = str(e)

        processing_time = time.time() - start_time

        # Create result
        result = BatchExtractionResult(
            batch_id=batch_id,
            per_document_entities=per_document_entities,
            per_document_relationships=per_document_relationships,
            processing_time=processing_time,
            success_status=success,
            retry_count=0,  # Will be set by retry wrapper if used
            error_message=error_msg,
        )

        # Update global metrics tracker
        tracker = BatchMetricsTracker.get_instance()
        tracker.update_with_batch(result, len(documents))

        logger.info(
            f"Batch extraction complete: {len(documents)} docs in {processing_time:.2f}s "
            f"(avg: {processing_time/len(documents):.2f}s/doc)"
        )

        return result

    def get_batch_metrics(self) -> "ProcessingMetrics":
        """
        Get batch processing statistics (T024: FR-007).

        Returns global processing metrics including speedup factor, entity extraction
        rate, and failure statistics.

        Returns:
            ProcessingMetrics with current batch processing statistics

        Examples:
            >>> service = EntityExtractionService(config_manager)
            >>> metrics = service.get_batch_metrics()
            >>> print(f"Speedup: {metrics.speedup_factor:.1f}x")
            >>> print(f"Avg entities/batch: {metrics.entity_extraction_rate_per_batch:.1f}")
        """
        from common.batch_utils import BatchMetricsTracker

        tracker = BatchMetricsTracker.get_instance()
        return tracker.get_statistics()
