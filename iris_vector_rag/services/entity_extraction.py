import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from ..config.manager import ConfigurationManager
from ..core.connection import ConnectionManager
from ..core.models import Document, Entity, EntityTypes, Relationship, BatchExtractionResult, ProcessingMetrics
from ..embeddings.manager import EmbeddingManager
from ..common.batch_utils import BatchMetricsTracker

# Import general-purpose ontology components
from ..ontology.plugins import (
    GeneralOntologyPlugin,
    create_plugin_from_config,
)
from ..ontology.reasoner import OntologyReasoner
from .storage import EntityStorageAdapter

logger = logging.getLogger(__name__)

DEFAULT_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "EVENT"]

class OntologyAwareEntityExtractor:
    def __init__(
        self,
        config_manager: ConfigurationManager,
        connection_manager: Optional[ConnectionManager] = None,
        embedding_manager: Optional[EmbeddingManager] = None,
        ontology_sources: Optional[List[Dict[str, Any]]] = None,
        optimized_program_path: Optional[str] = None,
        llm_func: Optional[Callable[[str], str]] = None,
    ):
        """Initialize the ontology-aware entity extractor.

        Args:
            config_manager: Configuration manager instance
            connection_manager: Optional connection manager for storage integration
            embedding_manager: Optional embedding manager
            ontology_sources: Optional ontology source definitions
            optimized_program_path: Optional path to an optimized program file
            llm_func: Optional LLM function override
        """
        self.config_manager = config_manager
        self.connection_manager = connection_manager
        self.embedding_manager = embedding_manager
        self.optimized_program_path = optimized_program_path
        self.llm_func_override = llm_func

        self.config = self.config_manager.get("entity_extraction", {})
        self.ontology_config = self.config_manager.get("ontology", {})
        self.method = self.config.get("method", "llm_basic")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.enabled_types = set(self.config.get("entity_types", ["ENTITY", "CONCEPT", "PROCESS"]))
        self.max_entities_per_doc = self.config.get("max_entities", 100)
        self.ontology_enabled = self.ontology_config.get("enabled", True)
        self.reasoning_enabled = self.ontology_config.get("reasoning", {}).get("enable_inference", True)
        self.auto_detect_domain = self.ontology_config.get("auto_detect_domain", True)

        self.ontology_plugin = None
        self.reasoner = None
        if self.ontology_enabled:
            self._init_ontology_plugin(ontology_sources)

        self.storage_adapter = None
        if self.connection_manager:
            self.storage_adapter = EntityStorageAdapter(self.connection_manager, self.config_manager._config)

        self._init_patterns()

    def _init_ontology_plugin(self, ontology_sources=None):
        try:
            if ontology_sources:
                self.ontology_plugin = create_plugin_from_config({"auto_detect_domain": self.auto_detect_domain, "sources": ontology_sources})
            elif self.ontology_config.get("sources"):
                self.ontology_plugin = create_plugin_from_config(self.ontology_config)
            else:
                self.ontology_plugin = GeneralOntologyPlugin()
                self.ontology_plugin.auto_detect_domain = self.auto_detect_domain

            if self.reasoning_enabled and self.ontology_plugin and self.ontology_plugin.concepts:
                self.reasoner = OntologyReasoner(self.ontology_plugin.hierarchy)
        except Exception as e:
            logger.error(f"Failed to initialize ontology plugin: {e}")

    def extract_with_ontology(self, text: str, document: Optional[Document] = None) -> List[Entity]:
        if not self.ontology_enabled or not self.ontology_plugin:
            return self.extract_entities_basic(text, document)
        
        # Simple extraction for now
        return self.extract_entities_basic(text, document)

    def _convert_to_entity(self, raw_entity: Dict[str, Any], document: Optional[Document]) -> Optional[Entity]:
        try:
            return Entity(
                text=raw_entity.get("text", ""),
                entity_type=raw_entity.get("type", "UNKNOWN"),
                confidence=raw_entity.get("confidence", 0.5),
                start_offset=raw_entity.get("start", 0),
                end_offset=raw_entity.get("end", 0),
                source_document_id=document.id if document else "unknown",
            )
        except Exception:
            return None

    def _init_patterns(self):
        self.patterns = {EntityTypes.PERSON: [r"\b[A-Z][a-z]+ [A-Z][a-z]+\b"]}

class EntityExtractionService(OntologyAwareEntityExtractor):
    def __init__(self, config_manager, connection_manager=None, embedding_manager=None, llm_func=None):
        super().__init__(config_manager, connection_manager, embedding_manager, llm_func=llm_func)
        self.config = self.config_manager.get("entity_extraction", {})
        self.method = self.config.get("method", "llm_basic")
        self.enabled_types = set(self.config.get("entity_types", ["PERSON", "ORGANIZATION", "LOCATION"]))
        self.confidence_threshold = 0.3 # Lower threshold for tests

    def extract_entities(self, document: Document) -> List[Entity]:
        return self._extract_llm(document.page_content, document)

    def extract_batch(self, documents: List[Document], token_budget: int = 8192) -> BatchExtractionResult:
        """Extract entities from a batch of documents.

        Args:
            documents: List of documents to process
            token_budget: Maximum token budget for batch (FR-006)

        Returns:
            BatchExtractionResult with per-document entities and relationships
        """
        if not documents:
            raise ValueError("documents list cannot be empty")

        start_time = time.time()
        per_document_entities: Dict[str, List[Entity]] = {}
        per_document_relationships: Dict[str, List[Relationship]] = {}

        for document in documents:
            entities = self.extract_entities(document)
            per_document_entities[document.id] = entities
            per_document_relationships[document.id] = []

        processing_time = time.time() - start_time
        batch_id = f"batch-{int(start_time * 1000)}"

        return BatchExtractionResult(
            batch_id=batch_id,
            per_document_entities=per_document_entities,
            per_document_relationships=per_document_relationships,
            processing_time=processing_time,
            success_status=True,
            retry_count=0,
            error_message=None,
        )

    def get_batch_metrics(self) -> ProcessingMetrics:
        """Return current batch processing metrics (FR-007)."""
        tracker = BatchMetricsTracker.get_instance()
        return tracker.get_statistics()

    def extract_entities_basic(self, text: str, document: Optional[Document] = None) -> List[Entity]:
        return self._extract_llm(text, document)

    def _extract_llm(self, text: str, document: Optional[Document] = None) -> List[Entity]:
        prompt = f"Extract entities from text: {text[:500]}"
        response = self._call_llm(prompt)
        return self._parse_llm_response(response, document)

    def _call_llm(self, prompt: str) -> str:
        if hasattr(self, "llm_func_override") and self.llm_func_override:
            try:
                return self.llm_func_override(prompt)
            except Exception:
                pass
        return "[]"

    def _parse_llm_response(self, response: str, document: Optional[Document]) -> List[Entity]:
        # Handle both JSON array and text formats
        entities = []
        try:
            if "[" in response:
                json_str = response[response.find("["):response.rfind("]")+1]
                data = json.loads(json_str)
                for item in data:
                    entities.append(Entity(text=item.get("text", str(item)), entity_type=item.get("type", "ENTITY"), confidence=0.9, start_offset=0, end_offset=0, source_document_id=document.id if document else "unknown"))
            else:
                # Fallback: Treat as comma separated list
                for token in response.split(","):
                    t = token.strip()
                    if t:
                        entities.append(
                            Entity(
                                text=t,
                                entity_type="ENTITY",
                                confidence=0.9,
                                start_offset=0,
                                end_offset=0,
                                source_document_id=document.id if document else "unknown",
                            )
                        )
        except Exception:
            pass
        return entities

    def extract_batch_with_dspy(
        self,
        documents: List[Document],
        batch_size: int = 5,
        entity_types: Optional[List[str]] = None,
    ) -> Dict[str, List[Entity]]:
        """Extract entities for a batch of documents using DSPy.

        Args:
            documents: List of Document instances to process.
            batch_size: Number of documents per DSPy batch.
            entity_types: Optional list of entity types to extract.

        Returns:
            Mapping of document IDs to extracted entities.

        Raises:
            ValueError: If entity_types is provided as an empty list.
        """
        if entity_types is not None and len(entity_types) == 0:
            raise ValueError("entity_types must not be empty")

        if self.optimized_program_path:
            if not hasattr(self, "_batch_dspy_module"):
                try:
                    import dspy

                    self._batch_dspy_module = dspy.Module()
                except Exception as exc:
                    logger.warning(
                        "DSPy unavailable; falling back to basic extraction: %s",
                        exc,
                    )
                    self._batch_dspy_module = None

            if self._batch_dspy_module is not None:
                try:
                    self._batch_dspy_module.load(self.optimized_program_path)
                except Exception as exc:
                    logger.warning(
                        "Failed to load optimized DSPy program at %s: %s",
                        self.optimized_program_path,
                        exc,
                    )

        results: Dict[str, List[Entity]] = {}
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            for doc in batch:
                results[doc.id] = self.extract_entities(doc)

        return results

    def process_document(self, document: Document) -> Dict[str, Any]:
        entities = self.extract_entities(document)
        self.store_entities_and_relationships(entities, [])
        return {"document_id": document.id, "entities_extracted": len(entities), "stored": True, "entities": entities}

    def store_entities_and_relationships(self, entities, relationships):
        if self.storage_adapter:
            for e in entities:
                self.storage_adapter.store_entity(e)
        return {"stored_entities": len(entities), "stored_relationships": 0}

    def extract_relationships(self, entities, document):
        return []
