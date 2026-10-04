import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..common.batch_utils import BatchMetricsTracker
from ..config.manager import ConfigurationManager
from ..core.connection import ConnectionManager
from ..core.models import (
    BatchExtractionResult,
    Document,
    Entity,
    EntityTypes,
    ProcessingMetrics,
    Relationship,
)
from ..embeddings.manager import EmbeddingManager
from ..exceptions import RAGException
from .storage import EntityStorageAdapter

logger = logging.getLogger(__name__)

DEFAULT_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "PRODUCT", "EVENT"]


class EntityExtractionError(RAGException):
    """Entity extraction or its storage failed and the caller must know."""


@dataclass
class ServiceBatchResult(BatchExtractionResult):
    """`BatchExtractionResult` plus the documents that failed.

    `failed` holds one `{"document_id", "reason"}` per document whose extraction
    raised. Those documents have no key in `per_document_entities`; the rest of the
    batch is unaffected.
    """

    failed: List[Dict[str, str]] = field(default_factory=list)


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
            ontology_sources: Ignored. The ontology package was removed; kept for
                signature compatibility
            optimized_program_path: Optional path to an optimized program file
            llm_func: Optional LLM function override
        """
        self.config_manager = config_manager
        self.connection_manager = connection_manager
        self.embedding_manager = embedding_manager
        self.optimized_program_path = optimized_program_path
        self.llm_func_override = llm_func

        self.config = self.config_manager.get("entity_extraction", {})
        self.method = self.config.get("method", "llm_basic")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.enabled_types = set(
            self.config.get("entity_types", ["ENTITY", "CONCEPT", "PROCESS"])
        )
        self.max_entities_per_doc = self.config.get("max_entities", 100)

        self.storage_adapter = None
        if self.connection_manager:
            self.storage_adapter = EntityStorageAdapter(
                self.connection_manager, self.config_manager._config
            )

        self._init_patterns()

    def _convert_to_entity(
        self, raw_entity: Dict[str, Any], document: Optional[Document]
    ) -> Optional[Entity]:
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
    def __init__(
        self,
        config_manager,
        connection_manager=None,
        embedding_manager=None,
        llm_func=None,
    ):
        super().__init__(
            config_manager, connection_manager, embedding_manager, llm_func=llm_func
        )
        self.config = self.config_manager.get("entity_extraction", {})
        self.method = self.config.get("method", "llm_basic")
        self.enabled_types = set(
            self.config.get("entity_types", ["PERSON", "ORGANIZATION", "LOCATION"])
        )
        self.confidence_threshold = self.config.get("confidence_threshold", 0.3)
        # Only a configured `entity_types` filters what the LLM returns.
        self._filter_types = "entity_types" in self.config

    def extract_entities(self, document: Document) -> List[Entity]:
        return self._extract_llm(document.page_content, document)

    def extract_batch(
        self, documents: List[Document], token_budget: int = 8192
    ) -> ServiceBatchResult:
        """Extract entities from a batch of documents.

        A document whose extraction raises is listed in `result.failed` with the
        reason and the batch carries on. If every document fails the batch raises
        `EntityExtractionError`: there is nothing to report but failure.

        Args:
            documents: List of documents to process
            token_budget: Advisory. Extraction runs one prompt per document, so
                there is no batch prompt to size; a batch whose text is estimated
                (4 characters per token) to exceed the budget is logged.

        Returns:
            ServiceBatchResult with per-document entities and relationships and
            the `failed` list. `success_status` is False when `failed` is not empty.

        Raises:
            ValueError: `documents` is empty.
            EntityExtractionError: every document failed.
        """
        if not documents:
            raise ValueError("documents list cannot be empty")

        estimated_tokens = sum(len(d.page_content) for d in documents) // 4
        if estimated_tokens > token_budget:
            logger.warning(
                "Batch of %d documents is about %d tokens, over the advisory "
                "token_budget of %d; documents are extracted one prompt each",
                len(documents),
                estimated_tokens,
                token_budget,
            )

        start_time = time.time()
        per_document_entities: Dict[str, List[Entity]] = {}
        per_document_relationships: Dict[str, List[Relationship]] = {}
        failed: List[Dict[str, str]] = []

        for document in documents:
            try:
                entities = self.extract_entities(document)
            except Exception as exc:
                logger.error("Entity extraction failed for %s: %s", document.id, exc)
                failed.append({"document_id": document.id, "reason": str(exc)})
                continue
            per_document_entities[document.id] = entities
            per_document_relationships[document.id] = []

        if len(failed) == len(documents):
            reasons = "; ".join(f"{f['document_id']}: {f['reason']}" for f in failed)
            raise EntityExtractionError(
                f"Entity extraction failed for every document in the batch ({reasons})"
            )

        processing_time = time.time() - start_time
        batch_id = f"batch-{int(start_time * 1000)}"
        error_message = None
        if failed:
            error_message = "Extraction failed for: " + ", ".join(
                f["document_id"] for f in failed
            )

        return ServiceBatchResult(
            batch_id=batch_id,
            per_document_entities=per_document_entities,
            per_document_relationships=per_document_relationships,
            processing_time=processing_time,
            success_status=not failed,
            retry_count=0,
            error_message=error_message,
            failed=failed,
        )

    def get_batch_metrics(self) -> ProcessingMetrics:
        """Return current batch processing metrics (FR-007)."""
        tracker = BatchMetricsTracker.get_instance()
        return tracker.get_statistics()

    def extract_entities_basic(
        self, text: str, document: Optional[Document] = None
    ) -> List[Entity]:
        return self._extract_llm(text, document)

    def extract_with_ontology(
        self, text: str, document: Optional[Document] = None
    ) -> List[Entity]:
        """Deprecated alias for LLM extraction; the ontology is retired.

        The ``ontology:`` config section and ``ontology_sources`` are ignored.
        Use `extract_entities`.
        """
        warnings.warn(
            "extract_with_ontology() is deprecated: the ontology is retired and "
            "this is plain LLM extraction. Use extract_entities().",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.extract_entities_basic(text, document)

    def _extract_llm(
        self, text: str, document: Optional[Document] = None
    ) -> List[Entity]:
        prompt = f"Extract entities from text: {text[:500]}"
        response = self._call_llm(prompt)
        return self._parse_llm_response(response, document)

    def _call_llm(self, prompt: str) -> str:
        if not self.llm_func_override:
            raise EntityExtractionError(
                "No llm_func configured for entity extraction; pass llm_func= to "
                "the service (or its pipeline)"
            )
        try:
            return self.llm_func_override(prompt)
        except Exception as exc:
            raise EntityExtractionError(f"LLM call failed: {exc}") from exc

    def _parse_llm_response(
        self, response: str, document: Optional[Document]
    ) -> List[Entity]:
        """Turn an LLM reply into entities, applying the configured limits.

        A JSON array (anywhere in the reply) or, failing that, a comma separated
        list. Items below `confidence_threshold`, of a type outside a configured
        `entity_types`, or beyond the `max_entities` best are dropped. A reply that
        cannot be parsed raises; it is not the same as "no entities".
        """
        source = document.id if document else "unknown"
        entities: List[Entity] = []
        try:
            if "[" in response:
                json_str = response[response.find("[") : response.rfind("]") + 1]
                data = json.loads(json_str)
                for item in data:
                    if isinstance(item, dict):
                        text = item.get("text", str(item))
                        etype = item.get("type")
                        confidence = float(item.get("confidence", 0.9))
                    else:
                        text, etype, confidence = str(item), None, 0.9
                    if (
                        self._filter_types
                        and etype is not None
                        and str(etype).upper()
                        not in {str(t).upper() for t in self.enabled_types}
                    ):
                        continue
                    entities.append(
                        Entity(
                            text=text,
                            entity_type=etype or "ENTITY",
                            confidence=confidence,
                            start_offset=0,
                            end_offset=0,
                            source_document_id=source,
                        )
                    )
            else:
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
                                source_document_id=source,
                            )
                        )
        except (ValueError, TypeError, AttributeError) as exc:
            raise EntityExtractionError(
                f"Could not parse LLM response for {source}: {exc}"
            ) from exc

        entities = [e for e in entities if e.confidence >= self.confidence_threshold]
        entities.sort(key=lambda e: -e.confidence)
        return entities[: self.max_entities_per_doc]

    def process_document(self, document: Document) -> Dict[str, Any]:
        """Extract a document's entities and store them when an adapter is set.

        `stored` is True only when the entities were written. Without a storage
        adapter nothing is written and `stored` is False.
        """
        entities = self.extract_entities(document)
        outcome = self.store_entities_and_relationships(entities, [])
        return {
            "document_id": document.id,
            "entities_extracted": len(entities),
            "stored": self.storage_adapter is not None,
            "entities": entities,
            "stored_entities": outcome["stored_entities"],
        }

    def store_entities_and_relationships(self, entities, relationships):
        """Write entities and relationships through the storage adapter.

        Without an adapter nothing is written (counts are 0). A write the adapter
        refuses raises `EntityExtractionError` naming what was not stored.
        """
        stored_entities = 0
        stored_relationships = 0
        if self.storage_adapter:
            entities_refused = [
                str(e.id) for e in entities if not self.storage_adapter.store_entity(e)
            ]
            relationships_refused = [
                str(r.id)
                for r in relationships
                if not self.storage_adapter.store_relationship(r)
            ]
            stored_entities = len(entities) - len(entities_refused)
            stored_relationships = len(relationships) - len(relationships_refused)
            not_stored = entities_refused + relationships_refused
            if not_stored:
                raise EntityExtractionError(
                    f"Could not store {len(not_stored)} item(s): {not_stored[:10]}"
                )
        return {
            "stored_entities": stored_entities,
            "stored_relationships": stored_relationships,
        }

    def extract_relationships(self, entities, document):
        return []
