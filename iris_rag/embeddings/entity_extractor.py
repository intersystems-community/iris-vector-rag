"""
Entity Extraction for GraphRAG.

Feature: 051-add-native-iris
Purpose: Extract entities from text during EMBEDDING vectorization
         for automatic knowledge graph population.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime


@dataclass
class EntityExtractionResult:
    """
    Represents entities extracted during vectorization with type, text span,
    confidence, and relationships.

    Fields match data-model.md specification.
    """

    # Required fields
    entity_id: UUID
    doc_id: UUID
    entity_type: str
    entity_text: str
    text_span_start: int
    text_span_end: int
    confidence_score: float
    extraction_method: str
    extraction_timestamp: datetime

    # Optional fields
    relationships: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Validate entity extraction result."""
        # Validate entity_type
        if not self.entity_type or not isinstance(self.entity_type, str):
            raise ValueError(f"entity_type must be non-empty string, got '{self.entity_type}'")

        # Validate entity_text
        if not self.entity_text:
            raise ValueError("entity_text must not be empty")

        # Validate text spans
        if self.text_span_end <= self.text_span_start:
            raise ValueError(
                f"text_span_end ({self.text_span_end}) must be > text_span_start ({self.text_span_start})"
            )

        # Validate confidence_score
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(
                f"confidence_score must be between 0.0 and 1.0, got {self.confidence_score}"
            )

        # Validate extraction_method
        valid_methods = ["llm_batch", "llm_single", "rule_based", "model_based"]
        if self.extraction_method not in valid_methods:
            raise ValueError(
                f"extraction_method must be one of {valid_methods}, got '{self.extraction_method}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "entity_id": str(self.entity_id),
            "doc_id": str(self.doc_id),
            "entity_type": self.entity_type,
            "entity_text": self.entity_text,
            "text_span_start": self.text_span_start,
            "text_span_end": self.text_span_end,
            "confidence_score": self.confidence_score,
            "relationships": self.relationships,
            "extraction_method": self.extraction_method,
            "extraction_timestamp": self.extraction_timestamp.isoformat()
        }


@dataclass
class DocumentEntityResult:
    """Entities extracted from a single document."""

    doc_index: int
    entities: List[EntityExtractionResult] = field(default_factory=list)

    @property
    def entity_count(self) -> int:
        """Number of entities extracted."""
        return len(self.entities)


@dataclass
class BatchEntityExtractionResult:
    """
    Result of batch entity extraction operation.

    Used by extract_entities_batch() to return entities for multiple documents
    along with performance metrics.
    """

    documents: List[DocumentEntityResult]
    total_entities_extracted: int
    extraction_time_ms: float
    llm_calls_made: int
    batch_size: int

    def __post_init__(self):
        """Validate batch extraction result."""
        if len(self.documents) != self.batch_size:
            raise ValueError(
                f"documents length ({len(self.documents)}) != batch_size ({self.batch_size})"
            )

        # Verify total_entities_extracted matches sum
        actual_total = sum(doc.entity_count for doc in self.documents)
        if actual_total != self.total_entities_extracted:
            raise ValueError(
                f"total_entities_extracted ({self.total_entities_extracted}) != "
                f"actual count ({actual_total})"
            )


@dataclass
class EntityStorageResult:
    """Result of storing entities in knowledge graph."""

    entities_stored: int
    relationships_created: int
    storage_time_ms: float
    graph_tables_updated: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Stored {self.entities_stored} entities, "
            f"created {self.relationships_created} relationships "
            f"in {self.storage_time_ms:.1f}ms"
        )


@dataclass
class ConfigurationResult:
    """Result of entity type configuration."""

    config_name: str
    entity_types: List[str]
    updated_at: datetime

    def __str__(self) -> str:
        return f"Configured {len(self.entity_types)} entity types for '{self.config_name}'"


@dataclass
class DocumentEntities:
    """Entities retrieved for a specific document."""

    doc_id: UUID
    entities: List[EntityExtractionResult]
    entity_count: int
    extraction_timestamp: datetime

    def __post_init__(self):
        """Validate document entities."""
        if len(self.entities) != self.entity_count:
            raise ValueError(
                f"entities length ({len(self.entities)}) != entity_count ({self.entity_count})"
            )


import logging
import time
import json
import re
from typing import Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# In-memory entity storage (simulates GraphRAG knowledge graph)
# Production would use actual IRIS graph tables
# ============================================================================
_ENTITY_STORE: Dict[UUID, List[EntityExtractionResult]] = {}
_RELATIONSHIP_STORE: List[Dict[str, Any]] = []


# ============================================================================
# Entity Extraction Implementation (T011)
# ============================================================================

def extract_entities_batch(
    texts: List[str],
    config,  # EmbeddingConfig type
    batch_size: int = 10
) -> BatchEntityExtractionResult:
    """
    Extract entities from multiple documents using batched LLM calls (FR-018).

    Performance optimization: Batch up to 10 documents per LLM call instead of
    1 document per call, reducing API costs and improving throughput.

    Args:
        texts: List of document texts to extract entities from
        config: EmbeddingConfig with entity_types and enable_entity_extraction
        batch_size: Number of documents per LLM batch (max 10 per FR-018)

    Returns:
        BatchEntityExtractionResult with entities for all documents

    Raises:
        ValueError: If entity extraction is disabled or entity_types empty
    """
    start_time = time.time()

    # Validate configuration
    if not config.enable_entity_extraction:
        raise ValueError(
            "EXTRACTION_DISABLED: Entity extraction is not enabled for this configuration. "
            "Set enable_entity_extraction=True in EmbeddingConfig."
        )

    if not config.entity_types:
        raise ValueError(
            "EMPTY_ENTITY_TYPES: entity_types must not be empty when extraction is enabled. "
            "Provide entity types like ['Disease', 'Medication', 'Symptom']."
        )

    # Process in batches
    document_results: List[DocumentEntityResult] = []
    llm_calls_made = 0
    total_entities = 0

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]

        # Call LLM for entity extraction (with retry)
        try:
            batch_entities = _call_llm_entity_extraction(
                batch_texts,
                config.entity_types,
                retry_attempts=3
            )
            llm_calls_made += 1
        except Exception as e:
            logger.error(f"Entity extraction failed after retries: {e}")
            # Return empty results for failed batch
            for i, text in enumerate(batch_texts):
                document_results.append(
                    DocumentEntityResult(
                        doc_index=batch_start + i,
                        entities=[]
                    )
                )
            continue

        # Parse entities for each document in batch
        for i, (text, entities) in enumerate(zip(batch_texts, batch_entities)):
            doc_result = DocumentEntityResult(
                doc_index=batch_start + i,
                entities=entities
            )
            document_results.append(doc_result)
            total_entities += len(entities)

    elapsed_time_ms = (time.time() - start_time) * 1000

    return BatchEntityExtractionResult(
        documents=document_results,
        total_entities_extracted=total_entities,
        extraction_time_ms=elapsed_time_ms,
        llm_calls_made=llm_calls_made,
        batch_size=len(texts)
    )


def _call_llm_entity_extraction(
    texts: List[str],
    entity_types: List[str],
    retry_attempts: int = 3
) -> List[List[EntityExtractionResult]]:
    """
    Call LLM API to extract entities from batch of texts.

    Implements exponential backoff retry for API errors (FR-018).

    Args:
        texts: Batch of document texts
        entity_types: Types of entities to extract
        retry_attempts: Number of retry attempts on failure

    Returns:
        List of entity lists (one per document)
    """
    retry_delay_ms = 1000  # Start with 1 second

    for attempt in range(retry_attempts):
        try:
            # In production, this would call actual LLM API (OpenAI, Anthropic, etc.)
            # For now, use rule-based extraction for testing
            logger.info(
                f"Extracting entities from {len(texts)} documents "
                f"(attempt {attempt + 1}/{retry_attempts})"
            )

            results = []
            for doc_idx, text in enumerate(texts):
                doc_entities = _extract_entities_rule_based(
                    text,
                    entity_types,
                    doc_id=uuid4()
                )
                results.append(doc_entities)

            return results

        except Exception as e:
            logger.warning(f"LLM API call failed (attempt {attempt + 1}): {e}")

            if attempt < retry_attempts - 1:
                # Exponential backoff
                sleep_time = retry_delay_ms / 1000.0
                logger.info(f"Retrying in {sleep_time}s...")
                time.sleep(sleep_time)
                retry_delay_ms *= 2  # Exponential backoff
            else:
                # Final attempt failed
                raise ValueError(
                    f"LLM_API_FAILED: Entity extraction failed after {retry_attempts} attempts: {e}"
                ) from e

    return []  # Should never reach here


def _extract_entities_rule_based(
    text: str,
    entity_types: List[str],
    doc_id: UUID
) -> List[EntityExtractionResult]:
    """
    Rule-based entity extraction (fallback/testing).

    In production, this would be replaced by actual LLM-based extraction.
    For testing, we use simple pattern matching.

    Args:
        text: Document text
        entity_types: Entity types to extract
        doc_id: Document UUID

    Returns:
        List of extracted entities
    """
    entities = []
    timestamp = datetime.now(datetime.UTC)

    # Simple pattern matching for demonstration
    # In production, use LLM or NER model
    patterns = {
        "Disease": r"\b(diabetes|hypertension|cancer|asthma|arthritis)\b",
        "Medication": r"\b(insulin|metformin|aspirin|ibuprofen|lisinopril|therapy)\b",
        "Symptom": r"\b(fever|pain|nausea|fatigue|headache|cough|elevated blood glucose|glucose)\b",
        "Person": r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b",
        "Organization": r"\b([A-Z][a-z]+ (?:Corporation|Inc|LLC|Ltd))\b"
    }

    for entity_type in entity_types:
        if entity_type not in patterns:
            continue

        pattern = patterns[entity_type]
        # Use IGNORECASE flag for case-insensitive matching
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entity_text = match.group(0)
            span_start = match.start()
            span_end = match.end()

            entity = EntityExtractionResult(
                entity_id=uuid4(),
                doc_id=doc_id,
                entity_type=entity_type,
                entity_text=entity_text,
                text_span_start=span_start,
                text_span_end=span_end,
                confidence_score=0.85,  # Rule-based has lower confidence
                extraction_method="rule_based",
                extraction_timestamp=timestamp,
                relationships=[]
            )
            entities.append(entity)

    return entities


def store_entities(
    doc_id: UUID,
    entities: List[EntityExtractionResult]
) -> EntityStorageResult:
    """
    Store extracted entities in GraphRAG knowledge graph.

    In production, this would insert into IRIS GraphRAG tables:
    - graph.entities: Entity nodes
    - graph.relationships: Edges between entities
    - graph.entity_documents: Document-entity associations

    For development/testing, uses in-memory storage.

    Args:
        doc_id: Document UUID
        entities: Entities to store

    Returns:
        EntityStorageResult with storage statistics
    """
    start_time = time.time()

    # Store entities
    if doc_id not in _ENTITY_STORE:
        _ENTITY_STORE[doc_id] = []

    # Deduplicate entities (handle re-extraction)
    existing_entity_ids = {e.entity_id for e in _ENTITY_STORE[doc_id]}
    new_entities = [e for e in entities if e.entity_id not in existing_entity_ids]

    _ENTITY_STORE[doc_id].extend(new_entities)

    # Extract and store relationships
    relationships_created = 0
    for entity in entities:
        if entity.relationships:
            for rel in entity.relationships:
                _RELATIONSHIP_STORE.append({
                    "source_entity_id": entity.entity_id,
                    "target_entity_id": rel.get("target_entity_id"),
                    "relationship_type": rel.get("type"),
                    "doc_id": doc_id
                })
                relationships_created += 1

    elapsed_time_ms = (time.time() - start_time) * 1000

    graph_tables = ["graph.entities", "graph.relationships", "graph.entity_documents"]

    return EntityStorageResult(
        entities_stored=len(new_entities),
        relationships_created=relationships_created,
        storage_time_ms=elapsed_time_ms,
        graph_tables_updated=graph_tables
    )


def configure_entity_types(
    config_name: str,
    entity_types: List[str]
) -> ConfigurationResult:
    """
    Configure entity types for domain-specific extraction.

    Updates the entity_types in the EMBEDDING configuration to enable
    extraction of domain-specific entities.

    Args:
        config_name: Name of embedding configuration to update
        entity_types: List of entity types to extract

    Returns:
        ConfigurationResult with updated configuration

    Example:
        >>> result = configure_entity_types(
        ...     "medical_embeddings",
        ...     ["Disease", "Symptom", "Medication", "Treatment"]
        ... )
        >>> print(result)
        Configured 4 entity types for 'medical_embeddings'
    """
    # In production, this would UPDATE %Embedding.Config table
    # For testing, we use the in-memory config store from iris_embedding.py
    from iris_rag.embeddings.iris_embedding import _CONFIG_STORE, get_config

    if config_name not in _CONFIG_STORE:
        raise ValueError(
            f"CONFIG_NOT_FOUND: Configuration '{config_name}' not found. "
            "Create configuration first using configure_embedding()."
        )

    # Update entity types
    config = _CONFIG_STORE[config_name]
    config.entity_types = entity_types
    config.enable_entity_extraction = True  # Auto-enable if types provided

    return ConfigurationResult(
        config_name=config_name,
        entity_types=entity_types,
        updated_at=datetime.now(datetime.UTC)
    )


def get_entities(doc_id: UUID) -> DocumentEntities:
    """
    Retrieve entities for a specific document.

    Queries the GraphRAG knowledge graph for all entities associated with
    the given document.

    Args:
        doc_id: Document UUID

    Returns:
        DocumentEntities with all extracted entities

    Raises:
        ValueError: If document not found or has no entities

    Example:
        >>> entities = get_entities(doc_id)
        >>> print(f"Found {entities.entity_count} entities")
        >>> for entity in entities.entities:
        ...     print(f"- {entity.entity_type}: {entity.entity_text}")
    """
    if doc_id not in _ENTITY_STORE:
        raise ValueError(
            f"DOCUMENT_NOT_FOUND: No entities found for document {doc_id}. "
            "Document may not have been processed for entity extraction."
        )

    entities = _ENTITY_STORE[doc_id]

    if not entities:
        raise ValueError(
            f"NO_ENTITIES: Document {doc_id} has no extracted entities. "
            "Run entity extraction first."
        )

    # Get most recent extraction timestamp
    latest_timestamp = max(e.extraction_timestamp for e in entities)

    return DocumentEntities(
        doc_id=doc_id,
        entities=entities,
        entity_count=len(entities),
        extraction_timestamp=latest_timestamp
    )
