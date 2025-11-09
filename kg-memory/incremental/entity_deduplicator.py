"""
Entity Deduplication Component for LightRAG-inspired incremental indexing.

This module provides vector similarity-based entity deduplication capabilities
to identify and merge duplicate entities in knowledge graphs during incremental updates.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from iris_vector_rag.core.connection import IRISConnectionManager
from iris_vector_rag.embeddings.manager import EmbeddingManager

logger = logging.getLogger(__name__)


class SimilarityThreshold(Enum):
    """Similarity threshold presets for entity deduplication."""

    STRICT = 0.95  # Very high confidence required
    HIGH = 0.85  # High confidence
    MODERATE = 0.75  # Moderate confidence
    LOOSE = 0.65  # Lower threshold for broader matching


class MergeStrategy(Enum):
    """Strategies for merging duplicate entities."""

    KEEP_FIRST = "keep_first"  # Keep the first entity found
    KEEP_LATEST = "keep_latest"  # Keep the most recently created
    MERGE_PROPERTIES = "merge_properties"  # Merge all properties together
    CUSTOM = "custom"  # Use custom merge function


@dataclass(frozen=True)
class Entity:
    """Represents an entity in the knowledge graph."""

    id: str
    text: str
    type: str
    properties: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: Optional[str] = None
    source_document: Optional[str] = None

    def __hash__(self) -> int:
        """Hash based on entity ID for set operations."""
        return hash(self.id)


@dataclass
class DuplicationCluster:
    """Represents a cluster of duplicate entities."""

    primary_entity: Entity
    duplicate_entities: List[Entity]
    similarity_scores: List[float]
    merge_strategy: MergeStrategy
    merged_entity: Optional[Entity] = None

    @property
    def cluster_size(self) -> int:
        """Total number of entities in the cluster."""
        return 1 + len(self.duplicate_entities)

    @property
    def average_similarity(self) -> float:
        """Average similarity score within the cluster."""
        if not self.similarity_scores:
            return 1.0
        return sum(self.similarity_scores) / len(self.similarity_scores)


@dataclass
class DeduplicationResult:
    """Results of entity deduplication operation."""

    clusters_found: List[DuplicationCluster]
    entities_processed: int
    duplicates_removed: int
    entities_merged: int
    processing_time_ms: float
    similarity_threshold: float

    @property
    def deduplication_rate(self) -> float:
        """Percentage of entities that were duplicates."""
        if self.entities_processed == 0:
            return 0.0
        return (self.duplicates_removed / self.entities_processed) * 100


class EntityDeduplicator:
    """
    Vector similarity-based entity deduplication for incremental knowledge graphs.

    Uses embedding similarity to identify duplicate entities and provides multiple
    merge strategies for consolidating duplicates while maintaining graph integrity.
    """

    def __init__(
        self,
        connection_manager: IRISConnectionManager,
        embedding_manager: EmbeddingManager,
        similarity_threshold: float = SimilarityThreshold.HIGH.value,
        merge_strategy: MergeStrategy = MergeStrategy.MERGE_PROPERTIES,
        batch_size: int = 100,
        custom_merge_fn: Optional[Callable[[List[Entity]], Entity]] = None,
    ):
        """
        Initialize the entity deduplicator.

        Args:
            connection_manager: IRIS database connection manager
            embedding_manager: Embedding generation manager
            similarity_threshold: Minimum similarity score for duplicates (0.0-1.0)
            merge_strategy: Strategy for merging duplicate entities
            batch_size: Batch size for processing entities
            custom_merge_fn: Custom function for merging entities (used with CUSTOM strategy)
        """
        self.connection_manager = connection_manager
        self.embedding_manager = embedding_manager
        self.similarity_threshold = similarity_threshold
        self.merge_strategy = merge_strategy
        self.batch_size = batch_size
        self.custom_merge_fn = custom_merge_fn

        # Performance tracking
        self._stats = {
            "total_comparisons": 0,
            "embedding_generation_time": 0.0,
            "similarity_search_time": 0.0,
            "merge_time": 0.0,
        }

        # Entity embedding cache
        self._embedding_cache: Dict[str, List[float]] = {}

        # Validate configuration
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate deduplicator configuration."""
        if not 0.0 <= self.similarity_threshold <= 1.0:
            raise ValueError(
                f"Similarity threshold must be between 0.0 and 1.0, got {self.similarity_threshold}"
            )

        if self.merge_strategy == MergeStrategy.CUSTOM and self.custom_merge_fn is None:
            raise ValueError(
                "Custom merge function required when using CUSTOM merge strategy"
            )

        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")

    def deduplicate_entities(
        self,
        entities: List[Entity],
        include_existing: bool = True,
        transaction_id: Optional[str] = None,
    ) -> DeduplicationResult:
        """
        Perform deduplication on a list of entities.

        Args:
            entities: List of entities to deduplicate
            include_existing: Whether to check against existing entities in DB
            transaction_id: Optional transaction ID for atomic operations

        Returns:
            DeduplicationResult with clustering and merge information
        """
        start_time = time.time()

        try:
            logger.info(f"Starting deduplication of {len(entities)} entities")

            # Generate embeddings for entities that don't have them
            entities_with_embeddings = self._ensure_embeddings(entities)

            # Find duplication clusters
            clusters = self._find_duplication_clusters(
                entities_with_embeddings, include_existing=include_existing
            )

            # Merge entities within clusters
            self._merge_entity_clusters(clusters, transaction_id)

            # Calculate results
            processing_time = (time.time() - start_time) * 1000
            duplicates_removed = sum(
                len(cluster.duplicate_entities) for cluster in clusters
            )
            entities_merged = len([c for c in clusters if c.merged_entity is not None])

            result = DeduplicationResult(
                clusters_found=clusters,
                entities_processed=len(entities),
                duplicates_removed=duplicates_removed,
                entities_merged=entities_merged,
                processing_time_ms=processing_time,
                similarity_threshold=self.similarity_threshold,
            )

            logger.info(
                f"Deduplication completed: {duplicates_removed} duplicates found, "
                f"{entities_merged} clusters merged in {processing_time:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Entity deduplication failed: {e}")
            raise

    def _ensure_embeddings(self, entities: List[Entity]) -> List[Entity]:
        """Ensure all entities have embeddings, generating them if needed."""
        start_time = time.time()
        entities_with_embeddings = []

        # Collect entities that need embeddings
        entities_needing_embeddings = []
        texts_to_embed = []

        for entity in entities:
            if entity.embedding is None:
                # Check cache first
                if entity.id in self._embedding_cache:
                    entity_with_embedding = Entity(
                        id=entity.id,
                        text=entity.text,
                        type=entity.type,
                        properties=entity.properties,
                        embedding=self._embedding_cache[entity.id],
                        created_at=entity.created_at,
                        source_document=entity.source_document,
                    )
                    entities_with_embeddings.append(entity_with_embedding)
                else:
                    entities_needing_embeddings.append(entity)
                    texts_to_embed.append(entity.text)
            else:
                entities_with_embeddings.append(entity)

        # Generate embeddings in batch
        if texts_to_embed:
            embeddings = self.embedding_manager.embed_texts(texts_to_embed)

            for entity, embedding in zip(entities_needing_embeddings, embeddings):
                # Cache the embedding
                self._embedding_cache[entity.id] = embedding

                # Create entity with embedding
                entity_with_embedding = Entity(
                    id=entity.id,
                    text=entity.text,
                    type=entity.type,
                    properties=entity.properties,
                    embedding=embedding,
                    created_at=entity.created_at,
                    source_document=entity.source_document,
                )
                entities_with_embeddings.append(entity_with_embedding)

        self._stats["embedding_generation_time"] += (time.time() - start_time) * 1000
        return entities_with_embeddings

    def _find_duplication_clusters(
        self, entities: List[Entity], include_existing: bool = True
    ) -> List[DuplicationCluster]:
        """Find clusters of duplicate entities using vector similarity."""
        start_time = time.time()
        clusters = []
        processed_entity_ids = set()

        for entity in entities:
            if entity.id in processed_entity_ids:
                continue

            # Find similar entities
            similar_entities = self._find_similar_entities(
                entity, entities, include_existing
            )

            if similar_entities:
                # Create cluster with this entity as primary
                cluster = DuplicationCluster(
                    primary_entity=entity,
                    duplicate_entities=[ent for ent, _ in similar_entities],
                    similarity_scores=[score for _, score in similar_entities],
                    merge_strategy=self.merge_strategy,
                )
                clusters.append(cluster)

                # Mark all entities in cluster as processed
                processed_entity_ids.add(entity.id)
                processed_entity_ids.update(ent.id for ent, _ in similar_entities)

        self._stats["similarity_search_time"] += (time.time() - start_time) * 1000
        return clusters

    def _find_similar_entities(
        self,
        target_entity: Entity,
        candidate_entities: List[Entity],
        include_existing: bool = True,
    ) -> List[Tuple[Entity, float]]:
        """Find entities similar to the target entity."""
        similar_entities = []

        # Compare with candidate entities
        for candidate in candidate_entities:
            if candidate.id == target_entity.id:
                continue

            # Skip if different types (configurable in future)
            if candidate.type != target_entity.type:
                continue

            similarity = self._calculate_similarity(target_entity, candidate)
            self._stats["total_comparisons"] += 1

            if similarity >= self.similarity_threshold:
                similar_entities.append((candidate, similarity))

        # TODO: If include_existing=True, also search existing entities in database
        # This would involve querying the vector store for similar entities

        # Sort by similarity descending
        similar_entities.sort(key=lambda x: x[1], reverse=True)

        return similar_entities

    def _calculate_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """Calculate cosine similarity between two entities."""
        if entity1.embedding is None or entity2.embedding is None:
            return 0.0

        # Convert to numpy arrays for efficient computation
        vec1 = np.array(entity1.embedding)
        vec2 = np.array(entity2.embedding)

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _merge_entity_clusters(
        self, clusters: List[DuplicationCluster], transaction_id: Optional[str] = None
    ) -> List[Entity]:
        """Merge entities within each cluster according to the merge strategy."""
        start_time = time.time()
        merged_entities = []

        for cluster in clusters:
            try:
                all_entities = [cluster.primary_entity] + cluster.duplicate_entities
                merged_entity = self._merge_entities(
                    all_entities, cluster.merge_strategy
                )
                cluster.merged_entity = merged_entity
                merged_entities.append(merged_entity)

            except Exception as e:
                logger.error(
                    f"Failed to merge cluster with primary {cluster.primary_entity.id}: {e}"
                )
                # Fall back to keeping primary entity
                cluster.merged_entity = cluster.primary_entity
                merged_entities.append(cluster.primary_entity)

        self._stats["merge_time"] += (time.time() - start_time) * 1000
        return merged_entities

    def _merge_entities(
        self, entities: List[Entity], strategy: MergeStrategy
    ) -> Entity:
        """Merge a list of duplicate entities according to the specified strategy."""
        if len(entities) == 1:
            return entities[0]

        if strategy == MergeStrategy.KEEP_FIRST:
            return entities[0]

        elif strategy == MergeStrategy.KEEP_LATEST:
            # Sort by created_at if available, otherwise keep first
            entities_with_time = [e for e in entities if e.created_at is not None]
            if entities_with_time:
                return max(entities_with_time, key=lambda e: e.created_at)
            return entities[0]

        elif strategy == MergeStrategy.MERGE_PROPERTIES:
            return self._merge_properties(entities)

        elif strategy == MergeStrategy.CUSTOM:
            if self.custom_merge_fn is None:
                raise ValueError("Custom merge function not provided")
            return self.custom_merge_fn(entities)

        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

    def _merge_properties(self, entities: List[Entity]) -> Entity:
        """Merge entities by combining their properties."""
        primary = entities[0]
        merged_properties = dict(primary.properties)

        # Merge properties from all entities
        for entity in entities[1:]:
            for key, value in entity.properties.items():
                if key not in merged_properties:
                    merged_properties[key] = value
                elif merged_properties[key] != value:
                    # Handle conflicts by creating lists
                    if not isinstance(merged_properties[key], list):
                        merged_properties[key] = [merged_properties[key]]
                    if value not in merged_properties[key]:
                        merged_properties[key].append(value)

        # Create merged entity with combined properties
        return Entity(
            id=primary.id,  # Keep primary ID
            text=primary.text,  # Keep primary text
            type=primary.type,
            properties=merged_properties,
            embedding=primary.embedding,
            created_at=primary.created_at,
            source_document=primary.source_document,
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the deduplicator."""
        return {
            "total_comparisons": self._stats["total_comparisons"],
            "embedding_generation_time_ms": self._stats["embedding_generation_time"],
            "similarity_search_time_ms": self._stats["similarity_search_time"],
            "merge_time_ms": self._stats["merge_time"],
            "cache_size": len(self._embedding_cache),
            "similarity_threshold": self.similarity_threshold,
            "merge_strategy": self.merge_strategy.value,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache to free memory."""
        self._embedding_cache.clear()
        logger.info("Entity embedding cache cleared")

    def set_similarity_threshold(self, threshold: float) -> None:
        """Update the similarity threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        self.similarity_threshold = threshold
        logger.info(f"Similarity threshold updated to {threshold}")

    def validate_performance(self, max_time_per_entity_ms: float = 30.0) -> bool:
        """
        Validate that deduplication performance meets requirements.

        Args:
            max_time_per_entity_ms: Maximum allowed processing time per entity

        Returns:
            True if performance requirements are met
        """
        total_time = (
            self._stats["embedding_generation_time"]
            + self._stats["similarity_search_time"]
            + self._stats["merge_time"]
        )

        if self._stats["total_comparisons"] == 0:
            return True

        avg_time_per_comparison = total_time / self._stats["total_comparisons"]

        performance_ok = avg_time_per_comparison <= max_time_per_entity_ms

        if not performance_ok:
            logger.warning(
                f"Performance validation failed: {avg_time_per_comparison:.2f}ms per entity "
                f"exceeds limit of {max_time_per_entity_ms}ms"
            )

        return performance_ok
