#!/usr/bin/env python3
"""
Incremental Learning Manager for RAG Memory Systems.

Demonstrates incremental learning patterns using existing M2 infrastructure.
Leverages CDC detector and graph union operations for efficient knowledge updates.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field

from iris_rag.memory.models import KnowledgePattern, LearningResult
from kg_memory.incremental.cdc_detector import CDCDetector, ChangeBatch
from kg_memory.incremental.graph_union import GraphUnionOperator
from iris_rag.core.models import Document

logger = logging.getLogger(__name__)


@dataclass
class IncrementalUpdate:
    """Represents an incremental knowledge update."""

    update_id: str
    new_documents: List[Document] = field(default_factory=list)
    updated_documents: List[Document] = field(default_factory=list)
    deleted_document_ids: List[str] = field(default_factory=list)
    knowledge_patterns: List[KnowledgePattern] = field(default_factory=list)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingUpdateResult:
    """Result of incremental embedding updates."""

    updated_embeddings: int = 0
    new_embeddings: int = 0
    deleted_embeddings: int = 0
    processing_time_ms: float = 0.0
    vector_store_updates: Dict[str, Any] = field(default_factory=dict)


class IncrementalLearningManager:
    """
    Incremental learning patterns for RAG memory systems.

    Features:
    - Leverages existing M2 CDC detector for change detection
    - Uses M2 graph union operations for knowledge merging
    - Efficient vector embedding updates
    - Knowledge pattern evolution tracking
    - Performance target: <30s for 1K document updates
    """

    def __init__(
        self,
        cdc_detector: CDCDetector,
        graph_union: GraphUnionOperator,
        knowledge_extractor: Any = None,
    ):
        """
        Initialize incremental learning manager.

        Args:
            cdc_detector: Change detection component from M2
            graph_union: Graph union operator from M2
            knowledge_extractor: Knowledge pattern extractor
        """
        self.cdc_detector = cdc_detector
        self.graph_union = graph_union
        self.knowledge_extractor = knowledge_extractor

        # Performance tracking
        self._learning_times: List[float] = []
        self._update_batch_sizes: List[int] = []
        self._embedding_update_times: List[float] = []

        logger.info("Incremental learning manager initialized")

    async def process_knowledge_updates(
        self, documents: List[Document]
    ) -> LearningResult:
        """
        Process incremental knowledge updates using CDC and graph union.

        Performance target: <30s for 1K documents

        Args:
            documents: List of documents to process incrementally

        Returns:
            LearningResult with update statistics
        """
        start_time = time.perf_counter()

        try:
            # Step 1: Detect changes using existing M2 CDC
            logger.debug("Detecting document changes...")
            change_batch = self.cdc_detector.detect_changes(documents)

            if not change_batch.has_changes:
                return LearningResult(
                    processing_time_ms=(time.perf_counter() - start_time) * 1000,
                    success=True,
                )

            # Step 2: Extract knowledge patterns from changed documents
            logger.debug(
                f"Extracting knowledge from {change_batch.total_changes} changes..."
            )
            new_patterns = await self._extract_knowledge_from_changes(change_batch)

            # Step 3: Merge knowledge using graph union operations
            logger.debug("Merging knowledge patterns...")
            learning_result = await self._merge_knowledge_incrementally(new_patterns)

            # Step 4: Update fingerprints for change tracking
            fingerprint_success = self.cdc_detector.update_fingerprints(change_batch)
            if not fingerprint_success:
                logger.warning("Failed to update document fingerprints")

            # Step 5: Update performance metrics
            processing_time = (time.perf_counter() - start_time) * 1000
            learning_result.processing_time_ms = processing_time
            self._learning_times.append(processing_time)
            self._update_batch_sizes.append(change_batch.total_changes)

            # Performance validation
            if (
                len(documents) >= 1000 and processing_time > 30000
            ):  # 30s target for 1K docs
                logger.warning(
                    f"Incremental learning took {processing_time/1000:.2f}s for {len(documents)} documents "
                    f"(target: <30s for 1K documents)"
                )

            logger.info(
                f"Incremental learning completed: {learning_result.total_changes} changes "
                f"in {processing_time:.2f}ms"
            )

            return learning_result

        except Exception as e:
            logger.error(f"Error in incremental knowledge updates: {e}")
            return LearningResult(
                processing_time_ms=(time.perf_counter() - start_time) * 1000,
                success=False,
                error_message=str(e),
            )

    async def update_embeddings_incrementally(
        self, changed_docs: List[Document], vector_store: Any = None
    ) -> EmbeddingUpdateResult:
        """
        Update embeddings without full recomputation.

        Args:
            changed_docs: Documents that have changed
            vector_store: Vector store interface (optional)

        Returns:
            EmbeddingUpdateResult with update statistics
        """
        start_time = time.perf_counter()

        try:
            result = EmbeddingUpdateResult()

            if not vector_store:
                logger.warning("No vector store provided for embedding updates")
                return result

            # Process each changed document
            for doc in changed_docs:
                try:
                    # Check if document already has embeddings
                    existing_embedding = await self._get_existing_embedding(
                        doc.id, vector_store
                    )

                    if existing_embedding:
                        # Update existing embedding
                        await self._update_document_embedding(doc, vector_store)
                        result.updated_embeddings += 1
                    else:
                        # Create new embedding
                        await self._create_document_embedding(doc, vector_store)
                        result.new_embeddings += 1

                except Exception as e:
                    logger.warning(
                        f"Error updating embedding for document {doc.id}: {e}"
                    )

            result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            self._embedding_update_times.append(result.processing_time_ms)

            logger.debug(
                f"Embedding updates completed: {result.new_embeddings} new, "
                f"{result.updated_embeddings} updated in {result.processing_time_ms:.2f}ms"
            )

            return result

        except Exception as e:
            logger.error(f"Error in incremental embedding updates: {e}")
            result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            return result

    async def learn_from_query_patterns(
        self, query_history: List[Dict[str, Any]]
    ) -> LearningResult:
        """
        Learn from query patterns to improve future responses.

        This demonstrates how applications can implement continuous learning
        from user behavior patterns.

        Args:
            query_history: List of query patterns with responses and feedback

        Returns:
            LearningResult with learning statistics
        """
        try:
            patterns = []

            # Analyze query patterns for common themes
            query_themes = self._analyze_query_themes(query_history)

            # Create knowledge patterns from frequent themes
            for theme, frequency in query_themes.items():
                if frequency >= 3:  # Minimum frequency threshold
                    pattern = KnowledgePattern(
                        pattern_id=f"query_theme_{hash(theme)}",
                        pattern_type="query_pattern",
                        source_rag_technique="incremental_learning",
                        context_summary=f"Common query theme: {theme}",
                        extraction_confidence=min(
                            1.0, frequency / 10.0
                        ),  # Scale confidence
                    )
                    patterns.append(pattern)

            return LearningResult(new_patterns=patterns, success=True)

        except Exception as e:
            logger.error(f"Error learning from query patterns: {e}")
            return LearningResult(success=False, error_message=str(e))

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for incremental learning."""
        return {
            "learning_performance": {
                "avg_time_ms": (
                    sum(self._learning_times) / len(self._learning_times)
                    if self._learning_times
                    else 0
                ),
                "max_time_ms": max(self._learning_times) if self._learning_times else 0,
                "total_learning_cycles": len(self._learning_times),
                "avg_batch_size": (
                    sum(self._update_batch_sizes) / len(self._update_batch_sizes)
                    if self._update_batch_sizes
                    else 0
                ),
            },
            "embedding_performance": {
                "avg_update_time_ms": (
                    sum(self._embedding_update_times)
                    / len(self._embedding_update_times)
                    if self._embedding_update_times
                    else 0
                ),
                "total_embedding_updates": len(self._embedding_update_times),
            },
            "performance_targets": {
                "target_met_1k_docs": (
                    len([t for t in self._learning_times if t <= 30000])
                    / len(self._learning_times)
                    * 100
                    if self._learning_times
                    else 0
                )
            },
        }

    async def _extract_knowledge_from_changes(
        self, change_batch: ChangeBatch
    ) -> List[KnowledgePattern]:
        """Extract knowledge patterns from document changes."""
        patterns = []

        if not self.knowledge_extractor:
            logger.warning("No knowledge extractor available")
            return patterns

        try:
            # Process new documents
            for change_info in change_batch.new_documents:
                # Create mock RAG response for knowledge extraction
                mock_response = self._create_mock_rag_response(change_info)
                extracted_patterns = self.knowledge_extractor.extract_patterns(
                    mock_response
                )
                patterns.extend(extracted_patterns)

            # Process modified documents
            for change_info in change_batch.modified_documents:
                mock_response = self._create_mock_rag_response(change_info)
                extracted_patterns = self.knowledge_extractor.extract_patterns(
                    mock_response
                )
                patterns.extend(extracted_patterns)

        except Exception as e:
            logger.warning(f"Error extracting knowledge from changes: {e}")

        return patterns

    async def _merge_knowledge_incrementally(
        self, new_patterns: List[KnowledgePattern]
    ) -> LearningResult:
        """Merge new knowledge patterns using graph union operations."""
        try:
            # Convert patterns to entities and relationships for graph operations
            new_entities, new_relationships = self._patterns_to_graph_elements(
                new_patterns
            )

            # Get existing knowledge (simplified - would query from storage)
            existing_entities, existing_relationships = (
                await self._get_existing_knowledge()
            )

            # Perform graph union operations using M2 infrastructure
            entity_result = self.graph_union.union_entities(
                existing_entities, new_entities
            )
            relationship_result = self.graph_union.union_relationships(
                existing_relationships, new_relationships
            )

            # Apply atomic update
            update_success = self.graph_union.atomic_graph_update(
                entity_result, relationship_result
            )

            if update_success:
                # Convert results back to knowledge patterns
                updated_patterns = self._graph_elements_to_patterns(
                    entity_result.updated_entities + entity_result.new_entities,
                    relationship_result.updated_relationships
                    + relationship_result.new_relationships,
                )

                return LearningResult(
                    new_patterns=new_patterns,
                    updated_patterns=updated_patterns,
                    merged_entities=entity_result.total_changes,
                    merged_relationships=relationship_result.total_changes,
                    success=True,
                )
            else:
                return LearningResult(
                    success=False, error_message="Atomic graph update failed"
                )

        except Exception as e:
            logger.error(f"Error in incremental knowledge merging: {e}")
            return LearningResult(success=False, error_message=str(e))

    def _patterns_to_graph_elements(
        self, patterns: List[KnowledgePattern]
    ) -> Tuple[Set[Any], Set[Any]]:
        """Convert knowledge patterns to graph entities and relationships."""
        entities = set()
        relationships = set()

        for pattern in patterns:
            entities.update(pattern.entities)
            relationships.update(pattern.relationships)

        return entities, relationships

    async def _get_existing_knowledge(self) -> Tuple[Set[Any], Set[Any]]:
        """Get existing knowledge from storage (simplified implementation)."""
        # This would query actual storage in a real implementation
        return set(), set()

    def _graph_elements_to_patterns(
        self, entities: List[Any], relationships: List[Any]
    ) -> List[KnowledgePattern]:
        """Convert graph elements back to knowledge patterns."""
        # Simplified conversion - real implementation would be more sophisticated
        if entities or relationships:
            pattern = KnowledgePattern(
                pattern_id=f"merged_pattern_{hash(str(entities))}",
                pattern_type="merged_pattern",
                source_rag_technique="incremental_learning",
                entities=list(entities),
                relationships=list(relationships),
            )
            return [pattern]
        return []

    def _create_mock_rag_response(self, change_info: Any) -> Dict[str, Any]:
        """Create mock RAG response for knowledge extraction."""
        return {
            "response_text": f"Document content for {change_info.document_id}",
            "query": "incremental_update",
            "technique_used": "incremental_learning",
            "retrieved_docs": [],
        }

    async def _get_existing_embedding(
        self, doc_id: str, vector_store: Any
    ) -> Optional[List[float]]:
        """Check if document already has embeddings in vector store."""
        try:
            # This would be implemented based on specific vector store interface
            return None
        except Exception:
            return None

    async def _update_document_embedding(
        self, doc: Document, vector_store: Any
    ) -> None:
        """Update existing document embedding."""
        # Implementation depends on vector store interface

    async def _create_document_embedding(
        self, doc: Document, vector_store: Any
    ) -> None:
        """Create new document embedding."""
        # Implementation depends on vector store interface

    def _analyze_query_themes(
        self, query_history: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Analyze query history for common themes."""
        themes = {}

        for query_data in query_history:
            query_text = query_data.get("query", "").lower()

            # Simple keyword-based theme extraction
            # Real implementation would use more sophisticated NLP
            keywords = query_text.split()
            for keyword in keywords:
                if len(keyword) > 3:  # Filter short words
                    themes[keyword] = themes.get(keyword, 0) + 1

        return themes
