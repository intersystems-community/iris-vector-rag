#!/usr/bin/env python3
"""
RAG-Memory Integration Architecture.

Demonstrates how to add memory capabilities to any RAG pipeline.
Provides base patterns for memory-enabled RAG systems.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from iris_rag.core.connection import ConnectionManager
from iris_rag.memory.knowledge_extractor import KnowledgePatternExtractor
from iris_rag.memory.models import (
    EnrichedRAGResponse,
    KnowledgePattern,
    MemoryConfig,
    MemoryItem,
    TemporalQuery,
    TemporalWindow,
)
from iris_rag.memory.temporal_manager import TemporalMemoryManager

logger = logging.getLogger(__name__)


@dataclass
class MemoryEnhancementConfig:
    """Configuration for memory enhancement in RAG pipelines."""

    enable_pattern_extraction: bool = True
    enable_temporal_storage: bool = True
    enable_context_enrichment: bool = True
    default_temporal_window: TemporalWindow = TemporalWindow.SHORT_TERM
    max_context_items: int = 5
    context_relevance_threshold: float = 0.6


class RAGPipelineWrapper:
    """
    Base wrapper for adding memory to existing RAG pipelines.

    This class shows how to wrap any RAG pipeline with memory capabilities
    without modifying the original pipeline implementation.
    """

    def __init__(
        self,
        base_pipeline: Any,
        memory_config: MemoryConfig,
        connection_manager: ConnectionManager,
    ):
        """
        Initialize memory-enabled RAG pipeline wrapper.

        Args:
            base_pipeline: Any existing RAG pipeline
            memory_config: Memory configuration
            connection_manager: Database connection manager
        """
        self.base_pipeline = base_pipeline
        self.memory_config = memory_config
        self.connection_manager = connection_manager

        # Initialize memory components
        self.knowledge_extractor = KnowledgePatternExtractor(
            memory_config.knowledge_extraction
        )
        self.temporal_manager = TemporalMemoryManager(
            connection_manager, memory_config.temporal_windows
        )

        # Enhancement configuration
        self.enhancement_config = MemoryEnhancementConfig()
        if "enhancement" in memory_config.custom_config:
            enhancement_dict = memory_config.custom_config["enhancement"]
            for key, value in enhancement_dict.items():
                if hasattr(self.enhancement_config, key):
                    setattr(self.enhancement_config, key, value)

        logger.info("RAG pipeline wrapper initialized with memory capabilities")

    async def query_with_memory(self, query: str, **kwargs) -> EnrichedRAGResponse:
        """
        Execute RAG query with memory enrichment.

        This is the main method that demonstrates how to add memory
        to any RAG pipeline.

        Args:
            query: User query
            **kwargs: Additional arguments for base pipeline

        Returns:
            Enriched RAG response with memory context
        """
        try:
            # Step 1: Get relevant memory context
            memory_context = []
            if self.enhancement_config.enable_context_enrichment:
                memory_context = await self._get_memory_context(query)

            # Step 2: Enrich query with memory context (optional)
            enriched_query = self._enrich_query_with_context(query, memory_context)

            # Step 3: Execute base RAG pipeline
            base_response = await self._execute_base_pipeline(enriched_query, **kwargs)

            # Step 4: Extract knowledge patterns from response
            extracted_patterns = []
            if self.enhancement_config.enable_pattern_extraction:
                extracted_patterns = self.knowledge_extractor.extract_patterns(
                    base_response
                )

            # Step 5: Store new patterns in temporal memory
            if self.enhancement_config.enable_temporal_storage and extracted_patterns:
                await self._store_patterns_in_memory(extracted_patterns, query)

            # Step 6: Create enriched response
            enriched_response = EnrichedRAGResponse(
                base_response=base_response,
                memory_context=memory_context,
                extracted_patterns=extracted_patterns,
                enrichment_metadata={
                    "memory_context_count": len(memory_context),
                    "extracted_patterns_count": len(extracted_patterns),
                    "enhancement_enabled": {
                        "pattern_extraction": self.enhancement_config.enable_pattern_extraction,
                        "temporal_storage": self.enhancement_config.enable_temporal_storage,
                        "context_enrichment": self.enhancement_config.enable_context_enrichment,
                    },
                },
            )

            logger.debug(
                f"RAG query with memory completed: {len(memory_context)} context items, "
                f"{len(extracted_patterns)} patterns extracted"
            )

            return enriched_response

        except Exception as e:
            logger.error(f"Error in memory-enhanced RAG query: {e}")
            # Fallback to base pipeline only
            base_response = await self._execute_base_pipeline(query, **kwargs)
            return EnrichedRAGResponse(
                base_response=base_response,
                enrichment_metadata={"error": str(e), "fallback_used": True},
            )

    async def _get_memory_context(self, query: str) -> List[MemoryItem]:
        """Retrieve relevant memory context for query."""
        try:
            temporal_query = TemporalQuery(
                query_text=query,
                window=self.enhancement_config.default_temporal_window,
                max_results=self.enhancement_config.max_context_items,
                relevance_threshold=self.enhancement_config.context_relevance_threshold,
            )

            temporal_context = await self.temporal_manager.retrieve_temporal_context(
                temporal_query
            )
            return temporal_context.items

        except Exception as e:
            logger.warning(f"Error retrieving memory context: {e}")
            return []

    def _enrich_query_with_context(
        self, query: str, memory_context: List[MemoryItem]
    ) -> str:
        """Enrich query with relevant memory context."""
        if not memory_context:
            return query

        # Simple context enrichment - applications can customize this
        context_snippets = []
        for item in memory_context[:3]:  # Top 3 most relevant
            if item.relevance_score > 0.7:  # High relevance only
                context_snippets.append(item.content[:100])  # First 100 chars

        if context_snippets:
            context_text = " ".join(context_snippets)
            enriched_query = f"Context: {context_text}\n\nQuery: {query}"
            logger.debug(
                f"Query enriched with {len(context_snippets)} context snippets"
            )
            return enriched_query

        return query

    async def _execute_base_pipeline(self, query: str, **kwargs) -> Any:
        """Execute the base RAG pipeline."""
        # Adapt to different pipeline interfaces
        if hasattr(self.base_pipeline, "query"):
            if asyncio.iscoroutinefunction(self.base_pipeline.query):
                return await self.base_pipeline.query(query, **kwargs)
            else:
                return self.base_pipeline.query(query, **kwargs)
        elif hasattr(self.base_pipeline, "run"):
            if asyncio.iscoroutinefunction(self.base_pipeline.run):
                return await self.base_pipeline.run(query, **kwargs)
            else:
                return self.base_pipeline.run(query, **kwargs)
        elif callable(self.base_pipeline):
            if asyncio.iscoroutinefunction(self.base_pipeline):
                return await self.base_pipeline(query, **kwargs)
            else:
                return self.base_pipeline(query, **kwargs)
        else:
            raise ValueError(
                f"Cannot determine how to execute pipeline: {type(self.base_pipeline)}"
            )

    async def _store_patterns_in_memory(
        self, patterns: List[KnowledgePattern], source_query: str
    ) -> None:
        """Store extracted patterns in temporal memory."""
        try:
            for pattern in patterns:
                # Convert pattern to memory content
                memory_content = {
                    "pattern_type": pattern.pattern_type,
                    "entities": [entity.name for entity in pattern.entities],
                    "relationships": [
                        f"{rel.source_entity_id}-{rel.relationship_type}-{rel.target_entity_id}"
                        for rel in pattern.relationships
                    ],
                    "context": pattern.context_summary,
                    "confidence": pattern.extraction_confidence,
                }

                await self.temporal_manager.store_with_window(
                    content=memory_content,
                    window=self.enhancement_config.default_temporal_window,
                    context={
                        "source_query": source_query,
                        "rag_technique": pattern.source_rag_technique,
                        "pattern_id": pattern.pattern_id,
                    },
                    source_query=source_query,
                )

        except Exception as e:
            logger.warning(f"Error storing patterns in memory: {e}")


class MemoryEnabledRAGPipeline:
    """
    Complete memory-enabled RAG pipeline implementation.

    This class provides a complete example of how to build a RAG pipeline
    with integrated memory capabilities from the ground up.
    """

    def __init__(
        self,
        base_pipeline: Any,
        memory_config: MemoryConfig,
        connection_manager: ConnectionManager,
    ):
        """Initialize memory-enabled RAG pipeline."""
        self.wrapper = RAGPipelineWrapper(
            base_pipeline, memory_config, connection_manager
        )
        self.performance_metrics = {
            "total_queries": 0,
            "memory_enhanced_queries": 0,
            "average_response_time_ms": 0.0,
            "memory_enhancement_time_ms": 0.0,
        }

    async def query(self, query: str, **kwargs) -> EnrichedRAGResponse:
        """Execute memory-enhanced RAG query."""
        start_time = time.perf_counter()

        try:
            # Execute with memory enhancement
            response = await self.wrapper.query_with_memory(query, **kwargs)

            # Update metrics
            self.performance_metrics["total_queries"] += 1
            if response.has_memory_enhancement:
                self.performance_metrics["memory_enhanced_queries"] += 1

            response_time = (time.perf_counter() - start_time) * 1000
            self._update_average_response_time(response_time)

            return response

        except Exception as e:
            logger.error(f"Error in memory-enabled RAG query: {e}")
            raise

    async def learn_from_feedback(
        self, query: str, response: EnrichedRAGResponse, feedback: Dict[str, Any]
    ) -> None:
        """
        Learn from user feedback to improve memory patterns.

        This demonstrates how applications can implement feedback loops
        for continuous improvement.
        """
        try:
            if feedback.get("helpful", False):
                # Increase confidence of used patterns
                for pattern in response.extracted_patterns:
                    pattern.extraction_confidence = min(
                        1.0, pattern.extraction_confidence + 0.1
                    )

                # Store positive feedback in memory
                feedback_content = {
                    "query": query,
                    "feedback": "positive",
                    "response_quality": feedback.get("quality", "unknown"),
                    "patterns_used": len(response.extracted_patterns),
                }

                await self.wrapper.temporal_manager.store_with_window(
                    content=feedback_content,
                    window=TemporalWindow.LONG_TERM,  # Store feedback longer
                    context={"type": "feedback", "sentiment": "positive"},
                    source_query=query,
                )

        except Exception as e:
            logger.warning(f"Error learning from feedback: {e}")

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory and performance statistics."""
        knowledge_metrics = self.wrapper.knowledge_extractor.get_performance_metrics()
        temporal_metrics = self.wrapper.temporal_manager.get_performance_metrics()

        return {
            "pipeline_metrics": self.performance_metrics,
            "knowledge_extraction": knowledge_metrics,
            "temporal_memory": temporal_metrics,
            "memory_enhancement_rate": (
                self.performance_metrics["memory_enhanced_queries"]
                / max(1, self.performance_metrics["total_queries"])
            )
            * 100,
        }

    def _update_average_response_time(self, response_time: float) -> None:
        """Update average response time metric."""
        current_avg = self.performance_metrics["average_response_time_ms"]
        total_queries = self.performance_metrics["total_queries"]

        # Calculate running average
        new_avg = ((current_avg * (total_queries - 1)) + response_time) / total_queries
        self.performance_metrics["average_response_time_ms"] = new_avg


# Factory functions for easy integration


def wrap_existing_pipeline(
    pipeline: Any, memory_config: MemoryConfig, connection_manager: ConnectionManager
) -> RAGPipelineWrapper:
    """
    Factory function to wrap existing RAG pipeline with memory.

    Args:
        pipeline: Existing RAG pipeline
        memory_config: Memory configuration
        connection_manager: Database connection manager

    Returns:
        Memory-enabled pipeline wrapper
    """
    return RAGPipelineWrapper(pipeline, memory_config, connection_manager)


def create_memory_enabled_pipeline(
    base_pipeline: Any,
    memory_config: MemoryConfig,
    connection_manager: ConnectionManager,
) -> MemoryEnabledRAGPipeline:
    """
    Factory function to create complete memory-enabled RAG pipeline.

    Args:
        base_pipeline: Base RAG pipeline
        memory_config: Memory configuration
        connection_manager: Database connection manager

    Returns:
        Complete memory-enabled RAG pipeline
    """
    return MemoryEnabledRAGPipeline(base_pipeline, memory_config, connection_manager)


# Example integration patterns


async def example_basic_rag_with_memory():
    """Example: Adding memory to BasicRAG."""
    from iris_rag.config.manager import ConfigurationManager
    from iris_rag.core.connection import ConnectionManager
    from iris_rag.pipelines.basic import BasicRAGPipeline

    # Initialize components
    config_manager = ConfigurationManager()
    connection_manager = ConnectionManager(config_manager)

    # Create base RAG pipeline
    basic_rag = BasicRAGPipeline(
        vector_store=None,  # Would be initialized with actual vector store
        llm_func=None,  # Would be initialized with actual LLM
    )

    # Create memory configuration
    memory_config = MemoryConfig.from_dict(
        {
            "temporal_windows": [
                {"name": "short_term", "duration_days": 7},
                {"name": "long_term", "duration_days": 90},
            ]
        }
    )

    # Wrap with memory
    memory_rag = wrap_existing_pipeline(basic_rag, memory_config, connection_manager)

    # Use memory-enhanced pipeline
    response = await memory_rag.query_with_memory("What are the symptoms of diabetes?")

    return response


async def example_graphrag_with_memory():
    """Example: Adding memory to GraphRAG."""

    # Similar pattern as above but with GraphRAG
    # This demonstrates the generic nature of the memory integration
