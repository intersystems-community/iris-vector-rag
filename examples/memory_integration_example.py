#!/usr/bin/env python3
"""
RAG Memory Integration Example

Demonstrates how to add memory capabilities to any RAG application
using the generic memory components from iris_rag.memory.
"""

import asyncio
import yaml
from pathlib import Path
from typing import Dict, Any

# Import RAG memory components
from iris_rag.memory import (
    KnowledgePatternExtractor,
    TemporalMemoryManager,
    IncrementalLearningManager,
    MemoryEnabledRAGPipeline,
    MemoryConfig,
    TemporalWindow,
    TemporalQuery,
)
from iris_rag.memory.rag_integration import wrap_existing_pipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from kg_memory.incremental.cdc_detector import CDCDetector
from kg_memory.incremental.graph_union import GraphUnionOperator


class MockRAGPipeline:
    """Mock RAG pipeline for demonstration purposes."""

    def __init__(self, name: str = "MockRAG"):
        self.name = name

    async def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Mock RAG query that returns a simulated response."""
        return {
            "response_text": f"Mock response for: {query}",
            "query": query,
            "technique_used": self.name,
            "retrieved_docs": [
                {"content": f"Document 1 content related to {query}", "score": 0.9},
                {"content": f"Document 2 content about {query}", "score": 0.8},
            ],
        }


async def demonstrate_basic_memory_integration():
    """
    Example 1: Basic memory integration with any RAG pipeline.

    Shows how to wrap an existing pipeline with memory capabilities.
    """
    print("\n=== Example 1: Basic Memory Integration ===")

    # 1. Load memory configuration
    config_path = Path(__file__).parent.parent / "config" / "memory_config.yaml"
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    memory_config = MemoryConfig.from_dict(config_dict["rag_memory_config"])

    # 2. Initialize infrastructure components
    config_manager = ConfigurationManager()
    connection_manager = ConnectionManager(config_manager)

    # 3. Create base RAG pipeline (any existing pipeline)
    base_pipeline = MockRAGPipeline("BasicRAG")

    # 4. Wrap with memory capabilities
    memory_wrapper = wrap_existing_pipeline(
        base_pipeline, memory_config, connection_manager
    )

    # 5. Use memory-enhanced pipeline
    query = "What are the symptoms of diabetes?"
    response = await memory_wrapper.query_with_memory(query)

    print(f"Query: {query}")
    print(f"Base Response: {response.base_response['response_text']}")
    print(f"Memory Enhancement: {response.has_memory_enhancement}")
    print(f"Extracted Patterns: {len(response.extracted_patterns)}")
    print(f"Memory Context Items: {len(response.memory_context)}")

    return memory_wrapper


async def demonstrate_temporal_memory_patterns():
    """
    Example 2: Temporal memory storage and retrieval patterns.

    Shows how applications can use temporal windows for different memory needs.
    """
    print("\n=== Example 2: Temporal Memory Patterns ===")

    # Initialize temporal memory manager
    config_manager = ConfigurationManager()
    connection_manager = ConnectionManager(config_manager)

    # Define temporal windows for different use cases
    temporal_windows = [
        {"name": "chat_session", "duration_days": 1, "retention_policy": "expire"},
        {"name": "user_preferences", "duration_days": 30, "retention_policy": "keep"},
        {
            "name": "domain_knowledge",
            "duration_days": 365,
            "retention_policy": "archive",
        },
    ]

    window_configs = [TemporalWindowConfig(**config) for config in temporal_windows]
    temporal_manager = TemporalMemoryManager(connection_manager, window_configs)

    # Store different types of memories
    memories = [
        (
            "User asked about diabetes symptoms",
            TemporalWindow.SHORT_TERM,
            "diabetes query",
        ),
        (
            "User prefers detailed explanations",
            TemporalWindow.MEDIUM_TERM,
            "user preference",
        ),
        (
            "Medical terminology definitions",
            TemporalWindow.LONG_TERM,
            "domain knowledge",
        ),
    ]

    for content, window, source in memories:
        await temporal_manager.store_with_window(
            content=content,
            window=window,
            context={"category": source},
            source_query=source,
        )
        print(f"Stored: {content[:50]}... in {window.value} window")

    # Retrieve temporal context
    query = TemporalQuery(
        query_text="diabetes", window=TemporalWindow.SHORT_TERM, max_results=5
    )

    context = await temporal_manager.retrieve_temporal_context(query)
    print(f"\nRetrieved {len(context.items)} items from {query.window.value} window")

    # Get performance metrics
    metrics = temporal_manager.get_performance_metrics()
    print(f"Cache hit rate: {metrics['cache_performance']['hit_rate']:.2%}")

    return temporal_manager


async def demonstrate_incremental_learning():
    """
    Example 3: Incremental learning with existing M2 infrastructure.

    Shows how to leverage CDC and graph union for efficient updates.
    """
    print("\n=== Example 3: Incremental Learning ===")

    # Initialize components (simplified for demonstration)
    config_manager = ConfigurationManager()
    connection_manager = ConnectionManager(config_manager)

    # Initialize M2 infrastructure components
    cdc_detector = CDCDetector(connection_manager, config_manager)

    # Mock schema manager for graph union
    class MockSchemaManager:
        def ensure_table_schema(self, table_name):
            pass

    schema_manager = MockSchemaManager()
    graph_union = GraphUnionOperator(connection_manager, schema_manager)

    # Initialize knowledge extractor
    from iris_rag.memory.models import KnowledgeExtractionConfig

    extraction_config = KnowledgeExtractionConfig()
    knowledge_extractor = KnowledgePatternExtractor(extraction_config)

    # Create incremental learning manager
    learning_manager = IncrementalLearningManager(
        cdc_detector, graph_union, knowledge_extractor
    )

    # Simulate document updates
    from iris_rag.core.models import Document

    new_documents = [
        Document(
            id="doc1",
            page_content="Diabetes is a metabolic disorder.",
            metadata={"source": "medical"},
        ),
        Document(
            id="doc2",
            page_content="Symptoms include increased thirst.",
            metadata={"source": "medical"},
        ),
        Document(
            id="doc3",
            page_content="Treatment involves insulin therapy.",
            metadata={"source": "medical"},
        ),
    ]

    # Process incremental updates
    print(f"Processing {len(new_documents)} document updates...")
    learning_result = await learning_manager.process_knowledge_updates(new_documents)

    print(f"Learning successful: {learning_result.success}")
    print(f"New patterns: {len(learning_result.new_patterns)}")
    print(f"Processing time: {learning_result.processing_time_ms:.2f}ms")

    # Get performance metrics
    metrics = learning_manager.get_performance_metrics()
    print(
        f"Average learning time: {metrics['learning_performance']['avg_time_ms']:.2f}ms"
    )

    return learning_manager


async def demonstrate_complete_integration():
    """
    Example 4: Complete integration showing all components working together.

    This demonstrates a realistic scenario where an application uses
    all memory components in a coordinated way.
    """
    print("\n=== Example 4: Complete Memory Integration ===")

    # 1. Initialize all components
    config_path = Path(__file__).parent.parent / "config" / "memory_config.yaml"
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    memory_config = MemoryConfig.from_dict(config_dict["rag_memory_config"])
    config_manager = ConfigurationManager()
    connection_manager = ConnectionManager(config_manager)

    # 2. Create different RAG pipelines
    pipelines = {
        "basic": MockRAGPipeline("BasicRAG"),
        "graph": MockRAGPipeline("GraphRAG"),
        "crag": MockRAGPipeline("CRAG"),
    }

    # 3. Create memory-enabled versions
    memory_pipelines = {}
    for name, pipeline in pipelines.items():
        memory_pipeline = MemoryEnabledRAGPipeline(
            pipeline, memory_config, connection_manager
        )
        memory_pipelines[name] = memory_pipeline

    # 4. Process a series of related queries
    queries = [
        "What is diabetes?",
        "What are the symptoms of diabetes?",
        "How is diabetes treated?",
        "What causes diabetes?",
    ]

    print("Processing query sequence with memory...")
    all_responses = []

    for i, query in enumerate(queries):
        # Use different techniques for variety
        technique = list(memory_pipelines.keys())[i % len(memory_pipelines)]
        pipeline = memory_pipelines[technique]

        response = await pipeline.query(query)
        all_responses.append(response)

        print(f"Query {i+1} ({technique}): {query}")
        print(f"  Memory Enhanced: {response.has_memory_enhancement}")
        print(f"  Context Items: {len(response.memory_context)}")
        print(f"  New Patterns: {len(response.extracted_patterns)}")

    # 5. Show memory evolution over queries
    print(f"\nMemory system learned from {len(queries)} queries:")

    for name, pipeline in memory_pipelines.items():
        stats = pipeline.get_memory_statistics()
        print(
            f"  {name}: {stats['pipeline_metrics']['total_queries']} queries, "
            f"{stats['memory_enhancement_rate']:.1f}% memory enhanced"
        )

    return memory_pipelines


async def demonstrate_performance_optimization():
    """
    Example 5: Performance optimization and monitoring.

    Shows how to monitor and optimize memory performance.
    """
    print("\n=== Example 5: Performance Optimization ===")

    # This would demonstrate performance monitoring patterns
    print("Performance monitoring capabilities:")
    print("- Knowledge extraction: <50ms target")
    print("- Temporal retrieval: <100ms target")
    print("- Incremental learning: <30s for 1K docs")
    print("- Memory cache hit rates: >80% target")
    print("- Automatic performance alerting")

    # Demonstrate configuration-driven optimization
    optimization_configs = {
        "development": {"cache_size": 100, "workers": 2},
        "production": {"cache_size": 5000, "workers": 8},
        "high_throughput": {"cache_size": 10000, "workers": 16},
    }

    for env, config in optimization_configs.items():
        print(f"  {env}: cache={config['cache_size']}, workers={config['workers']}")


async def main():
    """Run all memory integration examples."""
    print("RAG Memory Component Integration Examples")
    print("========================================")

    try:
        # Run all examples
        await demonstrate_basic_memory_integration()
        await demonstrate_temporal_memory_patterns()
        await demonstrate_incremental_learning()
        await demonstrate_complete_integration()
        await demonstrate_performance_optimization()

        print("\n=== Summary ===")
        print("✅ Basic memory integration - ANY RAG pipeline can be enhanced")
        print("✅ Temporal memory patterns - Configurable time windows")
        print("✅ Incremental learning - Leverages existing M2 infrastructure")
        print("✅ Complete integration - All components working together")
        print("✅ Performance optimization - Monitoring and tuning patterns")

        print("\n🎯 Key Benefits:")
        print("  • Zero application-specific assumptions")
        print("  • Configuration-driven behavior")
        print("  • Pluggable architecture for extensions")
        print("  • Performance-optimized with caching")
        print("  • Integrates with existing rag-templates infrastructure")

        print("\n📚 Applications can:")
        print("  • Wrap any existing RAG pipeline with memory")
        print("  • Configure temporal windows for their domain")
        print("  • Extend with custom knowledge extraction")
        print("  • Add domain-specific memory patterns")
        print("  • Monitor and optimize performance")

    except Exception as e:
        print(f"❌ Error in demonstration: {e}")
        print("Note: Some components require actual database connections")
        print("This example shows the integration patterns and APIs")


if __name__ == "__main__":
    # Run examples
    asyncio.run(main())
