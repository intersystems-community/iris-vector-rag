#!/usr/bin/env python3
"""
Example script demonstrating HybridGraphRAG capabilities.

This script shows how to use the enhanced GraphRAG pipeline with:
- RRF (Reciprocal Rank Fusion)
- HNSW-optimized vector search
- Multi-modal hybrid search
- Native IRIS iFind text search

Prerequisites:
- graph-ai project must be adjacent to rag-templates
- IRIS database with optimized vector tables
- Sample data loaded with entity extraction
"""

import logging
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_llm_func(prompt: str) -> str:
    """Simple LLM function that returns a placeholder response."""
    return f"[Generated Answer for: {prompt[:100]}...]"


def main():
    """Demonstrate HybridGraphRAG capabilities."""

    try:
        # Initialize components
        logger.info("Initializing HybridGraphRAG pipeline...")
        connection_manager = ConnectionManager()
        config_manager = ConfigurationManager()

        # Create pipeline with hybrid search capabilities
        pipeline = HybridGraphRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
            llm_func=simple_llm_func,
        )

        # Get performance statistics
        logger.info("Checking performance capabilities...")
        stats = pipeline.get_performance_statistics()

        print("=== HybridGraphRAG Performance Status ===")
        print(f"IRIS Graph Core Available: {pipeline.iris_engine is not None}")
        if "hnsw_status" in stats:
            hnsw = stats["hnsw_status"]
            print(f"HNSW Optimization: {hnsw.get('available', 'Unknown')}")
            if hnsw.get("available"):
                print(f"  - Vector Count: {hnsw.get('record_count', 'N/A')}")
                print(f"  - Query Time: {hnsw.get('query_time_ms', 'N/A')}ms")
                print(f"  - Performance Tier: {hnsw.get('performance_tier', 'N/A')}")

        print(f"Entity Count: {stats.get('entity_count', 'N/A')}")
        print(f"Relationship Count: {stats.get('relationship_count', 'N/A')}")
        print()

        # Test queries with different methods
        test_queries = [
            "protein interactions in cancer research",
            "drug targets for neurological disorders",
            "gene expression patterns in aging",
            "molecular mechanisms of inflammation",
        ]

        search_methods = ["hybrid", "rrf", "vector", "text", "kg"]

        for query in test_queries:
            print(f"=== Query: {query} ===")

            for method in search_methods:
                try:
                    # Skip methods that require iris_graph_core if not available
                    if (
                        method in ["hybrid", "rrf", "vector"]
                        and not pipeline.iris_engine
                    ):
                        print(
                            f"  {method.upper()}: Skipped (iris_graph_core not available)"
                        )
                        continue

                    # Execute query
                    result = pipeline.query(
                        query_text=query,
                        method=method,
                        top_k=5,
                        generate_answer=False,  # Skip answer generation for demo
                    )

                    # Display results
                    metadata = result["metadata"]
                    print(
                        f"  {method.upper()}: {metadata['num_retrieved']} docs, "
                        f"{metadata['processing_time_ms']:.1f}ms"
                    )

                    # Show top result
                    if result["retrieved_documents"]:
                        top_doc = result["retrieved_documents"][0]
                        preview = (
                            top_doc.page_content[:100] + "..."
                            if len(top_doc.page_content) > 100
                            else top_doc.page_content
                        )
                        print(f"    Top result: {preview}")

                except Exception as e:
                    print(f"  {method.upper()}: Error - {str(e)}")

            print()

        # Benchmark search methods if iris_graph_core is available
        if pipeline.iris_engine:
            print("=== Performance Benchmark ===")
            benchmark = pipeline.benchmark_search_methods(
                query_text="cancer research proteins", iterations=3
            )

            for method, metrics in benchmark.items():
                print(
                    f"{method.upper()}: {metrics['avg_time_ms']:.1f}ms average "
                    f"({metrics['min_time_ms']:.1f}-{metrics['max_time_ms']:.1f}ms range)"
                )

        print("\n=== Hybrid Search Example ===")

        # Demonstrate hybrid search with different parameters
        if pipeline.iris_engine:
            result = pipeline.query(
                query_text="alzheimer disease drug targets",
                method="hybrid",
                top_k=10,
                fusion_weights=[0.5, 0.3, 0.2],  # vector, text, graph
                generate_answer=True,
            )

            print(f"Query: {result['query']}")
            print(f"Retrieved: {len(result['retrieved_documents'])} documents")
            print(f"Processing time: {result['metadata']['processing_time_ms']:.1f}ms")

            if result["answer"]:
                print(f"Generated answer: {result['answer']}")

            # Show fusion details
            for i, doc in enumerate(result["retrieved_documents"][:3]):
                print(
                    f"  Doc {i+1}: {doc.metadata.get('fusion_score', 'N/A')} fusion score"
                )
                for mode in doc.metadata.get("search_modes", []):
                    print(
                        f"    - {mode['mode']}: score={mode['score']:.3f}, rank={mode['rank']}"
                    )

        print("\n=== Demo Complete ===")
        print(
            "Try different methods and parameters to explore hybrid search capabilities!"
        )

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure graph-ai project is adjacent to rag-templates")
        print("2. Check IRIS database connection")
        print("3. Load sample data with entity extraction")
        print("4. Run vector migration if needed")


if __name__ == "__main__":
    main()
