#!/usr/bin/env python3
"""
Test script for IRIS Global GraphRAG pipeline integration.

This script demonstrates how to use the new IRIS Global GraphRAG pipeline
that integrates the colleague's IRIS-Global-GraphRAG project with our
RAG framework.
"""

import logging
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.factory import create_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def test_pipeline_creation():
    """Test creating the IRIS Global GraphRAG pipeline."""
    logger.info("Testing IRIS Global GraphRAG pipeline creation...")

    try:
        # Create configuration manager
        config_manager = ConfigurationManager()

        # Create pipeline
        pipeline = create_pipeline(
            pipeline_type="IRISGlobalGraphRAG",
            config_manager=config_manager,
            validate_requirements=False,  # Skip validation for testing
        )

        logger.info("✅ Pipeline created successfully")

        # Get pipeline info
        info = pipeline.get_pipeline_info()
        logger.info(f"Pipeline info: {info}")

        return pipeline

    except Exception as e:
        logger.error(f"❌ Pipeline creation failed: {e}")
        return None


def test_document_loading(pipeline):
    """Test loading documents into the pipeline."""
    logger.info("Testing document loading...")

    try:
        # Create sample academic papers
        documents = [
            Document(
                id="paper_1",
                page_content="GraphRAG combines vector similarity with knowledge graph traversal for enhanced retrieval. This approach leverages entity relationships to provide more contextual and explainable results.",
                metadata={
                    "title": "Enhanced Retrieval with GraphRAG",
                    "authors": "John Doe, Jane Smith",
                    "published": "2024-01-15",
                    "url": "https://example.com/paper1",
                    "entities": [
                        {"name": "GraphRAG", "type": "Method"},
                        {"name": "vector similarity", "type": "Technique"},
                        {"name": "knowledge graph", "type": "Structure"},
                    ],
                    "relationships": [
                        {
                            "source": "GraphRAG",
                            "target": "vector similarity",
                            "relation": "USES",
                            "source_type": "Method",
                            "target_type": "Technique",
                        },
                        {
                            "source": "GraphRAG",
                            "target": "knowledge graph",
                            "relation": "LEVERAGES",
                            "source_type": "Method",
                            "target_type": "Structure",
                        },
                    ],
                },
            ),
            Document(
                id="paper_2",
                page_content="IRIS database provides native support for vector embeddings and graph storage using globals. The HNSW indexing enables fast approximate nearest neighbor search.",
                metadata={
                    "title": "IRIS Database for Vector-Graph Hybrid Storage",
                    "authors": "Alice Johnson, Bob Wilson",
                    "published": "2024-02-20",
                    "url": "https://example.com/paper2",
                    "entities": [
                        {"name": "IRIS", "type": "Database"},
                        {"name": "HNSW", "type": "Algorithm"},
                        {"name": "vector embeddings", "type": "Technique"},
                    ],
                    "relationships": [
                        {
                            "source": "IRIS",
                            "target": "vector embeddings",
                            "relation": "SUPPORTS",
                            "source_type": "Database",
                            "target_type": "Technique",
                        },
                        {
                            "source": "HNSW",
                            "target": "vector embeddings",
                            "relation": "INDEXES",
                            "source_type": "Algorithm",
                            "target_type": "Technique",
                        },
                    ],
                },
            ),
        ]

        # Load documents
        success = pipeline.load_documents(documents)

        if success:
            logger.info("✅ Documents loaded successfully")
        else:
            logger.error("❌ Document loading failed")

        return success

    except Exception as e:
        logger.error(f"❌ Document loading failed: {e}")
        return False


def test_query_processing(pipeline):
    """Test query processing with different modes."""
    logger.info("Testing query processing...")

    test_queries = [
        "What is GraphRAG and how does it work?",
        "How does IRIS database support vector embeddings?",
        "What are the benefits of hybrid search approaches?",
    ]

    for query in test_queries:
        logger.info(f"\nTesting query: '{query}'")

        try:
            # Test RAG mode
            logger.info("Testing RAG mode...")
            rag_result = pipeline.query(query, mode="rag", top_k=3)
            logger.info(f"RAG answer: {rag_result['answer'][:200]}...")

            # Test GraphRAG mode
            logger.info("Testing GraphRAG mode...")
            graphrag_result = pipeline.query(
                query, mode="graphrag", top_k=3, enable_visualization=True
            )
            logger.info(f"GraphRAG answer: {graphrag_result['answer'][:200]}...")

            if "graph_data" in graphrag_result:
                graph_data = graphrag_result["graph_data"]
                logger.info(
                    f"Graph data: {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('links', []))} links"
                )

            logger.info("✅ Query processing successful")

        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")


def test_comparison_functionality(pipeline):
    """Test side-by-side comparison functionality."""
    logger.info("Testing comparison functionality...")

    query = "What are the advantages of combining vector search with graph traversal?"

    try:
        comparison_result = pipeline.compare_modes(query, top_k=3)

        logger.info("Comparison results:")
        logger.info(f"LLM answer: {comparison_result['llm']['answer'][:150]}...")
        logger.info(f"RAG answer: {comparison_result['rag']['answer'][:150]}...")
        logger.info(
            f"GraphRAG answer: {comparison_result['graphrag']['answer'][:150]}..."
        )

        if "graph_data" in comparison_result["graphrag"]:
            graph_data = comparison_result["graphrag"]["graph_data"]
            logger.info(
                f"Graph visualization: {len(graph_data.get('nodes', []))} nodes, {len(graph_data.get('links', []))} links"
            )

        logger.info("✅ Comparison functionality successful")

    except Exception as e:
        logger.error(f"❌ Comparison functionality failed: {e}")


def test_web_interface(pipeline):
    """Test the web interface creation."""
    logger.info("Testing web interface...")

    try:
        from iris_vector_rag.visualization.iris_global_graphrag_interface import (
            IRISGlobalGraphRAGInterface,
        )

        # Create interface
        interface = IRISGlobalGraphRAGInterface(pipeline)

        # Create Flask app (don't run it)
        app = interface.create_flask_app()

        logger.info("✅ Web interface created successfully")
        logger.info(
            "To run the interface, use: python -m iris_rag.visualization.iris_global_graphrag_interface"
        )

    except Exception as e:
        logger.error(f"❌ Web interface creation failed: {e}")


def main():
    """Main test function."""
    logger.info("🚀 Starting IRIS Global GraphRAG integration tests...")

    # Test pipeline creation
    pipeline = test_pipeline_creation()
    if not pipeline:
        logger.error("Pipeline creation failed, stopping tests")
        return

    # Test document loading
    if test_document_loading(pipeline):
        # Test query processing
        test_query_processing(pipeline)

        # Test comparison functionality
        test_comparison_functionality(pipeline)

    # Test web interface
    test_web_interface(pipeline)

    logger.info("🎉 IRIS Global GraphRAG integration tests completed!")
    logger.info("\nTo run the web interface:")
    logger.info("  python scripts/test_iris_global_graphrag.py --web")
    logger.info("  # or")
    logger.info("  python -m iris_rag.visualization.iris_global_graphrag_interface")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--web":
        # Run web interface
        try:
            from iris_vector_rag.visualization.iris_global_graphrag_interface import (
                main as web_main,
            )

            web_main()
        except Exception as e:
            logger.error(f"Failed to start web interface: {e}")
    else:
        # Run tests
        main()
