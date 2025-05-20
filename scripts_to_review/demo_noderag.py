#!/usr/bin/env python3
"""
NodeRAG Demo Script

This script demonstrates the NodeRAG (Node-based Retrieval Augmented Generation)
pipeline, which uses a heterogeneous knowledge graph and SQL-based recursive CTEs
for efficient graph traversal.

NodeRAG retrieves information by:
1. Finding initial seed nodes based on vector similarity
2. Traversing the knowledge graph to find related information
3. Combining graph structure and vector similarity for ranking
4. Organizing information by node type for better context
"""

import os
import sys
import time
import argparse
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import pipeline components
from common.iris_connector import get_iris_connection
from noderag.pipeline import NodeRAGPipeline
from common.embedding_utils import get_embedding_model
from common.utils import timing_decorator, Document

# Mock LLM function for testing
def mock_llm_func(prompt: str) -> str:
    """Mock LLM function that just returns a simple response based on the prompt."""
    # In a real application, this would call an LLM API
    time.sleep(0.5)  # Simulate LLM latency
    
    # Extract query from prompt for the mock response
    query = "unknown query"
    if "Question:" in prompt:
        query = prompt.split("Question:")[1].split("\n")[0].strip()
    
    # Return a mock answer that references node types from the context
    node_types = []
    if "ENTITY INFORMATION:" in prompt:
        node_types.append("Entity")
    if "DOCUMENT INFORMATION:" in prompt:
        node_types.append("Document")
    if "CONCEPT INFORMATION:" in prompt:
        node_types.append("Concept")
    
    if node_types:
        type_str = ", ".join(node_types)
        return f"Answer to '{query}' based on information from {type_str} nodes in the knowledge graph."
    else:
        return f"Answer to '{query}' based on the knowledge graph."

@timing_decorator
def run_noderag_pipeline(query: str, use_mock: bool = True, max_depth: int = 2, top_k: int = 5):
    """
    Run the NodeRAG pipeline with the given query.
    
    Args:
        query: The query text
        use_mock: Whether to use mock components or real ones
        max_depth: Maximum traversal depth in the graph
        top_k: Number of nodes to return for context
        
    Returns:
        The pipeline result dictionary
    """
    logger.info(f"Running NodeRAG pipeline with query: '{query}'")
    
    # Get IRIS connection
    iris_connector = get_iris_connection(use_mock=use_mock)
    if not iris_connector:
        logger.error("Failed to establish IRIS connection")
        return {"error": "Failed to establish IRIS connection"}
    
    # Get embedding model
    embedding_model = get_embedding_model(mock=use_mock)
    embedding_func = lambda text: embedding_model.encode(text)
    
    # Use mock LLM function for demo
    llm_func = mock_llm_func
    
    # Create the pipeline
    pipeline = NodeRAGPipeline(
        iris_connector=iris_connector,
        embedding_func=embedding_func,
        llm_func=llm_func,
        max_depth=max_depth,
        top_k=top_k
    )
    
    # Run the pipeline
    try:
        result = pipeline.run(query)
        logger.info(f"Pipeline completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        return {"error": str(e), "query": query}

def print_results(result: Dict[str, Any]) -> None:
    """
    Print the pipeline results in a nicely formatted way.
    
    Args:
        result: The pipeline result dictionary
    """
    print("\n" + "="*80)
    print(f"QUERY: {result.get('query', 'N/A')}")
    print("="*80)
    
    if "error" in result:
        print(f"ERROR: {result['error']}")
        return
    
    print("\nKNOWLEDGE GRAPH TRAVERSAL RESULTS:")
    print("-"*80)
    
    # Group nodes by type for better display
    nodes_by_type = {}
    docs = result.get("retrieved_documents", [])
    
    if not docs:
        print("No nodes retrieved from knowledge graph")
    else:
        for doc in docs:
            node_type = getattr(doc, "node_type", "Unknown")
            if node_type not in nodes_by_type:
                nodes_by_type[node_type] = []
            nodes_by_type[node_type].append(doc)
        
        # Display nodes by type
        type_order = ["Entity", "Concept", "Summary", "Document", "Unknown"]
        for node_type in type_order:
            if node_type in nodes_by_type:
                print(f"\n## {node_type.upper()} NODES:")
                for i, node in enumerate(nodes_by_type[node_type]):
                    print(f"  {i+1}. {node_type} (ID: {node.id}, Score: {node.score:.4f}):")
                    # Print truncated content
                    content = node.content
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"     {content}")
    
    print("\nANSWER:")
    print("-"*80)
    print(result.get("answer", "No answer generated"))
    print("="*80 + "\n")

def main():
    """Main function to parse arguments and run the demo."""
    parser = argparse.ArgumentParser(description="NodeRAG Demo Script")
    
    parser.add_argument("--query", type=str, default="What is the relationship between diabetes and insulin?",
                       help="Query to test with the pipeline")
    
    parser.add_argument("--use-real", action="store_true",
                       help="Use real components instead of mocks")
    
    parser.add_argument("--max-depth", type=int, default=2,
                       help="Maximum traversal depth in the graph")
    
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of nodes to return for context")
    
    parser.add_argument("--sample-queries", action="store_true",
                       help="Run with a set of sample queries instead of a single query")
    
    args = parser.parse_args()
    
    # Sample queries if requested
    if args.sample_queries:
        sample_queries = [
            "What are the main treatments for diabetes?",
            "How does the immune system respond to viral infections?",
            "What is the relationship between amyloid plaques and Alzheimer's disease?",
            "How do neurons communicate with each other?",
            "What role does inflammation play in heart disease?"
        ]
        
        for query in sample_queries:
            result = run_noderag_pipeline(
                query, 
                use_mock=not args.use_real,
                max_depth=args.max_depth,
                top_k=args.top_k
            )
            print_results(result)
            print("\nWaiting before next query...\n")
            time.sleep(1)
    else:
        # Run with the single provided query
        result = run_noderag_pipeline(
            args.query, 
            use_mock=not args.use_real,
            max_depth=args.max_depth,
            top_k=args.top_k
        )
        print_results(result)

if __name__ == "__main__":
    main()
