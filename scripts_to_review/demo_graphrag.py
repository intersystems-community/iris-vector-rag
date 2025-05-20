#!/usr/bin/env python3
"""
GraphRAG Demo Script

This script demonstrates the GraphRAG (Graph-based Retrieval Augmented Generation)
pipeline, which uses a knowledge graph and recursive CTEs for graph traversal.

GraphRAG retrieves information by:
1. Finding starting nodes based on embedding similarity
2. Traversing the knowledge graph via connections between nodes
3. Combining graph structure and semantic similarity for ranking
4. Generating answers based on retrieved context
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
from graphrag.pipeline import GraphRAGPipeline
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
    
    return f"Answer to '{query}' based on knowledge graph information."

@timing_decorator
def run_graphrag_pipeline(query: str, use_mock: bool = True, max_depth: int = 2, top_k: int = 5):
    """
    Run the GraphRAG pipeline with the given query.
    
    Args:
        query: The query text
        use_mock: Whether to use mock components or real ones
        max_depth: Maximum traversal depth in the graph
        top_k: Number of nodes to return for context
        
    Returns:
        The pipeline result dictionary
    """
    logger.info(f"Running GraphRAG pipeline with query: '{query}'")
    
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
    pipeline = GraphRAGPipeline(
        iris_connector=iris_connector,
        embedding_func=embedding_func,
        llm_func=llm_func
    )
    
    # Run the pipeline
    try:
        result = pipeline.run(query)
        logger.info(f"Pipeline completed successfully")
        return result
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        return {"error": str(e), "query": query}

def print_graph_path(traversed_nodes: List[Dict[str, Any]]) -> None:
    """
    Print a visualization of the traversed graph path.
    
    Args:
        traversed_nodes: List of node dictionaries with path information
    """
    if not traversed_nodes:
        print("No graph path to display")
        return
    
    print("\nGRAPH TRAVERSAL PATH:")
    print("=====================")
    
    # Group nodes by depth
    nodes_by_depth = {}
    for node in traversed_nodes:
        depth = node.get("depth", 0)
        if depth not in nodes_by_depth:
            nodes_by_depth[depth] = []
        nodes_by_depth[depth].append(node)
    
    # Print nodes by depth level
    for depth in sorted(nodes_by_depth.keys()):
        print(f"\nDepth {depth}:")
        for i, node in enumerate(nodes_by_depth[depth]):
            node_id = node.get("id", "unknown")
            node_type = node.get("type", "unknown")
            print(f"  Node {i+1}: [{node_type}] {node_id}")
            
            # Print connections to previous depth if available
            if "connections" in node and depth > 0:
                print("    Connected to:")
                for conn in node["connections"]:
                    print(f"      - {conn}")

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
    
    print("\nRETRIEVED NODES:")
    print("-"*80)
    
    # Print retrieved documents/nodes
    docs = result.get("retrieved_documents", [])
    if not docs:
        print("No nodes retrieved from knowledge graph")
    else:
        for i, doc in enumerate(docs):
            # Extract node type if available (special GraphRAG attribute)
            node_type = "Node"
            if hasattr(doc, "node_type"):
                node_type = doc.node_type
            
            print(f"{i+1}. {node_type} (ID: {doc.id}, Score: {doc.score:.4f}):")
            
            # Print truncated content
            content = doc.content
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"   {content}\n")
    
    # If graph path information is available, print it
    if "graph_path" in result:
        print_graph_path(result["graph_path"])
    
    print("\nANSWER:")
    print("-"*80)
    print(result.get("answer", "No answer generated"))
    print("="*80 + "\n")

def main():
    """Main function to parse arguments and run the demo."""
    parser = argparse.ArgumentParser(description="GraphRAG Demo Script")
    
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
            "How are diabetes and insulin related?",
            "What are the key symptoms of diabetes?",
            "How does metformin help with diabetes treatment?",
            "What's the relationship between obesity and diabetes?",
            "How do statins affect cholesterol levels?"
        ]
        
        for query in sample_queries:
            result = run_graphrag_pipeline(
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
        result = run_graphrag_pipeline(
            args.query, 
            use_mock=not args.use_real,
            max_depth=args.max_depth,
            top_k=args.top_k
        )
        print_results(result)

if __name__ == "__main__":
    main()
