#!/usr/bin/env python3
"""
Comprehensive test runner for all RAG techniques with 1000+ documents.

This script runs all RAG techniques in sequence with a minimum of 1000 real or
synthetic documents to verify compliance with the project requirements.
"""

import os
import sys
import time
import logging
import random
from typing import Dict, List, Any, Optional, Callable
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_1000_docs_tester")

# Constants
MIN_DOCUMENTS = 1000
DEFAULT_TECHNIQUES = ["basic_rag", "hyde", "crag", "colbert", "noderag", "graphrag"]
TEST_QUERIES = [
    "What is the role of insulin in diabetes?",
    "How do statins affect cholesterol levels?",
    "What are the latest cancer treatment approaches?",
    "How does metformin work for type 2 diabetes?",
    "What is the relationship between obesity and insulin resistance?"
]

def setup_database_connection():
    """Set up a connection to the database (IRIS or mock)."""
    try:
        # Try to import the IRIS connector
        from common.iris_connector import get_iris_connection
        
        # Try to connect to a real IRIS instance
        logger.info("Trying to connect to IRIS database...")
        conn = get_iris_connection(use_mock=False)
        logger.info("Successfully connected to IRIS database")
        return conn
    except Exception as e:
        logger.warning(f"Failed to connect to real IRIS database: {e}")
        logger.info("Falling back to mock connection")
        
        # Import the mock connector
        from tests.mocks.db import MockDB
        
        # Create a mock connection with 1000+ documents
        mock_db = MockDB(document_count=MIN_DOCUMENTS)
        return mock_db

def generate_synthetic_documents(count: int = MIN_DOCUMENTS) -> List[Dict[str, Any]]:
    """Generate synthetic documents for testing."""
    logger.info(f"Generating {count} synthetic documents")
    
    documents = []
    topics = ["diabetes", "insulin", "cancer", "treatment", "vaccine", "research", 
              "medicine", "surgery", "therapy", "diagnosis", "prevention", "clinical"]
    
    for i in range(count):
        # Select random topics for more varied content
        doc_topics = random.sample(topics, k=min(3, len(topics)))
        topic_text = " and ".join(doc_topics)
        
        doc = {
            "doc_id": f"test_doc_{i:04d}",
            "title": f"Test Document {i:04d} about {topic_text}",
            "content": f"This is test document {i:04d} with information about {topic_text}. "
                      f"It contains medical research data related to {doc_topics[0]} studies.",
            "embedding": [random.random() for _ in range(10)]  # Simple random embedding
        }
        documents.append(doc)
    
    return documents

def ensure_min_document_count(conn, count: int = MIN_DOCUMENTS) -> int:
    """Ensure the database has at least the minimum number of documents."""
    logger.info(f"Ensuring database has at least {count} documents")
    
    # Check current document count
    current_count = 0
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
            current_count = cursor.fetchone()[0]
            logger.info(f"Current document count: {current_count}")
    except Exception as e:
        logger.warning(f"Error counting documents: {e}")
        logger.info("Creating SourceDocuments table...")
        
        # Create the table if it doesn't exist
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS SourceDocuments (
                    doc_id VARCHAR(100) PRIMARY KEY,
                    title VARCHAR(500),
                    content TEXT,
                    embedding VARCHAR(10000)
                )
            """)
    
    # Add documents if needed
    if current_count < count:
        docs_to_add = count - current_count
        logger.info(f"Adding {docs_to_add} documents to reach minimum count")
        
        # Generate documents
        documents = generate_synthetic_documents(docs_to_add)
        
        # Insert into database
        added_count = 0
        with conn.cursor() as cursor:
            for doc in documents:
                try:
                    # Convert embedding to string
                    embedding_str = str(doc["embedding"])
                    
                    cursor.execute(
                        "INSERT INTO SourceDocuments (doc_id, title, content, embedding) VALUES (?, ?, ?, ?)",
                        (doc["doc_id"], doc["title"], doc["content"], embedding_str)
                    )
                    added_count += 1
                    
                    if added_count % 100 == 0:
                        logger.info(f"Added {added_count}/{docs_to_add} documents")
                except Exception as e:
                    logger.warning(f"Error adding document {doc['doc_id']}: {e}")
        
        # Verify final count
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
            final_count = cursor.fetchone()[0]
            logger.info(f"Final document count: {final_count}")
            return final_count
    
    return current_count

def create_mock_functions():
    """Create mock functions for testing."""
    logger.info("Creating mock functions for embedding and LLM")
    
    def mock_embedding_func(text):
        """Simple mock embedding function."""
        return [random.random() for _ in range(10)]
    
    def mock_llm_func(prompt):
        """Simple mock LLM function."""
        return f"Mock answer about {prompt.split('Question:')[-1].split('\n')[0].strip()}"
    
    def mock_colbert_query_encoder(text):
        """Mock ColBERT query encoder."""
        # Return 5 token embeddings, each with 10 dimensions
        return [[random.random() for _ in range(10)] for _ in range(5)]
    
    def mock_web_search_func(query, num_results=3):
        """Mock web search function for CRAG."""
        return [f"Web result {i+1} for {query}" for i in range(num_results)]
    
    return {
        "embedding_func": mock_embedding_func,
        "llm_func": mock_llm_func,
        "colbert_query_encoder": mock_colbert_query_encoder,
        "web_search_func": mock_web_search_func
    }

def test_basic_rag(conn, mock_funcs, query: str) -> Dict[str, Any]:
    """Test the Basic RAG pipeline."""
    logger.info("Testing Basic RAG pipeline")
    
    try:
        from basic_rag.pipeline import BasicRAGPipeline
        
        # Create pipeline
        pipeline = BasicRAGPipeline(
            iris_connector=conn,
            embedding_func=mock_funcs["embedding_func"],
            llm_func=mock_funcs["llm_func"]
        )
        
        # Run query
        logger.info(f"Running BasicRAG query: {query}")
        start_time = time.time()
        result = pipeline.run(query, top_k=5)
        elapsed_time = time.time() - start_time
        
        # Log results
        logger.info(f"BasicRAG completed in {elapsed_time:.2f} seconds")
        logger.info(f"Retrieved {len(result.get('retrieved_documents', []))} documents")
        
        return {
            "technique": "basic_rag",
            "status": "success",
            "time": elapsed_time,
            "doc_count": len(result.get("retrieved_documents", [])),
            "result": result
        }
    except Exception as e:
        logger.error(f"Error testing BasicRAG: {e}", exc_info=True)
        return {
            "technique": "basic_rag",
            "status": "error",
            "error": str(e)
        }

def test_hyde(conn, mock_funcs, query: str) -> Dict[str, Any]:
    """Test the HyDE pipeline."""
    logger.info("Testing HyDE pipeline")
    
    try:
        from hyde.pipeline import HyDEPipeline
        
        # Create pipeline
        pipeline = HyDEPipeline(
            iris_connector=conn,
            embedding_func=mock_funcs["embedding_func"],
            llm_func=mock_funcs["llm_func"]
        )
        
        # Run query
        logger.info(f"Running HyDE query: {query}")
        start_time = time.time()
        result = pipeline.run(query, top_k=5)
        elapsed_time = time.time() - start_time
        
        # Log results
        logger.info(f"HyDE completed in {elapsed_time:.2f} seconds")
        logger.info(f"Retrieved {len(result.get('retrieved_documents', []))} documents")
        
        return {
            "technique": "hyde",
            "status": "success",
            "time": elapsed_time,
            "doc_count": len(result.get("retrieved_documents", [])),
            "result": result
        }
    except Exception as e:
        logger.error(f"Error testing HyDE: {e}", exc_info=True)
        return {
            "technique": "hyde",
            "status": "error",
            "error": str(e)
        }

def test_crag(conn, mock_funcs, query: str) -> Dict[str, Any]:
    """Test the CRAG pipeline."""
    logger.info("Testing CRAG pipeline")
    
    try:
        from crag.pipeline import CRAGPipeline
        
        # Create pipeline
        pipeline = CRAGPipeline(
            iris_connector=conn,
            embedding_func=mock_funcs["embedding_func"],
            llm_func=mock_funcs["llm_func"],
            web_search_func=mock_funcs["web_search_func"]
        )
        
        # Run query
        logger.info(f"Running CRAG query: {query}")
        start_time = time.time()
        result = pipeline.run(query, top_k=5)
        elapsed_time = time.time() - start_time
        
        # Log results
        logger.info(f"CRAG completed in {elapsed_time:.2f} seconds")
        logger.info(f"Retrieved {len(result.get('retrieved_context_chunks', []))} context chunks")
        
        return {
            "technique": "crag",
            "status": "success",
            "time": elapsed_time,
            "doc_count": len(result.get("retrieved_context_chunks", [])),
            "result": result
        }
    except Exception as e:
        logger.error(f"Error testing CRAG: {e}", exc_info=True)
        return {
            "technique": "crag",
            "status": "error",
            "error": str(e)
        }

def test_colbert(conn, mock_funcs, query: str) -> Dict[str, Any]:
    """Test the ColBERT pipeline."""
    logger.info("Testing ColBERT pipeline")
    
    try:
        from colbert.pipeline import ColBERTPipeline
        
        # Create pipeline
        pipeline = ColBERTPipeline(
            iris_connector=conn,
            colbert_query_encoder=mock_funcs["colbert_query_encoder"],
            llm_func=mock_funcs["llm_func"]
        )
        
        # Run query
        logger.info(f"Running ColBERT query: {query}")
        start_time = time.time()
        result = pipeline.run(query, top_k=5)
        elapsed_time = time.time() - start_time
        
        # Log results
        logger.info(f"ColBERT completed in {elapsed_time:.2f} seconds")
        logger.info(f"Retrieved {len(result.get('retrieved_documents', []))} documents")
        
        return {
            "technique": "colbert",
            "status": "success",
            "time": elapsed_time,
            "doc_count": len(result.get("retrieved_documents", [])),
            "result": result
        }
    except Exception as e:
        logger.error(f"Error testing ColBERT: {e}", exc_info=True)
        return {
            "technique": "colbert",
            "status": "error",
            "error": str(e)
        }

def test_noderag(conn, mock_funcs, query: str) -> Dict[str, Any]:
    """Test the NodeRAG pipeline."""
    logger.info("Testing NodeRAG pipeline")
    
    try:
        from noderag.pipeline import NodeRAGPipeline
        
        # Create pipeline
        pipeline = NodeRAGPipeline(
            iris_connector=conn,
            embedding_func=mock_funcs["embedding_func"],
            llm_func=mock_funcs["llm_func"]
        )
        
        # Run query
        logger.info(f"Running NodeRAG query: {query}")
        start_time = time.time()
        result = pipeline.run(query)
        elapsed_time = time.time() - start_time
        
        # Log results
        logger.info(f"NodeRAG completed in {elapsed_time:.2f} seconds")
        logger.info(f"Retrieved {len(result.get('retrieved_documents', []))} documents")
        
        return {
            "technique": "noderag",
            "status": "success",
            "time": elapsed_time,
            "doc_count": len(result.get("retrieved_documents", [])),
            "result": result
        }
    except Exception as e:
        logger.error(f"Error testing NodeRAG: {e}", exc_info=True)
        return {
            "technique": "noderag",
            "status": "error",
            "error": str(e)
        }

def test_graphrag(conn, mock_funcs, query: str) -> Dict[str, Any]:
    """Test the GraphRAG pipeline."""
    logger.info("Testing GraphRAG pipeline")
    
    try:
        from graphrag.pipeline import GraphRAGPipeline
        
        # Create pipeline
        pipeline = GraphRAGPipeline(
            iris_connector=conn,
            embedding_func=mock_funcs["embedding_func"],
            llm_func=mock_funcs["llm_func"]
        )
        
        # Run query
        logger.info(f"Running GraphRAG query: {query}")
        start_time = time.time()
        result = pipeline.run(query)
        elapsed_time = time.time() - start_time
        
        # Log results
        logger.info(f"GraphRAG completed in {elapsed_time:.2f} seconds")
        logger.info(f"Retrieved {len(result.get('retrieved_documents', []))} documents")
        
        return {
            "technique": "graphrag",
            "status": "success",
            "time": elapsed_time,
            "doc_count": len(result.get("retrieved_documents", [])),
            "result": result
        }
    except Exception as e:
        logger.error(f"Error testing GraphRAG: {e}", exc_info=True)
        return {
            "technique": "graphrag", 
            "status": "error",
            "error": str(e)
        }

def run_all_rag_tests(techniques: List[str], query: str) -> List[Dict[str, Any]]:
    """Run all RAG techniques and collect results."""
    # Set up database connection
    conn = setup_database_connection()
    
    # Ensure minimum document count
    doc_count = ensure_min_document_count(conn)
    logger.info(f"Running tests with {doc_count} documents")
    
    # Create mock functions
    mock_funcs = create_mock_functions()
    
    # Map technique names to test functions
    technique_functions = {
        "basic_rag": test_basic_rag,
        "hyde": test_hyde,
        "crag": test_crag,
        "colbert": test_colbert,
        "noderag": test_noderag,
        "graphrag": test_graphrag
    }
    
    # Run tests for selected techniques
    results = []
    for technique in techniques:
        if technique in technique_functions:
            logger.info(f"Running {technique} test")
            result = technique_functions[technique](conn, mock_funcs, query)
            results.append(result)
        else:
            logger.warning(f"Unknown technique: {technique}")
            results.append({
                "technique": technique,
                "status": "error",
                "error": "Unknown technique"
            })
    
    return results

def print_results_table(results: List[Dict[str, Any]]):
    """Print a formatted table of results."""
    logger.info("=== RAG Technique Test Results ===")
    logger.info(f"{'Technique':<15} {'Status':<10} {'Time (s)':<10} {'Doc Count':<10}")
    logger.info("-" * 50)
    
    for result in results:
        technique = result.get("technique", "unknown")
        status = result.get("status", "unknown")
        time_str = f"{result.get('time', 'N/A'):.2f}" if "time" in result else "N/A"
        doc_count = result.get("doc_count", "N/A")
        
        logger.info(f"{technique:<15} {status:<10} {time_str:<10} {doc_count:<10}")
    
    logger.info("-" * 50)
    
    # Count successes
    success_count = sum(1 for r in results if r.get("status") == "success")
    logger.info(f"Success: {success_count}/{len(results)} techniques")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test all RAG techniques with 1000+ documents")
    parser.add_argument(
        "--techniques", "-t",
        nargs="+",
        choices=DEFAULT_TECHNIQUES,
        default=DEFAULT_TECHNIQUES,
        help="RAG techniques to test"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        default=TEST_QUERIES[0],
        help="Query to test"
    )
    
    args = parser.parse_args()
    
    logger.info("=== RAG 1000+ Documents Test Runner ===")
    logger.info(f"Testing techniques: {', '.join(args.techniques)}")
    logger.info(f"Query: {args.query}")
    
    # Run all tests
    results = run_all_rag_tests(args.techniques, args.query)
    
    # Print results
    print_results_table(results)
    
    # Return success if all tests pass
    success_count = sum(1 for r in results if r.get("status") == "success")
    return 0 if success_count == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
