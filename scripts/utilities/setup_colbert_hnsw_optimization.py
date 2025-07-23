#!/usr/bin/env python3
"""
ColBERT HNSW Optimization Setup Script

This script:
1. Checks if HNSW index exists on DocumentTokenEmbeddings
2. Creates the index if it doesn't exist
3. Tests the optimized ColBERT pipeline performance
4. Compares performance with the original implementation
"""

import os
import sys
import time
import logging
import json
from typing import Dict, Any, List

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func # Updated import
from iris_rag.pipelines.colbert import ColbertRAGPipeline # Updated import - Original pipeline likely deprecated
from iris_rag.pipelines.colbert import ColBERTRAGPipeline # Updated import - Optimized is now the working version
from src.working.colbert.utils import check_hnsw_token_index_exists, create_hnsw_token_index # Assuming utils for HNSW functions

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_token_embeddings_data(iris_connector) -> Dict[str, Any]:
    """Check the current state of token embeddings data."""
    try:
        cursor = iris_connector.cursor()
        
        # Check if DocumentTokenEmbeddings table exists and has data
        cursor.execute("SELECT COUNT(*) FROM RAG_HNSW.DocumentTokenEmbeddings")
        token_count = cursor.fetchone()[0]
        
        # Check unique documents with token embeddings
        cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM RAG_HNSW.DocumentTokenEmbeddings")
        doc_count = cursor.fetchone()[0]
        
        # Check if any token embeddings are NULL
        cursor.execute("SELECT COUNT(*) FROM RAG_HNSW.DocumentTokenEmbeddings WHERE token_embedding IS NULL")
        null_count = cursor.fetchone()[0]
        
        cursor.close()
        
        return {
            "total_tokens": token_count,
            "documents_with_tokens": doc_count,
            "null_embeddings": null_count,
            "valid_embeddings": token_count - null_count
        }
        
    except Exception as e:
        logger.error(f"Error checking token embeddings data: {e}")
        return {"error": str(e)}

def setup_hnsw_index(iris_connector) -> bool:
    """Set up HNSW index on token embeddings."""
    logger.info("Checking HNSW index status on DocumentTokenEmbeddings...")
    
    if check_hnsw_token_index_exists(iris_connector):
        logger.info("HNSW index on DocumentTokenEmbeddings already exists")
        return True
    
    logger.info("HNSW index not found. Creating index...")
    success = create_hnsw_token_index(iris_connector)
    
    if success:
        logger.info("Successfully created HNSW index on DocumentTokenEmbeddings")
        return True
    else:
        logger.error("Failed to create HNSW index on DocumentTokenEmbeddings")
        return False

def create_mock_colbert_encoder(embedding_dim: int = 128):
    """Create a mock ColBERT encoder for testing."""
    def mock_encoder(text: str) -> List[List[float]]:
        import numpy as np
        # Create deterministic embeddings based on text
        words = text.split()[:10]  # Limit to 10 tokens
        embeddings = []
        
        for i, word in enumerate(words):
            # Create a deterministic but varied embedding
            np.random.seed(hash(word) % 10000)
            embedding = np.random.randn(embedding_dim)
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embeddings.append(embedding.tolist())
        
        return embeddings
    
    return mock_encoder

def test_pipeline_performance(pipeline, pipeline_name: str, test_queries: List[str]) -> Dict[str, Any]:
    """Test pipeline performance with multiple queries."""
    logger.info(f"Testing {pipeline_name} performance...")
    
    results = {
        "pipeline_name": pipeline_name,
        "total_time": 0,
        "query_results": [],
        "avg_time_per_query": 0,
        "success_count": 0,
        "error_count": 0
    }
    
    start_time = time.time()
    
    for i, query in enumerate(test_queries):
        query_start = time.time()
        try:
            result = pipeline.run(query, top_k=5, similarity_threshold=0.5)
            query_time = time.time() - query_start
            
            query_result = {
                "query_id": i,
                "query": query[:50] + "..." if len(query) > 50 else query,
                "time_seconds": query_time,
                "documents_found": len(result.get("retrieved_documents", [])),
                "success": True
            }
            
            results["query_results"].append(query_result)
            results["success_count"] += 1
            
            logger.info(f"  Query {i+1}: {query_time:.2f}s, {query_result['documents_found']} docs")
            
        except Exception as e:
            query_time = time.time() - query_start
            logger.error(f"  Query {i+1} failed: {e}")
            
            query_result = {
                "query_id": i,
                "query": query[:50] + "..." if len(query) > 50 else query,
                "time_seconds": query_time,
                "documents_found": 0,
                "success": False,
                "error": str(e)
            }
            
            results["query_results"].append(query_result)
            results["error_count"] += 1
    
    total_time = time.time() - start_time
    results["total_time"] = total_time
    results["avg_time_per_query"] = total_time / len(test_queries) if test_queries else 0
    
    logger.info(f"{pipeline_name} completed: {total_time:.2f}s total, {results['avg_time_per_query']:.2f}s avg per query")
    
    return results

def run_performance_comparison():
    """Run performance comparison between original and optimized ColBERT."""
    logger.info("Starting ColBERT performance comparison...")
    
    # Test queries
    test_queries = [
        "What are the latest treatments for diabetes?",
        "How does machine learning improve medical diagnosis?",
        "What are the mechanisms of cancer immunotherapy?",
        "How do genetic mutations contribute to disease development?",
        "What role does AI play in modern healthcare systems?"
    ]
    
    try:
        # Get database connection
        iris_connector = get_iris_connection()
        if not iris_connector:
            raise ConnectionError("Failed to get IRIS connection")
        
        # Check token embeddings data
        data_status = check_token_embeddings_data(iris_connector)
        logger.info(f"Token embeddings data status: {data_status}")
        
        if data_status.get("valid_embeddings", 0) == 0:
            logger.warning("No valid token embeddings found. ColBERT testing may not work properly.")
        
        # Set up HNSW index
        hnsw_success = setup_hnsw_index(iris_connector)
        
        # Create mock encoders
        mock_encoder = create_mock_colbert_encoder(128)
        llm_func = get_llm_func(provider="stub")
        
        # Test original pipeline
        logger.info("\n" + "="*50)
        logger.info("Testing Original ColBERT Pipeline")
        logger.info("="*50)
        
        original_pipeline = ColbertRAGPipeline(
            iris_connector=iris_connector,
            colbert_query_encoder_func=mock_encoder,
            colbert_doc_encoder_func=mock_encoder,
            llm_func=llm_func
        )
        
        original_results = test_pipeline_performance(original_pipeline, "Original ColBERT", test_queries)
        
        # Test optimized pipeline
        logger.info("\n" + "="*50)
        logger.info("Testing Optimized ColBERT Pipeline")
        logger.info("="*50)
        
        optimized_pipeline = ColBERTRAGPipeline(
            iris_connector=iris_connector,
            colbert_query_encoder_func=mock_encoder,
            colbert_doc_encoder_func=mock_encoder,
            llm_func=llm_func
        )
        
        optimized_results = test_pipeline_performance(optimized_pipeline, "Optimized ColBERT", test_queries)
        
        # Generate comparison report
        comparison_report = {
            "timestamp": time.time(),
            "data_status": data_status,
            "hnsw_index_created": hnsw_success,
            "original_pipeline": original_results,
            "optimized_pipeline": optimized_results,
            "performance_improvement": {
                "time_reduction_seconds": original_results["total_time"] - optimized_results["total_time"],
                "time_reduction_percent": ((original_results["total_time"] - optimized_results["total_time"]) / original_results["total_time"] * 100) if original_results["total_time"] > 0 else 0,
                "speedup_factor": original_results["total_time"] / optimized_results["total_time"] if optimized_results["total_time"] > 0 else float('inf')
            }
        }
        
        # Save results
        timestamp = int(time.time())
        results_file = f"colbert_performance_comparison_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE COMPARISON SUMMARY")
        logger.info("="*60)
        logger.info(f"Original ColBERT: {original_results['total_time']:.2f}s total, {original_results['avg_time_per_query']:.2f}s avg")
        logger.info(f"Optimized ColBERT: {optimized_results['total_time']:.2f}s total, {optimized_results['avg_time_per_query']:.2f}s avg")
        logger.info(f"Time reduction: {comparison_report['performance_improvement']['time_reduction_seconds']:.2f}s ({comparison_report['performance_improvement']['time_reduction_percent']:.1f}%)")
        logger.info(f"Speedup factor: {comparison_report['performance_improvement']['speedup_factor']:.2f}x")
        logger.info(f"Results saved to: {results_file}")
        
        iris_connector.close()
        
        return comparison_report
        
    except Exception as e:
        logger.error(f"Error during performance comparison: {e}", exc_info=True)
        return None

def main():
    """Main function."""
    logger.info("ColBERT HNSW Optimization Setup Starting...")
    
    try:
        # Run performance comparison
        results = run_performance_comparison()
        
        if results:
            logger.info("ColBERT optimization setup completed successfully!")
            
            # Check if optimization was successful
            improvement = results["performance_improvement"]
            if improvement["speedup_factor"] > 1.5:
                logger.info(f"✅ Significant performance improvement achieved: {improvement['speedup_factor']:.2f}x speedup")
            elif improvement["speedup_factor"] > 1.1:
                logger.info(f"✅ Moderate performance improvement achieved: {improvement['speedup_factor']:.2f}x speedup")
            else:
                logger.warning(f"⚠️  Limited performance improvement: {improvement['speedup_factor']:.2f}x speedup")
                logger.warning("This may indicate that HNSW indexing is not working as expected or data is limited")
        else:
            logger.error("❌ ColBERT optimization setup failed")
            
    except Exception as e:
        logger.error(f"❌ Fatal error during ColBERT optimization setup: {e}", exc_info=True)

if __name__ == "__main__":
    main()