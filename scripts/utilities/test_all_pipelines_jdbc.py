#!/usr/bin/env python3
"""
Test All Pipelines Performance with JDBC
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import logging
from typing import Dict, Any

# Import all pipelines
from src.deprecated.basic_rag.pipeline import BasicRAGPipeline # Updated import
from src.experimental.hyde.pipeline import HyDEPipeline # Updated import
from src.experimental.crag.pipeline import CRAGPipeline # Updated import
from src.experimental.noderag.pipeline import NodeRAGPipeline # Updated import
from src.deprecated.colbert.pipeline import OptimizedColbertRAGPipeline as ColBERTPipeline # Updated import
from src.experimental.graphrag.pipeline import GraphRAGPipeline # Updated import
from src.experimental.hybrid_ifind_rag.pipeline import HybridiFindRAGPipeline # Updated import

from common.iris_connector_jdbc import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func # Updated import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline(name: str, pipeline: Any, query: str) -> Dict[str, Any]:
    """Test a single pipeline"""
    logger.info(f"Testing {name}...")
    
    start_time = time.time()
    try:
        if name == "CRAG":
            # CRAG doesn't accept similarity_threshold
            result = pipeline.run(query, top_k=10)
        else:
            result = pipeline.run(query, top_k=10, similarity_threshold=0.1)
        
        elapsed = time.time() - start_time
        
        return {
            "success": True,
            "time": elapsed,
            "documents": len(result.get("retrieved_documents", [])),
            "answer_length": len(result.get("answer", ""))
        }
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"{name} failed: {e}")
        return {
            "success": False,
            "time": elapsed,
            "error": str(e)
        }

def main():
    """Test all pipelines"""
    print("🚀 Testing All Pipelines with JDBC")
    print("=" * 60)
    
    # Initialize connection and functions
    conn = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Initialize pipelines
    pipelines = {}
    
    try:
        pipelines["BasicRAG"] = BasicRAGPipeline(conn, embedding_func, llm_func)
    except Exception as e:
        logger.error(f"Failed to initialize BasicRAG: {e}")
    
    try:
        pipelines["HyDE"] = HyDEPipeline(conn, embedding_func, llm_func)
    except Exception as e:
        logger.error(f"Failed to initialize HyDE: {e}")
    
    try:
        pipelines["CRAG"] = CRAGPipeline(conn, embedding_func, llm_func)
    except Exception as e:
        logger.error(f"Failed to initialize CRAG: {e}")
    
    try:
        pipelines["NodeRAG"] = NodeRAGPipeline(conn, embedding_func, llm_func)
    except Exception as e:
        logger.error(f"Failed to initialize NodeRAG: {e}")
    
    try:
        pipelines["ColBERT"] = ColBERTPipeline(
            conn, embedding_func, embedding_func, llm_func
        )
    except Exception as e:
        logger.error(f"Failed to initialize ColBERT: {e}")
    
    try:
        pipelines["GraphRAG"] = GraphRAGPipeline(conn, embedding_func, llm_func)
    except Exception as e:
        logger.error(f"Failed to initialize GraphRAG: {e}")
    
    try:
        pipelines["HybridIFind"] = HybridiFindRAGPipeline(conn, embedding_func, llm_func)
    except Exception as e:
        logger.error(f"Failed to initialize HybridIFind: {e}")
    
    # Test query
    test_query = "What are the symptoms of diabetes?"
    
    # Test each pipeline
    results = {}
    for name, pipeline in pipelines.items():
        results[name] = test_pipeline(name, pipeline, test_query)
    
    # Print results
    print("\n📊 Results Summary")
    print("=" * 60)
    
    for name, result in results.items():
        if result["success"]:
            print(f"✅ {name}: {result['time']:.2f}s, {result['documents']} docs")
        else:
            print(f"❌ {name}: Failed - {result.get('error', 'Unknown error')}")
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    main()
