#!/usr/bin/env python3
"""
Test script to verify CRAG retrieval issues and test the fix
"""

import sys
import os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func # Updated import
from iris_rag.pipelines.crag import CRAGPipeline

def test_crag_retrieval():
    """Test CRAG document retrieval"""
    print("Testing CRAG document retrieval...")
    
    # Initialize components
    iris_conn = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Create CRAG pipeline
    crag_pipeline = CRAGPipeline(
        iris_connector=iris_conn,
        embedding_func=embedding_func,
        llm_func=llm_func
    )
    
    # Test query
    test_query = "What are the symptoms of diabetes?"
    
    print(f"Testing query: {test_query}")
    
    # Test retrieval with different thresholds
    for threshold in [0.0, 0.001, 0.01]:
        print(f"\n--- Testing with threshold {threshold} ---")
        try:
            docs = crag_pipeline.retrieve_documents(test_query, top_k=5, similarity_threshold=threshold)
            print(f"Retrieved {len(docs)} documents with threshold {threshold}")
            
            if docs:
                for i, doc in enumerate(docs[:3]):
                    print(f"  Doc {i+1}: ID={doc.id}, Score={doc.score:.4f}")
            else:
                print("  No documents retrieved!")
                
        except Exception as e:
            print(f"  Error with threshold {threshold}: {e}")
    
    # Test full pipeline
    print(f"\n--- Testing full CRAG pipeline ---")
    try:
        result = crag_pipeline.run(test_query, top_k=5)
        print(f"Full pipeline result:")
        print(f"  Query: {result['query']}")
        print(f"  Documents retrieved: {result['metadata']['num_documents_retrieved']}")
        print(f"  Answer length: {len(result['answer'])}")
    except Exception as e:
        print(f"  Error in full pipeline: {e}")

if __name__ == "__main__":
    test_crag_retrieval()