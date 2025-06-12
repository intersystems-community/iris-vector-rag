#!/usr/bin/env python3
"""
Test BasicRAG to see if it works and compare with CRAG
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) # Assuming script is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func # Updated import
from src.deprecated.basic_rag.pipeline_v2_fixed import BasicRAGPipelineV2Fixed as BasicRAGPipelineV2 # Updated import

def test_basic_rag():
    """Test BasicRAG to see if it works"""
    print("Testing BasicRAG document retrieval...")
    
    # Initialize components
    iris_conn = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Create BasicRAG pipeline
    basic_rag_pipeline = BasicRAGPipelineV2(
        iris_connector=iris_conn,
        embedding_func=embedding_func,
        llm_func=llm_func
    )
    
    # Test query
    test_query = "What are the symptoms of diabetes?"
    
    print(f"Testing query: {test_query}")
    
    try:
        docs = basic_rag_pipeline.retrieve_documents(test_query, top_k=5, similarity_threshold=0.0)
        print(f"BasicRAG retrieved {len(docs)} documents")
        
        if docs:
            for i, doc in enumerate(docs[:3]):
                print(f"  Doc {i+1}: ID={doc.id}, Score={doc.score:.4f}")
        else:
            print("  No documents retrieved!")
            
    except Exception as e:
        print(f"  Error with BasicRAG: {e}")

if __name__ == "__main__":
    test_basic_rag()