#!/usr/bin/env python3
"""
Test to identify and fix NodeRAG stream handling issues
"""

import os
import sys
# Old path insert - keep for now if it serves a specific purpose for this test file
sys.path.insert(0, os.path.abspath('.'))
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.common.iris_connector import get_iris_connection # Updated import
from src.common.utils import get_embedding_func, get_llm_func # Updated import
from src.deprecated.noderag.pipeline_v2 import NodeRAGPipelineV2 # Updated import

def test_noderag_current_state():
    """Test NodeRAG current state to identify stream issues"""
    print("Testing NodeRAG Current State...")
    
    # Initialize components
    iris_conn = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Create NodeRAG pipeline
    pipeline = NodeRAGPipelineV2(
        iris_connector=iris_conn,
        embedding_func=embedding_func,
        llm_func=llm_func
    )
    
    # Test simple query
    test_query = "What are the symptoms of diabetes?"
    
    try:
        print(f"\nTesting query: {test_query}")
        result = pipeline.run(test_query, top_k=3)
        
        print(f"✓ NodeRAG completed successfully")
        print(f"  - Retrieved {len(result.get('retrieved_nodes', []))} nodes")
        print(f"  - Answer length: {len(result.get('answer', ''))}")
        
        # Check for stream issues in content
        for i, node in enumerate(result.get('retrieved_nodes', [])):
            content = node.get('content', '')
            if 'IRISInputStream' in str(content):
                print(f"  ⚠️  Node {i} has stream issue: {content[:100]}")
            else:
                print(f"  ✓ Node {i} content OK: {content[:50]}...")
                
        return True
        
    except Exception as e:
        print(f"✗ NodeRAG failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_noderag_current_state()