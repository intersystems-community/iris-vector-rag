#!/usr/bin/env python3
"""
Comprehensive test for NodeRAG functionality
"""

import os
import sys
# Old path insert - keep for now if it serves a specific purpose for this test file
sys.path.insert(0, os.path.abspath('.'))
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func # Updated import
from iris_rag.pipelines.noderag import NodeRAGPipeline # Corrected import path and class name

def test_noderag_comprehensive():
    """Test NodeRAG with comprehensive debugging"""
    print("Testing NodeRAG Comprehensive...")
    
    # Check database state first
    iris_conn = get_iris_connection()
    cursor = iris_conn.cursor()
    
    # Check document count
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
    doc_count = cursor.fetchone()[0]
    print(f"Documents with embeddings: {doc_count}")
    
    # Check chunk count
    cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks WHERE embedding IS NOT NULL")
    chunk_count = cursor.fetchone()[0]
    print(f"Chunks with embeddings: {chunk_count}")
    
    cursor.close()
    
    if doc_count == 0 and chunk_count == 0:
        print("No embeddings found - cannot test retrieval")
        return False
    
    # Initialize components
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Create NodeRAG pipeline
    pipeline = NodeRAGPipeline(
        iris_connector=iris_conn,
        embedding_func=embedding_func,
        llm_func=llm_func
    )
    
    # Test with very low threshold
    test_query = "What are the symptoms of diabetes?"
    
    try:
        print(f"\nTesting query: {test_query}")
        print("Testing with threshold 0.0...")
        
        # Test document retrieval directly
        docs = pipeline.retrieve_documents(test_query, top_k=5, similarity_threshold=0.0)
        print(f"Documents retrieved: {len(docs)}")
        
        # Test chunk retrieval directly
        chunks = pipeline.retrieve_chunks(test_query, top_k=10, similarity_threshold=0.0)
        print(f"Chunks retrieved: {len(chunks)}")
        
        if len(docs) > 0 or len(chunks) > 0:
            # Run full pipeline
            result = pipeline.run(test_query, top_k=3)
            
            print(f"✓ NodeRAG completed successfully")
            print(f"  - Retrieved {len(result.get('retrieved_nodes', []))} nodes")
            print(f"  - Answer length: {len(result.get('answer', ''))}")
            
            # Show sample content
            for i, node in enumerate(result.get('retrieved_nodes', [])[:2]):
                content = node.get('content', '')
                print(f"  - Node {i}: {content[:100]}...")
                
            return True
        else:
            print("No documents or chunks retrieved even with threshold 0.0")
            return False
        
    except Exception as e:
        print(f"✗ NodeRAG failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_noderag_comprehensive()