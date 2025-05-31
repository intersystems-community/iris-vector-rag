#!/usr/bin/env python3
"""
Test the corrected BasicRAG pipeline using the verified TO_VECTOR(embedding) approach
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import logging
from basic_rag.pipeline import BasicRAGPipeline
from common.iris_connector_jdbc import get_iris_connection
from common.utils import get_embedding_func, get_llm_func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_corrected_basicrag():
    """Test BasicRAG with the verified TO_VECTOR(embedding) syntax"""
    
    print("🔧 Testing Corrected BasicRAG with Verified TO_VECTOR(embedding) Approach")
    print("=" * 70)
    
    try:
        # Setup connections
        db_conn = get_iris_connection()
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub")
        
        # Initialize pipeline
        pipeline = BasicRAGPipeline(
            iris_connector=db_conn,
            embedding_func=embed_fn,
            llm_func=llm_fn,
            schema="RAG"
        )
        
        # Test query
        test_query = "What is diabetes?"
        print(f"📝 Test Query: {test_query}")
        
        # Run the pipeline
        print("\n🚀 Running BasicRAG pipeline...")
        result = pipeline.run(test_query, top_k=5, similarity_threshold=0.1)
        
        # Verify results
        print("\n📊 Results:")
        print(f"✅ Query: {result['query']}")
        print(f"✅ Answer: {result['answer'][:100]}...")
        print(f"✅ Retrieved Documents: {result['document_count']}")
        print(f"✅ Similarity Threshold: {result['similarity_threshold']}")
        
        # Check document details
        if result['retrieved_documents']:
            print(f"\n📋 Document Details:")
            for i, doc in enumerate(result['retrieved_documents'][:3]):
                print(f"  Doc {i+1}: ID={doc['id']}, Score={doc.get('score', 0):.4f}")
                if 'metadata' in doc and 'title' in doc['metadata']:
                    print(f"         Title: {doc['metadata']['title'][:60]}...")
        
        # Success verification
        if result['document_count'] > 0:
            print(f"\n✅ SUCCESS: BasicRAG retrieved {result['document_count']} documents using verified TO_VECTOR(embedding) syntax!")
            print("✅ The corrected approach is working as expected!")
            return True
        else:
            print("\n❌ FAILURE: No documents retrieved")
            return False
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'db_conn' in locals() and db_conn:
            db_conn.close()

if __name__ == "__main__":
    success = test_corrected_basicrag()
    if success:
        print("\n🎉 BasicRAG correction test PASSED!")
    else:
        print("\n💥 BasicRAG correction test FAILED!")
    
    sys.exit(0 if success else 1)