#!/usr/bin/env python3
"""
Test all corrected RAG techniques using the verified TO_VECTOR(embedding) approach
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import logging
from basic_rag.pipeline import BasicRAGPipeline
from noderag.pipeline_v2 import NodeRAGPipelineV2
from crag.pipeline_v2 import CRAGPipelineV2
from hyde.pipeline import HyDEPipeline
from graphrag.pipeline_jdbc_fixed import JDBCFixedGraphRAGPipeline
from common.iris_connector_jdbc import get_iris_connection
from common.utils import get_embedding_func, get_llm_func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_all_corrected_techniques():
    """Test all corrected RAG techniques"""
    
    print("🔧 Testing All Corrected RAG Techniques with Verified TO_VECTOR(embedding) Approach")
    print("=" * 80)
    
    results = {}
    
    try:
        # Setup connections
        db_conn = get_iris_connection()
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub")
        
        # Test query
        test_query = "What is diabetes?"
        print(f"📝 Test Query: {test_query}")
        
        # Test techniques
        techniques = [
            ("BasicRAG", BasicRAGPipeline(db_conn, embed_fn, llm_fn)),
            ("NodeRAG", NodeRAGPipelineV2(db_conn, embed_fn, llm_fn)),
            ("CRAG", CRAGPipelineV2(db_conn, embed_fn, llm_fn)),
            ("HyDE", HyDEPipeline(db_conn, embed_fn, llm_fn)),
            ("GraphRAG", JDBCFixedGraphRAGPipeline(db_conn, embed_fn, llm_fn))
        ]
        
        for name, pipeline in techniques:
            print(f"\n🚀 Testing {name}...")
            try:
                result = pipeline.run(test_query, top_k=5, similarity_threshold=0.1)
                doc_count = result.get('document_count', len(result.get('retrieved_documents', [])))
                
                if doc_count > 0:
                    print(f"✅ {name}: Retrieved {doc_count} documents")
                    results[name] = {"status": "SUCCESS", "documents": doc_count}
                else:
                    print(f"❌ {name}: No documents retrieved")
                    results[name] = {"status": "FAILED", "documents": 0}
                    
            except Exception as e:
                print(f"❌ {name}: Error - {e}")
                results[name] = {"status": "ERROR", "error": str(e)}
        
        # Summary
        print(f"\n📊 SUMMARY:")
        print("=" * 40)
        successful = 0
        for name, result in results.items():
            status = result["status"]
            if status == "SUCCESS":
                print(f"✅ {name}: {result['documents']} documents")
                successful += 1
            elif status == "FAILED":
                print(f"❌ {name}: No documents")
            else:
                print(f"💥 {name}: Error")
        
        print(f"\n🎯 Success Rate: {successful}/{len(techniques)} techniques working")
        
        return successful == len(techniques)
        
    except Exception as e:
        print(f"\n❌ SETUP ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'db_conn' in locals() and db_conn:
            db_conn.close()

if __name__ == "__main__":
    success = test_all_corrected_techniques()
    if success:
        print("\n🎉 All RAG techniques are working with the verified TO_VECTOR(embedding) approach!")
    else:
        print("\n💥 Some RAG techniques still have issues!")
    
    sys.exit(0 if success else 1)