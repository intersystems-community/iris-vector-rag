#!/usr/bin/env python3
"""
Test all corrected RAG techniques using the verified TO_VECTOR(embedding) approach
"""

import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) # Assuming script is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
from src.deprecated.basic_rag.pipeline import BasicRAGPipeline # Updated import
from src.experimental.noderag.pipeline import NodeRAGPipeline as NodeRAGPipelineV2 # Updated import
from src.experimental.crag.pipeline import CRAGPipeline as CRAGPipelineV2 # Updated import
from src.experimental.hyde.pipeline import HyDEPipeline # Updated import
from src.experimental.graphrag.pipeline import GraphRAGPipeline as GraphRAGPipelineV2 # Updated import
from src.common.iris_connector_jdbc import get_iris_connection # Updated import
from src.common.utils import get_embedding_func, get_llm_func # Updated import

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
        all_techniques = [
            ("BasicRAG", BasicRAGPipeline(db_conn, embed_fn, llm_fn)),
            ("NodeRAG", NodeRAGPipelineV2(db_conn, embed_fn, llm_fn)),
            ("CRAG", CRAGPipelineV2(db_conn, embed_fn, llm_fn)),
            ("HyDE", HyDEPipeline(db_conn, embed_fn, llm_fn)),
            ("GraphRAG", GraphRAGPipelineV2(db_conn, embed_fn, llm_fn))
        ]

        techniques_to_test = all_techniques
        if os.environ.get("TEST_ONLY_GRAPHRAG") == "1":
            print("🔬 Running ONLY GraphRAG V2 test due to TEST_ONLY_GRAPHRAG environment variable.")
            techniques_to_test = [tech for tech in all_techniques if tech[0] == "GraphRAG"]
        
        for name, pipeline in techniques_to_test:
            print(f"\n🚀 Testing {name}...")
            try:
                if name == "GraphRAG":
                    # GraphRAGPipelineV2.run does not take similarity_threshold
                    result = pipeline.run(test_query, top_k=5)
                else:
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
        
        print(f"\n🎯 Success Rate: {successful}/{len(techniques_to_test)} techniques working")
        
        return successful == len(techniques_to_test)
        
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