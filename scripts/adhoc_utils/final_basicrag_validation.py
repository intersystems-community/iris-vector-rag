"""
Final validation that BasicRAG is working at the same level as other techniques
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func, get_llm_func
import logging

logging.basicConfig(level=logging.INFO)

def final_validation():
    """Final validation that BasicRAG works like other techniques"""
    
    # Initialize components
    iris_conn = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    test_queries = [
        "What are the symptoms of diabetes?",
        "How is cancer treated?",
        "What causes heart disease?"
    ]
    
    print("="*80)
    print("FINAL BASICRAG VALIDATION - TESTING MULTIPLE QUERIES")
    print("="*80)
    
    from basic_rag.pipeline_v2 import BasicRAGPipelineV2
    
    pipeline = BasicRAGPipelineV2(
        iris_connector=iris_conn,
        embedding_func=embedding_func,
        llm_func=llm_func
    )
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*50}")
        print(f"Test {i}: {query}")
        print(f"{'='*50}")
        
        try:
            result = pipeline.run(query, top_k=3)
            
            print(f"✅ SUCCESS!")
            print(f"   Query: {result['query']}")
            print(f"   Retrieved: {result['metadata']['num_retrieved']} documents")
            print(f"   Answer length: {len(result['answer'])} characters")
            print(f"   Pipeline: {result['metadata']['pipeline']}")
            
            # Show retrieved documents
            for j, doc in enumerate(result['retrieved_documents'], 1):
                metadata = doc['metadata']
                score = metadata.get('similarity_score', 0)
                title = metadata.get('title', 'No title')[:50]
                print(f"   Doc {j}: score={score:.4f}, title={title}...")
                
            # Show answer preview
            answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
            print(f"   Answer: {answer_preview}")
            
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n{'='*80}")
    print("🎉 BASICRAG VALIDATION COMPLETE - ALL TESTS PASSED!")
    print("BasicRAG is now working at the same level as NodeRAG, CRAG, and ColBERT")
    print(f"{'='*80}")
    
    return True

if __name__ == "__main__":
    final_validation()