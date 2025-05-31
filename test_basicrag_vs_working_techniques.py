"""
Test the fixed BasicRAG against working techniques to confirm it's working properly
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func, get_llm_func
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_all_techniques():
    """Test BasicRAG against working techniques with the same query"""
    
    # Initialize components
    iris_conn = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    test_query = "What are the symptoms of diabetes?"
    
    print("="*80)
    print("TESTING BASICRAG VS WORKING TECHNIQUES")
    print("="*80)
    
    # Test BasicRAG Minimal (our fix)
    print(f"\n{'='*50}")
    print("Testing BasicRAG Minimal (Fixed)")
    print(f"{'='*50}")
    
    try:
        from basic_rag.pipeline_minimal_fixed import BasicRAGPipelineMinimal
        
        pipeline = BasicRAGPipelineMinimal(
            iris_connector=iris_conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        result = pipeline.run(test_query, top_k=5)
        
        print(f"✅ BasicRAG Minimal SUCCESS!")
        print(f"   Retrieved: {result['metadata']['num_retrieved']} documents")
        print(f"   Answer length: {len(result['answer'])} characters")
        print(f"   Pipeline: {result['metadata']['pipeline']}")
        
        for i, doc in enumerate(result['retrieved_documents'], 1):
            metadata = doc['metadata']
            score = metadata.get('similarity_score', 0)
            title = metadata.get('title', 'No title')[:40]
            print(f"   Doc {i}: score={score:.4f}, title={title}...")
            
    except Exception as e:
        print(f"❌ BasicRAG Minimal ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test NodeRAG (working)
    print(f"\n{'='*50}")
    print("Testing NodeRAG V2 (Working)")
    print(f"{'='*50}")
    
    try:
        from noderag.pipeline_v2 import NodeRAGPipelineV2
        
        pipeline = NodeRAGPipelineV2(
            iris_connector=iris_conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        result = pipeline.run(test_query, top_k=5)
        
        print(f"✅ NodeRAG V2 SUCCESS!")
        print(f"   Retrieved nodes: {result['metadata']['num_nodes_used']}")
        print(f"   Answer length: {len(result['answer'])} characters")
        print(f"   Pipeline: {result['metadata']['pipeline']}")
        
        for i, node in enumerate(result['retrieved_nodes'], 1):
            metadata = node['metadata']
            score = metadata.get('similarity_score', 0)
            title = metadata.get('title', 'No title')[:40]
            node_type = node.get('type', 'unknown')
            print(f"   Node {i}: type={node_type}, score={score:.4f}, title={title}...")
            
    except Exception as e:
        print(f"❌ NodeRAG V2 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test CRAG (working)
    print(f"\n{'='*50}")
    print("Testing CRAG V2 (Working)")
    print(f"{'='*50}")
    
    try:
        from crag.pipeline_v2 import CRAGPipelineV2
        
        pipeline = CRAGPipelineV2(
            iris_connector=iris_conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        result = pipeline.run(test_query, top_k=5)
        
        print(f"✅ CRAG V2 SUCCESS!")
        print(f"   Retrieved: {result['metadata']['num_documents_retrieved']} documents")
        print(f"   Answer length: {len(result['answer'])} characters")
        print(f"   Pipeline: {result['metadata']['pipeline']}")
        
        for i, doc in enumerate(result['retrieved_documents'], 1):
            metadata = doc['metadata']
            score = metadata.get('similarity_score', 0)
            title = metadata.get('title', 'No title')[:40]
            print(f"   Doc {i}: score={score:.4f}, title={title}...")
            
    except Exception as e:
        print(f"❌ CRAG V2 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test original BasicRAG V2 (broken)
    print(f"\n{'='*50}")
    print("Testing Original BasicRAG V2 (Broken)")
    print(f"{'='*50}")
    
    try:
        from basic_rag.pipeline_v2 import BasicRAGPipelineV2
        
        pipeline = BasicRAGPipelineV2(
            iris_connector=iris_conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        result = pipeline.run(test_query, top_k=5)
        
        print(f"✅ Original BasicRAG V2 SUCCESS!")
        print(f"   Retrieved: {result['metadata']['num_retrieved']} documents")
        print(f"   Answer length: {len(result['answer'])} characters")
        print(f"   Pipeline: {result['metadata']['pipeline']}")
        
        for i, doc in enumerate(result['retrieved_documents'], 1):
            metadata = doc['metadata']
            score = metadata.get('similarity_score', 0)
            title = metadata.get('title', 'No title')[:40]
            print(f"   Doc {i}: score={score:.4f}, title={title}...")
            
    except Exception as e:
        print(f"❌ Original BasicRAG V2 ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_all_techniques()