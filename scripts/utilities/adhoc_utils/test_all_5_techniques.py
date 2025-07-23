#!/usr/bin/env python3
"""
Test all 5 major RAG techniques to validate the complete system
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func, get_llm_func

def test_technique(name, pipeline_class, module_path):
    """Test a single RAG technique"""
    print(f"\n{'='*20} Testing {name} {'='*20}")
    
    try:
        # Import the pipeline
        module = __import__(module_path, fromlist=[pipeline_class])
        PipelineClass = getattr(module, pipeline_class)
        
        # Initialize components
        iris_conn = get_iris_connection()
        embedding_func = get_embedding_func()
        llm_func = get_llm_func()
        
        # Create pipeline
        pipeline = PipelineClass(
            iris_connector=iris_conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        # Test query
        test_query = "What are the symptoms of diabetes?"
        result = pipeline.run(test_query, top_k=3)
        
        # Check results
        retrieved_count = 0
        if 'retrieved_documents' in result:
            retrieved_count = len(result['retrieved_documents'])
        elif 'retrieved_nodes' in result:
            retrieved_count = len(result['retrieved_nodes'])
        elif 'metadata' in result and 'num_retrieved' in result['metadata']:
            retrieved_count = result['metadata']['num_retrieved']
        
        answer_length = len(result.get('answer', ''))
        
        print(f"✓ {name} completed successfully")
        print(f"  - Retrieved: {retrieved_count} items")
        print(f"  - Answer length: {answer_length}")
        
        success = retrieved_count > 0
        if success:
            print(f"  ✓ SUCCESS: {name} is working!")
        else:
            print(f"  ⚠️  WARNING: {name} retrieved 0 items")
            
        return success
        
    except Exception as e:
        print(f"✗ {name} failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_techniques():
    """Test all 5 major RAG techniques"""
    print("Testing All 5 Major RAG Techniques")
    print("="*60)
    
    techniques = [
        ("BasicRAG", "BasicRAGPipeline", "basic_rag.pipeline"),
        ("HyDE", "HyDERAGPipeline", "hyde.pipeline"),
        ("HybridIFindRAG", "HybridIFindRAGPipeline", "iris_rag.pipelines.hybrid_ifind"),
        ("CRAG", "CRAGPipelineV2", "crag.pipeline_v2"),
        ("NodeRAG", "NodeRAGPipelineV2", "noderag.pipeline_v2"),
    ]
    
    results = {}
    
    for name, pipeline_class, module_path in techniques:
        results[name] = test_technique(name, pipeline_class, module_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY:")
    print("="*60)
    
    working_count = 0
    for name, success in results.items():
        status = "✓ WORKING" if success else "✗ FAILED"
        print(f"  {name:20} {status}")
        if success:
            working_count += 1
    
    print(f"\nWorking techniques: {working_count}/5")
    
    if working_count == 5:
        print("🎉 SUCCESS: All 5 major RAG techniques are working!")
    else:
        print(f"⚠️  {5-working_count} technique(s) need fixing")
    
    return working_count == 5

if __name__ == "__main__":
    test_all_techniques()