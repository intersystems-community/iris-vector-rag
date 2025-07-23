#!/usr/bin/env python3
"""
Test all 7 RAG techniques to verify they're working with the V2 pattern
"""

import os
import sys
import logging
import time
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func, get_llm_func

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_technique(technique_name: str, pipeline_class, test_query: str) -> Dict[str, Any]:
    """Test a single RAG technique"""
    print(f"\n{'='*60}")
    print(f"Testing {technique_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize components
        iris_conn = get_iris_connection()
        embedding_func = get_embedding_func()
        llm_func = get_llm_func()
        
        # Create pipeline
        pipeline = pipeline_class(
            iris_connector=iris_conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        # Run test
        start_time = time.time()
        result = pipeline.run(test_query, top_k=3)
        execution_time = time.time() - start_time
        
        # Validate result - handle different return formats
        retrieved_items = []
        if result and isinstance(result, dict):
            if 'retrieved_documents' in result:
                retrieved_items = result['retrieved_documents']
            elif 'retrieved_nodes' in result:
                retrieved_items = result['retrieved_nodes']
        
        success = (
            result is not None and
            isinstance(result, dict) and
            'query' in result and
            'answer' in result and
            len(retrieved_items) > 0
        )
        
        if success:
            print(f"✅ {technique_name} SUCCESS")
            print(f"   - Retrieved {len(retrieved_items)} documents/nodes")
            print(f"   - Execution time: {execution_time:.2f}s")
            print(f"   - Answer preview: {result['answer'][:100]}...")
        else:
            print(f"❌ {technique_name} FAILED - Invalid result structure")
            if result:
                print(f"   - Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
        return {
            'technique': technique_name,
            'success': success,
            'execution_time': execution_time,
            'num_documents': len(retrieved_items),
            'result': result if success else None,
            'error': None
        }
        
    except Exception as e:
        print(f"❌ {technique_name} FAILED - Exception: {str(e)}")
        logger.error(f"Error testing {technique_name}: {e}", exc_info=True)
        return {
            'technique': technique_name,
            'success': False,
            'execution_time': 0,
            'num_documents': 0,
            'result': None,
            'error': str(e)
        }

def main():
    """Test all 7 RAG techniques"""
    print("🚀 Testing All 7 RAG Techniques with V2 Pattern")
    print("=" * 80)
    
    test_query = "What are the symptoms of diabetes?"
    
    # Define all techniques to test
    techniques = [
        ("BasicRAG V2", "basic_rag.pipeline_v2", "BasicRAGPipelineV2"),
        ("CRAG V2", "crag.pipeline_v2", "CRAGPipelineV2"),
        ("NodeRAG V2", "noderag.pipeline_v2", "NodeRAGPipelineV2"),
        ("ColBERT V2", "colbert.pipeline_v2", "ColBERTPipelineV2"),
        ("HyDE V2", "hyde.pipeline_v2", "HyDERAGPipelineV2"),
        ("GraphRAG V2", "graphrag.pipeline_v2", "GraphRAGPipelineV2"),
        ("HybridIFindRAG V2", "hybrid_ifind_rag.pipeline_v2", "HybridIFindRAGPipelineV2"),
    ]
    
    results = []
    successful_techniques = []
    failed_techniques = []
    
    for technique_name, module_path, class_name in techniques:
        try:
            # Dynamic import
            module = __import__(module_path, fromlist=[class_name])
            pipeline_class = getattr(module, class_name)
            
            # Test the technique
            result = test_technique(technique_name, pipeline_class, test_query)
            results.append(result)
            
            if result['success']:
                successful_techniques.append(technique_name)
            else:
                failed_techniques.append(technique_name)
                
        except ImportError as e:
            print(f"❌ {technique_name} FAILED - Import Error: {str(e)}")
            failed_techniques.append(technique_name)
            results.append({
                'technique': technique_name,
                'success': False,
                'execution_time': 0,
                'num_documents': 0,
                'result': None,
                'error': f"Import Error: {str(e)}"
            })
        except Exception as e:
            print(f"❌ {technique_name} FAILED - Unexpected Error: {str(e)}")
            failed_techniques.append(technique_name)
            results.append({
                'technique': technique_name,
                'success': False,
                'execution_time': 0,
                'num_documents': 0,
                'result': None,
                'error': f"Unexpected Error: {str(e)}"
            })
    
    # Summary
    print(f"\n{'='*80}")
    print("🎯 FINAL RESULTS")
    print(f"{'='*80}")
    
    print(f"✅ Successful Techniques ({len(successful_techniques)}/7):")
    for technique in successful_techniques:
        print(f"   - {technique}")
    
    if failed_techniques:
        print(f"\n❌ Failed Techniques ({len(failed_techniques)}/7):")
        for technique in failed_techniques:
            print(f"   - {technique}")
    
    # Detailed results
    print(f"\n📊 Detailed Results:")
    print("-" * 80)
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"{result['technique']:<25} {status:<12} "
              f"Docs: {result['num_documents']:<3} "
              f"Time: {result['execution_time']:.2f}s")
        if result['error']:
            print(f"   Error: {result['error']}")
    
    # Overall success
    success_rate = len(successful_techniques) / 7 * 100
    print(f"\n🎉 Overall Success Rate: {success_rate:.1f}% ({len(successful_techniques)}/7)")
    
    if len(successful_techniques) == 7:
        print("\n🏆 ALL 7 RAG TECHNIQUES ARE WORKING! 🏆")
        print("Ready for comprehensive evaluation and scaling tests!")
    else:
        print(f"\n⚠️  Need to fix {len(failed_techniques)} more technique(s)")
    
    return results

if __name__ == "__main__":
    results = main()