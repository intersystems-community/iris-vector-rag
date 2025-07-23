#!/usr/bin/env python3
"""
Test Fixed Vector Pipelines
============================

Quick test to verify that the vector datatype fixes work correctly.
Tests the main pipelines with proper TO_VECTOR(?, DOUBLE) syntax.
"""

import os
import sys
import logging

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Assuming scripts is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func # Updated import

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_fixed_pipelines():
    """Test all fixed pipelines"""
    logger.info("🧪 Testing fixed vector pipelines")
    
    # Initialize functions
    embedding_func = get_embedding_func()
    llm_func = get_llm_func(provider="stub")
    test_query = "What is diabetes?"
    results = {}
    
    # Test BasicRAG
    logger.info("   🔬 Testing BasicRAG...")
    try:
        from iris_rag.pipelines.basic import BasicRAGPipeline # Updated import
        
        conn = get_iris_connection()
        pipeline = BasicRAGPipeline(
            iris_connector=conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        result = pipeline.run(test_query, top_k=5)
        results['BasicRAG'] = {
            'success': True,
            'docs_retrieved': result.get('document_count', 0),
            'error': None
        }
        logger.info(f"      ✅ BasicRAG: {result.get('document_count', 0)} docs retrieved")
        conn.close()
        
    except Exception as e:
        results['BasicRAG'] = {'success': False, 'docs_retrieved': 0, 'error': str(e)}
        logger.error(f"      ❌ BasicRAG failed: {e}")
    
    # Test HyDE
    logger.info("   🔬 Testing HyDE...")
    try:
        from iris_rag.pipelines.hyde import HyDERAGPipeline # Updated import
        
        conn = get_iris_connection()
        pipeline = HyDERAGPipeline(
            iris_connector=conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        result = pipeline.run(test_query, top_k=5)
        results['HyDE'] = {
            'success': True,
            'docs_retrieved': result.get('document_count', 0),
            'error': None
        }
        logger.info(f"      ✅ HyDE: {result.get('document_count', 0)} docs retrieved")
        conn.close()
        
    except Exception as e:
        results['HyDE'] = {'success': False, 'docs_retrieved': 0, 'error': str(e)}
        logger.error(f"      ❌ HyDE failed: {e}")
    
    # Test HybridiFindRAG
    logger.info("   🔬 Testing HybridiFindRAG...")
    try:
        from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline # Updated import
        
        conn = get_iris_connection()
        pipeline = HybridIFindRAGPipeline(
            iris_connector=conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        result = pipeline.query(test_query)
        results['HybridiFindRAG'] = {
            'success': True,
            'docs_retrieved': len(result.get('retrieved_documents', [])),
            'error': None
        }
        logger.info(f"      ✅ HybridiFindRAG: {len(result.get('retrieved_documents', []))} docs retrieved")
        conn.close()
        
    except Exception as e:
        results['HybridiFindRAG'] = {'success': False, 'docs_retrieved': 0, 'error': str(e)}
        logger.error(f"      ❌ HybridiFindRAG failed: {e}")
    
    # Test CRAG with correct class name
    logger.info("   🔬 Testing CRAG...")
    try:
        from iris_rag.pipelines.crag import CRAGPipeline as CRAGPipelineV2 # Updated import
        
        conn = get_iris_connection()
        pipeline = CRAGPipelineV2(
            iris_connector=conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        result = pipeline.run(test_query, top_k=5)
        results['CRAG'] = {
            'success': True,
            'docs_retrieved': result.get('document_count', 0),
            'error': None
        }
        logger.info(f"      ✅ CRAG: {result.get('document_count', 0)} docs retrieved")
        conn.close()
        
    except Exception as e:
        results['CRAG'] = {'success': False, 'docs_retrieved': 0, 'error': str(e)}
        logger.error(f"      ❌ CRAG failed: {e}")
    
    # Test NodeRAG with correct class name
    logger.info("   🔬 Testing NodeRAG...")
    try:
        from iris_rag.pipelines.noderag import NodeRAGPipeline as NodeRAGPipelineV2 # Updated import
        
        conn = get_iris_connection()
        pipeline = NodeRAGPipelineV2(
            iris_connector=conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        
        result = pipeline.run(test_query, top_k=5)
        results['NodeRAG'] = {
            'success': True,
            'docs_retrieved': result.get('document_count', 0),
            'error': None
        }
        logger.info(f"      ✅ NodeRAG: {result.get('document_count', 0)} docs retrieved")
        conn.close()
        
    except Exception as e:
        results['NodeRAG'] = {'success': False, 'docs_retrieved': 0, 'error': str(e)}
        logger.error(f"      ❌ NodeRAG failed: {e}")
    
    # Summary
    successful_pipelines = [name for name, result in results.items() if result['success']]
    failed_pipelines = [name for name, result in results.items() if not result['success']]
    
    logger.info(f"✅ TESTING COMPLETE: {len(successful_pipelines)}/{len(results)} pipelines working")
    logger.info(f"   ✅ Working: {', '.join(successful_pipelines)}")
    if failed_pipelines:
        logger.info(f"   ❌ Failed: {', '.join(failed_pipelines)}")
    
    # Check if we have documents being retrieved
    docs_retrieved = sum(result['docs_retrieved'] for result in results.values() if result['success'])
    if docs_retrieved > 0:
        logger.info(f"🎉 SUCCESS: {docs_retrieved} total documents retrieved across all pipelines!")
        return True
    else:
        logger.error("❌ FAILURE: No documents retrieved by any pipeline")
        return False

if __name__ == "__main__":
    success = test_fixed_pipelines()
    sys.exit(0 if success else 1)