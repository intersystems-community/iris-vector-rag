#!/usr/bin/env python3
"""
Quick test to verify CRAG now provides proper contexts for RAGAS evaluation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import logging
from crag.pipeline import CRAGPipeline
from common.iris_connector_jdbc import get_iris_connection
from common.utils import get_embedding_func, get_llm_func

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_crag_contexts():
    """Test CRAG to ensure it returns proper contexts."""
    logger.info("=== Testing CRAG Context Extraction ===")
    
    try:
        # Setup pipeline
        db_conn = get_iris_connection()
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub")  # Use stub for testing
        
        pipeline = CRAGPipeline(
            iris_connector=db_conn,
            embedding_func=embed_fn,
            llm_func=llm_fn
        )
        
        # Test query
        test_query = "What are the cardiovascular benefits of SGLT2 inhibitors?"
        logger.info(f"Running CRAG with query: '{test_query}'")
        
        # Run pipeline
        result = pipeline.run(test_query, top_k=3)
        
        # Check contexts field
        contexts = result.get('contexts', [])
        logger.info(f"Contexts field present: {'contexts' in result}")
        logger.info(f"Contexts count: {len(contexts)}")
        
        # Check for IRISInputStream objects
        issues_found = []
        for i, context in enumerate(contexts):
            if not isinstance(context, str):
                issues_found.append(f"Context {i+1} is not a string: {type(context)}")
            elif "IRISInputStream" in str(context):
                issues_found.append(f"Context {i+1} contains IRISInputStream object")
            elif context.strip().isdigit():
                issues_found.append(f"Context {i+1} appears to be numeric ID: '{context}'")
            else:
                logger.info(f"✅ Context {i+1}: {len(context)} chars, preview: '{context[:50]}...'")
        
        if issues_found:
            logger.error("Issues found:")
            for issue in issues_found:
                logger.error(f"  - {issue}")
            return False
        else:
            logger.info("✅ All CRAG context checks passed!")
            return True
            
    except Exception as e:
        logger.error(f"Error during CRAG test: {e}", exc_info=True)
        return False
    finally:
        if 'db_conn' in locals() and db_conn:
            db_conn.close()

if __name__ == "__main__":
    success = test_crag_contexts()
    if success:
        print("\n🎉 CRAG context extraction test PASSED!")
    else:
        print("\n❌ CRAG context extraction test FAILED!")
    
    sys.exit(0 if success else 1)