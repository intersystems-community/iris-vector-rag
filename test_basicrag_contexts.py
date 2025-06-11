#!/usr/bin/env python3
"""
Quick test to verify BasicRAG now provides proper contexts for RAGAS evaluation.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

import logging
from basic_rag.pipeline import BasicRAGPipeline
from common.iris_connector_jdbc import get_iris_connection
from common.utils import get_embedding_func, get_llm_func

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basicrag_contexts():
    """Test BasicRAG to ensure it returns proper contexts."""
    logger.info("=== Testing BasicRAG Context Extraction ===")
    
    try:
        # Setup pipeline
        db_conn = get_iris_connection()
        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub")  # Use stub for testing
        
        pipeline = BasicRAGPipeline(
            iris_connector=db_conn,
            embedding_func=embed_fn,
            llm_func=llm_fn
        )
        
        # Test query
        test_query = "What are the effects of metformin on type 2 diabetes?"
        logger.info(f"Running BasicRAG with query: '{test_query}'")
        
        # Run pipeline
        result = pipeline.run(test_query, top_k=3)
        
        # Check result structure
        logger.info("\n=== BasicRAG Result Analysis ===")
        logger.info(f"Query: {result.get('query', 'MISSING')}")
        logger.info(f"Answer length: {len(result.get('answer', ''))}")
        logger.info(f"Retrieved documents count: {result.get('document_count', 0)}")
        
        # Check contexts field
        contexts = result.get('contexts', [])
        logger.info(f"Contexts field present: {'contexts' in result}")
        logger.info(f"Contexts type: {type(contexts)}")
        logger.info(f"Contexts count: {len(contexts)}")
        
        # Analyze each context
        if contexts:
            logger.info("\n=== Context Analysis ===")
            for i, context in enumerate(contexts):
                logger.info(f"Context {i+1}:")
                logger.info(f"  Type: {type(context)}")
                logger.info(f"  Length: {len(context) if isinstance(context, str) else 'N/A'}")
                logger.info(f"  Is string: {isinstance(context, str)}")
                logger.info(f"  Is empty: {not context.strip() if isinstance(context, str) else 'N/A'}")
                
                # Show preview of content
                if isinstance(context, str) and context.strip():
                    preview = context.strip()[:100] + "..." if len(context.strip()) > 100 else context.strip()
                    logger.info(f"  Preview: '{preview}'")
                else:
                    logger.info(f"  Content: {repr(context)}")
        else:
            logger.warning("No contexts found!")
        
        # Check for common issues
        logger.info("\n=== Issue Detection ===")
        issues_found = []
        
        if 'contexts' not in result:
            issues_found.append("Missing 'contexts' field")
        
        if not contexts:
            issues_found.append("Empty contexts list")
        
        for i, context in enumerate(contexts):
            if not isinstance(context, str):
                issues_found.append(f"Context {i+1} is not a string: {type(context)}")
            elif not context.strip():
                issues_found.append(f"Context {i+1} is empty or whitespace")
            elif context.strip().isdigit():
                issues_found.append(f"Context {i+1} appears to be a numeric ID: '{context}'")
            elif "IRISInputStream" in str(context):
                issues_found.append(f"Context {i+1} contains IRISInputStream object")
        
        if issues_found:
            logger.error("Issues found:")
            for issue in issues_found:
                logger.error(f"  - {issue}")
            return False
        else:
            logger.info("✅ All context checks passed!")
            return True
            
    except Exception as e:
        logger.error(f"Error during BasicRAG test: {e}", exc_info=True)
        return False
    finally:
        if 'db_conn' in locals() and db_conn:
            db_conn.close()

if __name__ == "__main__":
    success = test_basicrag_contexts()
    if success:
        print("\n🎉 BasicRAG context extraction test PASSED!")
        print("The pipeline should now work properly with RAGAS evaluation.")
    else:
        print("\n❌ BasicRAG context extraction test FAILED!")
        print("Additional fixes may be needed.")
    
    sys.exit(0 if success else 1)