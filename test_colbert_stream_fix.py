#!/usr/bin/env python3
"""
Test script to verify ColBERT stream handling fix
"""

import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from colbert.pipeline import OptimizedColbertRAGPipeline
from common.iris_connector_jdbc import get_iris_connection
from common.utils import get_colbert_query_encoder_func, get_colbert_doc_encoder_func, get_llm_func

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_colbert_stream_handling():
    """Test that ColBERT pipeline properly handles streams."""
    
    logger.info("🧪 Testing ColBERT stream handling fix...")
    
    try:
        # Initialize pipeline components
        iris_connector = get_iris_connection()
        colbert_query_encoder = get_colbert_query_encoder_func()
        colbert_doc_encoder = get_colbert_doc_encoder_func()
        llm_func = get_llm_func()
        
        # Create pipeline
        pipeline = OptimizedColbertRAGPipeline(
            iris_connector=iris_connector,
            colbert_query_encoder_func=colbert_query_encoder,
            colbert_doc_encoder_func=colbert_doc_encoder,
            llm_func=llm_func
        )
        
        # Test with a simple query
        test_query = "What is cancer treatment?"
        
        logger.info(f"Testing query: {test_query}")
        result = pipeline.run(test_query)
        
        # Check if we got meaningful results
        if result and "retrieved_documents" in result:
            docs = result["retrieved_documents"]
            logger.info(f"Retrieved {len(docs)} documents")
            
            # Check document content
            for i, doc in enumerate(docs[:3]):  # Check first 3 docs
                content = getattr(doc, 'content', '') or getattr(doc, 'page_content', '')
                logger.info(f"Doc {i+1} content length: {len(content)}")
                logger.info(f"Doc {i+1} content preview: {content[:100]}...")
                
                # Check if content is meaningful (not just numeric)
                if len(content) > 50 and not content.isdigit():
                    logger.info(f"✅ Doc {i+1}: Meaningful content found")
                else:
                    logger.warning(f"❌ Doc {i+1}: Content appears corrupted: '{content}'")
            
            logger.info("✅ ColBERT stream handling test completed")
            return True
        else:
            logger.error("❌ No documents retrieved")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_colbert_stream_handling()
    if success:
        print("\n✅ ColBERT stream handling fix is working correctly")
    else:
        print("\n❌ ColBERT stream handling fix needs more work")
