#!/usr/bin/env python3
"""
Quick focused test to fix the 3 remaining failing RAG techniques.
Minimal setup, fast execution, clear error identification.
"""

import pytest
import logging
from typing import Dict, Any

# Configure minimal logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_colbert_connection_fix():
    """Test ColBERT connection manager fix."""
    logger.info("Testing ColBERT connection manager fix...")
    
    try:
        from iris_rag.validation.factory import ValidatedPipelineFactory
        from iris_rag.config.manager import ConfigurationManager
        from common.iris_connection_manager import get_iris_connection
        
        # Setup minimal components
        config_manager = ConfigurationManager()
        iris_connector = get_iris_connection()
        
        # Create connection manager wrapper
        connection_manager = type('ConnectionManager', (), {
            'get_connection': lambda: iris_connector
        })()
        
        # Test pipeline creation
        factory = ValidatedPipelineFactory(
            config_manager=config_manager,
            connection_manager=connection_manager
        )
        
        pipeline = factory.create_pipeline('colbert')
        logger.info("✓ ColBERT pipeline created successfully")
        
        # Test single query
        result = pipeline.query("What are the effects of BRCA1 mutations?")
        logger.info(f"✓ ColBERT query result: {type(result)}")
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'query' in result, "Result should contain query"
        
        logger.info("✓ ColBERT test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"✗ ColBERT test FAILED: {e}")
        return False

def test_noderag_retrieval_fix():
    """Test NodeRAG retrieval issue."""
    logger.info("Testing NodeRAG retrieval fix...")
    
    try:
        from iris_rag.validation.factory import ValidatedPipelineFactory
        from iris_rag.config.manager import ConfigurationManager
        from common.iris_connection_manager import get_iris_connection
        
        # Setup minimal components
        config_manager = ConfigurationManager()
        iris_connector = get_iris_connection()
        
        connection_manager = type('ConnectionManager', (), {
            'get_connection': lambda: iris_connector
        })()
        
        # Test pipeline creation
        factory = ValidatedPipelineFactory(
            config_manager=config_manager,
            connection_manager=connection_manager
        )
        
        pipeline = factory.create_pipeline('noderag')
        logger.info("✓ NodeRAG pipeline created successfully")
        
        # Test single query
        result = pipeline.query("What are the effects of BRCA1 mutations?")
        logger.info(f"✓ NodeRAG query result: {type(result)}")
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'retrieved_documents' in result, "Result should contain retrieved_documents"
        
        retrieved_docs = result.get('retrieved_documents', [])
        logger.info(f"✓ NodeRAG retrieved {len(retrieved_docs)} documents")
        
        if len(retrieved_docs) == 0:
            logger.warning("⚠️ NodeRAG retrieved 0 documents - this is the issue to fix")
            return False
        
        logger.info("✓ NodeRAG test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"✗ NodeRAG test FAILED: {e}")
        return False

def test_hybrid_ifind_document_fix():
    """Test HybridIFind Document.content issue."""
    logger.info("Testing HybridIFind Document.content fix...")
    
    try:
        from iris_rag.validation.factory import ValidatedPipelineFactory
        from iris_rag.config.manager import ConfigurationManager
        from common.iris_connection_manager import get_iris_connection
        
        # Setup minimal components
        config_manager = ConfigurationManager()
        iris_connector = get_iris_connection()
        
        connection_manager = type('ConnectionManager', (), {
            'get_connection': lambda: iris_connector
        })()
        
        # Test pipeline creation
        factory = ValidatedPipelineFactory(
            config_manager=config_manager,
            connection_manager=connection_manager
        )
        
        pipeline = factory.create_pipeline('hybrid_ifind')
        logger.info("✓ HybridIFind pipeline created successfully")
        
        # Test single query
        result = pipeline.query("What are the effects of BRCA1 mutations?")
        logger.info(f"✓ HybridIFind query result: {type(result)}")
        
        assert isinstance(result, dict), "Result should be a dictionary"
        assert 'retrieved_documents' in result, "Result should contain retrieved_documents"
        
        # Check for Document.content error
        if 'error' in result and 'Document' in str(result['error']) and 'content' in str(result['error']):
            logger.warning("⚠️ HybridIFind has Document.content attribute error - this is the issue to fix")
            return False
        
        logger.info("✓ HybridIFind test PASSED")
        return True
        
    except Exception as e:
        logger.error(f"✗ HybridIFind test FAILED: {e}")
        return False

def test_all_failing_techniques():
    """Run all failing technique tests."""
    logger.info("=" * 60)
    logger.info("QUICK FIX TEST FOR 3 FAILING RAG TECHNIQUES")
    logger.info("=" * 60)
    
    results = {
        'colbert': test_colbert_connection_fix(),
        'noderag': test_noderag_retrieval_fix(),
        'hybrid_ifind': test_hybrid_ifind_document_fix()
    }
    
    logger.info("=" * 60)
    logger.info("QUICK FIX TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for technique, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{technique:15} {status}")
    
    logger.info(f"Success Rate: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TECHNIQUES FIXED!")
    else:
        logger.info("🔧 Some techniques still need fixes")
    
    return results

if __name__ == "__main__":
    test_all_failing_techniques()