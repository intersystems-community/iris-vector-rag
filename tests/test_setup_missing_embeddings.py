#!/usr/bin/env python3
"""
Setup script to generate missing embeddings for failing RAG techniques.
This addresses the validation failures by ensuring proper database setup.
"""

import pytest
import logging
from typing import Dict, Any

# Configure minimal logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_setup_missing_embeddings_for_failing_techniques():
    """Setup missing embeddings for ColBERT, NodeRAG, and HybridIFind."""
    logger.info("=" * 60)
    logger.info("SETTING UP MISSING EMBEDDINGS FOR FAILING TECHNIQUES")
    logger.info("=" * 60)
    
    try:
        from iris_rag.validation.orchestrator import SetupOrchestrator
        from iris_rag.config.manager import ConfigurationManager
        from common.iris_connection_manager import get_iris_connection
        
        # Setup components
        config_manager = ConfigurationManager()
        iris_connector = get_iris_connection()
        
        connection_manager = type('ConnectionManager', (), {
            'get_connection': lambda: iris_connector
        })()
        
        # Create setup orchestrator
        setup_orchestrator = SetupOrchestrator(
            config_manager=config_manager,
            connection_manager=connection_manager
        )
        
        # Setup for each failing technique
        failing_techniques = ['colbert', 'noderag', 'hybrid_ifind']
        
        for technique in failing_techniques:
            logger.info(f"Setting up {technique}...")
            
            try:
                # Generate missing embeddings
                setup_orchestrator.generate_missing_embeddings(technique)
                logger.info(f"✓ {technique} setup completed")
                
            except Exception as e:
                logger.error(f"✗ {technique} setup failed: {e}")
                # Continue with other techniques
                continue
        
        logger.info("=" * 60)
        logger.info("SETUP COMPLETED - Now testing techniques...")
        logger.info("=" * 60)
        
        # Now test if the techniques work
        return test_techniques_after_setup()
        
    except Exception as e:
        logger.error(f"Setup orchestrator failed: {e}")
        return False

def test_techniques_after_setup():
    """Test the techniques after setup to verify they work."""
    from iris_rag.validation.factory import ValidatedPipelineFactory
    from iris_rag.config.manager import ConfigurationManager
    from common.iris_connection_manager import get_iris_connection
    
    # Setup components
    config_manager = ConfigurationManager()
    iris_connector = get_iris_connection()
    
    connection_manager = type('ConnectionManager', (), {
        'get_connection': lambda *args, **kwargs: iris_connector
    })()
    
    factory = ValidatedPipelineFactory(
        config_manager=config_manager,
        connection_manager=connection_manager
    )
    
    results = {}
    techniques = ['colbert', 'noderag', 'hybrid_ifind']
    
    for technique in techniques:
        logger.info(f"Testing {technique} after setup...")
        
        try:
            # Create pipeline (should pass validation now)
            pipeline = factory.create_pipeline(technique)
            logger.info(f"✓ {technique} pipeline created successfully")
            
            # Test single query
            result = pipeline.query("What are the effects of BRCA1 mutations?")
            logger.info(f"✓ {technique} query executed successfully")
            
            # Basic validation
            assert isinstance(result, dict), f"{technique}: Result should be a dictionary"
            assert 'query' in result, f"{technique}: Result should contain query"
            
            results[technique] = True
            logger.info(f"✓ {technique} test PASSED")
            
        except Exception as e:
            logger.error(f"✗ {technique} test FAILED: {e}")
            results[technique] = False
    
    # Summary
    logger.info("=" * 60)
    logger.info("POST-SETUP TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for technique, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{technique:15} {status}")
    
    logger.info(f"Success Rate: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 ALL TECHNIQUES WORKING AFTER SETUP!")
    else:
        logger.info("🔧 Some techniques still need fixes")
    
    return results

if __name__ == "__main__":
    test_setup_missing_embeddings_for_failing_techniques()