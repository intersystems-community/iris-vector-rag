#!/usr/bin/env python3
"""
Script to test the fixed pipeline implementations.

This script tests all the pipelines that were fixed to ensure they now have
the required abstract methods implemented.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_pipeline_abstract_methods():
    """Test that all pipelines implement the required abstract methods."""
    
    logger.info("=== TESTING FIXED PIPELINE IMPLEMENTATIONS ===")
    
    # Import the pipeline classes
    try:
        from iris_rag.pipelines.crag import CRAGPipeline
        from iris_rag.pipelines.hyde import HyDERAGPipeline
        from iris_rag.pipelines.graphrag import GraphRAGPipeline
        from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline
        from iris_rag.pipelines.basic import BasicRAGPipeline
        from iris_rag.pipelines.noderag import NodeRAGPipeline
        
        logger.info("✓ All pipeline classes imported successfully")
        
    except ImportError as e:
        logger.error(f"Failed to import pipeline classes: {e}")
        return False
    
    # Test each pipeline class for required methods
    pipelines_to_test = [
        ("CRAG", CRAGPipeline),
        ("HyDE", HyDERAGPipeline),
        ("GraphRAG", GraphRAGPipeline),
        ("Hybrid IFind", HybridIFindRAGPipeline),
        ("Basic RAG", BasicRAGPipeline),
        ("NodeRAG", NodeRAGPipeline)
    ]
    
    required_methods = ["execute", "load_documents", "query"]
    
    all_passed = True
    
    for pipeline_name, pipeline_class in pipelines_to_test:
        logger.info(f"\nTesting {pipeline_name} pipeline...")
        
        # Check if all required methods exist
        missing_methods = []
        for method_name in required_methods:
            if not hasattr(pipeline_class, method_name):
                missing_methods.append(method_name)
        
        if missing_methods:
            logger.error(f"✗ {pipeline_name} missing methods: {missing_methods}")
            all_passed = False
        else:
            logger.info(f"✓ {pipeline_name} has all required methods: {required_methods}")
            
            # Check if methods are callable
            for method_name in required_methods:
                method = getattr(pipeline_class, method_name)
                if not callable(method):
                    logger.error(f"✗ {pipeline_name}.{method_name} is not callable")
                    all_passed = False
                else:
                    logger.info(f"  ✓ {method_name} is callable")
    
    return all_passed

def test_pipeline_instantiation():
    """Test that pipelines can be instantiated with mock components."""
    
    logger.info("\n=== TESTING PIPELINE INSTANTIATION ===")
    
    # Create mock components
    try:
        from iris_rag.core.connection import ConnectionManager
        from iris_rag.config.manager import ConfigurationManager
        
        # Mock connection manager
        class MockConnectionManager:
            def get_connection(self):
                return None
        
        # Mock config manager
        class MockConfigManager:
            def get(self, key, default=None):
                return default
        
        connection_manager = MockConnectionManager()
        config_manager = MockConfigManager()
        
        logger.info("✓ Mock components created")
        
    except Exception as e:
        logger.error(f"Failed to create mock components: {e}")
        return False
    
    # Test instantiation of each pipeline
    pipelines_to_test = [
        ("CRAG", "iris_rag.pipelines.crag", "CRAGPipeline"),
        ("HyDE", "iris_rag.pipelines.hyde", "HyDERAGPipeline"),
        ("GraphRAG", "iris_rag.pipelines.graphrag", "GraphRAGPipeline"),
        ("Hybrid IFind", "iris_rag.pipelines.hybrid_ifind", "HybridIFindRAGPipeline"),
        ("Basic RAG", "iris_rag.pipelines.basic", "BasicRAGPipeline")
    ]
    
    all_passed = True
    
    for pipeline_name, module_name, class_name in pipelines_to_test:
        try:
            # Import the pipeline class
            module = __import__(module_name, fromlist=[class_name])
            pipeline_class = getattr(module, class_name)
            
            # Try to instantiate
            pipeline = pipeline_class(
                connection_manager=connection_manager,
                config_manager=config_manager,
                llm_func=lambda x: "Mock response"
            )
            
            logger.info(f"✓ {pipeline_name} instantiated successfully")
            
        except Exception as e:
            logger.error(f"✗ Failed to instantiate {pipeline_name}: {e}")
            all_passed = False
    
    return all_passed

def main():
    """Main function to run all tests."""
    
    logger.info("=== PIPELINE IMPLEMENTATION VALIDATION ===")
    
    # Test 1: Check abstract methods
    methods_test = test_pipeline_abstract_methods()
    
    # Test 2: Check instantiation
    instantiation_test = test_pipeline_instantiation()
    
    # Summary
    logger.info("\n=== TEST SUMMARY ===")
    
    if methods_test:
        logger.info("✓ All pipelines have required abstract methods")
    else:
        logger.error("✗ Some pipelines are missing required abstract methods")
    
    if instantiation_test:
        logger.info("✓ All pipelines can be instantiated")
    else:
        logger.error("✗ Some pipelines failed instantiation")
    
    overall_success = methods_test and instantiation_test
    
    if overall_success:
        logger.info("🎉 ALL TESTS PASSED - Pipelines are ready for auto-setup!")
        logger.info("\nNext steps:")
        logger.info("1. Run the auto-setup again to test all 7 pipelines")
        logger.info("2. All pipelines should now complete setup without abstract method errors")
        logger.info("3. Move from 2/7 working to 7/7 working pipelines")
    else:
        logger.error("❌ SOME TESTS FAILED - Additional fixes needed")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)