#!/usr/bin/env python3
"""
Test script to verify the CRAG pipeline fix for ConfigurationManager callable error.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_crag_initialization():
    """Test that CRAG pipeline can be initialized without the ConfigurationManager error."""
    try:
        # Import required components
        from iris_rag.core.connection import ConnectionManager
        from iris_rag.config.manager import ConfigurationManager
        from iris_rag.pipelines.crag import CRAGPipeline
        
        print("✅ Successfully imported required modules")
        
        # Initialize managers
        connection_manager = ConnectionManager()
        config_manager = ConfigurationManager()
        
        print("✅ Successfully created connection and config managers")
        
        # Initialize CRAG pipeline with new architecture
        crag_pipeline = CRAGPipeline(connection_manager, config_manager)
        
        print("✅ Successfully initialized CRAG pipeline")
        
        # Test that the pipeline has the expected attributes
        assert hasattr(crag_pipeline, 'connection_manager'), "Missing connection_manager attribute"
        assert hasattr(crag_pipeline, 'config_manager'), "Missing config_manager attribute"
        assert hasattr(crag_pipeline, 'embedding_func'), "Missing embedding_func attribute"
        assert hasattr(crag_pipeline, 'llm_func'), "Missing llm_func attribute"
        assert hasattr(crag_pipeline, 'evaluator'), "Missing evaluator attribute"
        
        print("✅ All expected attributes are present")
        
        # Test that config_manager is not being called as a function
        assert callable(crag_pipeline.config_manager.get), "config_manager.get should be callable"
        
        print("✅ ConfigurationManager has proper method interface")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_crag_legacy_initialization():
    """Test that CRAG pipeline still works with legacy initialization."""
    try:
        from iris_rag.pipelines.crag import CRAGPipeline
        
        # Test legacy initialization (should still work)
        crag_pipeline = CRAGPipeline()
        
        print("✅ Successfully initialized CRAG pipeline with legacy parameters")
        
        return True
        
    except Exception as e:
        print(f"❌ Legacy test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing CRAG pipeline ConfigurationManager fix...")
    print("=" * 60)
    
    # Test new architecture initialization
    print("\n1. Testing new architecture initialization:")
    success1 = test_crag_initialization()
    
    # Test legacy initialization
    print("\n2. Testing legacy initialization:")
    success2 = test_crag_legacy_initialization()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All tests passed! The ConfigurationManager callable error has been fixed.")
        sys.exit(0)
    else:
        print("💥 Some tests failed. Please check the error messages above.")
        sys.exit(1)