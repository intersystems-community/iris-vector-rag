#!/usr/bin/env python3
"""
Test to verify BasicRAG pipeline works after the TypeError fix.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import iris_rag

def test_basicrag_pipeline():
    """Test that BasicRAG pipeline works without TypeError."""
    print("🔍 Testing BasicRAG pipeline after fix...")
    
    try:
        # Create BasicRAG pipeline
        pipeline = iris_rag.create_pipeline(
            pipeline_type="basic",
            validate=False  # Skip validation to focus on the TypeError fix
        )
        print("✅ BasicRAG pipeline created successfully!")
        
        # Test a simple query
        test_query = "What is machine learning?"
        result = pipeline.query(test_query)
        
        print("✅ BasicRAG pipeline query completed successfully!")
        print(f"   Query: {test_query}")
        print(f"   Result type: {type(result)}")
        if isinstance(result, dict):
            print(f"   Answer length: {len(result.get('answer', ''))}")
            print(f"   Retrieved documents: {len(result.get('retrieved_documents', []))}")
        elif isinstance(result, list):
            print(f"   Result list length: {len(result)}")
        else:
            print(f"   Result: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ BasicRAG pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basicrag_pipeline()
    sys.exit(0 if success else 1)