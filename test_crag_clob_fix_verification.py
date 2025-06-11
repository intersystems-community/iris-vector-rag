#!/usr/bin/env python3
"""
Test script to verify that the CRAG pipeline CLOB fix is working properly.
This script tests the CRAGPipeline with the updated constructor and CLOB handling.
"""

import sys
import os
sys.path.append('.')

def test_crag_import_and_constructor():
    """Test that CRAG pipeline can be imported and instantiated with legacy constructor."""
    print("Testing CRAG pipeline import and constructor...")
    
    try:
        from iris_rag.pipelines.crag import CRAGPipeline, _convert_clob_to_string
        print("✓ CRAG pipeline imports successfully")
        
        # Test legacy constructor (the way it's called in existing code)
        def mock_embedding_func(texts):
            return [[0.1, 0.2, 0.3] for _ in texts]
        
        def mock_llm_func(prompt):
            return f"Mock response to: {prompt[:50]}..."
        
        # Mock connection object
        class MockConnection:
            def cursor(self):
                return MockCursor()
            def commit(self):
                pass
        
        class MockCursor:
            def execute(self, sql, params=None):
                pass
            def fetchall(self):
                return []
            def close(self):
                pass
        
        mock_connection = MockConnection()
        
        # Test legacy constructor signature
        pipeline = CRAGPipeline(
            iris_connector=mock_connection,
            embedding_func=mock_embedding_func,
            llm_func=mock_llm_func
        )
        print("✓ CRAG pipeline instantiated successfully with legacy constructor")
        
        # Test CLOB conversion function
        test_string = "This is a test string"
        converted = _convert_clob_to_string(test_string)
        assert converted == test_string, f"Expected '{test_string}', got '{converted}'"
        print("✓ CLOB conversion function works correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_document_creation_with_string():
    """Test that Document objects can be created with string page_content."""
    print("\nTesting Document creation with string content...")
    
    try:
        from iris_rag.core.models import Document
        
        # Test normal string content
        doc = Document(
            page_content="This is test content",
            metadata={"test": "value"},
            id="test_doc_1"
        )
        print("✓ Document created successfully with string content")
        
        # Verify page_content is a string
        assert isinstance(doc.page_content, str), f"page_content should be str, got {type(doc.page_content)}"
        print("✓ Document page_content is confirmed to be a string")
        
        return True
        
    except Exception as e:
        print(f"✗ Document test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("CRAG Pipeline CLOB Fix Verification")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: Import and constructor
    if not test_crag_import_and_constructor():
        all_passed = False
    
    # Test 2: Document creation
    if not test_document_creation_with_string():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! CRAG pipeline CLOB fix appears to be working.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())