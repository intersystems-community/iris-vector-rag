#!/usr/bin/env python3
"""
Test script to validate corrected imports for existing test files.
This demonstrates how to fix the existing test files to work with the refactored iris_rag package.
"""

def test_corrected_e2e_imports():
    """Test the corrected imports that should be used in the E2E test files."""
    print("Testing corrected imports for E2E test files...")
    
    try:
        # Corrected imports for test_e2e_iris_rag_integration.py
        from iris_rag.config.manager import ConfigurationManager as ConfigManager  # Alias for compatibility
        from iris_rag.core.connection import ConnectionManager as IRISConnectionManager  # Alias for compatibility
        from iris_rag.embeddings.manager import EmbeddingManager
        from iris_rag.storage.iris import IRISStorage as IRISVectorStorage  # Alias for compatibility
        from iris_rag.pipelines.basic import BasicRAGPipeline
        from iris_rag.core.models import Document
        
        print("✓ All corrected imports successful")
        
        # Test Document creation with corrected parameter name
        # OLD: Document(id='test', content='test content')
        # NEW: Document(id='test', page_content='test content')
        doc = Document(id='test', page_content='test content', metadata={"source": "test"})
        print(f"✓ Document created with corrected parameters: {doc.id}")
        
        # Test class instantiation with aliases
        config_manager = ConfigManager()
        print("✓ ConfigManager (ConfigurationManager) instantiated")
        
        # Test that aliases work as expected
        assert ConfigManager == ConfigurationManager
        assert IRISConnectionManager == ConnectionManager
        assert IRISVectorStorage == IRISStorage
        print("✓ All class aliases work correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Corrected imports failed: {e}")
        return False

def test_mock_llm_function():
    """Test the mock LLM function used in tests."""
    def mock_llm_func_integration(query: str, context: str) -> str:
        """A simple mock LLM function for integration testing, returns context for inspection."""
        if not context:
            return f"NO_CONTEXT_PROVIDED_FOR_QUERY:{query}"
        return f"CONTEXT_FOR_QUERY:{query}|CONTEXT_START:{context}CONTEXT_END"
    
    # Test the mock function
    result1 = mock_llm_func_integration("test query", "test context")
    assert "CONTEXT_FOR_QUERY:test query" in result1
    assert "CONTEXT_START:test context" in result1
    print("✓ Mock LLM function works correctly")
    
    result2 = mock_llm_func_integration("test query", "")
    assert "NO_CONTEXT_PROVIDED_FOR_QUERY:test query" in result2
    print("✓ Mock LLM function handles empty context correctly")
    
    return True

def test_sample_documents_loading():
    """Test loading sample documents as done in the E2E tests."""
    import os
    import glob
    import xml.etree.ElementTree as ET
    from iris_rag.core.models import Document
    
    try:
        docs = []
        sample_docs_path = os.path.join("data", "sample_10_docs", "*.xml")
        
        for filepath in glob.glob(sample_docs_path):
            doc_id = os.path.basename(filepath).replace(".xml", "")
            try:
                tree = ET.parse(filepath)
                root = tree.getroot()
                content_parts = [elem.text for elem in root.iter() if elem.text]
                content = "\n".join(filter(None, content_parts)).strip()
                
                if not content:
                    content = "Placeholder content for " + doc_id
                
                # Use corrected Document constructor
                docs.append(Document(id=doc_id, page_content=content, metadata={"source": filepath}))
            except Exception as e:
                print(f"Warning: Could not load document {filepath}: {e}")
        
        print(f"✓ Successfully loaded {len(docs)} sample documents")
        
        if docs:
            # Test first document
            first_doc = docs[0]
            print(f"✓ First document: ID={first_doc.id}, content_length={len(first_doc.page_content)}")
            assert hasattr(first_doc, 'page_content')
            assert hasattr(first_doc, 'metadata')
            assert hasattr(first_doc, 'id')
        
        return len(docs) > 0
        
    except Exception as e:
        print(f"✗ Sample document loading failed: {e}")
        return False

def main():
    """Run all corrected import tests."""
    print("=== TESTING CORRECTED IMPORTS FOR EXISTING TEST FILES ===\n")
    
    tests = [
        ("Corrected E2E Imports", test_corrected_e2e_imports),
        ("Mock LLM Function", test_mock_llm_function),
        ("Sample Documents Loading", test_sample_documents_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            result = test_func()
            results.append(result)
            status = "PASSED" if result else "FAILED"
            print(f"{status}: {test_name}\n")
        except Exception as e:
            print(f"FAILED: {test_name} - {e}\n")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"=== SUMMARY ===")
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - The existing test files can be fixed with these corrections")
        print("\nRequired changes for existing test files:")
        print("1. Update imports:")
        print("   - ConfigurationManager as ConfigManager")
        print("   - ConnectionManager as IRISConnectionManager") 
        print("   - IRISStorage as IRISVectorStorage")
        print("2. Update Document constructor:")
        print("   - Document(page_content=...) instead of Document(content=...)")
        print("3. All other functionality should work as expected")
    else:
        print(f"\n✗ {total-passed} tests failed - Additional fixes needed")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)