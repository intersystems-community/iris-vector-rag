"""
Test to reproduce the document format inconsistency bug in rag-templates library.

This test demonstrates the bug where:
1. rag_templates/simple.py and standard.py create dictionaries with "page_content" keys
2. iris_rag/storage/vector_store_iris.py expects Document objects with .page_content attributes
3. The pipeline doesn't convert dictionaries to Document objects before passing to vector store

Expected failure: AttributeError: 'dict' object has no attribute 'page_content'
"""

import pytest
from unittest.mock import Mock, patch
from rag_templates.simple import RAG
from rag_templates.standard import ConfigurableRAG


class TestDocumentFormatInconsistencyBug:
    """
    Test class to reproduce and document the document format inconsistency bug.
    
    The bug occurs when users pass documents to rag.add_documents() and the
    pipeline creates dictionaries with "page_content" keys, but the vector store
    expects Document objects with .page_content attributes.
    """
    
    def test_simple_rag_document_format_bug_with_string_documents(self):
        """
        Test that demonstrates the bug in simple.py RAG class with string documents.
        
        This test should FAIL with AttributeError: 'dict' object has no attribute 'page_content'
        when the pipeline tries to pass dictionary documents to the vector store.
        """
        # Arrange: Create RAG instance and test documents
        rag = RAG()
        test_documents = [
            "This is the first test document about machine learning.",
            "This is the second test document about artificial intelligence."
        ]
        
        # Act & Assert: This should now pass without AttributeError
        try:
            rag.add_documents(test_documents)
        except AttributeError:
            pytest.fail("AttributeError was raised unexpectedly after the fix.")
    
    def test_simple_rag_document_format_bug_with_dict_documents(self):
        """
        Test that demonstrates the bug in simple.py RAG class with dictionary documents.
        
        This test should FAIL with AttributeError: 'dict' object has no attribute 'page_content'
        when the pipeline tries to pass dictionary documents to the vector store.
        """
        # Arrange: Create RAG instance and test documents as dictionaries
        rag = RAG()
        test_documents = [
            {
                "page_content": "This is the first test document about machine learning.",
                "metadata": {"source": "test1.txt", "topic": "ML"}
            },
            {
                "page_content": "This is the second test document about artificial intelligence.",
                "metadata": {"source": "test2.txt", "topic": "AI"}
            }
        ]
        
        # Act & Assert: This should now pass without AttributeError
        try:
            rag.add_documents(test_documents)
        except AttributeError:
            pytest.fail("AttributeError was raised unexpectedly after the fix.")
    
    def test_configurable_rag_document_format_bug_with_string_documents(self):
        """
        Test that demonstrates the bug in standard.py ConfigurableRAG class with string documents.
        
        This test should FAIL with AttributeError: 'dict' object has no attribute 'page_content'
        when the pipeline tries to pass dictionary documents to the vector store.
        """
        # Arrange: Create ConfigurableRAG instance and test documents
        config = {"technique": "basic"}
        rag = ConfigurableRAG(config)
        test_documents = [
            "This is the first test document about machine learning.",
            "This is the second test document about artificial intelligence."
        ]
        
        # Act & Assert: This should now pass without AttributeError
        try:
            rag.add_documents(test_documents)
        except AttributeError:
            pytest.fail("AttributeError was raised unexpectedly after the fix.")
    
    def test_configurable_rag_document_format_bug_with_dict_documents(self):
        """
        Test that demonstrates the bug in standard.py ConfigurableRAG class with dictionary documents.
        
        This test should FAIL with AttributeError: 'dict' object has no attribute 'page_content'
        when the pipeline tries to pass dictionary documents to the vector store.
        """
        # Arrange: Create ConfigurableRAG instance and test documents as dictionaries
        config = {"technique": "basic"}
        rag = ConfigurableRAG(config)
        test_documents = [
            {
                "page_content": "This is the first test document about machine learning.",
                "metadata": {"source": "test1.txt", "topic": "ML"}
            },
            {
                "page_content": "This is the second test document about artificial intelligence.",
                "metadata": {"source": "test2.txt", "topic": "AI"}
            }
        ]
        
        # Act & Assert: This should now pass without AttributeError
        try:
            rag.add_documents(test_documents)
        except AttributeError:
            pytest.fail("AttributeError was raised unexpectedly after the fix.")
    
    def test_document_processing_creates_correct_format(self):
        """
        Test that verifies the _process_documents methods create dictionaries with page_content keys.
        
        This test documents the current behavior that creates the inconsistency.
        """
        # Test simple.py _process_documents
        rag_simple = RAG()
        simple_result = rag_simple._process_documents(["Test document"])
        
        assert len(simple_result) == 1
        from iris_rag.core.models import Document
        assert isinstance(simple_result[0], Document)
        assert simple_result[0].page_content == "Test document"
        assert isinstance(simple_result[0].metadata, dict)
        
        # Test standard.py _process_documents
        config = {"technique": "basic"}
        rag_configurable = ConfigurableRAG(config)
        configurable_result = rag_configurable._process_documents(["Test document"])
        
        assert len(configurable_result) == 1
        from iris_rag.core.models import Document
        assert isinstance(configurable_result[0], Document)
        assert configurable_result[0].page_content == "Test document"
        assert isinstance(configurable_result[0].metadata, dict)
    
    def test_expected_behavior_should_work_with_document_objects(self):
        """
        Test that documents the expected behavior - the system should work when
        Document objects are passed to the vector store instead of dictionaries.
        
        This test shows what the fix should achieve.
        """
        # This test documents what should happen after the fix:
        # 1. rag.add_documents() should accept strings or dicts
        # 2. _process_documents() should convert them to proper Document objects
        # 3. The vector store should receive Document objects with .page_content attributes
        
        # After the fix, this test should pass without issue.
        
        # Test simple RAG
        rag = RAG()
        test_documents = ["Test document 1", "Test document 2"]
        try:
            rag.add_documents(test_documents)
        except AttributeError:
            pytest.fail("Simple RAG failed with AttributeError after fix.")
            
        # Test configurable RAG
        config = {"technique": "basic"}
        configurable_rag = ConfigurableRAG(config)
        try:
            configurable_rag.add_documents(test_documents)
        except AttributeError:
            pytest.fail("Configurable RAG failed with AttributeError after fix.")


class TestBugRootCauseAnalysis:
    """
    Additional tests to analyze the root cause of the document format bug.
    """
    
    def test_vector_store_expects_document_objects(self):
        """
        Test that demonstrates the vector store expects Document objects, not dictionaries.
        
        This test isolates the vector store behavior to show it expects .page_content attributes.
        """
        from iris_rag.storage.vector_store_iris import IRISVectorStore
        from iris_rag.config.manager import ConfigurationManager
        
        # Create a mock vector store to test the validation logic
        config_manager = ConfigurationManager()
        
        # Mock the connection to avoid database dependency
        with patch('iris_rag.storage.vector_store_iris.IRISVectorStore._get_connection'):
            vector_store = IRISVectorStore(config_manager=config_manager)
            
            # Create test documents as dictionaries (what rag_templates creates)
            dict_documents = [
                {"page_content": "Test content", "metadata": {"source": "test"}}
            ]
            
            # This should still fail because we are directly passing a dict
            # This test validates the vector store's expectation, not the fix itself.
            with pytest.raises(AttributeError, match="'dict' object has no attribute 'page_content'"):
                vector_store.add_documents(dict_documents)
    
    def test_pipeline_document_flow_analysis(self):
        """
        Test that analyzes the document flow through the pipeline to identify
        where the format conversion should happen.
        """
        # This test documents the current flow:
        # 1. User calls rag.add_documents(["string"] or [{"page_content": "..."}])
        # 2. rag._process_documents() converts to [{"page_content": "...", "metadata": {...}}]
        # 3. pipeline.load_documents() receives dictionaries
        # 4. pipeline eventually calls vector_store.add_documents() with dictionaries
        # 5. vector_store expects Document objects with .page_content attributes
        # 6. AttributeError occurs at line 235 in vector_store_iris.py
        
        pytest.skip("This test documents the current document flow for analysis")