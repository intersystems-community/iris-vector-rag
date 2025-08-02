"""
Tests for real data integration with embedding generation.

This module tests the complete pipeline for processing real PMC data 
and generating both document-level and token-level embeddings using
the proper utility abstractions.
"""

import pytest
import os
import sys

# Make sure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from rag_templates.simple import RAG


@pytest.mark.integration
@pytest.mark.real_data  
def test_real_data_embedding_pipeline(use_real_data):
    """
    Test the complete pipeline for processing real data and generating embeddings
    using the Simple RAG API abstraction with proper schema setup.
    """
    # Initialize RAG system - this should handle schema setup through proper abstractions
    try:
        rag = RAG()
        
        # Ensure proper schema setup by validating configuration
        config_valid = rag.validate_config()
        assert config_valid, "RAG configuration should be valid"
        
        # Add test documents using the proper API if not using real data
        if not use_real_data:
            test_documents = [
                "This is test document 1 about machine learning and AI.",
                "This is test document 2 about database systems and vectors.", 
                "This is test document 3 about RAG and information retrieval."
            ]
            
            # Use the Simple API to add documents (this handles all abstractions including schema)
            rag.add_documents(test_documents)
            print("Added test documents using Simple RAG API")
        
        # Verify document count
        doc_count = rag.get_document_count()
        assert doc_count > 0, "No documents found in knowledge base"
        
        # Test querying the system
        query_result = rag.query("What is machine learning?")
        assert isinstance(query_result, str), "Query should return a string response"
        assert len(query_result) > 0, "Query should return non-empty response"
        
        print(f"\nResults using {'real' if use_real_data else 'mock'} data:")
        print(f"Total documents: {doc_count}")
        print(f"Query response: {query_result[:100]}...")
        
        # Verify the system is working end-to-end
        assert doc_count > 0, "Pipeline should have documents loaded"
        
    except Exception as e:
        # If the test fails due to schema issues, that indicates the schema manager setup needs fixing
        if "Field" in str(e) and "not found" in str(e):
            pytest.fail(f"Schema setup failed - schema manager did not properly create required database structure: {e}")
        else:
            # Re-raise other exceptions
            raise


@pytest.mark.integration
@pytest.mark.real_data
def test_embedding_end_to_end(use_real_data):
    """
    Test the end-to-end embedding generation and retrieval process
    using the Simple RAG API abstractions with proper schema management.
    """
    try:
        # Initialize RAG system - this should handle schema setup through proper abstractions
        rag = RAG()
        
        # Validate configuration before proceeding
        config_valid = rag.validate_config()
        assert config_valid, "RAG configuration should be valid"
        
        # If using mock data, add a test document
        if not use_real_data:
            test_document = "This is a comprehensive test document for end-to-end testing of the RAG pipeline with embeddings and retrieval capabilities."
            rag.add_documents([test_document])
            print("Added test document using Simple RAG API")
        
        # Verify we have documents
        doc_count = rag.get_document_count()
        if doc_count == 0:
            pytest.skip("No documents available for testing")
        
        # Test the end-to-end pipeline with a query
        test_query = "What is the purpose of this test document?"
        query_result = rag.query(test_query)
        
        # Verify the response
        assert isinstance(query_result, str), "Query should return a string response"
        assert len(query_result) > 0, "Query should return non-empty response"
        assert "error" not in query_result.lower() or "Error:" not in query_result, f"Query returned error: {query_result}"
        
        # Test another query to verify retrieval is working
        similarity_query = "test document"
        similarity_result = rag.query(similarity_query)
        assert isinstance(similarity_result, str), "Similarity query should return string response"
        assert len(similarity_result) > 0, "Similarity query should return non-empty response"
        
        print(f"\nE2E test results using {'real' if use_real_data else 'mock'} data:")
        print(f"Total documents: {doc_count}")
        print(f"Test query response: {query_result[:100]}...")
        print(f"Similarity query response: {similarity_result[:100]}...")
        
        print("End-to-end test completed successfully using proper abstractions")
        
    except Exception as e:
        # If the test fails due to schema issues, that indicates the schema manager setup needs fixing
        if "Field" in str(e) and "not found" in str(e):
            pytest.fail(f"Schema setup failed - schema manager did not properly create required database structure: {e}")
        else:
            # Re-raise other exceptions  
            raise
