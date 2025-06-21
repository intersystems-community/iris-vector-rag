"""
Tests for context reduction strategies.

This module tests the effectiveness of various context reduction strategies
when dealing with large documents or multiple retrieved documents that exceed
the LLM context window size.
"""

import pytest
import os
import sys
import numpy as np
from unittest.mock import MagicMock, patch

# Make sure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.iris_connector import get_iris_connection
from common.utils import Document
from common.embedding_utils import get_embedding_model

# Import the module we'll create
from common.context_reduction import (
    simple_truncation,
    recursive_summarization,
    embeddings_reranking,
    map_reduce_approach
)

class TestContextReduction:
    """Test suite for context reduction strategies."""

    @pytest.fixture
    def large_documents(self):
        """Fixture to create a set of large test documents."""
        # Create a list of documents that would exceed a typical context window
        docs = []
        for i in range(5):
            # Each document is about 2000 tokens
            content = f"Document {i+1}: " + "This is a test sentence. " * 400
            docs.append(Document(id=f"doc{i+1}", content=content, score=0.9 - i*0.1))
        return docs
            
    @pytest.fixture
    def query_text(self):
        """Fixture for a test query."""
        return "What are the key treatments for diabetes discussed in these medical papers?"
    
    def test_simple_truncation_strategy(self, large_documents, query_text):
        """Test the simple truncation strategy."""
        # Apply simple truncation
        max_tokens = 3000
        truncated_context = simple_truncation(large_documents, max_tokens)
        
        # The truncated context should be no more than max_tokens
        total_tokens = len(truncated_context.split())
        assert total_tokens <= max_tokens, f"Truncated context exceeds max tokens: {total_tokens} > {max_tokens}"
        
        # The truncated context should contain content from the beginning documents
        assert "Document 1:" in truncated_context
        
        # The context should be properly formatted
        assert "\n\n" in truncated_context, "Expected separator between documents"
    
    def test_recursive_summarization_strategy(self, large_documents, query_text):
        """Test the recursive summarization strategy."""
        # This strategy uses an LLM to summarize documents
        mock_llm = MagicMock()
        mock_llm.return_value = "This is a summarized version of the document."
        
        with patch('common.context_reduction.generate_summary', mock_llm):
            # Apply recursive summarization
            max_tokens = 3000
            summarized_context = recursive_summarization(large_documents, query_text, max_tokens)
            
            # Verify the LLM was called to generate summaries
            assert mock_llm.called
            
            # The summarized context should be no more than max_tokens
            assert len(summarized_context.split()) <= max_tokens
            
            # The context should contain summaries
            assert "summarized version" in summarized_context
    
    def test_embeddings_reranking_strategy(self, large_documents, query_text):
        """Test the embeddings reranking strategy."""
        # This strategy reranks document chunks based on relevance to query
        embedding_model = get_embedding_model(mock=True)
        
        # Apply embedding reranking
        max_tokens = 3000
        reranked_context = embeddings_reranking(large_documents, query_text, embedding_model, max_tokens)
        
        # The reranked context should be no more than max_tokens
        assert len(reranked_context.split()) <= max_tokens
        
        # For this test, the content should still be from our documents
        assert "Document" in reranked_context
    
    def test_map_reduce_approach(self, large_documents, query_text):
        """Test the map-reduce approach for context reduction."""
        # This approach processes each document separately and combines results
        mock_llm = MagicMock()
        mock_llm.return_value = "This is a processed document summary."
        
        with patch('common.context_reduction.process_document', mock_llm):
            # Apply map-reduce
            reduced_context = map_reduce_approach(large_documents, query_text)
            
            # Verify the LLM was called for each document
            assert mock_llm.call_count == len(large_documents)
            
            # The result should contain combined information
            assert "processed document summary" in reduced_context
            
    @pytest.mark.integration
    def test_context_reduction_end_to_end(self, iris_connection, use_real_data):
        """Test the end-to-end pipeline with context reduction."""
        if not use_real_data:
            pytest.skip("This test requires real data")
            
        # Get some large documents from the database
        cursor = iris_connection.cursor()
        cursor.execute("""
            SELECT TOP 5 doc_id, content FROM SourceDocuments 
            ORDER BY LENGTH(content) DESC
        """)
        results = cursor.fetchall()
        cursor.close()
        
        assert len(results) > 0, "No documents found in database"
        
        # Create Document objects
        documents = [Document(id=row[0], content=row[1], score=0.9) for row in results]
        
        # Apply different reduction strategies
        query = "What are the key treatments for diabetes?"
        max_tokens = 3000
        
        # Simple truncation
        truncated = simple_truncation(documents, max_tokens)
        assert len(truncated.split()) <= max_tokens
        
        # Embeddings reranking (if real embedding model available)
        if use_real_data:
            try:
                embedding_model = get_embedding_model()
                reranked = embeddings_reranking(documents, query, embedding_model, max_tokens)
                assert len(reranked.split()) <= max_tokens
            except ImportError:
                pytest.skip("Real embedding model not available")
        
        # For LLM-based strategies, we'd use mocks in real tests
        # but here we're skipping the actual tests if they require real LLMs
