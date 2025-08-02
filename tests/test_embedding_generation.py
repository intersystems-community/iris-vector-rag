"""
Tests for embedding generation.

These tests verify that we can correctly generate both document-level
and token-level embeddings for documents in the database.
"""

import os
import sys

# Make sure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.iris_connector import get_iris_connection

# Import our embedding generation functions
# This will fail initially since we haven't moved these functions out of 
# generate_embeddings.py into a proper module yet
from common.embedding_utils import (
    generate_document_embeddings,
    generate_token_embeddings,
    get_embedding_model,
    get_colbert_model
)

class TestEmbeddingGeneration:
    """Test suite for embedding generation capabilities."""

    def test_get_embedding_model(self):
        """Test that we can get an embedding model."""
        model = get_embedding_model(mock=True)
        assert model is not None
        
        # Test that the model can generate embeddings
        text = "This is a test document"
        embeddings = model.encode([text])
        assert embeddings is not None
        assert len(embeddings) == 1
        assert embeddings.shape[1] >= 128  # Should have a reasonable embedding dimension

    def test_get_colbert_model(self):
        """Test that we can get a ColBERT model for token-level embeddings."""
        model = get_colbert_model(mock=True)
        assert model is not None
        
        # Test that the model can generate token-level embeddings
        text = "This is a test document"
        tokens, embeddings = model.encode(text)
        assert tokens is not None
        assert embeddings is not None
        assert len(tokens) == len(embeddings)
        assert embeddings.shape[1] >= 64  # Should have a reasonable embedding dimension

    def test_generate_document_embeddings_mock(self):
        """Test that we can generate document-level embeddings with a mock connection."""
        # Get a connection (mock)
        connection = get_iris_connection()
        assert connection is not None
        
        # Add some documents
        cursor = connection.cursor()
        mock_docs = [
            ("doc1", "Test Title 1", "Test Content 1", "[]", "[]"),
            ("doc2", "Test Title 2", "Test Content 2", "[]", "[]")
        ]
        cursor.executemany(
            "INSERT INTO SourceDocuments (doc_id, title, content, authors, keywords) VALUES (?, ?, ?, ?, ?)",
            mock_docs
        )
        cursor.close()
        
        # Get an embedding model
        model = get_embedding_model(mock=True)
        
        # Generate embeddings
        stats = generate_document_embeddings(connection, model, batch_size=2, limit=5)
        
        # Verify results
        assert stats is not None
        assert stats["type"] == "document_embeddings"
        assert stats["processed_count"] > 0

    def test_generate_token_embeddings_mock(self):
        """Test that we can generate token-level embeddings with a mock connection."""
        # Get a connection (mock)
        connection = get_iris_connection()
        assert connection is not None
        
        # Add some documents
        cursor = connection.cursor()
        mock_docs = [
            ("doc1", "Test Title 1", "Test Content 1", "[]", "[]"),
            ("doc2", "Test Title 2", "Test Content 2", "[]", "[]")
        ]
        cursor.executemany(
            "INSERT INTO SourceDocuments (doc_id, title, content, authors, keywords) VALUES (?, ?, ?, ?, ?)",
            mock_docs
        )
        cursor.close()
        
        # Get a token encoder model
        model = get_colbert_model(mock=True)
        
        # Generate token embeddings
        stats = generate_token_embeddings(connection, model, batch_size=1, limit=2)
        
        # Verify results
        assert stats is not None
        assert stats["type"] == "token_embeddings"
        assert stats["processed_count"] > 0
        assert stats["tokens_count"] > 0
