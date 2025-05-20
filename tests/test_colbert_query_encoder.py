"""
Test for ColBERT query encoder functionality.

This module tests both the mock and real implementations of the ColBERT query encoder,
ensuring it correctly generates token-level embeddings for queries.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Make sure the project root is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from colbert.query_encoder import ColBERTQueryEncoder, get_colbert_query_encoder


class TestColBERTQueryEncoder:
    """Test suite for the ColBERT query encoder."""
    
    def test_mock_query_encoder_initialization(self):
        """Test that the mock query encoder initializes correctly."""
        encoder = ColBERTQueryEncoder(mock=True)
        assert encoder.mock == True
        assert encoder.embedding_dim == 128
        assert encoder.max_query_length == 32
        
    def test_mock_tokenization(self):
        """Test that the mock tokenizer works correctly."""
        encoder = ColBERTQueryEncoder(mock=True)
        query = "What is ColBERT?"
        
        tokenizer_output = encoder._mock_tokenize(query)
        
        assert "tokens" in tokenizer_output
        assert len(tokenizer_output["tokens"]) == 3  # what, is, colbert
        assert tokenizer_output["tokens"][0] == "what"
        assert tokenizer_output["attention_mask"].shape[1] == 3
        
    def test_mock_encoder_output_shape(self):
        """Test that the mock encoder produces correctly shaped outputs."""
        encoder = ColBERTQueryEncoder(mock=True, embedding_dim=64)
        query = "What is ColBERT?"
        
        token_embeddings = encoder.encode(query)
        
        assert isinstance(token_embeddings, list)
        assert len(token_embeddings) == 3  # 3 tokens
        assert len(token_embeddings[0]) == 64  # 64-dimensional embeddings
        
    def test_mock_encoder_normalization(self):
        """Test that the mock encoder produces normalized embeddings."""
        encoder = ColBERTQueryEncoder(mock=True)
        query = "What is ColBERT?"
        
        token_embeddings = encoder.encode(query)
        
        # Check each embedding is normalized (L2 norm ≈ 1.0)
        for embedding in token_embeddings:
            norm = np.linalg.norm(embedding)
            assert 0.99 <= norm <= 1.01, f"Embedding not normalized, norm = {norm}"
            
    def test_mock_encoder_deterministic(self):
        """Test that the mock encoder produces deterministic results for the same input."""
        encoder = ColBERTQueryEncoder(mock=True)
        query = "What is ColBERT?"
        
        embeddings1 = encoder.encode(query)
        embeddings2 = encoder.encode(query)
        
        # Check that embeddings are the same for identical queries
        for emb1, emb2 in zip(embeddings1, embeddings2):
            assert np.allclose(emb1, emb2)
            
    def test_mock_encoder_callable(self):
        """Test that the encoder object is callable as a function."""
        encoder = ColBERTQueryEncoder(mock=True)
        query = "What is ColBERT?"
        
        # Can call the encoder directly
        token_embeddings = encoder(query)
        
        assert isinstance(token_embeddings, list)
        assert len(token_embeddings) > 0
        
    def test_get_colbert_query_encoder(self):
        """Test that the get_colbert_query_encoder function returns a callable."""
        encoder_func = get_colbert_query_encoder(mock=True)
        
        assert callable(encoder_func)
        
        # Test the returned function
        query = "What is ColBERT?"
        token_embeddings = encoder_func(query)
        
        assert isinstance(token_embeddings, list)
        assert len(token_embeddings) > 0
        
    @pytest.mark.skipif(True, reason="Requires transformers package and real model")
    def test_real_encoder_initialization(self):
        """Test that the real query encoder initializes correctly."""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
             patch('transformers.AutoModel.from_pretrained') as mock_model:
            
            # Setup mocks
            mock_tokenizer.return_value = MagicMock()
            mock_model.return_value = MagicMock()
            mock_model.return_value.to.return_value = mock_model.return_value
            
            # Initialize encoder
            encoder = ColBERTQueryEncoder(mock=False)
            
            # Verify mocks were called
            mock_tokenizer.assert_called_once()
            mock_model.assert_called_once()
            assert not encoder.mock
            
    @pytest.mark.skipif(True, reason="Requires transformers package and real model")
    def test_real_encoder_fallback(self):
        """Test that the real encoder falls back to mock if initialization fails."""
        with patch('colbert.query_encoder.ColBERTQueryEncoder._initialize_real_model') as mock_init:
            # Make initialization fail
            mock_init.side_effect = ImportError("No module named 'transformers'")
            
            # Initialize encoder
            encoder = ColBERTQueryEncoder(mock=False)
            
            # Verify fallback to mock
            assert encoder.mock
            
            # Should still work
            token_embeddings = encoder.encode("test query")
            assert isinstance(token_embeddings, list)
            assert len(token_embeddings) > 0
            
    @pytest.mark.skipif(True, reason="Integration test requiring real transformer model")
    def test_real_encoder_with_transformers_integration(self):
        """Integration test with real transformers model."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            # Skip if CUDA/MPS is not available
            if not torch.cuda.is_available() and not hasattr(torch.backends, 'mps') and not torch.backends.mps.is_available():
                device = "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "mps"
                
            # Initialize with a small model for testing
            encoder = ColBERTQueryEncoder(
                model_name="distilbert-base-uncased",  # Smaller model for tests
                device=device,
                mock=False
            )
            
            query = "What is ColBERT?"
            token_embeddings = encoder.encode(query)
            
            # Basic checks
            assert isinstance(token_embeddings, list)
            assert len(token_embeddings) > 0
            
            # Should have embeddings for each token
            tokenizer_output = encoder._tokenize_query(query)
            expected_token_count = sum(tokenizer_output["attention_mask"][0].tolist())
            assert len(token_embeddings) == expected_token_count
            
            # Check normalization
            for embedding in token_embeddings:
                norm = np.linalg.norm(embedding)
                assert 0.99 <= norm <= 1.01, f"Embedding not normalized, norm = {norm}"
                
        except ImportError:
            pytest.skip("Transformers package not installed")
            
    def test_long_query_truncation(self):
        """Test that long queries are properly truncated."""
        encoder = ColBERTQueryEncoder(mock=True, max_query_length=5)
        long_query = "This is a very long query that exceeds the maximum length"
        
        token_embeddings = encoder.encode(long_query)
        
        # Should be truncated to max_query_length
        assert len(token_embeddings) <= 5
