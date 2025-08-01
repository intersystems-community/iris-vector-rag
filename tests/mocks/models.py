"""
Model mock functions for testing.

This module provides mock implementations of embedding functions, LLM functions,
and other model-related components used in tests.
"""

from typing import List, Union, Tuple
import numpy as np


def mock_embedding_func(text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
    """
    Mock embedding function that returns deterministic embeddings.
    
    Args:
        text: Single text string or list of text strings
        
    Returns:
        Single embedding vector or list of embedding vectors
    """
    if isinstance(text, list):
        return [[0.1] * 768 for _ in text]
    return [0.1] * 768


def mock_llm_func(prompt: str) -> str:
    """
    Mock LLM function that returns a deterministic response.
    
    Args:
        prompt: Input prompt string
        
    Returns:
        Mock response string
    """
    return "Mock LLM response for testing purposes."


def mock_colbert_doc_encoder(text: str) -> List[Tuple[str, List[float]]]:
    """
    Mock ColBERT document encoder that returns token-level embeddings.
    
    Args:
        text: Input document text
        
    Returns:
        List of (token, embedding) tuples
    """
    # Simple tokenization for testing
    tokens = text.split()[:10]  # Limit to 10 tokens for testing
    return [(token, [0.5] * 10) for token in tokens]


def mock_colbert_query_encoder(text: str) -> List[List[float]]:
    """
    Mock ColBERT query encoder that returns token-level embeddings.
    
    Args:
        text: Input query text
        
    Returns:
        List of embedding vectors (one per token)
    """
    # Simple tokenization for testing
    tokens = text.split()[:5]  # Limit to 5 tokens for testing
    return [[0.1] * 10 for _ in tokens]


def mock_colbert_query_encoder_with_tokens(text: str) -> Tuple[List[str], List[List[float]]]:
    """
    Mock ColBERT query encoder that returns both tokens and embeddings.
    
    Args:
        text: Input query text
        
    Returns:
        Tuple of (tokens, embeddings)
    """
    # Simple tokenization for testing
    tokens = text.split()[:5]  # Limit to 5 tokens for testing
    embeddings = [[0.1] * 10 for _ in tokens]
    return tokens, embeddings


def create_mock_embedding_matrix(num_docs: int = 5, embedding_dim: int = 768) -> np.ndarray:
    """
    Create a mock embedding matrix for testing.
    
    Args:
        num_docs: Number of document embeddings
        embedding_dim: Dimension of each embedding
        
    Returns:
        NumPy array of shape (num_docs, embedding_dim)
    """
    # Create deterministic embeddings for consistent testing
    embeddings = []
    for i in range(num_docs):
        # Create a unique pattern for each document
        embedding = [0.1 + (i * 0.1)] * embedding_dim
        embeddings.append(embedding)
    
    return np.array(embeddings)


def create_mock_colbert_embeddings(num_docs: int = 5, tokens_per_doc: int = 10, 
                                  embedding_dim: int = 10) -> List[List[List[float]]]:
    """
    Create mock ColBERT token embeddings for multiple documents.
    
    Args:
        num_docs: Number of documents
        tokens_per_doc: Number of tokens per document
        embedding_dim: Dimension of each token embedding
        
    Returns:
        List of documents, each containing a list of token embeddings
    """
    doc_embeddings = []
    for doc_idx in range(num_docs):
        token_embeddings = []
        for token_idx in range(tokens_per_doc):
            # Create unique embeddings based on doc and token indices
            embedding = [(doc_idx + 1) * 0.1 + (token_idx + 1) * 0.01] * embedding_dim
            token_embeddings.append(embedding)
        doc_embeddings.append(token_embeddings)
    
    return doc_embeddings