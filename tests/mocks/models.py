"""
Standardized mock implementations for machine learning models.
These mocks provide consistent behavior for testing model interactions
without requiring actual model loading or inference.
"""

import numpy as np
from typing import List, Dict, Any, Callable, Union, Optional
import random

def mock_embedding_func(
    text_input: Union[str, List[str]], 
    dimensions: int = 768, # Changed default to match e5-base-v2
    seed: int = 42
) -> List[List[float]]:
    """
    A configurable mock embedding function that returns deterministic embeddings.
    
    Args:
        text_input: A string or list of strings to embed
        dimensions: Dimensionality of the generated embeddings
        seed: Random seed for reproducibility
        
    Returns:
        A list of embedding vectors (each vector is a list of floats)
    """
    np.random.seed(seed)
    
    # Convert single string to list for consistent handling
    if isinstance(text_input, str):
        text_input = [text_input]
    
    # Generate deterministic embeddings based on input text length
    embeddings = []
    for text in text_input:
        # Use text length as a factor to make embeddings somewhat meaningful
        text_factor = len(text) / 100.0
        # Create deterministic but "random-looking" embedding
        embedding = np.random.randn(dimensions) * text_factor
        # Normalize embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        embeddings.append(embedding.tolist())
    
    return embeddings


def mock_llm_func(
    prompt: str,
    predefined_responses: Dict[str, str] = None,
    default_response: str = "This is a mock response from the LLM.",
    response_words_min: int = 50,
    response_words_max: int = 200,
    seed: int = 24
) -> str:
    """
    A configurable mock LLM function that returns deterministic responses.
    
    Args:
        prompt: The prompt text
        predefined_responses: Dictionary mapping prompt substrings to responses
        default_response: Default response if no predefined response matches
        response_words_min: Minimum words in generated response
        response_words_max: Maximum words in generated response
        seed: Random seed for reproducibility
        
    Returns:
        A string response
    """
    random.seed(seed)
    
    # Check for predefined responses
    if predefined_responses:
        for key, response in predefined_responses.items():
            if key.lower() in prompt.lower():
                return response
    
    # Generate a more realistic but deterministic response based on the prompt
    prompt_words = prompt.split()
    num_words = min(
        response_words_max, 
        max(response_words_min, len(prompt_words) // 2)
    )
    
    # Some filler sentences to make responses look realistic
    fillers = [
        "This is an important consideration.",
        "The data suggests several interpretations.",
        "Based on the available information, we can conclude the following.",
        "Let's examine this question from multiple perspectives.",
        "The research indicates a correlation between these factors.",
        "We should consider both the advantages and limitations.",
        "This approach offers several benefits worth considering.",
        "The evidence supports this conclusion.",
        "Further analysis may be required to fully understand the implications.",
        "There are several key takeaways from this analysis."
    ]
    
    # Extract some words from the prompt to make the response contextual
    prompt_extract = [word for word in prompt_words if len(word) > 4]
    if not prompt_extract:
        prompt_extract = prompt_words
    
    # Generate response
    response_parts = []
    words_added = 0
    
    # Start with a contextual opening
    if prompt_extract:
        contextual_opener = f"Regarding {random.choice(prompt_extract)}, {random.choice(fillers).lower()}"
        response_parts.append(contextual_opener)
        words_added += len(contextual_opener.split())
    
    # Add filler content
    while words_added < num_words:
        filler = random.choice(fillers)
        response_parts.append(filler)
        words_added += len(filler.split())
    
    return " ".join(response_parts)


def mock_colbert_doc_encoder(
    text: str,
    token_count: int = 20,
    dimensions: int = 10,
    seed: int = 42
) -> List[List[float]]:
    """
    A mock ColBERT document encoder that returns token-level embeddings.
    
    Args:
        text: Document text to encode
        token_count: Maximum number of tokens to encode
        dimensions: Dimensionality of the generated embeddings
        seed: Random seed for reproducibility
        
    Returns:
        A list of token embedding vectors
    """
    np.random.seed(seed)
    
    # Simple tokenization by splitting on spaces
    tokens = text.split()
    
    # Limit tokens to the specified count
    tokens = tokens[:token_count]
    
    # Generate token embeddings
    token_embeddings = []
    for token in tokens:
        # Use token length as a factor for some variance
        token_factor = len(token) / 10.0
        # Create embedding
        embedding = np.random.randn(dimensions) * token_factor
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        token_embeddings.append(embedding.tolist())
    
    return token_embeddings


def mock_colbert_query_encoder(
    text: str,
    token_count: int = 5,
    dimensions: int = 10,
    seed: int = 24
) -> List[List[float]]:
    """
    A mock ColBERT query encoder that returns token-level embeddings.
    
    Args:
        text: Query text to encode
        token_count: Maximum number of tokens to encode
        dimensions: Dimensionality of the generated embeddings
        seed: Random seed for reproducibility
        
    Returns:
        A list of token embedding vectors
    """
    np.random.seed(seed)
    
    # Simple tokenization by splitting on spaces
    tokens = text.split()
    
    # Limit tokens to the specified count
    tokens = tokens[:token_count]
    
    # Generate token embeddings with different characteristics than document encoder
    token_embeddings = []
    for token in tokens:
        # Use different factor for query tokens
        token_factor = len(token) / 8.0 
        # Create embedding with slightly different distribution
        embedding = np.random.randn(dimensions) * token_factor + 0.1
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        token_embeddings.append(embedding.tolist())
    
    return token_embeddings
