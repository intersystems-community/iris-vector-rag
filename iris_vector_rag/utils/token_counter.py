"""
Token counting utilities for batch sizing.

This module provides token estimation functionality using tiktoken to support
dynamic batch sizing based on token budgets.

Implementation:
- Uses tiktoken library for accurate token counting (~1M tokens/sec, Rust-based)
- Supports multiple OpenAI model encodings (gpt-3.5-turbo, gpt-4, etc.)
- Provides edge case handling (None, empty strings, special characters)
"""

from typing import Optional
import tiktoken


# Supported models and their default encoding
_MODEL_ENCODINGS = {
    "gpt-3.5-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4o": "o200k_base",
    "text-embedding-ada-002": "cl100k_base",
    "text-embedding-3-small": "cl100k_base",
    "text-embedding-3-large": "cl100k_base",
}


def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate token count for text using tiktoken.

    This function provides accurate token counting for batch sizing (FR-006).
    It uses tiktoken's encoding_for_model() to match the exact tokenization
    used by OpenAI models.

    Args:
        text: Text to count tokens for
        model: Model name to use for encoding (default: "gpt-3.5-turbo")

    Returns:
        Integer token count

    Raises:
        ValueError: If text is None or model is unsupported

    Examples:
        >>> estimate_tokens("Hello world")
        2
        >>> estimate_tokens("This is a test.", model="gpt-4")
        5
        >>> estimate_tokens("")
        0
    """
    # Validate input
    if text is None:
        raise ValueError("text parameter cannot be None")

    # Handle empty string
    if text == "":
        return 0

    # Get encoding for the model
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Model not recognized - raise descriptive error
        supported_models = ", ".join(_MODEL_ENCODINGS.keys())
        raise ValueError(
            f"unsupported model: '{model}'. "
            f"Supported models: {supported_models}"
        )

    # Encode and count tokens
    tokens = encoding.encode(text)
    return len(tokens)


def estimate_tokens_bulk(
    texts: list[str], model: str = "gpt-3.5-turbo"
) -> list[int]:
    """
    Estimate token counts for multiple texts efficiently.

    This is more efficient than calling estimate_tokens() multiple times
    because it reuses the same encoding instance.

    Args:
        texts: List of texts to count tokens for
        model: Model name to use for encoding (default: "gpt-3.5-turbo")

    Returns:
        List of integer token counts (same length as texts)

    Raises:
        ValueError: If any text is None or model is unsupported
    """
    # Validate inputs
    if any(text is None for text in texts):
        raise ValueError("text parameter cannot be None")

    # Get encoding once for all texts
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        supported_models = ", ".join(_MODEL_ENCODINGS.keys())
        raise ValueError(
            f"unsupported model: '{model}'. "
            f"Supported models: {supported_models}"
        )

    # Count tokens for each text
    return [len(encoding.encode(text)) for text in texts]
