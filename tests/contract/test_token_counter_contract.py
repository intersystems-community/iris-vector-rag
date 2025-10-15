"""
Contract tests for token counting utility.

These tests validate the estimate_tokens() API contract before implementation.
Tests MUST fail initially, then pass after implementation (TDD).
"""

import pytest


class TestTokenCounterContract:
    """Contract tests for token counting utility."""

    def test_estimate_tokens_function_exists(self):
        """Validate estimate_tokens() function exists."""
        from iris_rag.utils.token_counter import estimate_tokens

        assert callable(estimate_tokens), "estimate_tokens must be a callable function"

    def test_estimate_tokens_signature(self):
        """Validate estimate_tokens() has correct signature."""
        from iris_rag.utils.token_counter import estimate_tokens
        import inspect

        sig = inspect.signature(estimate_tokens)
        params = sig.parameters

        # Validate required parameters
        assert 'text' in params, "estimate_tokens must accept 'text' parameter"

        # Validate optional parameters with defaults
        assert 'model' in params, "estimate_tokens must accept 'model' parameter"
        assert params['model'].default == "gpt-3.5-turbo", \
            "model default must be 'gpt-3.5-turbo' per research.md"

    def test_estimate_tokens_accuracy_short_text(self):
        """Validate token estimation accuracy within ±10% tolerance for short text."""
        from iris_rag.utils.token_counter import estimate_tokens

        # Known test text: "This is a test document with multiple words."
        # Expected tokens (from tiktoken for gpt-3.5-turbo): ~9 tokens
        test_text = "This is a test document with multiple words."

        estimated = estimate_tokens(test_text)

        # Validate within ±10% tolerance
        expected = 9
        tolerance = expected * 0.1  # ±10%

        assert abs(estimated - expected) <= tolerance, \
            f"Token estimation must be within ±10% (expected ~{expected}, got {estimated})"

    def test_estimate_tokens_empty_string_returns_zero(self):
        """Validate empty string returns 0 tokens."""
        from iris_rag.utils.token_counter import estimate_tokens

        assert estimate_tokens("") == 0, "Empty string must return 0 tokens"

    def test_estimate_tokens_large_document(self):
        """Validate token estimation for large documents (5000 words)."""
        from iris_rag.utils.token_counter import estimate_tokens

        # Generate large text (~5000 tokens)
        large_text = "word " * 5000  # ~5000 tokens

        estimated = estimate_tokens(large_text)

        # Validate within reasonable range (4500-5500 tokens, ±10%)
        assert 4500 <= estimated <= 5500, \
            f"Large document estimation must be accurate (expected ~5000, got {estimated})"

    def test_estimate_tokens_none_input_raises_error(self):
        """Validate None input raises ValueError."""
        from iris_rag.utils.token_counter import estimate_tokens

        with pytest.raises(ValueError, match="text.*cannot be None"):
            estimate_tokens(None)

    def test_estimate_tokens_unsupported_model_raises_error(self):
        """Validate unsupported model raises ValueError."""
        from iris_rag.utils.token_counter import estimate_tokens

        with pytest.raises(ValueError, match="unsupported model"):
            estimate_tokens("test text", model="invalid-model-xyz")

    def test_estimate_tokens_different_models(self):
        """Validate token estimation works for different model encodings."""
        from iris_rag.utils.token_counter import estimate_tokens

        test_text = "This is a test."

        # Should work for different models
        gpt35_tokens = estimate_tokens(test_text, model="gpt-3.5-turbo")
        gpt4_tokens = estimate_tokens(test_text, model="gpt-4")

        # Both should return valid token counts (may differ slightly)
        assert gpt35_tokens > 0, "gpt-3.5-turbo encoding must work"
        assert gpt4_tokens > 0, "gpt-4 encoding must work"

    def test_estimate_tokens_special_characters(self):
        """Validate token estimation handles special characters correctly."""
        from iris_rag.utils.token_counter import estimate_tokens

        # Text with special characters
        special_text = "Test with émojis 🚀 and spëcial çharacters!"

        estimated = estimate_tokens(special_text)

        # Should return valid token count (no crash)
        assert estimated > 0, "Must handle special characters without error"

    def test_estimate_tokens_performance(self):
        """Validate token estimation is fast (< 10ms for 1000 chars)."""
        from iris_rag.utils.token_counter import estimate_tokens
        import time

        # Create medium-sized text
        text = "This is a test document. " * 40  # ~1000 characters

        # Measure estimation time
        start = time.time()
        estimate_tokens(text)
        elapsed = time.time() - start

        # Should be very fast (tiktoken is Rust-based, ~1M tokens/sec)
        assert elapsed < 0.01, \
            f"Token estimation must be fast (expected <10ms, got {elapsed * 1000:.2f}ms)"
