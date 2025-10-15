"""
Unit tests for token counter utility.

Tests token estimation logic in isolation without dependencies.
"""

import pytest
from iris_rag.utils.token_counter import estimate_tokens


class TestTokenCounter:
    """Unit tests for token counting utility."""

    def test_tiktoken_integration(self):
        """Validate tiktoken library integration."""
        # Simple text should return non-zero token count
        result = estimate_tokens("Hello world")
        assert result > 0, "Should return positive token count"
        assert isinstance(result, int), "Should return integer token count"

    def test_different_model_encodings(self):
        """Validate token estimation works for different model encodings."""
        test_text = "This is a test sentence for token counting."

        # Test different models
        gpt35_tokens = estimate_tokens(test_text, model="gpt-3.5-turbo")
        gpt4_tokens = estimate_tokens(test_text, model="gpt-4")

        # Both should work
        assert gpt35_tokens > 0, "gpt-3.5-turbo encoding should work"
        assert gpt4_tokens > 0, "gpt-4 encoding should work"

        # Token counts may differ slightly between models
        # but should be in same ballpark
        assert abs(gpt35_tokens - gpt4_tokens) < 5, \
            "Token counts should be similar across models"

    def test_edge_case_none_input(self):
        """Validate None input raises ValueError."""
        with pytest.raises(ValueError, match="text.*cannot be None"):
            estimate_tokens(None)

    def test_edge_case_empty_string(self):
        """Validate empty string returns 0 tokens."""
        assert estimate_tokens("") == 0, "Empty string should return 0 tokens"

    def test_edge_case_special_characters(self):
        """Validate special characters handled correctly."""
        special_texts = [
            "Hello! How are you? 😊",
            "Code: print('hello')",
            "Math: 1 + 1 = 2",
            "Symbols: @#$%^&*()",
            "Unicode: café résumé naïve",
        ]

        for text in special_texts:
            tokens = estimate_tokens(text)
            assert tokens > 0, f"Should handle special characters: {text}"
            assert isinstance(tokens, int), "Should return integer"

    def test_whitespace_handling(self):
        """Validate whitespace is counted correctly."""
        # Multiple spaces
        assert estimate_tokens("word  word") > 0

        # Tabs and newlines
        assert estimate_tokens("word\tword\nword") > 0

        # Leading/trailing whitespace
        assert estimate_tokens("  word  ") > 0

    def test_very_long_text(self):
        """Validate token estimation for very long text."""
        long_text = "word " * 10000  # ~10,000 tokens

        tokens = estimate_tokens(long_text)

        # Should be in reasonable range
        assert 9000 <= tokens <= 11000, \
            f"Long text estimation (expected ~10K, got {tokens})"

    def test_token_count_consistency(self):
        """Validate same text always returns same token count."""
        text = "Consistent token counting test."

        count1 = estimate_tokens(text)
        count2 = estimate_tokens(text)
        count3 = estimate_tokens(text)

        assert count1 == count2 == count3, \
            "Token count should be deterministic"

    def test_known_token_counts(self):
        """Validate against known token counts for specific texts."""
        test_cases = [
            ("Hello", 1),  # Single word
            ("Hello world", 2),  # Two words
            ("This is a test.", 5),  # Simple sentence
        ]

        for text, expected in test_cases:
            actual = estimate_tokens(text)
            # Allow ±1 token tolerance
            assert abs(actual - expected) <= 1, \
                f"Token count for '{text}' (expected ~{expected}, got {actual})"

    def test_default_model_parameter(self):
        """Validate default model is gpt-3.5-turbo."""
        text = "Test text"

        # Call without model parameter
        default_tokens = estimate_tokens(text)

        # Call with explicit gpt-3.5-turbo
        explicit_tokens = estimate_tokens(text, model="gpt-3.5-turbo")

        assert default_tokens == explicit_tokens, \
            "Default model should be gpt-3.5-turbo"

    def test_unsupported_model_raises_error(self):
        """Validate unsupported model raises ValueError."""
        with pytest.raises(ValueError, match="unsupported model"):
            estimate_tokens("test", model="invalid-model-xyz")

    def test_performance_fast_estimation(self):
        """Validate token estimation is fast."""
        import time

        text = "Performance test text. " * 100  # ~300 tokens

        start = time.time()
        for _ in range(100):  # 100 iterations
            estimate_tokens(text)
        elapsed = time.time() - start

        # Should be very fast (< 100ms for 100 iterations)
        assert elapsed < 0.1, \
            f"Token estimation should be fast (got {elapsed*1000:.1f}ms for 100 iterations)"

    def test_text_with_numbers(self):
        """Validate text with numbers is counted correctly."""
        numeric_texts = [
            "The year 2024 is here.",
            "Price: $99.99",
            "Version 2.1.5",
            "1234567890",
        ]

        for text in numeric_texts:
            tokens = estimate_tokens(text)
            assert tokens > 0, f"Should handle numbers: {text}"

    def test_multilingual_text(self):
        """Validate multilingual text is counted correctly."""
        multilingual_texts = [
            "Hello world",  # English
            "Bonjour monde",  # French
            "Hola mundo",  # Spanish
            "你好世界",  # Chinese
            "こんにちは世界",  # Japanese
        ]

        for text in multilingual_texts:
            tokens = estimate_tokens(text)
            assert tokens > 0, f"Should handle multilingual text: {text}"

    def test_code_snippets(self):
        """Validate code snippets are counted correctly."""
        code_texts = [
            "def hello(): return 'world'",
            "SELECT * FROM users WHERE id = 1;",
            "const x = { key: 'value' };",
        ]

        for text in code_texts:
            tokens = estimate_tokens(text)
            assert tokens > 0, f"Should handle code: {text}"
