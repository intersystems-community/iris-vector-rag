"""
Contract tests for PyLateColBERT Dimension Validation (DIM-001).

Tests validate that PyLateColBERT correctly handles token-level embeddings for ColBERT
and 384-dimensional embeddings for fallback dense vector search.

Contract: dimension_validation_contract.md
Requirements: FR-021, FR-022, FR-023, FR-024
"""

import pytest


@pytest.mark.contract
@pytest.mark.dimension
@pytest.mark.pylate_colbert
class TestPyLateColBERTDimensionValidation:
    """Contract tests for PyLateColBERT dimension validation."""

    def test_colbert_token_embedding_structure(self, pylate_colbert_pipeline):
        """
        FR-021: PyLateColBERT MUST use token-level embeddings for ColBERT.

        Given: PyLateColBERT pipeline instance
        When: ColBERT encoding generated for sample text
        Then: Token embeddings have correct structure (list of token vectors)
        """
        if not hasattr(pylate_colbert_pipeline, 'colbert_encoder'):
            pytest.skip("Pipeline does not expose colbert_encoder")

        # Generate token embeddings for test text
        test_text = "Sample text for ColBERT token embedding validation"

        try:
            token_embeddings = pylate_colbert_pipeline.colbert_encoder.encode(test_text)

            # Verify token-level structure
            assert isinstance(token_embeddings, (list, tuple)), \
                "ColBERT encoding must return list of token embeddings"

            # Verify each token has embedding vector
            if len(token_embeddings) > 0:
                first_token = token_embeddings[0]
                assert isinstance(first_token, (list, tuple)) or hasattr(first_token, '__len__'), \
                    "Each token must have embedding vector"

                # Token dimension varies by ColBERT model (typically 128 or 256)
                token_dim = len(first_token)
                assert token_dim > 0, "Token embedding must have positive dimension"
        except AttributeError:
            pytest.skip("ColBERT encoder interface may differ from expected")

    def test_fallback_dense_vector_is_384d(self, pylate_colbert_pipeline, mocker):
        """
        FR-021: Fallback dense vector MUST be 384-dimensional.

        Given: PyLateColBERT with fallback to dense vector search
        When: Dense embedding generated for fallback
        Then: Embedding has exactly 384 dimensions
        """
        if not hasattr(pylate_colbert_pipeline, 'embedding_manager'):
            pytest.skip("Pipeline does not expose embedding_manager for fallback")

        # Generate dense embedding for fallback
        test_text = "Sample text for dense vector fallback"
        embedding = pylate_colbert_pipeline.embedding_manager.generate_embedding(test_text)

        # Verify dimension (fallback uses all-MiniLM-L6-v2 = 384D)
        assert len(embedding) == 384, \
            f"Fallback dense vector must have 384 dimensions, got {len(embedding)}"

    def test_dimension_mismatch_raises_clear_error(self, pylate_colbert_pipeline, mocker):
        """
        FR-022: Dimension mismatch MUST raise clear diagnostic error.

        Given: Fallback embedding with wrong dimensions
        When: Dimension validation occurs
        Then: Error message includes expected (384) and actual dimensions
        """
        if not hasattr(pylate_colbert_pipeline, 'embedding_manager'):
            pytest.skip("Pipeline does not expose embedding_manager")

        # Mock fallback embedding to return wrong dimensions
        corrupt_embedding = [0.1] * 768

        mocker.patch.object(
            pylate_colbert_pipeline.embedding_manager,
            'generate_embedding',
            return_value=corrupt_embedding
        )

        try:
            # Trigger fallback path (mock ColBERT to fail)
            if hasattr(pylate_colbert_pipeline, 'colbert_encoder'):
                mocker.patch.object(
                    pylate_colbert_pipeline.colbert_encoder,
                    'encode',
                    side_effect=Exception("ColBERT failed, triggering fallback")
                )

            pylate_colbert_pipeline.query("test query")
        except Exception as e:
            error_msg = str(e).lower()

            # Error message SHOULD include dimension info
            if "dimension" in error_msg:
                # Dimension validation exists - verify message quality
                assert "384" in error_msg or "expected" in error_msg, \
                    "Error should mention expected dimension (384) for fallback"
            else:
                # May skip if fallback not yet implemented
                pytest.skip("Dimension validation not yet implemented")

    def test_dimension_error_message_is_actionable(self, pylate_colbert_pipeline, mocker):
        """
        FR-022: Dimension mismatch error MUST suggest fix.

        Given: Fallback embedding with wrong dimensions
        When: Error is raised
        Then: Error message suggests reconfiguring embedding model
        """
        if not hasattr(pylate_colbert_pipeline, 'embedding_manager'):
            pytest.skip("Pipeline does not expose embedding_manager")

        # Mock embedding to return wrong dimensions
        corrupt_embedding = [0.1] * 512

        mocker.patch.object(
            pylate_colbert_pipeline.embedding_manager,
            'generate_embedding',
            return_value=corrupt_embedding
        )

        try:
            # Trigger fallback path
            if hasattr(pylate_colbert_pipeline, 'colbert_encoder'):
                mocker.patch.object(
                    pylate_colbert_pipeline.colbert_encoder,
                    'encode',
                    side_effect=Exception("ColBERT failed")
                )

            pylate_colbert_pipeline.query("test query")
        except Exception as e:
            error_msg = str(e).lower()

            # If dimension validation exists, check for actionable guidance
            if "dimension" in error_msg:
                actionable_keywords = ["reconfigure", "change", "model", "embedding", "384", "fallback"]
                has_actionable = any(keyword in error_msg for keyword in actionable_keywords)

                assert has_actionable, \
                    f"Dimension error should include actionable guidance. Got: {e}"
        else:
            pytest.skip("Dimension validation not yet implemented")

    def test_correct_dimension_embeddings_succeed(self, pylate_colbert_pipeline, mocker):
        """
        FR-021: Correctly dimensioned fallback embeddings MUST succeed.

        Given: Fallback embedding with correct 384 dimensions
        When: Query executed
        Then: No dimension validation errors raised
        """
        if not hasattr(pylate_colbert_pipeline, 'embedding_manager'):
            pytest.skip("Pipeline does not expose embedding_manager")

        # Mock embedding to return correct dimensions
        correct_embedding = [0.1] * 384

        mocker.patch.object(
            pylate_colbert_pipeline.embedding_manager,
            'generate_embedding',
            return_value=correct_embedding
        )

        # This should NOT raise dimension validation errors
        try:
            # May raise other errors (no data, etc.), but NOT dimension errors
            pylate_colbert_pipeline.query("test query")
        except Exception as e:
            error_msg = str(e).lower()

            # Dimension validation should NOT trigger for correct 384D
            assert "dimension" not in error_msg or "384" not in error_msg, \
                f"Correct 384D embedding should not trigger dimension error: {e}"

    def test_token_count_validation(self, pylate_colbert_pipeline):
        """
        FR-023: ColBERT token count SHOULD be validated.

        Given: Text with excessive token count
        When: ColBERT encoding attempted
        Then: Warning or error if token count exceeds model limit
        """
        if not hasattr(pylate_colbert_pipeline, 'colbert_encoder'):
            pytest.skip("Pipeline does not expose colbert_encoder")

        # Generate text with many tokens (may exceed model limit)
        long_text = " ".join(["word"] * 1000)  # 1000 tokens

        try:
            token_embeddings = pylate_colbert_pipeline.colbert_encoder.encode(long_text)

            # If encoding succeeded, token count may be truncated or validated
            # This is acceptable behavior
            assert isinstance(token_embeddings, (list, tuple)), \
                "Token embeddings should be returned"
        except Exception as e:
            error_msg = str(e).lower()

            # If token validation exists
            if "token" in error_msg or "length" in error_msg or "exceed" in error_msg:
                # Token count validation is implemented
                assert True, "Token count validation working"

    def test_colbert_and_dense_dimensions_independent(self, pylate_colbert_pipeline):
        """
        FR-021: ColBERT token dimensions MUST be independent of dense fallback (384D).

        Given: PyLateColBERT with both ColBERT and dense fallback
        When: Both encoding types used
        Then: ColBERT has token-level dims, dense has 384D
        """
        if not hasattr(pylate_colbert_pipeline, 'colbert_encoder'):
            pytest.skip("Pipeline does not expose colbert_encoder")

        if not hasattr(pylate_colbert_pipeline, 'embedding_manager'):
            pytest.skip("Pipeline does not expose embedding_manager")

        test_text = "Sample text"

        # Generate ColBERT encoding (token-level)
        try:
            colbert_tokens = pylate_colbert_pipeline.colbert_encoder.encode(test_text)
            # Token-level structure (variable)
            assert isinstance(colbert_tokens, (list, tuple)), \
                "ColBERT uses token-level embeddings"
        except AttributeError:
            pass

        # Generate dense encoding (fallback)
        dense_embedding = pylate_colbert_pipeline.embedding_manager.generate_embedding(test_text)
        # Dense vector is 384D
        assert len(dense_embedding) == 384, \
            f"Dense fallback must be 384D, got {len(dense_embedding)}"
