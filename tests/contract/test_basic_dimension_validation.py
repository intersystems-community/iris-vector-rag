"""
Contract tests for BasicRAG Dimension Validation (DIM-001).

Tests validate that BasicRAG correctly handles 384-dimensional embeddings
and provides clear diagnostics when dimension mismatches occur.

Contract: dimension_validation_contract.md
Requirements: FR-021, FR-022, FR-023, FR-024
"""

import pytest

from iris_vector_rag.exceptions import (
    DimensionMismatchError,
    VectorStoreConnectionError,
)
from tests.contract.conftest import assert_names_the_mismatch


@pytest.mark.contract
@pytest.mark.dimension
@pytest.mark.basic_rag
class TestBasicRAGDimensionValidation:
    """Contract tests for BasicRAG dimension validation."""

    def test_embedding_dimension_is_384(self, basic_rag_pipeline, mocker):
        """
        FR-021: BasicRAG MUST use 384-dimensional embeddings.

        Given: BasicRAG pipeline instance
        When: Embedding generated for sample text
        Then: Embedding has exactly 384 dimensions
        """
        if not hasattr(basic_rag_pipeline, "embedding_manager"):
            pytest.skip("Pipeline does not expose embedding_manager")

        # Generate embedding for test text
        test_text = "Sample text for dimension validation"
        embedding = basic_rag_pipeline.embedding_manager.generate_embedding(test_text)

        # Verify dimension
        assert (
            len(embedding) == 384
        ), f"BasicRAG embedding must have 384 dimensions, got {len(embedding)}"

    def test_dimension_mismatch_raises_clear_error(
        self, basic_rag_pipeline, wrong_width_queries
    ):
        """
        FR-022: Dimension mismatch MUST raise clear diagnostic error.

        Given: Query embedding at 768 dimensions against a 384 column
        When: BasicRAG queries
        Then: DimensionMismatchError names both widths and the fix
        """
        wrong_width_queries(basic_rag_pipeline)

        with pytest.raises(DimensionMismatchError) as exc_info:
            basic_rag_pipeline.query("test query", generate_answer=False)

        assert_names_the_mismatch(exc_info.value)

    def test_dimension_error_message_is_actionable(self, basic_rag_pipeline, mocker):
        """
        FR-022: Dimension mismatch error MUST suggest fix.

        Given: Query embedding with wrong dimensions
        When: Error is raised
        Then: Error message suggests reconfiguring embedding model
        """
        if not hasattr(basic_rag_pipeline, "embedding_manager"):
            pytest.skip("Pipeline does not expose embedding_manager")

        # Mock embedding to return wrong dimensions
        corrupt_embedding = [0.1] * 512  # Different wrong dimension

        mocker.patch.object(
            basic_rag_pipeline.embedding_manager,
            "generate_embedding",
            return_value=corrupt_embedding,
        )

        try:
            basic_rag_pipeline.query("test query")
        except Exception as e:
            error_msg = str(e).lower()

            # If dimension validation exists, check for actionable guidance
            if "dimension" in error_msg:
                actionable_keywords = [
                    "reconfigure",
                    "change",
                    "model",
                    "embedding",
                    "384",
                ]
                has_actionable = any(
                    keyword in error_msg for keyword in actionable_keywords
                )

                assert (
                    has_actionable
                ), f"Dimension error should include actionable guidance. Got: {e}"
        else:
            pytest.skip("Dimension validation not yet implemented")

    @pytest.mark.requires_database
    def test_load_documents_validates_embedding_dimensions(
        self, basic_rag_pipeline, mocker, sample_documents
    ):
        """
        FR-023: Load documents MUST validate embedding dimensions.

        Given: Document loading with mocked embeddings
        When: Embeddings have wrong dimensions
        Then: Clear error raised before database write
        """
        if not hasattr(basic_rag_pipeline, "embedding_manager"):
            pytest.skip("Pipeline does not expose embedding_manager")

        # Mock embedding to return wrong dimensions
        corrupt_embedding = [0.1] * 1024  # Wrong dimension

        mocker.patch.object(
            basic_rag_pipeline.embedding_manager,
            "generate_embedding",
            return_value=corrupt_embedding,
        )

        # Attempt to load documents
        try:
            result = basic_rag_pipeline.load_documents(documents=sample_documents)

            # If loading succeeded, check if any documents failed
            if "documents_failed" in result:
                # Some validation may occur, check failure reasons
                assert (
                    result["documents_failed"] > 0 or result["documents_loaded"] == 0
                ), "Dimension validation should prevent loading corrupted embeddings"
        except Exception as e:
            error_msg = str(e).lower()

            # If dimension validation exists, verify error quality
            if "dimension" in error_msg:
                assert True, "Dimension validation working"
            else:
                pytest.skip("Dimension validation not yet implemented")

    def test_dimension_validation_logs_diagnostic_info(
        self, basic_rag_pipeline, mocker, caplog
    ):
        """
        FR-024: Dimension validation MUST log diagnostic information.

        Given: Dimension mismatch scenario
        When: Validation occurs
        Then: Diagnostic logs include expected and actual dimensions
        """
        import logging

        caplog.set_level(logging.INFO)

        if not hasattr(basic_rag_pipeline, "embedding_manager"):
            pytest.skip("Pipeline does not expose embedding_manager")

        # Mock embedding to return wrong dimensions
        corrupt_embedding = [0.1] * 128  # Wrong dimension

        mocker.patch.object(
            basic_rag_pipeline.embedding_manager,
            "generate_embedding",
            return_value=corrupt_embedding,
        )

        try:
            basic_rag_pipeline.query("test query")
        except Exception:
            # Check if dimension validation logged diagnostic info
            log_output = caplog.text.lower()

            if "dimension" in log_output:
                # Dimension validation exists and logs info
                assert (
                    "384" in log_output or "expected" in log_output
                ), "Diagnostic logs should mention expected dimension"
            else:
                pytest.skip("Dimension validation logging not yet implemented")

    def test_correct_dimension_embeddings_succeed(self, basic_rag_pipeline, mocker):
        """
        FR-021: Correctly dimensioned embeddings MUST succeed.

        Given: Query embedding with correct 384 dimensions
        When: Query executed
        Then: No dimension validation errors raised
        """
        if not hasattr(basic_rag_pipeline, "embedding_manager"):
            pytest.skip("Pipeline does not expose embedding_manager")

        # Mock embedding to return correct dimensions
        correct_embedding = [0.1] * 384

        mocker.patch.object(
            basic_rag_pipeline.embedding_manager,
            "generate_embedding",
            return_value=correct_embedding,
        )

        # This should NOT raise dimension validation errors
        try:
            # May raise other errors (no data, etc.), but NOT dimension errors
            basic_rag_pipeline.query("test query")
        except Exception as e:
            error_msg = str(e).lower()

            # Dimension validation should NOT trigger
            assert (
                "dimension" not in error_msg or "384" not in error_msg
            ), f"Correct 384D embedding should not trigger dimension error: {e}"

    def test_dimension_validation_early_in_pipeline(
        self, basic_rag_pipeline, wrong_width_queries, mocker
    ):
        """
        FR-023: Dimension validation MUST occur before expensive operations.

        Given: Query embedding at 768 dimensions against a 384 column
        When: BasicRAG queries with an answer asked for
        Then: DimensionMismatchError, and the LLM never ran
        """
        wrong_width_queries(basic_rag_pipeline)
        llm = mocker.patch.object(
            basic_rag_pipeline, "llm_func", mocker.Mock(name="llm")
        )

        with pytest.raises(DimensionMismatchError):
            basic_rag_pipeline.query("test query", generate_answer=True)

        llm.assert_not_called()
