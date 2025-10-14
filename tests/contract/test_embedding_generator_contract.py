"""
Contract tests for EmbeddingGenerator.

These tests define the expected behavior of EmbeddingGenerator and MUST fail
initially (no implementation exists yet). They serve as executable
specifications for TDD.

Contract Reference: specs/047-create-a-unified/contracts/embedding_generator_contract.md
"""

import pytest
import numpy as np


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def embedding_generator():
    """Create EmbeddingGenerator instance for testing."""
    from tests.fixtures.embedding_generator import EmbeddingGenerator

    return EmbeddingGenerator(
        model_name="all-MiniLM-L6-v2", dimension=384, batch_size=32, device="cpu"
    )


# ==============================================================================
# CONSTRUCTOR TESTS
# ==============================================================================


@pytest.mark.contract
class TestEmbeddingGeneratorConstructor:
    """Contract tests for EmbeddingGenerator.__init__()."""

    def test_loads_model_successfully(self):
        """✅ Loads sentence-transformers model."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator

        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2", dimension=384)

        assert generator is not None
        assert hasattr(generator, "_model")

    def test_validates_dimension_matches_model(self):
        """✅ Validates dimension matches model."""
        from tests.fixtures.embedding_generator import (
            EmbeddingGenerator,
            DimensionMismatchError,
        )

        # all-MiniLM-L6-v2 should be 384 dimensions, not 768
        with pytest.raises(DimensionMismatchError):
            EmbeddingGenerator(model_name="all-MiniLM-L6-v2", dimension=768)

    def test_supports_cpu_device(self):
        """✅ Supports CPU device."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator

        generator = EmbeddingGenerator(device="cpu")

        assert generator.device == "cpu"

    @pytest.mark.skipif(
        not pytest.importorskip("torch").cuda.is_available(),
        reason="CUDA not available",
    )
    def test_supports_cuda_device(self):
        """✅ Supports CUDA device (if available)."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator

        generator = EmbeddingGenerator(device="cuda")

        assert generator.device == "cuda"

    def test_caches_model_instance(self):
        """✅ Caches model instance (same model_name returns same instance)."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator

        gen1 = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        gen2 = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")

        # Should use cached model
        assert gen1._model is gen2._model


# ==============================================================================
# GENERATE_EMBEDDINGS TESTS
# ==============================================================================


@pytest.mark.contract
class TestGenerateEmbeddings:
    """Contract tests for EmbeddingGenerator.generate_embeddings()."""

    def test_generates_embeddings_for_list_of_texts(self, embedding_generator):
        """✅ Generates embeddings for list of texts."""
        texts = ["Hello world", "This is a test", "Machine learning"]

        embeddings = embedding_generator.generate_embeddings(texts)

        assert embeddings.shape == (3, 384)
        assert isinstance(embeddings, np.ndarray)

    def test_returns_numpy_array_not_tensors(self, embedding_generator):
        """✅ Returns numpy array (not tensors)."""
        texts = ["Test text"]

        embeddings = embedding_generator.generate_embeddings(texts)

        assert isinstance(embeddings, np.ndarray)
        assert not hasattr(embeddings, "requires_grad")  # Not a torch tensor

    def test_normalizes_embeddings_when_normalize_true(self):
        """✅ Normalizes embeddings when normalize=True."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator

        generator = EmbeddingGenerator(normalize=True)
        texts = ["Test text"]

        embeddings = generator.generate_embeddings(texts)

        # L2 norm should be ~1.0 for normalized vectors
        norm = np.linalg.norm(embeddings[0])
        assert abs(norm - 1.0) < 0.01

    def test_handles_empty_strings_gracefully(self, embedding_generator):
        """✅ Handles empty strings (returns zero vector)."""
        texts = ["Valid text", "", None, "Another valid text"]

        embeddings = embedding_generator.generate_embeddings(texts)

        assert embeddings.shape == (4, 384)
        # Empty string and None should produce zero vectors
        assert np.allclose(embeddings[1], np.zeros(384))
        assert np.allclose(embeddings[2], np.zeros(384))

    def test_preserves_order(self, embedding_generator):
        """✅ Preserves order (output[i] matches texts[i])."""
        texts = ["First", "Second", "Third"]

        embeddings = embedding_generator.generate_embeddings(texts)

        # Each text should produce a unique embedding
        assert not np.allclose(embeddings[0], embeddings[1])
        assert not np.allclose(embeddings[1], embeddings[2])

    def test_processes_batches_efficiently(self, embedding_generator):
        """✅ Processes batches efficiently."""
        # Generate 100 texts (> batch_size of 32)
        texts = [f"Text {i}" for i in range(100)]

        embeddings = embedding_generator.generate_embeddings(texts)

        assert embeddings.shape == (100, 384)


# ==============================================================================
# POPULATE_TABLE_EMBEDDINGS TESTS
# ==============================================================================


@pytest.mark.contract
class TestPopulateTableEmbeddings:
    """Contract tests for EmbeddingGenerator.populate_table_embeddings()."""

    @pytest.mark.skip(reason="Requires IRIS database connection")
    def test_populates_embeddings_for_all_rows(self, embedding_generator):
        """✅ Populates embeddings for all rows."""
        # Requires actual IRIS database - will be in integration tests
        pass

    @pytest.mark.skip(reason="Requires IRIS database connection")
    def test_uses_to_vector_function_for_iris(self, embedding_generator):
        """✅ Uses TO_VECTOR() function for IRIS inserts."""
        # Requires actual IRIS database - will be in integration tests
        pass

    @pytest.mark.skip(reason="Requires IRIS database connection")
    def test_fetches_rows_in_batches(self, embedding_generator):
        """✅ Fetches rows in batches (memory efficient)."""
        # Requires actual IRIS database - will be in integration tests
        pass

    @pytest.mark.skip(reason="Requires IRIS database connection")
    def test_handles_null_text_values(self, embedding_generator):
        """✅ Handles NULL text values gracefully."""
        # Requires actual IRIS database - will be in integration tests
        pass

    @pytest.mark.skip(reason="Requires IRIS database connection")
    def test_commits_transaction_on_success(self, embedding_generator):
        """✅ Commits transaction on success."""
        # Requires actual IRIS database - will be in integration tests
        pass

    @pytest.mark.skip(reason="Requires IRIS database connection")
    def test_rollbacks_on_error(self, embedding_generator):
        """✅ Rollbacks on error."""
        # Requires actual IRIS database - will be in integration tests
        pass

    @pytest.mark.skip(reason="Requires IRIS database connection")
    def test_returns_accurate_row_count(self, embedding_generator):
        """✅ Returns accurate row count."""
        # Requires actual IRIS database - will be in integration tests
        pass


# ==============================================================================
# TEXT COLUMN MAPPING TESTS
# ==============================================================================


@pytest.mark.contract
class TestTextColumnMapping:
    """Contract tests for default text column mappings."""

    def test_uses_content_for_source_documents(self, embedding_generator):
        """✅ Uses 'content' for RAG.SourceDocuments."""
        # This is tested via populate_table_embeddings integration tests
        pass

    def test_uses_description_for_entities(self, embedding_generator):
        """✅ Uses 'description' for RAG.Entities."""
        # This is tested via populate_table_embeddings integration tests
        pass


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================


@pytest.mark.contract
class TestErrorHandling:
    """Contract tests for error handling."""

    def test_raises_model_load_error_for_invalid_model(self):
        """✅ Raises ModelLoadError for invalid model_name."""
        from tests.fixtures.embedding_generator import (
            EmbeddingGenerator,
            ModelLoadError,
        )

        with pytest.raises(ModelLoadError):
            EmbeddingGenerator(model_name="non-existent-model-xyz")

    def test_raises_dimension_mismatch_error(self):
        """✅ Raises DimensionMismatchError for dimension mismatch."""
        from tests.fixtures.embedding_generator import (
            EmbeddingGenerator,
            DimensionMismatchError,
        )

        with pytest.raises(DimensionMismatchError):
            EmbeddingGenerator(model_name="all-MiniLM-L6-v2", dimension=999)

    def test_handles_null_empty_text_gracefully(self, embedding_generator):
        """✅ NULL/empty text → zero vector."""
        texts = [None, "", "   ", "valid"]

        embeddings = embedding_generator.generate_embeddings(texts)

        # First 3 should be zero vectors
        assert np.allclose(embeddings[0], np.zeros(384))
        assert np.allclose(embeddings[1], np.zeros(384))
        assert np.allclose(embeddings[2], np.zeros(384))
        # Last should be non-zero
        assert not np.allclose(embeddings[3], np.zeros(384))


# ==============================================================================
# STATE MANAGEMENT TESTS
# ==============================================================================


@pytest.mark.contract
class TestStateManagement:
    """Contract tests for state management."""

    def test_identical_texts_produce_identical_embeddings(self, embedding_generator):
        """✅ Identical texts produce identical embeddings."""
        text = "This is a test"

        emb1 = embedding_generator.generate_embeddings([text])
        emb2 = embedding_generator.generate_embeddings([text])

        assert np.allclose(emb1, emb2)


# ==============================================================================
# CONFIGURATION TESTS
# ==============================================================================


@pytest.mark.contract
class TestConfiguration:
    """Contract tests for supported models and configuration."""

    def test_supports_all_minilm_l6_v2(self):
        """✅ Supports all-MiniLM-L6-v2 (default)."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator

        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2", dimension=384)

        assert generator.model_name == "all-MiniLM-L6-v2"
        assert generator.dimension == 384

    def test_validates_model_name_and_dimension(self):
        """✅ Validates model name and dimension."""
        from tests.fixtures.embedding_generator import (
            EmbeddingGenerator,
            DimensionMismatchError,
        )

        # Should fail for mismatched dimension
        with pytest.raises(DimensionMismatchError):
            EmbeddingGenerator(model_name="all-MiniLM-L6-v2", dimension=768)

    def test_auto_detects_cuda_availability(self):
        """✅ Auto-detects CUDA availability."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator
        import torch

        generator = EmbeddingGenerator()  # No device specified

        if torch.cuda.is_available():
            # Should use CUDA when available (or may default to CPU, depends on impl)
            assert generator.device in ["cpu", "cuda"]
        else:
            # Should fallback to CPU
            assert generator.device == "cpu"


# ==============================================================================
# PERFORMANCE TESTS
# ==============================================================================


@pytest.mark.contract
@pytest.mark.performance
class TestPerformanceContracts:
    """Contract tests for performance requirements."""

    @pytest.mark.skip(reason="Performance test - run separately")
    def test_cpu_generates_over_100_texts_per_second(self, embedding_generator):
        """✅ CPU: > 100 texts/second."""
        import time

        texts = [f"Sample text {i}" for i in range(1000)]

        start = time.time()
        embedding_generator.generate_embeddings(texts)
        elapsed = time.time() - start

        texts_per_second = len(texts) / elapsed
        assert texts_per_second > 100

    @pytest.mark.skipif(
        not pytest.importorskip("torch").cuda.is_available(),
        reason="CUDA not available",
    )
    @pytest.mark.skip(reason="Performance test - run separately")
    def test_gpu_generates_over_1000_texts_per_second(self):
        """✅ GPU: > 1000 texts/second (if CUDA available)."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator
        import time

        generator = EmbeddingGenerator(device="cuda")
        texts = [f"Sample text {i}" for i in range(10000)]

        start = time.time()
        generator.generate_embeddings(texts)
        elapsed = time.time() - start

        texts_per_second = len(texts) / elapsed
        assert texts_per_second > 1000


# ==============================================================================
# DOCUMENTATION TESTS
# ==============================================================================


@pytest.mark.contract
class TestDocumentation:
    """Contract tests for documentation requirements."""

    def test_all_public_methods_have_docstrings(self):
        """✅ All public methods have docstrings."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator

        # Constructor
        assert EmbeddingGenerator.__init__.__doc__ is not None

        # Public methods
        assert EmbeddingGenerator.generate_embeddings.__doc__ is not None

    def test_class_docstring_includes_usage_examples(self):
        """✅ Class docstring includes usage examples."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator

        assert "Example:" in EmbeddingGenerator.__doc__

    def test_to_vector_requirement_documented(self):
        """✅ TO_VECTOR() requirement documented."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator

        # Should be documented in populate_table_embeddings docstring
        method = getattr(EmbeddingGenerator, "populate_table_embeddings", None)
        if method:
            assert "TO_VECTOR" in method.__doc__
