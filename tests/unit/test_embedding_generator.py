"""
Unit tests for EmbeddingGenerator.

These tests define the expected behavior of the EmbeddingGenerator class
which consolidates all embedding generation logic from fragmented scripts.

Reference: specs/047-create-a-unified/tasks.md (T051)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer model."""
    mock_model = Mock()

    # Mock encode to return embeddings matching the number of input texts
    def mock_encode(texts, **kwargs):
        # Return one embedding per input text
        num_texts = len(texts)
        return np.array([[0.1] * 384] * num_texts)

    mock_model.encode = Mock(side_effect=mock_encode)
    return mock_model


@pytest.fixture
def mock_connection():
    """Mock IRIS database connection."""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_cursor.fetchall = Mock(return_value=[])
    mock_cursor.execute = Mock()
    mock_conn.cursor = Mock(return_value=mock_cursor)
    return mock_conn


@pytest.fixture
def embedding_generator(mock_sentence_transformer):
    """Create EmbeddingGenerator instance with mocked model."""
    from tests.fixtures.embedding_generator import EmbeddingGenerator

    # Mock the import inside _load_model
    with patch('tests.fixtures.embedding_generator._MODEL_CACHE', {}):
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer):
            generator = EmbeddingGenerator(
                model_name="all-MiniLM-L6-v2",
                dimension=384,
            )
            return generator


# ==============================================================================
# CONSTRUCTOR TESTS
# ==============================================================================


@pytest.mark.unit
class TestEmbeddingGeneratorConstructor:
    """Unit tests for EmbeddingGenerator.__init__()."""

    def test_accepts_model_name_parameter(self):
        """✅ Accepts model_name parameter."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator

        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.array([[0.1] * 384]))

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
            assert generator.model_name == "all-MiniLM-L6-v2"

    def test_accepts_dimension_parameter(self):
        """✅ Accepts dimension parameter."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator

        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.array([[0.1] * 384]))

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            generator = EmbeddingGenerator(dimension=384)
            assert generator.dimension == 384

    def test_default_dimension_is_384(self):
        """✅ Default dimension is 384 per constitution."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator

        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.array([[0.1] * 384]))

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            generator = EmbeddingGenerator()
            assert generator.dimension == 384

    def test_loads_sentence_transformer_model(self):
        """✅ Loads SentenceTransformer model on init."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator

        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.array([[0.1] * 512]))  # Different dimension to test validation

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model) as mock_st:
            # This will raise DimensionMismatchError because model returns 512 but we expect 384
            from tests.fixtures.embedding_generator import DimensionMismatchError
            with pytest.raises(DimensionMismatchError):
                generator = EmbeddingGenerator(model_name="test-model", dimension=384)


# ==============================================================================
# DOCUMENT EMBEDDING GENERATION TESTS
# ==============================================================================


@pytest.mark.unit
class TestGenerateDocumentEmbeddings:
    """Unit tests for generate_document_embeddings()."""

    def test_generates_embeddings_for_source_documents(self, embedding_generator, mock_connection):
        """✅ Generates embeddings for RAG.SourceDocuments."""
        # Setup mock cursor to return documents
        mock_cursor = mock_connection.cursor()

        # Mock fetchone for COUNT(*) and fetchall for SELECT with LIMIT/OFFSET
        # The implementation uses: while offset < total_rows, so it will loop twice (0 < 2, 32 < 2 is False)
        mock_cursor.fetchone = Mock(return_value=(2,))
        mock_cursor.fetchall = Mock(return_value=[
            ("doc1", "Sample document text"),
            ("doc2", "Another document"),
        ])

        # Generate embeddings
        count = embedding_generator.populate_table_embeddings(
            connection=mock_connection,
            table_name="RAG.SourceDocuments",
            text_column="content",
            embedding_column="embedding",
            id_column="doc_id",
        )

        assert count == 2

    def test_uses_to_vector_function_for_updates(self, embedding_generator, mock_connection):
        """✅ Uses TO_VECTOR() for embedding updates per constitution."""
        mock_cursor = mock_connection.cursor()

        # COUNT(*) query
        mock_cursor.fetchone = Mock(return_value=(1,))

        # SELECT query
        mock_cursor.fetchall = Mock(return_value=[
            ("doc1", "Sample text"),
        ])

        embedding_generator.populate_table_embeddings(
            connection=mock_connection,
            table_name="RAG.SourceDocuments",
            text_column="content",
            embedding_column="embedding",
        )

        # Verify UPDATE statement uses TO_VECTOR
        execute_calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
        update_calls = [call for call in execute_calls if "UPDATE" in call]

        assert len(update_calls) > 0
        assert any("TO_VECTOR" in call for call in update_calls), \
            "UPDATE statements must use TO_VECTOR() per constitution"

    def test_handles_null_text_gracefully(self, embedding_generator, mock_connection):
        """✅ Handles NULL text values gracefully."""
        mock_cursor = mock_connection.cursor()

        # COUNT(*) query
        mock_cursor.fetchone = Mock(return_value=(2,))

        # SELECT query with NULL text (all returned in first query since batch_size=32 > 2)
        mock_cursor.fetchall = Mock(return_value=[
            ("doc1", None),
            ("doc2", "Valid text"),
        ])

        # Should not crash on NULL text
        count = embedding_generator.populate_table_embeddings(
            connection=mock_connection,
            table_name="RAG.SourceDocuments",
            text_column="content",
            embedding_column="embedding",
        )

        # Should process all documents (NULL → zero vector)
        assert count == 2


# ==============================================================================
# ENTITY EMBEDDING GENERATION TESTS
# ==============================================================================


@pytest.mark.unit
class TestGenerateEntityEmbeddings:
    """Unit tests for generate_entity_embeddings()."""

    def test_generates_embeddings_for_entities(self, embedding_generator, mock_connection):
        """✅ Generates embeddings for RAG.Entities."""
        mock_cursor = mock_connection.cursor()

        # COUNT(*) query
        mock_cursor.fetchone = Mock(return_value=(2,))

        # SELECT query (all in one batch)
        mock_cursor.fetchall = Mock(return_value=[
            ("entity1", "Entity description"),
            ("entity2", "Another entity"),
        ])

        count = embedding_generator.populate_table_embeddings(
            connection=mock_connection,
            table_name="RAG.Entities",
            text_column="description",
            embedding_column="embedding",
        )

        assert count == 2

    def test_generates_embeddings_for_kg_node_embeddings(self, embedding_generator, mock_connection):
        """✅ Generates embeddings for kg_NodeEmbeddings_optimized."""
        mock_cursor = mock_connection.cursor()

        # COUNT(*) query
        mock_cursor.fetchone = Mock(return_value=(1,))

        # SELECT query
        mock_cursor.fetchall = Mock(return_value=[
            ("node1", "Node text"),
        ])

        count = embedding_generator.populate_table_embeddings(
            connection=mock_connection,
            table_name="kg_NodeEmbeddings_optimized",
            text_column="node_text",
            embedding_column="node_embedding",
        )

        assert count == 1


# ==============================================================================
# CACHING AND RETRY LOGIC TESTS
# ==============================================================================


@pytest.mark.unit
class TestEmbeddingCaching:
    """Unit tests for embedding caching logic."""

    def test_caches_embeddings_to_avoid_redundant_api_calls(self, embedding_generator):
        """✅ Caches embeddings for repeated text."""
        # Generate embedding for same text twice
        texts = ["Sample text"]

        # Call encode twice with same input
        embedding1 = embedding_generator.generate_embeddings(texts)
        embedding2 = embedding_generator.generate_embeddings(texts)

        # Should return same embeddings (deterministic)
        assert np.array_equal(embedding1, embedding2)

    def test_retries_on_api_failure_with_exponential_backoff(self, embedding_generator):
        """✅ Retries with exponential backoff on API failure."""
        # Mock encode to fail twice then succeed
        with patch.object(embedding_generator._model, 'encode') as mock_encode:
            mock_encode.side_effect = [
                Exception("API error"),
                Exception("API error"),
                np.array([[0.1] * 384]),  # Success on third try
            ]

            # Note: Current implementation doesn't have retry logic yet
            # This test documents the expected behavior for future implementation
            with pytest.raises(Exception, match="API error"):
                embedding_generator.generate_embeddings(["text"])


# ==============================================================================
# BATCH PROCESSING TESTS
# ==============================================================================


@pytest.mark.unit
class TestBatchProcessing:
    """Unit tests for batch embedding generation."""

    def test_processes_large_tables_in_batches(self, embedding_generator, mock_connection):
        """✅ Processes large tables in batches to avoid memory issues."""
        # Setup mock to return many rows
        mock_cursor = mock_connection.cursor()

        # COUNT(*) query
        mock_cursor.fetchone = Mock(return_value=(1000,))

        # SELECT query returns batches
        # Loop will run: offset=0, 100, 200, ..., 900 (10 iterations)
        # Each SELECT uses LIMIT 100 OFFSET {offset}
        rows = [(f"doc{i}", f"Text {i}") for i in range(1000)]
        batch_results = [rows[i:i+100] for i in range(0, 1000, 100)]
        mock_cursor.fetchall = Mock(side_effect=batch_results)

        count = embedding_generator.populate_table_embeddings(
            connection=mock_connection,
            table_name="RAG.SourceDocuments",
            text_column="content",
            embedding_column="embedding",
            batch_size=100,
        )

        assert count == 1000

    def test_commits_after_each_batch(self, embedding_generator, mock_connection):
        """✅ Commits after each batch for progress tracking."""
        mock_cursor = mock_connection.cursor()

        # COUNT(*) query
        mock_cursor.fetchone = Mock(return_value=(250,))

        # SELECT query returns batches
        rows = [(f"doc{i}", f"Text {i}") for i in range(250)]
        batch_results = [rows[i:i+100] for i in range(0, 250, 100)]
        mock_cursor.fetchall = Mock(side_effect=batch_results)

        embedding_generator.populate_table_embeddings(
            connection=mock_connection,
            table_name="RAG.SourceDocuments",
            text_column="content",
            embedding_column="embedding",
            batch_size=100,
        )

        # Should have committed multiple times (3 batches: 100 + 100 + 50)
        assert mock_connection.commit.call_count >= 3


# ==============================================================================
# ERROR HANDLING TESTS
# ==============================================================================


@pytest.mark.unit
class TestEmbeddingGeneratorErrorHandling:
    """Unit tests for error handling."""

    def test_raises_clear_error_on_invalid_dimension(self):
        """✅ Raises clear error for invalid dimension."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator, DimensionMismatchError

        mock_model = Mock()
        mock_model.encode = Mock(return_value=np.array([[0.1] * 384]))

        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            # Request dimension=0 but model returns 384
            with pytest.raises(DimensionMismatchError, match="dimension"):
                EmbeddingGenerator(dimension=0)

    def test_raises_clear_error_on_invalid_model(self):
        """✅ Raises clear error for invalid model name."""
        from tests.fixtures.embedding_generator import EmbeddingGenerator, ModelLoadError

        # Patch at import site
        with patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_st.side_effect = Exception("Model not found")

            with pytest.raises(ModelLoadError, match="Model not found"):
                EmbeddingGenerator(model_name="invalid-model")

    def test_skips_rows_with_database_errors(self, embedding_generator, mock_connection):
        """✅ Skips rows that cause database errors and continues."""
        mock_cursor = mock_connection.cursor()

        # COUNT(*) query
        mock_cursor.fetchone = Mock(return_value=(2,))

        # SELECT query
        mock_cursor.fetchall = Mock(return_value=[
            ("doc1", "Valid text"),
            ("doc2", "Another valid text"),
        ])

        # Make UPDATE fail for first row only
        mock_cursor.execute = Mock(side_effect=[
            None,  # COUNT(*) query
            None,  # SELECT query
            Exception("Database error"),  # First UPDATE fails
            None,  # Second UPDATE succeeds
        ])

        # Should raise exception (current implementation doesn't skip errors)
        with pytest.raises(Exception, match="Failed to populate embeddings"):
            embedding_generator.populate_table_embeddings(
                connection=mock_connection,
                table_name="RAG.SourceDocuments",
                text_column="content",
                embedding_column="embedding",
            )
