"""
Unit tests for AUD-002 bug fix: proper error handling for embedding and retrieval failures.

Tests verify that:
1. Embedding failures raise EmbeddingError (not zero vectors)
2. Retrieval failures are captured in response.error (not empty list)
3. Generation failures are captured in response.error (not placeholder strings)
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from iris_vector_rag.core.models import Document
from iris_vector_rag.exceptions import EmbeddingError, RetrievalError, GenerationError
from iris_vector_rag.pipelines.basic import BasicRAGPipeline
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager


class TestEmbeddingErrorHandling:
    """Tests for proper EmbeddingError propagation (Part A)."""

    def test_embedding_failure_raises_error_not_zero_vector(self):
        """
        Test that embedding generation failure raises EmbeddingError
        instead of returning zero vectors.
        """
        # Setup
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        vector_store = IRISVectorStore(connection_manager, config_manager)

        # Create test documents
        documents = [
            Document(page_content="Test content 1", metadata={}),
            Document(page_content="Test content 2", metadata={}),
        ]

        # Patch EmbeddingManager where it's imported in the module
        with patch(
            "iris_vector_rag.embeddings.manager.EmbeddingManager"
        ) as mock_embedding_class:
            mock_embedding_manager = MagicMock()
            mock_embedding_class.return_value = mock_embedding_manager
            mock_embedding_manager.embed_text.side_effect = RuntimeError(
                "Embedding model failed"
            )

            # Assert: EmbeddingError is raised, not zero vectors
            with pytest.raises(EmbeddingError) as exc_info:
                vector_store._generate_embeddings(documents)

            # Verify error details
            error = exc_info.value
            assert "Failed to generate embeddings" in str(error)
            assert error.details["doc_count"] == 2
            assert error.details["error_type"] == "RuntimeError"

    def test_embedding_error_has_context(self):
        """Test that EmbeddingError includes machine-readable context."""
        error = EmbeddingError(
            "Model initialization failed",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="gpu",
        )
        assert error.message == "Model initialization failed"
        assert error.details["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"
        assert error.details["device"] == "gpu"


class TestRetrievalErrorHandling:
    """Tests for proper retrieval error handling (Part B)."""

    def test_retrieval_failure_captured_in_response_error(self):
        """
        Test that retrieval failure is captured in response["error"]
        instead of being swallowed to an empty list.
        """
        # Setup
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)

        # Create pipeline with mocked vector store that raises
        pipeline = BasicRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
        )

        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.side_effect = RuntimeError(
            "Database connection failed"
        )
        pipeline.vector_store = mock_vector_store

        # Execute query
        result = pipeline.query("What is diabetes?", generate_answer=False)

        # Assert: error is captured, retrieved_documents is empty, error != None
        assert result["retrieved_documents"] == []
        assert result["error"] is not None
        assert result["error"]["type"] == "RetrievalError"
        assert "Database connection failed" in result["error"]["message"]
        assert result["error"]["error_class"] == "RuntimeError"

    def test_empty_result_has_no_error(self):
        """
        Test that legitimate empty results (zero hits) have error=None,
        distinguishing them from actual failures.
        """
        # Setup
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)

        pipeline = BasicRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
        )

        # Mock vector store that returns empty list (legitimate case)
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = []  # Valid empty result
        pipeline.vector_store = mock_vector_store

        # Execute query
        result = pipeline.query("unlikely query", generate_answer=False)

        # Assert: no error for legitimate empty result
        assert result["retrieved_documents"] == []
        assert result["error"] is None  # Key: error is None, not an error dict
        assert result["answer"] is None

    def test_retrieval_error_has_context(self):
        """Test that RetrievalError includes machine-readable context."""
        error = RetrievalError(
            "Similarity search failed",
            query="medical query",
            top_k=5,
            store_type="iris",
        )
        assert error.message == "Similarity search failed"
        assert error.details["query"] == "medical query"
        assert error.details["top_k"] == 5
        assert error.details["store_type"] == "iris"


class TestGenerationErrorHandling:
    """Tests for proper generation error handling (Part B)."""

    def test_generation_failure_captured_in_response_error(self):
        """
        Test that generation failure is captured in response["error"]
        instead of returning "Error generating answer" placeholder.
        """
        # Setup
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)

        pipeline = BasicRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
        )

        # Mock LLM that raises
        def failing_llm(text):
            raise RuntimeError("OpenAI API timeout")

        pipeline.llm_func = failing_llm

        # Mock vector store that returns docs
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = [
            Document(page_content="Retrieved content", metadata={})
        ]
        pipeline.vector_store = mock_vector_store

        # Execute query with generation enabled
        result = pipeline.query("What is diabetes?", generate_answer=True)

        # Assert: error is captured, answer is None (not placeholder string)
        assert result["answer"] is None
        assert result["error"] is not None
        assert result["error"]["type"] == "GenerationError"
        assert "OpenAI API timeout" in result["error"]["message"]
        assert result["error"]["error_class"] == "RuntimeError"
        # Key: answer is None, not "Error generating answer"

    def test_generation_with_docs_but_no_error(self):
        """
        Test that successful generation with docs and no error
        produces a valid answer.
        """
        # Setup
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)

        pipeline = BasicRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
        )

        # Mock LLM that succeeds
        def working_llm(text):
            return "Diabetes is a metabolic disease."

        pipeline.llm_func = working_llm

        # Mock vector store
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = [
            Document(page_content="Diabetes info", metadata={})
        ]
        pipeline.vector_store = mock_vector_store

        # Execute query
        result = pipeline.query("What is diabetes?", generate_answer=True)

        # Assert: successful generation
        assert result["answer"] == "Diabetes is a metabolic disease."
        assert result["error"] is None
        assert len(result["retrieved_documents"]) == 1

    def test_generation_error_has_context(self):
        """Test that GenerationError includes machine-readable context."""
        error = GenerationError(
            "LLM call timed out",
            llm_model="gpt-4",
            query="complex medical question",
            doc_count=5,
        )
        assert error.message == "LLM call timed out"
        assert error.details["llm_model"] == "gpt-4"
        assert error.details["query"] == "complex medical question"
        assert error.details["doc_count"] == 5


class TestComprehensiveErrorScenarios:
    """Integration tests for complex error scenarios."""

    def test_retrieval_error_prevents_generation_attempt(self):
        """
        Test that if retrieval fails, generation is not attempted,
        and both errors are not present.
        """
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)

        pipeline = BasicRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
        )

        # Mock retrieval failure
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.side_effect = RuntimeError(
            "Vector DB offline"
        )
        pipeline.vector_store = mock_vector_store

        # Mock LLM (should not be called)
        gen_call_count = 0

        def tracking_llm(text):
            nonlocal gen_call_count
            gen_call_count += 1
            raise RuntimeError("Should not be called")

        pipeline.llm_func = tracking_llm

        # Execute query
        result = pipeline.query("Query", generate_answer=True)

        # Assert: only retrieval error, generation not attempted
        assert result["error"] is not None
        assert result["error"]["type"] == "RetrievalError"
        assert result["answer"] is None
        assert gen_call_count == 0  # LLM never called

    def test_successful_path_has_no_error(self):
        """Test that successful execution has error=None."""
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)

        pipeline = BasicRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
        )

        # Mock successful retrieval
        mock_vector_store = MagicMock()
        mock_vector_store.similarity_search.return_value = [
            Document(page_content="Relevant content", metadata={})
        ]
        pipeline.vector_store = mock_vector_store

        # Mock successful generation
        pipeline.llm_func = lambda text: "Generated answer"

        # Execute query
        result = pipeline.query("Query", generate_answer=True)

        # Assert: successful path
        assert result["answer"] == "Generated answer"
        assert result["error"] is None
        assert len(result["retrieved_documents"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
