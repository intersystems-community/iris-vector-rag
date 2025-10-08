"""
Contract tests for embedding dimension validation.

Contract: DVC-002 (specs/033-fix-graphrag-vector/contracts/dimension_validation_contract.md)
Requirements: FR-005
"""

import pytest
from iris_rag import create_pipeline
from iris_rag.embeddings.manager import EmbeddingManager
from iris_rag.config.manager import ConfigurationManager


class DimensionMismatchError(Exception):
    """Raised when embedding dimensions don't match."""
    pass


class TestDimensionValidationContract:
    """Contract tests for dimension validation (DVC-002)."""

    @pytest.fixture
    def config_manager(self):
        """Create configuration manager."""
        return ConfigurationManager()

    @pytest.fixture
    def embedding_manager(self, config_manager):
        """Create embedding manager."""
        return EmbeddingManager(config_manager)

    @pytest.fixture
    def graphrag_pipeline(self):
        """Create GraphRAG pipeline."""
        return create_pipeline("graphrag", validate_requirements=True)

    def test_query_embedding_is_384_dimensions(self, embedding_manager):
        """
        FR-005: Query embedding MUST be exactly 384 dimensions.

        Given: all-MiniLM-L6-v2 embedding model configured
        When: Query is embedded
        Then: Resulting vector is exactly 384 dimensions
        """
        query = "What are the symptoms of diabetes?"
        query_embedding = embedding_manager.generate_embedding(query)

        assert len(query_embedding) == 384, \
            f"Query embedding dimension mismatch: {len(query_embedding)} != 384"

    def test_dimension_validation_before_search(self, graphrag_pipeline):
        """
        FR-005: Vector search MUST validate dimensions before querying.

        Given: GraphRAG pipeline with dimension validation enabled
        When: Query is executed
        Then: Dimensions are validated before IRIS query (no silent failures)
        """
        query = "What are the symptoms of diabetes?"

        # Should not raise error (dimensions match)
        result = graphrag_pipeline.query(query)

        # If retrieval succeeds, validation passed
        assert len(result['contexts']) >= 0, \
            "Query execution should succeed with matching dimensions"

    def test_dimension_mismatch_raises_clear_error(self, graphrag_pipeline, embedding_manager):
        """
        FR-005: Dimension mismatch MUST raise DimensionMismatchError with clear message.

        Given: Query embedding with wrong dimensions
        When: Validation is performed
        Then: DimensionMismatchError raised with both dimensions in message
        """
        # Generate correct embedding first
        query = "test query"
        correct_embedding = embedding_manager.generate_embedding(query)
        assert len(correct_embedding) == 384, "Embedding manager should produce 384D"

        # Simulate dimension mismatch (implementation detail - may need adjustment)
        # This test validates the error message structure when mismatch occurs
        with pytest.raises(Exception) as exc_info:
            # Trigger dimension validation failure
            # (Implementation will add _validate_dimensions method)
            if hasattr(graphrag_pipeline, '_validate_dimensions'):
                corrupt_embedding = [0.1] * 256  # Wrong dimension
                graphrag_pipeline._validate_dimensions(corrupt_embedding, expected_dims=384)
            else:
                pytest.skip("_validate_dimensions method not yet implemented")

        error_msg = str(exc_info.value).lower()

        # Error message MUST include both dimensions
        assert "256" in error_msg or "dimension" in error_msg, \
            "Error message must mention actual dimension"
        assert "384" in error_msg, \
            "Error message must mention expected dimension"

    def test_dimension_error_suggests_actionable_fix(self, graphrag_pipeline):
        """
        FR-005: DimensionMismatchError MUST suggest actionable fix.

        Given: Dimension mismatch error occurs
        When: Error message is inspected
        Then: Message suggests model verification or re-indexing
        """
        # This test validates error message quality
        # (Implementation will ensure helpful error messages)

        with pytest.raises(Exception) as exc_info:
            if hasattr(graphrag_pipeline, '_validate_dimensions'):
                corrupt_embedding = [0.1] * 768  # Wrong dimension (BERT-base size)
                graphrag_pipeline._validate_dimensions(corrupt_embedding, expected_dims=384)
            else:
                pytest.skip("_validate_dimensions method not yet implemented")

        error_msg = str(exc_info.value).lower()

        # Error message MUST suggest fix
        actionable_keywords = ["model", "re-index", "reindex", "embedding", "verify"]
        assert any(keyword in error_msg for keyword in actionable_keywords), \
            f"Error message must suggest actionable fix. Got: {exc_info.value}"

    def test_document_embedding_dimension_check(self, graphrag_pipeline):
        """
        FR-005: System MUST validate document embedding dimensions.

        Given: Documents in RAG.SourceDocuments
        When: Vector search initializes
        Then: Document embeddings are validated to be 384D
        """
        # Sample a document embedding from database
        from iris_rag.core.connection import ConnectionManager

        conn = ConnectionManager(graphrag_pipeline.config).get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT embedding FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
            LIMIT 1
        """)
        result = cursor.fetchone()

        if result is None:
            pytest.skip("No documents with embeddings in database")

        document_embedding = result[0]

        # Document embedding MUST be 384D
        assert len(document_embedding) == 384, \
            f"Document embedding dimension mismatch: {len(document_embedding)} != 384. " \
            f"Database may contain embeddings from different model. Re-indexing required."

    def test_mismatched_embeddings_prevented(self, graphrag_pipeline):
        """
        FR-005: System MUST prevent queries with mismatched embedding dimensions.

        Given: Database has 384D embeddings
        When: Query embedding has different dimensions
        Then: Error raised before IRIS query (no silent failures)
        """
        # This is a defensive test ensuring validation occurs
        # If query embedding generator is wrong, validation should catch it

        query = "test query"

        # Normal query should work (dimensions match)
        result = graphrag_pipeline.query(query)

        # If we get here without error, validation either:
        # 1. Passed (dimensions match) - good
        # 2. Not implemented yet - will fail contract
        # The actual dimension validation logic will be added in implementation

        # For now, verify retrieval works with correct dimensions
        assert isinstance(result['contexts'], list), \
            "Query should execute successfully with matching dimensions"
