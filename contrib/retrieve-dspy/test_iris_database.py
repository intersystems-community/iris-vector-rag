"""Tests for IRIS database adapter."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from retrieve_dspy.models import ObjectFromDB


class TestIRISSearchTool:
    """Tests for iris_search_tool function."""

    def test_returns_object_from_db_list(self):
        """Test that iris_search_tool returns list of ObjectFromDB."""
        from retrieve_dspy.database.iris_database import iris_search_tool

        # Mock IRIS connection and cursor
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor

        # Mock database results: (id, content, score, vector)
        mock_cursor.fetchall.return_value = [
            ('doc_1', 'Sample document about diabetes symptoms', 0.95, None),
            ('doc_2', 'Another document about treatment', 0.85, None),
            ('doc_3', 'Third document about prevention', 0.75, None),
        ]

        # Call function with mocked connection
        with patch('retrieve_dspy.database.iris_database._get_query_embedding') as mock_embed:
            mock_embed.return_value = [0.1] * 384  # Mock embedding

            results = iris_search_tool(
                query="diabetes symptoms",
                collection_name="RAG.Documents",
                target_property_name="text_content",
                iris_connection=mock_connection,
                retrieved_k=3
            )

        # Assertions
        assert len(results) == 3
        assert all(isinstance(obj, ObjectFromDB) for obj in results)

        # Check first result
        assert results[0].object_id == 'doc_1'
        assert results[0].content == 'Sample document about diabetes symptoms'
        assert results[0].relevance_score == 0.95
        assert results[0].relevance_rank == 1
        assert results[0].vector is None

        # Check ranking
        assert results[1].relevance_rank == 2
        assert results[2].relevance_rank == 3

    def test_returns_vectors_when_requested(self):
        """Test that vectors are returned when return_vector=True."""
        from retrieve_dspy.database.iris_database import iris_search_tool

        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor

        # Mock results with vector (as comma-separated string)
        vector_string = ','.join(str(x) for x in [0.1, 0.2, 0.3])
        mock_cursor.fetchall.return_value = [
            ('doc_1', 'Content', 0.9, vector_string),
        ]

        with patch('retrieve_dspy.database.iris_database._get_query_embedding') as mock_embed:
            mock_embed.return_value = [0.1] * 3

            results = iris_search_tool(
                query="test",
                collection_name="RAG.Documents",
                target_property_name="text_content",
                iris_connection=mock_connection,
                retrieved_k=1,
                return_vector=True
            )

        assert len(results) == 1
        assert results[0].vector is not None
        assert len(results[0].vector) == 3
        assert results[0].vector == [0.1, 0.2, 0.3]

    def test_handles_tag_filter(self):
        """Test that tag filtering is applied in SQL."""
        from retrieve_dspy.database.iris_database import iris_search_tool

        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        with patch('retrieve_dspy.database.iris_database._get_query_embedding') as mock_embed:
            mock_embed.return_value = [0.1] * 384

            iris_search_tool(
                query="test",
                collection_name="RAG.Documents",
                target_property_name="text_content",
                iris_connection=mock_connection,
                retrieved_k=5,
                tag_filter_value="medical"
            )

        # Check that SQL was executed with tag filter
        call_args = mock_cursor.execute.call_args[0][0]
        assert "tags LIKE '%medical%'" in call_args


class TestVectorSearch:
    """Tests for _vector_search helper function."""

    def test_builds_correct_sql(self):
        """Test that correct SQL is generated."""
        from retrieve_dspy.database.iris_database import _vector_search

        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        embedding = [0.1, 0.2, 0.3]

        _vector_search(
            mock_connection,
            "RAG.Documents",
            "text_content",
            embedding,
            top_k=5,
            tag_filter=None,
            return_vector=False
        )

        # Get the SQL that was executed
        sql = mock_cursor.execute.call_args[0][0]

        # Check SQL structure
        assert "SELECT" in sql
        assert "VECTOR_COSINE" in sql
        assert "TO_VECTOR" in sql
        assert "FLOAT" in sql
        assert "ORDER BY score DESC" in sql
        assert "LIMIT 5" in sql

    def test_includes_vector_column_when_requested(self):
        """Test that vector column is included when return_vector=True."""
        from retrieve_dspy.database.iris_database import _vector_search

        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []

        _vector_search(
            mock_connection,
            "RAG.Documents",
            "text_content",
            [0.1, 0.2],
            top_k=5,
            tag_filter=None,
            return_vector=True
        )

        sql = mock_cursor.execute.call_args[0][0]
        assert "text_content_embedding as vector" in sql


class TestAsyncSearch:
    """Tests for async_iris_search_tool."""

    @pytest.mark.asyncio
    async def test_async_search_returns_results(self):
        """Test that async search works."""
        from retrieve_dspy.database.iris_database import async_iris_search_tool

        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [
            ('doc_1', 'Content', 0.9, None),
        ]

        with patch('retrieve_dspy.database.iris_database._get_query_embedding') as mock_embed:
            mock_embed.return_value = [0.1] * 384

            results = await async_iris_search_tool(
                query="test",
                collection_name="RAG.Documents",
                target_property_name="text_content",
                iris_connection=mock_connection,
                retrieved_k=1
            )

        assert len(results) == 1
        assert isinstance(results[0], ObjectFromDB)


class TestEmbeddingGeneration:
    """Tests for _get_query_embedding."""

    def test_uses_iris_rag_if_available(self):
        """Test that iris_rag is preferred when available."""
        from retrieve_dspy.database.iris_database import _get_query_embedding

        with patch('retrieve_dspy.database.iris_database.EmbeddingManager') as mock_em:
            mock_manager = Mock()
            mock_manager.embed_texts.return_value = [[0.1, 0.2, 0.3]]
            mock_em.return_value = mock_manager

            with patch('retrieve_dspy.database.iris_database.ConfigurationManager'):
                embedding = _get_query_embedding("test query")

        assert embedding == [0.1, 0.2, 0.3]

    def test_falls_back_to_sentence_transformers(self):
        """Test fallback to sentence-transformers."""
        from retrieve_dspy.database.iris_database import _get_query_embedding

        # Mock iris_rag import to fail
        with patch.dict('sys.modules', {'iris_rag': None, 'iris_rag.embeddings': None}):
            with patch('retrieve_dspy.database.iris_database.SentenceTransformer') as mock_st:
                mock_model = Mock()
                mock_model.encode.return_value = Mock(tolist=lambda: [0.5, 0.6])
                mock_st.return_value = mock_model

                embedding = _get_query_embedding("test")

        assert len(embedding) == 2


# Integration test marker
@pytest.mark.integration
class TestIRISIntegration:
    """Integration tests requiring live IRIS database."""

    def test_real_iris_search(self):
        """
        Test with real IRIS database (skipped if not available).

        To run: pytest tests/database/test_iris_database.py -m integration
        """
        import os
        if not os.getenv("IRIS_PASSWORD"):
            pytest.skip("IRIS database not configured")

        from retrieve_dspy.database.iris_database import iris_search_tool

        results = iris_search_tool(
            query="test query",
            collection_name="RAG.Documents",
            target_property_name="text_content",
            retrieved_k=5
        )

        assert isinstance(results, list)
        # Results may be empty if no documents match
        if len(results) > 0:
            assert isinstance(results[0], ObjectFromDB)
