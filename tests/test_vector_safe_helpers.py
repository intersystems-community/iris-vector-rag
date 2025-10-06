"""
Unit tests for safe vector SQL helpers.

Tests the safe vector query building and execution utilities that use
the proven single-parameter pattern for IRIS vector operations.
"""

import re
from unittest.mock import Mock, patch

import pytest

from common.vector_sql_utils import (
    build_safe_vector_dot_sql,
    execute_safe_vector_search,
)


class TestBuildSafeVectorDotSQL:
    """Tests for build_safe_vector_dot_sql function."""

    def test_build_safe_vector_dot_sql_shape(self):
        """Test that build_safe_vector_dot_sql generates SQL with correct shape."""
        # Basic test
        sql = build_safe_vector_dot_sql("RAG.SourceDocuments", "embedding")

        # Check SQL contains expected components
        assert "SELECT TOP 5 doc_id" in sql
        assert "VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?))" in sql
        assert "FROM RAG.SourceDocuments" in sql
        assert "WHERE embedding IS NOT NULL" in sql
        assert "ORDER BY score DESC" in sql

        # Check parameter placeholder count (should be exactly 1)
        assert sql.count("?") == 1

    def test_build_safe_vector_dot_sql_with_extra_columns(self):
        """Test build_safe_vector_dot_sql with extra columns."""
        sql = build_safe_vector_dot_sql(
            "RAG.SourceDocuments", "embedding", "doc_id", ["title", "abstract"], 10
        )

        assert "SELECT TOP 10 doc_id, title, abstract" in sql
        assert "VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?)) AS score" in sql
        assert sql.count("?") == 1

    def test_build_safe_vector_dot_sql_with_additional_where(self):
        """Test build_safe_vector_dot_sql with additional WHERE clause."""
        sql = build_safe_vector_dot_sql(
            "RAG.SourceDocuments", "embedding", additional_where="title IS NOT NULL"
        )

        assert "WHERE embedding IS NOT NULL AND (title IS NOT NULL)" in sql
        assert sql.count("?") == 1

    def test_build_safe_vector_dot_sql_entities_table(self):
        """Test build_safe_vector_dot_sql with Entities table."""
        sql = build_safe_vector_dot_sql(
            "RAG.Entities", "embedding", "entity_id", ["entity_name", "entity_type"], 3
        )

        assert "SELECT TOP 3 entity_id, entity_name, entity_type" in sql
        assert "VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?)) AS score" in sql
        assert "FROM RAG.Entities" in sql
        assert sql.count("?") == 1

    def test_build_safe_vector_dot_sql_validation_errors(self):
        """Test validation errors in build_safe_vector_dot_sql."""
        # Invalid table name
        with pytest.raises(ValueError, match="Invalid table name"):
            build_safe_vector_dot_sql("DROP TABLE users;", "embedding")

        # Invalid vector column name
        with pytest.raises(ValueError, match="Invalid column name"):
            build_safe_vector_dot_sql(
                "RAG.SourceDocuments", "embedding'; DROP TABLE users; --"
            )

        # Invalid id column name
        with pytest.raises(ValueError, match="Invalid column name"):
            build_safe_vector_dot_sql(
                "RAG.SourceDocuments", "embedding", "id; DELETE FROM users"
            )

        # Invalid extra column name
        with pytest.raises(ValueError, match="Invalid extra column name"):
            build_safe_vector_dot_sql(
                "RAG.SourceDocuments",
                "embedding",
                extra_columns=["title; DROP TABLE users"],
            )

        # Invalid top_k value
        with pytest.raises(ValueError, match="Invalid top_k value"):
            build_safe_vector_dot_sql("RAG.SourceDocuments", "embedding", top_k=0)

        with pytest.raises(ValueError, match="Invalid top_k value"):
            build_safe_vector_dot_sql(
                "RAG.SourceDocuments", "embedding", top_k="5; DROP TABLE users"
            )

    def test_build_safe_vector_dot_sql_allowed_table_formats(self):
        """Test that build_safe_vector_dot_sql accepts valid table name formats."""
        # Schema.Table format
        sql1 = build_safe_vector_dot_sql("RAG.SourceDocuments", "embedding")
        assert "FROM RAG.SourceDocuments" in sql1

        # Simple table name
        sql2 = build_safe_vector_dot_sql("Documents", "embedding")
        assert "FROM Documents" in sql2

        # Table with underscores
        sql3 = build_safe_vector_dot_sql("RAG.Source_Documents", "embedding")
        assert "FROM RAG.Source_Documents" in sql3


class TestExecuteSafeVectorSearch:
    """Tests for execute_safe_vector_search function."""

    def test_execute_safe_vector_search_mocks_param_and_returns_rows(self):
        """Test that execute_safe_vector_search properly mocks parameters and returns rows."""
        # Setup mock cursor
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ("doc1", "title1", 0.95),
            ("doc2", "title2", 0.87),
        ]

        # Test vector
        test_vector = [0.1, 0.2, 0.3, 0.4]

        # Execute function
        sql = "SELECT TOP 5 doc_id, title, VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?)) AS score FROM RAG.SourceDocuments WHERE embedding IS NOT NULL ORDER BY score DESC"
        results = execute_safe_vector_search(mock_cursor, sql, test_vector)

        # Verify cursor.execute was called with correct parameters
        mock_cursor.execute.assert_called_once_with(sql, ("0.1,0.2,0.3,0.4",))

        # Verify fetchall was called
        mock_cursor.fetchall.assert_called_once()

        # Verify results
        assert len(results) == 2
        assert results[0] == ("doc1", "title1", 0.95)
        assert results[1] == ("doc2", "title2", 0.87)

    def test_execute_safe_vector_search_vector_conversion(self):
        """Test vector list to string conversion in execute_safe_vector_search."""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []

        # Test different vector formats
        test_vectors = [
            [1.0, 2.0, 3.0],
            [0.1, -0.2, 0.3, -0.4],
            [1, 2, 3],  # integers
            [1.5e-3, 2.7e2],  # scientific notation
        ]

        expected_strings = ["1.0,2.0,3.0", "0.1,-0.2,0.3,-0.4", "1,2,3", "0.0015,270.0"]

        sql = "SELECT doc_id FROM RAG.SourceDocuments WHERE embedding IS NOT NULL"

        for vector, expected_str in zip(test_vectors, expected_strings):
            mock_cursor.reset_mock()
            execute_safe_vector_search(mock_cursor, sql, vector)
            mock_cursor.execute.assert_called_once_with(sql, (expected_str,))

    def test_execute_safe_vector_search_empty_results(self):
        """Test execute_safe_vector_search with empty results."""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []

        sql = "SELECT doc_id FROM RAG.SourceDocuments WHERE embedding IS NOT NULL"
        results = execute_safe_vector_search(mock_cursor, sql, [0.1, 0.2])

        assert results == []
        mock_cursor.execute.assert_called_once()
        mock_cursor.fetchall.assert_called_once()

    def test_execute_safe_vector_search_384_dimension_vector(self):
        """Test execute_safe_vector_search with 384-dimension vector (typical model size)."""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [("doc1", 0.92)]

        # Create 384-dimension vector
        vector_384d = [0.1] * 384
        expected_str = ",".join(["0.1"] * 384)

        sql = (
            "SELECT doc_id, score FROM RAG.SourceDocuments WHERE embedding IS NOT NULL"
        )
        results = execute_safe_vector_search(mock_cursor, sql, vector_384d)

        mock_cursor.execute.assert_called_once_with(sql, (expected_str,))
        assert len(results) == 1
        assert results[0] == ("doc1", 0.92)


class TestSafeVectorIntegration:
    """Integration tests for safe vector utilities."""

    def test_build_and_execute_integration(self):
        """Test building SQL and executing it with safe vector utilities."""
        # Build SQL
        sql = build_safe_vector_dot_sql(
            "RAG.SourceDocuments", "embedding", "doc_id", ["title"], 5
        )

        # Mock cursor
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ("doc1", "Title 1", 0.95),
            ("doc2", "Title 2", 0.88),
        ]

        # Execute
        test_vector = [0.1, 0.2, 0.3]
        results = execute_safe_vector_search(mock_cursor, sql, test_vector)

        # Verify the complete flow
        expected_sql = "SELECT TOP 5 doc_id, title, VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?)) AS score FROM RAG.SourceDocuments WHERE embedding IS NOT NULL ORDER BY score DESC"
        assert sql == expected_sql

        mock_cursor.execute.assert_called_once_with(sql, ("0.1,0.2,0.3",))
        assert len(results) == 2

    def test_entities_table_integration(self):
        """Test safe vector utilities with Entities table."""
        # Build SQL for entities
        sql = build_safe_vector_dot_sql(
            "RAG.Entities",
            "embedding",
            "entity_id",
            ["entity_name", "entity_type"],
            3,
            "entity_type = 'PERSON'",
        )

        # Mock cursor
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [("entity1", "John Doe", "PERSON", 0.91)]

        # Execute
        test_vector = [0.5] * 384  # 384-dimension vector
        results = execute_safe_vector_search(mock_cursor, sql, test_vector)

        # Verify
        assert "entity_name, entity_type" in sql
        assert "FROM RAG.Entities" in sql
        assert "entity_type = 'PERSON'" in sql
        assert len(results) == 1
