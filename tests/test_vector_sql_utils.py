"""
Tests for the vector_sql_utils module.
"""
import pytest
from unittest.mock import MagicMock
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.vector_sql_utils import (
    validate_vector_string,
    validate_top_k,
    format_vector_search_sql,
    execute_vector_search
)


class TestVectorSqlUtils:
    """Test suite for vector_sql_utils module."""

    def test_validate_vector_string(self):
        """Test the validate_vector_string function."""
        # Valid vector strings
        assert validate_vector_string("[0.1,0.2,0.3]") is True
        assert validate_vector_string("[0,1,2]") is True
        assert validate_vector_string("[]") is True
        assert validate_vector_string("[0.123456789]") is True

        # Invalid vector strings
        assert validate_vector_string("'; DROP TABLE users; --") is False
        assert validate_vector_string("<script>alert('xss')</script>") is False
        assert validate_vector_string("not a vector") is False
        assert validate_vector_string("SELECT * FROM users") is False

    def test_validate_top_k(self):
        """Test the validate_top_k function."""
        # Valid top_k values
        assert validate_top_k(1) is True
        assert validate_top_k(10) is True
        assert validate_top_k(100) is True
        assert validate_top_k(1000) is True

        # Invalid top_k values
        assert validate_top_k(0) is False
        assert validate_top_k(-1) is False
        assert validate_top_k("10") is False
        assert validate_top_k("10; DROP TABLE users; --") is False
        assert validate_top_k(None) is False
        assert validate_top_k([10]) is False
        assert validate_top_k({"top_k": 10}) is False

    def test_format_vector_search_sql(self):
        """Test the format_vector_search_sql function."""
        # Test with minimal parameters
        sql = format_vector_search_sql(
            "SourceDocuments",
            "embedding",
            "[0.1,0.2,0.3]",
            768,
            10
        )
        assert "SELECT TOP 10 doc_id" in sql
        assert "text_content" in sql
        assert "VECTOR_COSINE(embedding, TO_VECTOR('[0.1,0.2,0.3]', 'DOUBLE', 768))" in sql
        assert "FROM SourceDocuments" in sql
        assert "WHERE embedding IS NOT NULL" in sql
        assert "ORDER BY score DESC" in sql

        # Test with custom column names
        sql = format_vector_search_sql(
            "CustomTable",
            "vector_col",
            "[0.1,0.2,0.3]",
            512,
            5,
            "item_id",
            "item_text"
        )
        assert "SELECT TOP 5 item_id" in sql
        assert "item_text" in sql
        assert "VECTOR_COSINE(vector_col, TO_VECTOR('[0.1,0.2,0.3]', 'DOUBLE', 512))" in sql
        assert "FROM CustomTable" in sql

        # Test with no content column
        sql = format_vector_search_sql(
            "Embeddings",
            "vector",
            "[0.1,0.2,0.3]",
            1024,
            20,
            "id",
            None
        )
        assert "SELECT TOP 20 id" in sql
        assert "text_content" not in sql
        assert "VECTOR_COSINE(vector, TO_VECTOR('[0.1,0.2,0.3]', 'DOUBLE', 1024))" in sql

        # Test with additional WHERE clause
        sql = format_vector_search_sql(
            "Documents",
            "embedding",
            "[0.1,0.2,0.3]",
            768,
            10,
            additional_where="category = 'science'"
        )
        assert "WHERE embedding IS NOT NULL AND (category = 'science')" in sql

    def test_format_vector_search_sql_validation(self):
        """Test that format_vector_search_sql validates inputs correctly."""
        # Invalid table name
        with pytest.raises(ValueError, match="Invalid table name"):
            format_vector_search_sql(
                "Source; DROP TABLE users; --",
                "embedding",
                "[0.1,0.2,0.3]",
                768,
                10
            )

        # Invalid column name
        with pytest.raises(ValueError, match="Invalid column name"):
            format_vector_search_sql(
                "SourceDocuments",
                "embedding; DROP TABLE users; --",
                "[0.1,0.2,0.3]",
                768,
                10
            )

        # Invalid vector string
        with pytest.raises(ValueError, match="Invalid vector string"):
            format_vector_search_sql(
                "SourceDocuments",
                "embedding",
                "'; DROP TABLE users; --",
                768,
                10
            )

        # Invalid embedding dimension
        with pytest.raises(ValueError, match="Invalid embedding dimension"):
            format_vector_search_sql(
                "SourceDocuments",
                "embedding",
                "[0.1,0.2,0.3]",
                -1,
                10
            )

        # Invalid top_k
        with pytest.raises(ValueError, match="Invalid top_k value"):
            format_vector_search_sql(
                "SourceDocuments",
                "embedding",
                "[0.1,0.2,0.3]",
                768,
                0
            )

    def test_execute_vector_search(self):
        """Test the execute_vector_search function."""
        # Create a mock cursor
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            ("doc1", "content1", 0.9),
            ("doc2", "content2", 0.8),
            ("doc3", "content3", 0.7)
        ]

        # Test with a valid SQL query
        sql = "SELECT * FROM SourceDocuments"
        results = execute_vector_search(mock_cursor, sql)

        # Verify that the cursor was called correctly
        mock_cursor.execute.assert_called_once_with(sql)
        mock_cursor.fetchall.assert_called_once()

        # Verify the results
        assert len(results) == 3
        assert results[0][0] == "doc1"
        assert results[0][1] == "content1"
        assert results[0][2] == 0.9

        # Test with an exception
        mock_cursor.reset_mock()
        mock_cursor.execute.side_effect = Exception("Database error")

        # Verify that the exception is re-raised
        with pytest.raises(Exception, match="Database error"):
            execute_vector_search(mock_cursor, sql)


if __name__ == "__main__":
    pytest.main(["-v", __file__])