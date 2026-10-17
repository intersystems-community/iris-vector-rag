"""
Unit tests for safe vector SQL helpers.

``build_safe_vector_dot_sql`` embeds a validated vector literal in the SQL text
(``TO_VECTOR('[..]', FLOAT, dim)``); its injection defence is ``validate_vector_string``,
the identifier regexes and the dimension/top_k checks. ``build_bound_vector_dot_sql``
binds the vector as ``TO_VECTOR(?, <type>, <dim>)`` instead, which works for every
element type on IRIS 2026.1 and later (the earlier report that it did not is corrected
in docs/reports/IRIS_VECTOR_SQL_PARAMETERIZATION_REPRO.md). The library uses the bound
form.
"""

from unittest.mock import Mock

import pytest

from iris_vector_rag.common.vector_sql_utils import (
    build_safe_vector_dot_sql,
    execute_safe_vector_search,
    validate_vector_string,
)

VEC3 = "[0.1,0.2,0.3]"
LIT3 = "TO_VECTOR('[0.1,0.2,0.3]', FLOAT, 3)"


class TestBuildSafeVectorDotSQL:
    """Tests for build_safe_vector_dot_sql function."""

    def test_build_safe_vector_dot_sql_shape(self):
        """SQL has the expected shape and carries the vector as a literal."""
        sql = build_safe_vector_dot_sql("RAG.SourceDocuments", "embedding", VEC3, 3)

        assert "SELECT TOP 5 doc_id" in sql
        assert f"VECTOR_DOT_PRODUCT(embedding, {LIT3})" in sql
        assert "FROM RAG.SourceDocuments" in sql
        assert "WHERE embedding IS NOT NULL" in sql
        assert "ORDER BY score DESC" in sql

        # The vector is embedded, so no bound parameter remains.
        assert sql.count("?") == 0

    def test_build_safe_vector_dot_sql_with_extra_columns(self):
        """Test build_safe_vector_dot_sql with extra columns."""
        sql = build_safe_vector_dot_sql(
            "RAG.SourceDocuments",
            "embedding",
            VEC3,
            3,
            "doc_id",
            ["title", "abstract"],
            10,
        )

        assert "SELECT TOP 10 doc_id, title, abstract" in sql
        assert f"VECTOR_DOT_PRODUCT(embedding, {LIT3}) AS score" in sql
        assert sql.count("?") == 0

    def test_build_safe_vector_dot_sql_with_additional_where(self):
        """Test build_safe_vector_dot_sql with additional WHERE clause."""
        sql = build_safe_vector_dot_sql(
            "RAG.SourceDocuments",
            "embedding",
            VEC3,
            3,
            additional_where="title IS NOT NULL",
        )

        assert "WHERE embedding IS NOT NULL AND (title IS NOT NULL)" in sql

    def test_build_safe_vector_dot_sql_entities_table(self):
        """Test build_safe_vector_dot_sql with Entities table."""
        sql = build_safe_vector_dot_sql(
            "RAG.Entities",
            "embedding",
            VEC3,
            3,
            "entity_id",
            ["entity_name", "entity_type"],
            3,
        )

        assert "SELECT TOP 3 entity_id, entity_name, entity_type" in sql
        assert f"VECTOR_DOT_PRODUCT(embedding, {LIT3}) AS score" in sql
        assert "FROM RAG.Entities" in sql

    def test_build_safe_vector_dot_sql_validation_errors(self):
        """Test validation errors in build_safe_vector_dot_sql."""
        with pytest.raises(ValueError, match="Invalid table name"):
            build_safe_vector_dot_sql("DROP TABLE users;", "embedding", VEC3, 3)

        with pytest.raises(ValueError, match="Invalid column name"):
            build_safe_vector_dot_sql(
                "RAG.SourceDocuments", "embedding'; DROP TABLE users; --", VEC3, 3
            )

        with pytest.raises(ValueError, match="Invalid column name"):
            build_safe_vector_dot_sql(
                "RAG.SourceDocuments", "embedding", VEC3, 3, "id; DELETE FROM users"
            )

        with pytest.raises(ValueError, match="Invalid extra column name"):
            build_safe_vector_dot_sql(
                "RAG.SourceDocuments",
                "embedding",
                VEC3,
                3,
                extra_columns=["title; DROP TABLE users"],
            )

        with pytest.raises(ValueError, match="Invalid top_k value"):
            build_safe_vector_dot_sql(
                "RAG.SourceDocuments", "embedding", VEC3, 3, top_k=0
            )

        with pytest.raises(ValueError, match="Invalid top_k value"):
            build_safe_vector_dot_sql(
                "RAG.SourceDocuments",
                "embedding",
                VEC3,
                3,
                top_k="5; DROP TABLE users",
            )

        # The vector literal is spliced into the SQL, so it is validated too.
        with pytest.raises(ValueError, match="Invalid vector string"):
            build_safe_vector_dot_sql(
                "RAG.SourceDocuments",
                "embedding",
                "[0.1]'); DROP TABLE users; --",
                1,
            )

        with pytest.raises(ValueError, match="Invalid vector_dimension"):
            build_safe_vector_dot_sql("RAG.SourceDocuments", "embedding", VEC3, 0)

    def test_build_safe_vector_dot_sql_allowed_table_formats(self):
        """Test that build_safe_vector_dot_sql accepts valid table name formats."""
        sql1 = build_safe_vector_dot_sql("RAG.SourceDocuments", "embedding", VEC3, 3)
        assert "FROM RAG.SourceDocuments" in sql1

        sql2 = build_safe_vector_dot_sql("Documents", "embedding", VEC3, 3)
        assert "FROM Documents" in sql2

        sql3 = build_safe_vector_dot_sql("RAG.Source_Documents", "embedding", VEC3, 3)
        assert "FROM RAG.Source_Documents" in sql3


class TestValidateVectorString:
    """The vector literal formats the builder accepts and rejects."""

    @pytest.mark.parametrize(
        "vector_string",
        ["[1.0,2.0,3.0]", "[0.1,-0.2,0.3,-0.4]", "[1,2,3]", "[1.5e-3,2.7e2]"],
    )
    def test_accepts_floats_negatives_integers_and_scientific(self, vector_string):
        assert validate_vector_string(vector_string) is True

    @pytest.mark.parametrize(
        "vector_string",
        ["", "[]", "0.1,0.2", "[0.1,abc]", "[0.1;DROP]", "[0.1,--1]", "[SELECT]"],
    )
    def test_rejects_malformed_and_hostile_strings(self, vector_string):
        assert validate_vector_string(vector_string) is False


class TestExecuteSafeVectorSearch:
    """Tests for execute_safe_vector_search function."""

    def test_execute_safe_vector_search_mocks_param_and_returns_rows(self):
        """The SQL is executed as given (nothing bound) and the rows returned."""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ("doc1", "title1", 0.95),
            ("doc2", "title2", 0.87),
        ]

        sql = build_safe_vector_dot_sql(
            "RAG.SourceDocuments", "embedding", VEC3, 3, "doc_id", ["title"]
        )
        results = execute_safe_vector_search(mock_cursor, sql)

        mock_cursor.execute.assert_called_once_with(sql)
        mock_cursor.fetchall.assert_called_once()
        assert results == [("doc1", "title1", 0.95), ("doc2", "title2", 0.87)]

    def test_execute_safe_vector_search_empty_results(self):
        """Test execute_safe_vector_search with empty results."""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []

        sql = "SELECT doc_id FROM RAG.SourceDocuments WHERE embedding IS NOT NULL"
        results = execute_safe_vector_search(mock_cursor, sql)

        assert results == []
        mock_cursor.execute.assert_called_once_with(sql)
        mock_cursor.fetchall.assert_called_once()

    def test_execute_safe_vector_search_384_dimension_vector(self):
        """A 384-wide vector (typical model size) is embedded whole and executed."""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [("doc1", 0.92)]

        vector_string = "[" + ",".join(["0.1"] * 384) + "]"
        sql = build_safe_vector_dot_sql(
            "RAG.SourceDocuments", "embedding", vector_string, 384
        )
        assert f"TO_VECTOR('{vector_string}', FLOAT, 384)" in sql

        results = execute_safe_vector_search(mock_cursor, sql)

        mock_cursor.execute.assert_called_once_with(sql)
        assert results == [("doc1", 0.92)]

    def test_execute_safe_vector_search_vector_conversion(self):
        """Each accepted vector format reaches the executed SQL unchanged."""
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = []

        for vector_string, dim in [
            ("[1.0,2.0,3.0]", 3),
            ("[0.1,-0.2,0.3,-0.4]", 4),
            ("[1,2,3]", 3),
            ("[1.5e-3,2.7e2]", 2),
        ]:
            mock_cursor.reset_mock()
            sql = build_safe_vector_dot_sql(
                "RAG.SourceDocuments", "embedding", vector_string, dim
            )
            execute_safe_vector_search(mock_cursor, sql)
            mock_cursor.execute.assert_called_once_with(sql)
            assert f"TO_VECTOR('{vector_string}', FLOAT, {dim})" in sql


class TestSafeVectorIntegration:
    """Integration tests for safe vector utilities."""

    def test_build_and_execute_integration(self):
        """Test building SQL and executing it with safe vector utilities."""
        sql = build_safe_vector_dot_sql(
            "RAG.SourceDocuments", "embedding", VEC3, 3, "doc_id", ["title"], 5
        )

        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [
            ("doc1", "Title 1", 0.95),
            ("doc2", "Title 2", 0.88),
        ]

        results = execute_safe_vector_search(mock_cursor, sql)

        expected_sql = (
            "SELECT TOP 5 doc_id, title, "
            f"VECTOR_DOT_PRODUCT(embedding, {LIT3}) AS score "
            "FROM RAG.SourceDocuments WHERE embedding IS NOT NULL "
            "ORDER BY score DESC"
        )
        assert sql == expected_sql
        mock_cursor.execute.assert_called_once_with(sql)
        assert len(results) == 2

    def test_entities_table_integration(self):
        """Test safe vector utilities with Entities table."""
        vector_string = "[" + ",".join(["0.5"] * 384) + "]"
        sql = build_safe_vector_dot_sql(
            "RAG.Entities",
            "embedding",
            vector_string,
            384,
            "entity_id",
            ["entity_name", "entity_type"],
            3,
            "entity_type = 'PERSON'",
        )

        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [("entity1", "John Doe", "PERSON", 0.91)]

        results = execute_safe_vector_search(mock_cursor, sql)

        assert "entity_name, entity_type" in sql
        assert "FROM RAG.Entities" in sql
        assert "entity_type = 'PERSON'" in sql
        assert len(results) == 1


# --- the bound form: the query vector travels as a parameter -------------------


class TestBuildBoundVectorDotSQL:
    """``TO_VECTOR(?, <type>, <dim>)``: measured to work for FLOAT and DOUBLE columns on
    IRIS 2026.1 (Build 234U) and 2026.3 with intersystems-irispython 5.3.1 and 5.4.1,
    including the cached second run. Only the vector is a parameter: the element type
    and dimension must be literals (``TO_VECTOR(?, ?, ?)`` is SQLCODE -262), and the
    untyped ``TO_VECTOR(?)`` makes a FLOAT vector, so it fails on a DOUBLE column
    (SQLCODE -259)."""

    def test_the_vector_is_one_parameter_and_type_and_dimension_are_literals(self):
        from iris_vector_rag.common.vector_sql_utils import build_bound_vector_dot_sql

        sql = build_bound_vector_dot_sql("RAG.SourceDocuments", "embedding", 3)
        assert "VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?, FLOAT, 3)) AS score" in sql
        assert sql.count("?") == 1
        assert "[" not in sql and "'" not in sql

    def test_the_element_type_is_the_columns(self):
        from iris_vector_rag.common.vector_sql_utils import build_bound_vector_dot_sql

        sql = build_bound_vector_dot_sql(
            "RAG.SourceDocuments", "embedding", 384, vector_data_type="DOUBLE"
        )
        assert "TO_VECTOR(?, DOUBLE, 384)" in sql

    def test_columns_where_and_top_k_are_as_in_the_literal_form(self):
        from iris_vector_rag.common.vector_sql_utils import build_bound_vector_dot_sql

        sql = build_bound_vector_dot_sql(
            "RAG.SourceDocuments",
            "embedding",
            3,
            id_column="doc_id",
            extra_columns=["title"],
            top_k=7,
            additional_where="title IS NOT NULL",
        )
        assert sql.startswith("SELECT TOP 7 doc_id, title, VECTOR_DOT_PRODUCT(")
        assert "WHERE embedding IS NOT NULL AND (title IS NOT NULL)" in sql
        assert sql.endswith("ORDER BY score DESC")

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"table": "RAG.Docs; DROP TABLE x"},
            {"vector_column": "embedding)--"},
            {"vector_dimension": 0},
            {"top_k": 0},
            {"vector_data_type": "FLOAT, 3)) --"},
            {"vector_data_type": "?"},
        ],
    )
    def test_unsafe_input_is_refused(self, kwargs):
        from iris_vector_rag.common.vector_sql_utils import build_bound_vector_dot_sql

        args = {
            "table": "RAG.SourceDocuments",
            "vector_column": "embedding",
            "vector_dimension": 3,
        }
        args.update(kwargs)
        with pytest.raises(ValueError):
            build_bound_vector_dot_sql(**args)

    def test_execute_passes_the_parameters(self):
        cursor = Mock()
        cursor.fetchall.return_value = [("d1", 0.5)]
        assert execute_safe_vector_search(cursor, "SQL ?", [VEC3]) == [("d1", 0.5)]
        cursor.execute.assert_called_once_with("SQL ?", [VEC3])

    def test_execute_without_parameters_is_unchanged(self):
        cursor = Mock()
        cursor.fetchall.return_value = []
        execute_safe_vector_search(cursor, "SQL")
        cursor.execute.assert_called_once_with("SQL")
