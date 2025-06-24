"""
Comprehensive unit tests for the IrisSQLTool class.

This test suite follows TDD principles and covers:
1. SQL rewriting functionality with different dialects and edge cases
2. SQL execution with mocked IRIS connector
3. Complete search workflow integration
4. Error handling and edge cases
5. LLM response parsing

Tests are designed to be isolated, comprehensive, and maintainable.
"""

import pytest
import unittest.mock as mock
from typing import Dict, List, Any, Tuple
import logging

# Import the class under test
from iris_rag.tools.iris_sql_tool import IrisSQLTool

# Import test fixtures and mocks
from tests.mocks.db import MockIRISConnector, MockIRISCursor
from tests.mocks.models import mock_llm_func

logger = logging.getLogger(__name__)


class TestIrisSQLToolInitialization:
    """Test IrisSQLTool initialization and validation."""
    
    def test_init_with_valid_parameters(self):
        """Test successful initialization with valid parameters."""
        # Arrange
        mock_iris_connector = MockIRISConnector()
        mock_llm = mock_llm_func
        
        # Act
        tool = IrisSQLTool(iris_connector=mock_iris_connector, llm_func=mock_llm)
        
        # Assert
        assert tool.iris_connector == mock_iris_connector
        assert tool.llm_func == mock_llm
    
    def test_init_with_none_iris_connector(self):
        """Test initialization fails with None iris_connector."""
        # Arrange
        mock_llm = mock_llm_func
        
        # Act & Assert
        with pytest.raises(ValueError, match="iris_connector cannot be None"):
            IrisSQLTool(iris_connector=None, llm_func=mock_llm)
    
    def test_init_with_none_llm_func(self):
        """Test initialization fails with None llm_func."""
        # Arrange
        mock_iris_connector = MockIRISConnector()
        
        # Act & Assert
        with pytest.raises(ValueError, match="llm_func cannot be None"):
            IrisSQLTool(iris_connector=mock_iris_connector, llm_func=None)


class TestRewriteSQL:
    """Test the rewrite_sql method with various SQL dialects and scenarios."""
    
    @pytest.fixture
    def iris_sql_tool(self):
        """Create an IrisSQLTool instance for testing."""
        mock_iris_connector = MockIRISConnector()
        mock_llm = mock_llm_func
        return IrisSQLTool(iris_connector=mock_iris_connector, llm_func=mock_llm)
    
    def test_rewrite_sql_basic_query(self, iris_sql_tool):
        """Test rewriting a basic SQL query."""
        # Arrange
        original_query = "SELECT * FROM users LIMIT 10"
        expected_rewritten = "SELECT TOP 10 * FROM users"
        expected_explanation = "Changed LIMIT to TOP for IRIS compatibility"
        
        # Mock LLM response
        mock_llm_response = f"""REWRITTEN_SQL:
{expected_rewritten}

EXPLANATION:
{expected_explanation}"""
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=mock_llm_response):
            # Act
            rewritten_sql, explanation = iris_sql_tool.rewrite_sql(original_query)
            
            # Assert
            assert rewritten_sql == expected_rewritten
            assert explanation == expected_explanation
    
    def test_rewrite_sql_limit_to_top_conversion(self, iris_sql_tool):
        """Test conversion of LIMIT to TOP syntax."""
        # Arrange
        original_query = "SELECT name, email FROM customers WHERE active = 1 LIMIT 50"
        expected_rewritten = "SELECT TOP 50 name, email FROM customers WHERE active = 1"
        expected_explanation = "Converted LIMIT 50 to TOP 50 for IRIS SQL compatibility"
        
        mock_llm_response = f"""REWRITTEN_SQL:
{expected_rewritten}

EXPLANATION:
{expected_explanation}"""
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=mock_llm_response):
            # Act
            rewritten_sql, explanation = iris_sql_tool.rewrite_sql(original_query)
            
            # Assert
            assert "TOP 50" in rewritten_sql
            assert "LIMIT" not in rewritten_sql
            assert "IRIS" in explanation
    
    def test_rewrite_sql_vector_operations(self, iris_sql_tool):
        """Test rewriting SQL with vector operations."""
        # Arrange
        original_query = "SELECT * FROM documents WHERE VECTOR_SIMILARITY(embedding, ?) > 0.8"
        expected_rewritten = "SELECT * FROM documents WHERE VECTOR_COSINE_SIMILARITY(TO_VECTOR(?), embedding) > 0.8"
        expected_explanation = "Added TO_VECTOR() function and used IRIS vector similarity syntax"
        
        mock_llm_response = f"""REWRITTEN_SQL:
{expected_rewritten}

EXPLANATION:
{expected_explanation}"""
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=mock_llm_response):
            # Act
            rewritten_sql, explanation = iris_sql_tool.rewrite_sql(original_query)
            
            # Assert
            assert "TO_VECTOR" in rewritten_sql
            assert "VECTOR_COSINE_SIMILARITY" in rewritten_sql
            assert "vector" in explanation.lower()
    
    def test_rewrite_sql_string_concatenation(self, iris_sql_tool):
        """Test rewriting SQL with string concatenation."""
        # Arrange
        original_query = "SELECT CONCAT(first_name, ' ', last_name) AS full_name FROM users"
        expected_rewritten = "SELECT (first_name || ' ' || last_name) AS full_name FROM users"
        expected_explanation = "Replaced CONCAT with || operator for IRIS string concatenation"
        
        mock_llm_response = f"""REWRITTEN_SQL:
{expected_rewritten}

EXPLANATION:
{expected_explanation}"""
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=mock_llm_response):
            # Act
            rewritten_sql, explanation = iris_sql_tool.rewrite_sql(original_query)
            
            # Assert
            assert "||" in rewritten_sql
            assert "CONCAT" not in rewritten_sql
            assert "concatenation" in explanation.lower()
    
    def test_rewrite_sql_date_functions(self, iris_sql_tool):
        """Test rewriting SQL with date/time functions."""
        # Arrange
        original_query = "SELECT * FROM orders WHERE created_at >= NOW() - INTERVAL 7 DAY"
        expected_rewritten = "SELECT * FROM orders WHERE created_at >= DATEADD(day, -7, GETDATE())"
        expected_explanation = "Converted NOW() and INTERVAL to IRIS date functions"
        
        mock_llm_response = f"""REWRITTEN_SQL:
{expected_rewritten}

EXPLANATION:
{expected_explanation}"""
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=mock_llm_response):
            # Act
            rewritten_sql, explanation = iris_sql_tool.rewrite_sql(original_query)
            
            # Assert
            assert "DATEADD" in rewritten_sql or "GETDATE" in rewritten_sql
            assert "date" in explanation.lower()
    
    def test_rewrite_sql_complex_query(self, iris_sql_tool):
        """Test rewriting a complex SQL query with multiple IRIS-specific changes."""
        # Arrange
        original_query = """
        SELECT CONCAT(u.first_name, ' ', u.last_name) as full_name,
               COUNT(*) as order_count
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE o.created_at >= NOW() - INTERVAL 30 DAY
        GROUP BY u.id, u.first_name, u.last_name
        ORDER BY order_count DESC
        LIMIT 20
        """
        expected_rewritten = """
        SELECT TOP 20 (u.first_name || ' ' || u.last_name) as full_name,
               COUNT(*) as order_count
        FROM users u
        JOIN orders o ON u.id = o.user_id
        WHERE o.created_at >= DATEADD(day, -30, GETDATE())
        GROUP BY u.id, u.first_name, u.last_name
        ORDER BY order_count DESC
        """
        expected_explanation = "Multiple IRIS compatibility changes: LIMIT to TOP, CONCAT to ||, date functions"
        
        mock_llm_response = f"""REWRITTEN_SQL:
{expected_rewritten}

EXPLANATION:
{expected_explanation}"""
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=mock_llm_response):
            # Act
            rewritten_sql, explanation = iris_sql_tool.rewrite_sql(original_query)
            
            # Assert
            assert "TOP 20" in rewritten_sql
            assert "||" in rewritten_sql
            assert "LIMIT" not in rewritten_sql
            assert "CONCAT" not in rewritten_sql
    
    def test_rewrite_sql_empty_query(self, iris_sql_tool):
        """Test rewrite_sql with empty query raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Original query cannot be empty or None"):
            iris_sql_tool.rewrite_sql("")
    
    def test_rewrite_sql_none_query(self, iris_sql_tool):
        """Test rewrite_sql with None query raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Original query cannot be empty or None"):
            iris_sql_tool.rewrite_sql(None)
    
    def test_rewrite_sql_whitespace_only_query(self, iris_sql_tool):
        """Test rewrite_sql with whitespace-only query raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="Original query cannot be empty or None"):
            iris_sql_tool.rewrite_sql("   \n\t   ")
    
    def test_rewrite_sql_llm_empty_response(self, iris_sql_tool):
        """Test rewrite_sql handles empty LLM response."""
        # Arrange
        original_query = "SELECT * FROM users"
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=""):
            # Act & Assert
            with pytest.raises(RuntimeError, match="LLM returned empty response"):
                iris_sql_tool.rewrite_sql(original_query)
    
    def test_rewrite_sql_llm_exception(self, iris_sql_tool):
        """Test rewrite_sql handles LLM function exceptions."""
        # Arrange
        original_query = "SELECT * FROM users"
        
        with mock.patch.object(iris_sql_tool, 'llm_func', side_effect=Exception("LLM service unavailable")):
            # Act & Assert
            with pytest.raises(RuntimeError, match="SQL rewriting failed"):
                iris_sql_tool.rewrite_sql(original_query)


class TestParseLLMResponse:
    """Test the _parse_llm_response method."""
    
    @pytest.fixture
    def iris_sql_tool(self):
        """Create an IrisSQLTool instance for testing."""
        mock_iris_connector = MockIRISConnector()
        mock_llm = mock_llm_func
        return IrisSQLTool(iris_connector=mock_iris_connector, llm_func=mock_llm)
    
    def test_parse_llm_response_valid_format(self, iris_sql_tool):
        """Test parsing a well-formatted LLM response."""
        # Arrange
        llm_response = """REWRITTEN_SQL:
SELECT TOP 10 * FROM users

EXPLANATION:
Changed LIMIT to TOP for IRIS compatibility"""
        
        # Act
        rewritten_sql, explanation = iris_sql_tool._parse_llm_response(llm_response)
        
        # Assert
        assert rewritten_sql == "SELECT TOP 10 * FROM users"
        assert explanation == "Changed LIMIT to TOP for IRIS compatibility"
    
    def test_parse_llm_response_multiline_sql(self, iris_sql_tool):
        """Test parsing LLM response with multiline SQL."""
        # Arrange
        llm_response = """REWRITTEN_SQL:
SELECT TOP 10 u.name,
       u.email,
       COUNT(*) as order_count
FROM users u
JOIN orders o ON u.id = o.user_id
GROUP BY u.name, u.email

EXPLANATION:
Converted LIMIT to TOP and restructured for IRIS compatibility"""
        
        # Act
        rewritten_sql, explanation = iris_sql_tool._parse_llm_response(llm_response)
        
        # Assert
        assert "SELECT TOP 10 u.name," in rewritten_sql
        assert "FROM users u" in rewritten_sql
        assert "JOIN orders o" in rewritten_sql
        assert "IRIS compatibility" in explanation
    
    def test_parse_llm_response_no_sql_section(self, iris_sql_tool):
        """Test parsing LLM response without REWRITTEN_SQL section."""
        # Arrange
        llm_response = """EXPLANATION:
This is just an explanation without SQL"""
        
        # Act
        rewritten_sql, explanation = iris_sql_tool._parse_llm_response(llm_response)
        
        # Assert - Should fallback gracefully
        assert rewritten_sql == llm_response.strip()
        assert explanation == "Failed to parse LLM response format"
    
    def test_parse_llm_response_no_explanation_section(self, iris_sql_tool):
        """Test parsing LLM response without EXPLANATION section."""
        # Arrange
        llm_response = """REWRITTEN_SQL:
SELECT TOP 10 * FROM users"""
        
        # Act
        rewritten_sql, explanation = iris_sql_tool._parse_llm_response(llm_response)
        
        # Assert
        assert rewritten_sql == "SELECT TOP 10 * FROM users"
        assert explanation == "No explanation provided by LLM"
    
    def test_parse_llm_response_malformed_fallback(self, iris_sql_tool):
        """Test parsing completely malformed LLM response falls back gracefully."""
        # Arrange
        llm_response = "This is completely malformed response"
        
        # Act
        rewritten_sql, explanation = iris_sql_tool._parse_llm_response(llm_response)
        
        # Assert
        assert rewritten_sql == "This is completely malformed response"
        assert explanation == "Failed to parse LLM response format"


class TestExecuteSQL:
    """Test the execute_sql method with mocked IRIS connector."""
    
    @pytest.fixture
    def iris_sql_tool(self):
        """Create an IrisSQLTool instance with mock connector for testing."""
        mock_iris_connector = MockIRISConnector()
        mock_llm = mock_llm_func
        return IrisSQLTool(iris_connector=mock_iris_connector, llm_func=mock_llm)
    
    def test_execute_sql_basic_query(self, iris_sql_tool):
        """Test executing a basic SQL query."""
        # Arrange
        sql_query = "SELECT TOP 5 * FROM users"
        
        # Mock the cursor method to return a cursor with test data
        with mock.patch.object(iris_sql_tool.iris_connector, 'cursor') as mock_cursor_method:
            mock_cursor = mock.Mock()
            mock_cursor.execute.return_value = None
            mock_cursor.fetchall.return_value = [
                ("user1", "John Doe", "john@example.com"),
                ("user2", "Jane Smith", "jane@example.com")
            ]
            mock_cursor.description = [("id",), ("name",), ("email",)]
            mock_cursor.close.return_value = None
            mock_cursor_method.return_value = mock_cursor
            
            # Act
            results = iris_sql_tool.execute_sql(sql_query)
            
            # Assert
            assert len(results) == 2
            assert results[0]["id"] == "user1"
            assert results[0]["name"] == "John Doe"
            assert results[0]["email"] == "john@example.com"
            assert results[1]["id"] == "user2"
            assert results[1]["name"] == "Jane Smith"
            assert results[1]["email"] == "jane@example.com"
    
    def test_execute_sql_empty_results(self, iris_sql_tool):
        """Test executing SQL query that returns no results."""
        # Arrange
        sql_query = "SELECT * FROM users WHERE id = 'nonexistent'"
        
        # Setup mock cursor to return empty results
        mock_cursor = iris_sql_tool.iris_connector.cursor()
        mock_cursor.results = []
        mock_cursor.description = [("id",), ("name",), ("email",)]
        
        # Act
        results = iris_sql_tool.execute_sql(sql_query)
        
        # Assert
        assert results == []
    
    def test_execute_sql_no_description(self, iris_sql_tool):
        """Test executing SQL query with no column description."""
        # Arrange
        sql_query = "INSERT INTO users (name) VALUES ('Test User')"
        
        # Setup mock cursor for INSERT operation
        mock_cursor = iris_sql_tool.iris_connector.cursor()
        mock_cursor.results = []
        mock_cursor.description = None
        
        # Act
        results = iris_sql_tool.execute_sql(sql_query)
        
        # Assert
        assert results == []
    
    def test_execute_sql_mismatched_columns(self, iris_sql_tool):
        """Test executing SQL query with mismatched column count."""
        # Arrange
        sql_query = "SELECT * FROM users"
        
        # Mock the cursor method to return a cursor with mismatched data
        with mock.patch.object(iris_sql_tool.iris_connector, 'cursor') as mock_cursor_method:
            mock_cursor = mock.Mock()
            mock_cursor.execute.return_value = None
            mock_cursor.fetchall.return_value = [
                ("user1", "John Doe", "john@example.com", "extra_data")
            ]
            mock_cursor.description = [("id",), ("name",), ("email",)]
            mock_cursor.close.return_value = None
            mock_cursor_method.return_value = mock_cursor
            
            # Act
            results = iris_sql_tool.execute_sql(sql_query)
            
            # Assert
            assert len(results) == 1
            assert results[0]["id"] == "user1"
            assert results[0]["name"] == "John Doe"
            assert results[0]["email"] == "john@example.com"
            assert results[0]["column_3"] == "extra_data"  # Extra column gets generic name
    
    def test_execute_sql_empty_query(self, iris_sql_tool):
        """Test execute_sql with empty query raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="SQL query cannot be empty or None"):
            iris_sql_tool.execute_sql("")
    
    def test_execute_sql_none_query(self, iris_sql_tool):
        """Test execute_sql with None query raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="SQL query cannot be empty or None"):
            iris_sql_tool.execute_sql(None)
    
    def test_execute_sql_whitespace_only_query(self, iris_sql_tool):
        """Test execute_sql with whitespace-only query raises ValueError."""
        # Act & Assert
        with pytest.raises(ValueError, match="SQL query cannot be empty or None"):
            iris_sql_tool.execute_sql("   \n\t   ")
    
    def test_execute_sql_cursor_exception(self, iris_sql_tool):
        """Test execute_sql handles cursor execution exceptions."""
        # Arrange
        sql_query = "SELECT * FROM nonexistent_table"
        
        # Mock cursor to raise exception
        with mock.patch.object(iris_sql_tool.iris_connector, 'cursor') as mock_cursor_method:
            mock_cursor = mock.Mock()
            mock_cursor.execute.side_effect = Exception("Table does not exist")
            mock_cursor_method.return_value = mock_cursor
            
            # Act & Assert
            with pytest.raises(RuntimeError, match="SQL execution failed"):
                iris_sql_tool.execute_sql(sql_query)
    
    def test_execute_sql_cursor_cleanup_on_exception(self, iris_sql_tool):
        """Test execute_sql properly cleans up cursor on exception."""
        # Arrange
        sql_query = "SELECT * FROM users"
        
        # Mock cursor to raise exception during fetchall
        with mock.patch.object(iris_sql_tool.iris_connector, 'cursor') as mock_cursor_method:
            mock_cursor = mock.Mock()
            mock_cursor.execute.return_value = None
            mock_cursor.fetchall.side_effect = Exception("Fetch failed")
            mock_cursor_method.return_value = mock_cursor
            
            # Act & Assert
            with pytest.raises(RuntimeError, match="SQL execution failed"):
                iris_sql_tool.execute_sql(sql_query)
            
            # Verify cursor.close() was called
            mock_cursor.close.assert_called_once()


class TestSearchIntegration:
    """Test the complete search workflow integration."""
    
    @pytest.fixture
    def iris_sql_tool(self):
        """Create an IrisSQLTool instance for integration testing."""
        mock_iris_connector = MockIRISConnector()
        mock_llm = mock_llm_func
        return IrisSQLTool(iris_connector=mock_iris_connector, llm_func=mock_llm)
    
    def test_search_successful_workflow(self, iris_sql_tool):
        """Test complete successful search workflow."""
        # Arrange
        original_query = "SELECT * FROM users LIMIT 5"
        rewritten_query = "SELECT TOP 5 * FROM users"
        explanation = "Changed LIMIT to TOP for IRIS compatibility"
        
        mock_llm_response = f"""REWRITTEN_SQL:
{rewritten_query}

EXPLANATION:
{explanation}"""
        
        # Mock both LLM and cursor
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=mock_llm_response):
            with mock.patch.object(iris_sql_tool.iris_connector, 'cursor') as mock_cursor_method:
                mock_cursor = mock.Mock()
                mock_cursor.execute.return_value = None
                mock_cursor.fetchall.return_value = [
                    ("user1", "John Doe"),
                    ("user2", "Jane Smith")
                ]
                mock_cursor.description = [("id",), ("name",)]
                mock_cursor.close.return_value = None
                mock_cursor_method.return_value = mock_cursor
                
                # Act
                result = iris_sql_tool.search(original_query)
                
                # Assert
                assert result["success"] is True
                assert result["error"] is None
                assert result["original_query"] == original_query
                assert result["rewritten_query"] == rewritten_query
                assert result["explanation"] == explanation
                assert len(result["results"]) == 2
                assert result["results"][0]["id"] == "user1"
                assert result["results"][0]["name"] == "John Doe"
    
    def test_search_rewrite_failure(self, iris_sql_tool):
        """Test search workflow when SQL rewriting fails."""
        # Arrange
        original_query = "SELECT * FROM users"
        
        with mock.patch.object(iris_sql_tool, 'llm_func', side_effect=Exception("LLM unavailable")):
            # Act
            result = iris_sql_tool.search(original_query)
            
            # Assert
            assert result["success"] is False
            assert "LLM unavailable" in result["error"]
            assert result["original_query"] == original_query
            assert result["rewritten_query"] is None
            assert result["explanation"] is None
            assert result["results"] == []
    
    def test_search_execution_failure(self, iris_sql_tool):
        """Test search workflow when SQL execution fails."""
        # Arrange
        original_query = "SELECT * FROM users"
        rewritten_query = "SELECT * FROM users"
        explanation = "No changes needed"
        
        mock_llm_response = f"""REWRITTEN_SQL:
{rewritten_query}

EXPLANATION:
{explanation}"""
        
        # Mock successful rewrite but failed execution
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=mock_llm_response):
            with mock.patch.object(iris_sql_tool.iris_connector, 'cursor') as mock_cursor_method:
                mock_cursor = mock.Mock()
                mock_cursor.execute.side_effect = Exception("Database connection lost")
                mock_cursor_method.return_value = mock_cursor
                
                # Act
                result = iris_sql_tool.search(original_query)
                
                # Assert
                assert result["success"] is False
                assert "Database connection lost" in result["error"]
                assert result["original_query"] == original_query
                assert result["rewritten_query"] == rewritten_query
                assert result["explanation"] == explanation
                assert result["results"] == []
    
    def test_search_empty_query(self, iris_sql_tool):
        """Test search workflow with empty query."""
        # Act
        result = iris_sql_tool.search("")
        
        # Assert
        assert result["success"] is False
        assert "Original query cannot be empty or None" in result["error"]
        assert result["original_query"] == ""
        assert result["rewritten_query"] is None
        assert result["explanation"] is None
        assert result["results"] == []


class TestSQLDialectCompatibility:
    """Test SQL rewriting for different database dialects."""
    
    @pytest.fixture
    def iris_sql_tool(self):
        """Create an IrisSQLTool instance for dialect testing."""
        mock_iris_connector = MockIRISConnector()
        mock_llm = mock_llm_func
        return IrisSQLTool(iris_connector=mock_iris_connector, llm_func=mock_llm)
    
    def test_mysql_to_iris_conversion(self, iris_sql_tool):
        """Test converting MySQL-specific syntax to IRIS."""
        # Arrange
        mysql_query = """
        SELECT CONCAT(first_name, ' ', last_name) as full_name,
               DATE_SUB(NOW(), INTERVAL 30 DAY) as cutoff_date
        FROM users
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        LIMIT 20
        """
        
        iris_query = """
        SELECT (first_name || ' ' || last_name) as full_name,
               DATEADD(day, -30, GETDATE()) as cutoff_date
        FROM users
        WHERE created_at >= DATEADD(day, -30, GETDATE())
        ORDER BY created_at DESC
        FETCH FIRST 20 ROWS ONLY
        """
        
        explanation = "Converted MySQL CONCAT, DATE_SUB, NOW(), and LIMIT to IRIS equivalents"
        
        mock_llm_response = f"""REWRITTEN_SQL:
{iris_query}

EXPLANATION:
{explanation}"""
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=mock_llm_response):
            # Act
            rewritten_sql, explanation_result = iris_sql_tool.rewrite_sql(mysql_query)
            
            # Assert
            assert "||" in rewritten_sql  # String concatenation
            assert "DATEADD" in rewritten_sql  # Date functions
            assert "CONCAT" not in rewritten_sql
            assert "DATE_SUB" not in rewritten_sql
            assert "MySQL" in explanation_result
    
    def test_postgresql_to_iris_conversion(self, iris_sql_tool):
        """Test converting PostgreSQL-specific syntax to IRIS."""
        # Arrange
        postgresql_query = """
        SELECT u.name,
               EXTRACT(YEAR FROM u.created_at) as year_created,
               ARRAY_AGG(o.id) as order_ids
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.created_at >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY u.id, u.name, EXTRACT(YEAR FROM u.created_at)
        LIMIT 15
        """
        
        iris_query = """
        SELECT TOP 15 u.name,
               YEAR(u.created_at) as year_created,
               STRING_AGG(CAST(o.id AS VARCHAR), ',') as order_ids
        FROM users u
        LEFT JOIN orders o ON u.id = o.user_id
        WHERE u.created_at >= DATEADD(day, -7, GETDATE())
        GROUP BY u.id, u.name, YEAR(u.created_at)
        """
        
        explanation = "Converted PostgreSQL EXTRACT, ARRAY_AGG, CURRENT_DATE, and INTERVAL to IRIS functions"
        
        mock_llm_response = f"""REWRITTEN_SQL:
{iris_query}

EXPLANATION:
{explanation}"""
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=mock_llm_response):
            # Act
            rewritten_sql, explanation_result = iris_sql_tool.rewrite_sql(postgresql_query)
            
            # Assert
            assert "TOP 15" in rewritten_sql
            assert "YEAR(" in rewritten_sql
            assert "STRING_AGG" in rewritten_sql
            assert "EXTRACT" not in rewritten_sql
            assert "ARRAY_AGG" not in rewritten_sql
            assert "PostgreSQL" in explanation_result
    
    def test_sql_server_to_iris_conversion(self, iris_sql_tool):
        """Test converting SQL Server syntax to IRIS (minimal changes expected)."""
        # Arrange
        sqlserver_query = """
        SELECT TOP 10 u.name,
               DATEDIFF(day, u.created_at, GETDATE()) as days_since_created
        FROM users u
        WHERE u.active = 1
        ORDER BY u.created_at DESC
        """
        
        iris_query = """
        SELECT TOP 10 u.name,
               DATEDIFF(day, u.created_at, GETDATE()) as days_since_created
        FROM users u
        WHERE u.active = 1
        ORDER BY u.created_at DESC
        """
        
        explanation = "SQL Server syntax is largely compatible with IRIS, minimal changes needed"
        
        mock_llm_response = f"""REWRITTEN_SQL:
{iris_query}

EXPLANATION:
{explanation}"""
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=mock_llm_response):
            # Act
            rewritten_sql, explanation = iris_sql_tool.rewrite_sql(sqlserver_query)
            
            # Assert
            # The rewrite_sql method returns a tuple (sql, explanation)
            assert rewritten_sql is not None
            assert "SELECT TOP 10 u.name" in rewritten_sql
            assert "DATEDIFF(day, u.created_at, GETDATE())" in rewritten_sql
            assert "FROM users u" in rewritten_sql
            assert "WHERE u.active = 1" in rewritten_sql
            assert "ORDER BY u.created_at DESC" in rewritten_sql
            assert "compatible" in explanation.lower()


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and comprehensive error handling."""
    
    @pytest.fixture
    def iris_sql_tool(self):
        """Create an IrisSQLTool instance for edge case testing."""
        mock_iris_connector = MockIRISConnector()
        mock_llm = mock_llm_func
        return IrisSQLTool(iris_connector=mock_iris_connector, llm_func=mock_llm)
    
    def test_very_long_query_handling(self, iris_sql_tool):
        """Test handling of very long SQL queries."""
        # Arrange
        long_query = "SELECT " + ", ".join([f"col{i}" for i in range(100)]) + " FROM large_table LIMIT 1000"
        expected_rewritten = "SELECT TOP 1000 " + ", ".join([f"col{i}" for i in range(100)]) + " FROM large_table"
        expected_explanation = "Converted LIMIT to TOP for IRIS compatibility in large query"
        
        mock_llm_response = f"""REWRITTEN_SQL:
{expected_rewritten}

EXPLANATION:
{expected_explanation}"""
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=mock_llm_response):
            # Act
            rewritten_sql, explanation = iris_sql_tool.rewrite_sql(long_query)
            
            # Assert
            assert "TOP 1000" in rewritten_sql
            assert "LIMIT" not in rewritten_sql
            assert len(rewritten_sql) > 500  # Verify it's still a reasonably long query
    
    def test_special_characters_in_query(self, iris_sql_tool):
        """Test handling of SQL queries with special characters."""
        # Arrange
        special_query = "SELECT name FROM users WHERE description LIKE '%test's \"data\"% AND id > 100'"
        expected_rewritten = "SELECT name FROM users WHERE description LIKE '%test''s \"data\"% AND id > 100'"
        expected_explanation = "Escaped single quotes for IRIS compatibility"
        
        mock_llm_response = f"""REWRITTEN_SQL:
{expected_rewritten}

EXPLANATION:
{expected_explanation}"""
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=mock_llm_response):
            # Act
            rewritten_sql, explanation = iris_sql_tool.rewrite_sql(special_query)
            
            # Assert
            assert "test''s" in rewritten_sql
            assert "special" in explanation.lower() or "escape" in explanation.lower()
    
    def test_unicode_characters_in_query(self, iris_sql_tool):
        """Test handling of SQL queries with Unicode characters."""
        # Arrange
        unicode_query = "SELECT name FROM users WHERE city = 'São Paulo' OR city = '北京'"
        expected_rewritten = "SELECT name FROM users WHERE city = 'São Paulo' OR city = '北京'"
        expected_explanation = "Unicode characters preserved in IRIS-compatible query"
        
        mock_llm_response = f"""REWRITTEN_SQL:
{expected_rewritten}

EXPLANATION:
{expected_explanation}"""
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=mock_llm_response):
            # Act
            rewritten_sql, explanation = iris_sql_tool.rewrite_sql(unicode_query)
            
            # Assert
            assert "São Paulo" in rewritten_sql
            assert "北京" in rewritten_sql
    
    def test_sql_injection_patterns(self, iris_sql_tool):
        """Test handling of potentially malicious SQL patterns."""
        # Arrange
        malicious_query = "SELECT * FROM users WHERE id = 1; DROP TABLE users; --"
        expected_rewritten = "SELECT * FROM users WHERE id = 1"
        expected_explanation = "Removed potentially dangerous SQL injection patterns"
        
        mock_llm_response = f"""REWRITTEN_SQL:
{expected_rewritten}

EXPLANATION:
{expected_explanation}"""
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=mock_llm_response):
            # Act
            rewritten_sql, explanation = iris_sql_tool.rewrite_sql(malicious_query)
            
            # Assert
            assert "DROP TABLE" not in rewritten_sql
            assert "injection" in explanation.lower() or "dangerous" in explanation.lower()
    
    def test_nested_subqueries(self, iris_sql_tool):
        """Test handling of complex nested subqueries."""
        # Arrange
        nested_query = """
        SELECT u.name,
               (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) as order_count,
               (SELECT MAX(amount) FROM orders o WHERE o.user_id = u.id) as max_order
        FROM users u
        WHERE u.id IN (SELECT user_id FROM orders WHERE amount > 100)
        LIMIT 50
        """
        expected_rewritten = """
        SELECT TOP 50 u.name,
               (SELECT COUNT(*) FROM orders o WHERE o.user_id = u.id) as order_count,
               (SELECT MAX(amount) FROM orders o WHERE o.user_id = u.id) as max_order
        FROM users u
        WHERE u.id IN (SELECT user_id FROM orders WHERE amount > 100)
        """
        expected_explanation = "Converted LIMIT to TOP in complex nested query"
        
        mock_llm_response = f"""REWRITTEN_SQL:
{expected_rewritten}

EXPLANATION:
{expected_explanation}"""
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=mock_llm_response):
            # Act
            rewritten_sql, explanation = iris_sql_tool.rewrite_sql(nested_query)
            
            # Assert
            assert "TOP 50" in rewritten_sql
            assert "LIMIT" not in rewritten_sql
            assert rewritten_sql.count("SELECT") == 4  # Main query + 3 subqueries
    
    def test_performance_with_large_result_set(self, iris_sql_tool):
        """Test performance handling with large result sets."""
        # Arrange
        sql_query = "SELECT TOP 10000 * FROM large_table"
        
        # Mock the cursor method to return a cursor with large result set
        with mock.patch.object(iris_sql_tool.iris_connector, 'cursor') as mock_cursor_method:
            mock_cursor = mock.Mock()
            mock_cursor.execute.return_value = None
            large_results = [(f"id_{i}", f"data_{i}") for i in range(10000)]
            mock_cursor.fetchall.return_value = large_results
            mock_cursor.description = [("id",), ("data",)]
            mock_cursor.close.return_value = None
            mock_cursor_method.return_value = mock_cursor
            
            # Act
            results = iris_sql_tool.execute_sql(sql_query)
            
            # Assert
            assert len(results) == 10000
            assert results[0]["id"] == "id_0"
            assert results[9999]["id"] == "id_9999"
            assert all("id" in result and "data" in result for result in results)
    
    def test_concurrent_execution_safety(self, iris_sql_tool):
        """Test that the tool handles concurrent-like execution safely."""
        # Arrange
        sql_query = "SELECT * FROM users"
        
        # Mock the cursor method to return consistent results
        with mock.patch.object(iris_sql_tool.iris_connector, 'cursor') as mock_cursor_method:
            mock_cursor = mock.Mock()
            mock_cursor.execute.return_value = None
            mock_cursor.fetchall.return_value = [("user1", "John")]
            mock_cursor.description = [("id",), ("name",)]
            mock_cursor.close.return_value = None
            mock_cursor_method.return_value = mock_cursor
            
            # Act - Execute multiple times to simulate concurrent usage
            results1 = iris_sql_tool.execute_sql(sql_query)
            results2 = iris_sql_tool.execute_sql(sql_query)
            results3 = iris_sql_tool.execute_sql(sql_query)
            
            # Assert - All executions should return consistent results
            assert results1 == results2 == results3
            assert len(results1) == 1
            assert results1[0]["id"] == "user1"


class TestPromptTemplateValidation:
    """Test the SQL rewrite prompt template and its effectiveness."""
    
    @pytest.fixture
    def iris_sql_tool(self):
        """Create an IrisSQLTool instance for prompt testing."""
        mock_iris_connector = MockIRISConnector()
        mock_llm = mock_llm_func
        return IrisSQLTool(iris_connector=mock_iris_connector, llm_func=mock_llm)
    
    def test_prompt_template_contains_iris_rules(self, iris_sql_tool):
        """Test that the prompt template contains key IRIS SQL rules."""
        # Act
        prompt_template = iris_sql_tool.SQL_REWRITE_PROMPT_TEMPLATE
        
        # Assert
        assert "TOP instead of LIMIT" in prompt_template
        assert "TO_VECTOR" in prompt_template
        assert "|| operator" in prompt_template
        assert "IRIS" in prompt_template
        assert "REWRITTEN_SQL:" in prompt_template
        assert "EXPLANATION:" in prompt_template
    
    def test_prompt_formatting_with_query(self, iris_sql_tool):
        """Test that the prompt template formats correctly with a query."""
        # Arrange
        test_query = "SELECT * FROM test_table LIMIT 10"
        
        # Act
        formatted_prompt = iris_sql_tool.SQL_REWRITE_PROMPT_TEMPLATE.format(
            original_query=test_query
        )
        
        # Assert
        assert test_query in formatted_prompt
        assert "Original SQL Query:" in formatted_prompt
        assert "Please rewrite this query" in formatted_prompt
    
    def test_llm_function_receives_correct_prompt(self, iris_sql_tool):
        """Test that the LLM function receives the correctly formatted prompt."""
        # Arrange
        original_query = "SELECT * FROM users LIMIT 5"
        expected_prompt_content = [
            "InterSystems IRIS SQL syntax",
            "TOP instead of LIMIT",
            original_query,
            "REWRITTEN_SQL:",
            "EXPLANATION:"
        ]
        
        mock_llm_response = """REWRITTEN_SQL:
SELECT TOP 5 * FROM users

EXPLANATION:
Changed LIMIT to TOP"""
        
        # Mock the LLM function to capture the prompt
        captured_prompt = None
        def capture_llm_func(prompt):
            nonlocal captured_prompt
            captured_prompt = prompt
            return mock_llm_response
        
        with mock.patch.object(iris_sql_tool, 'llm_func', side_effect=capture_llm_func):
            # Act
            iris_sql_tool.rewrite_sql(original_query)
            
            # Assert
            assert captured_prompt is not None
            for expected_content in expected_prompt_content:
                assert expected_content in captured_prompt


class TestErrorRecoveryAndResilience:
    """Test error recovery and system resilience."""
    
    @pytest.fixture
    def iris_sql_tool(self):
        """Create an IrisSQLTool instance for resilience testing."""
        mock_iris_connector = MockIRISConnector()
        mock_llm = mock_llm_func
        return IrisSQLTool(iris_connector=mock_iris_connector, llm_func=mock_llm)
    
    def test_recovery_from_partial_llm_response(self, iris_sql_tool):
        """Test recovery when LLM provides partial response."""
        # Arrange
        original_query = "SELECT * FROM users LIMIT 10"
        partial_response = "REWRITTEN_SQL:\nSELECT TOP 10 * FROM users\n\nEXPLANA"  # Cut off
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=partial_response):
            # Act
            rewritten_sql, explanation = iris_sql_tool.rewrite_sql(original_query)
            
            # Assert
            assert "SELECT TOP 10 * FROM users" in rewritten_sql
            assert explanation == "No explanation provided by LLM"
    
    def test_recovery_from_malformed_llm_response(self, iris_sql_tool):
        """Test recovery when LLM provides completely malformed response."""
        # Arrange
        original_query = "SELECT * FROM users LIMIT 10"
        malformed_response = "This is not a properly formatted response at all!"
        
        with mock.patch.object(iris_sql_tool, 'llm_func', return_value=malformed_response):
            # Act
            rewritten_sql, explanation = iris_sql_tool.rewrite_sql(original_query)
            
            # Assert
            assert rewritten_sql == malformed_response
            assert explanation == "Failed to parse LLM response format"
    
    def test_database_connection_resilience(self, iris_sql_tool):
        """Test resilience when database connection is unstable."""
        # Arrange
        sql_query = "SELECT * FROM users"
        
        # Mock unstable connection that fails then succeeds
        call_count = 0
        def unstable_cursor():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Connection timeout")
            else:
                mock_cursor = mock.Mock()
                mock_cursor.execute.return_value = None
                mock_cursor.fetchall.return_value = [("user1", "John")]
                mock_cursor.description = [("id",), ("name",)]
                mock_cursor.close.return_value = None
                return mock_cursor
        
        with mock.patch.object(iris_sql_tool.iris_connector, 'cursor', side_effect=unstable_cursor):
            # Act & Assert - First call should fail
            with pytest.raises(RuntimeError, match="SQL execution failed"):
                iris_sql_tool.execute_sql(sql_query)
            
            # Second call should succeed (in a real scenario, this would be a retry)
            # For this test, we'll just verify the mock behavior
            assert call_count == 1
    
    def test_memory_cleanup_on_large_operations(self, iris_sql_tool):
        """Test that memory is properly cleaned up during large operations."""
        # Arrange
        sql_query = "SELECT * FROM large_table"
        
        # Mock the cursor method to return a cursor with large result set
        with mock.patch.object(iris_sql_tool.iris_connector, 'cursor') as mock_cursor_method:
            mock_cursor = mock.Mock()
            mock_cursor.execute.return_value = None
            large_results = [(f"id_{i}", f"data_{i}" * 100) for i in range(1000)]  # Large strings
            mock_cursor.fetchall.return_value = large_results
            mock_cursor.description = [("id",), ("data",)]
            mock_cursor.close.return_value = None
            mock_cursor_method.return_value = mock_cursor
            
            # Act
            results = iris_sql_tool.execute_sql(sql_query)
            
            # Assert
            assert len(results) == 1000
            # Verify that the cursor was properly closed (cleanup)
            mock_cursor.close.assert_called_once()
            # In a real implementation, we'd check memory usage, but here we verify structure
            assert all(isinstance(result, dict) for result in results)
            assert all("id" in result and "data" in result for result in results)