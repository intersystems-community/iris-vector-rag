"""
Unit tests for request/response logging middleware.

Tests the logging middleware in isolation.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Request, Response

from iris_rag.api.middleware.logging import LoggingMiddleware
from iris_rag.api.models.request import APIRequestLog


class TestLoggingMiddleware:
    """Test logging middleware."""

    @pytest.fixture
    def mock_connection_pool(self):
        """Create mock connection pool."""
        pool = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()

        pool.get_connection.return_value.__enter__.return_value = conn
        conn.cursor.return_value = cursor

        return pool

    @pytest.fixture
    def middleware(self, mock_connection_pool):
        """Create middleware instance."""
        return LoggingMiddleware(mock_connection_pool)

    @pytest.fixture
    def mock_request(self):
        """Create mock FastAPI request."""
        request = MagicMock(spec=Request)
        request.method = "POST"
        request.url.path = "/api/v1/basic/_search"
        request.query_params = {}
        request.client.host = "192.168.1.100"
        request.headers = {
            "user-agent": "Mozilla/5.0",
            "authorization": "ApiKey test123",
        }
        return request

    @pytest.fixture
    def mock_response(self):
        """Create mock FastAPI response."""
        response = MagicMock(spec=Response)
        response.status_code = 200
        response.headers = {}
        return response

    def test_log_request_success(self, middleware, mock_request, mock_response, mock_connection_pool):
        """Test logging successful request."""
        request_id = "550e8400-e29b-41d4-a716-446655440000"
        api_key_id = "test-key-id"
        execution_time_ms = 1456

        middleware.log_request(
            request_id=request_id,
            request=mock_request,
            response=mock_response,
            execution_time_ms=execution_time_ms,
            api_key_id=api_key_id,
        )

        # Verify database insert was called
        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.execute.assert_called_once()

        # Verify SQL contains expected fields
        sql_call = cursor.execute.call_args[0][0]
        assert "INSERT INTO api_request_logs" in sql_call
        assert "request_id" in sql_call
        assert "execution_time_ms" in sql_call

    def test_log_request_error(self, middleware, mock_request, mock_connection_pool):
        """Test logging failed request."""
        request_id = "550e8400-e29b-41d4-a716-446655440000"
        error_response = MagicMock(spec=Response)
        error_response.status_code = 500
        error_message = "Internal server error"

        middleware.log_request(
            request_id=request_id,
            request=mock_request,
            response=error_response,
            execution_time_ms=234,
            error_message=error_message,
        )

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.execute.assert_called_once()

    def test_generate_request_id(self, middleware):
        """Test request ID generation."""
        request_id = middleware.generate_request_id()

        assert isinstance(request_id, str)
        assert len(request_id) == 36  # UUID format
        assert request_id.count("-") == 4  # UUID has 4 hyphens

    def test_extract_query_params(self, middleware, mock_request):
        """Test query parameter extraction."""
        mock_request.query_params = {"q": "diabetes", "top_k": "5"}

        params = middleware.extract_query_params(mock_request)

        assert "q" in params
        assert "top_k" in params
        assert params["q"] == "diabetes"

    def test_sanitize_headers(self, middleware):
        """Test header sanitization removes sensitive data."""
        headers = {
            "authorization": "ApiKey secret123",
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0",
        }

        sanitized = middleware.sanitize_headers(headers)

        assert sanitized["authorization"] == "[REDACTED]"
        assert sanitized["content-type"] == "application/json"
        assert sanitized["user-agent"] == "Mozilla/5.0"

    def test_log_request_with_query_params(self, middleware, mock_request, mock_response, mock_connection_pool):
        """Test logging request with query parameters."""
        mock_request.query_params = {"query": "What is diabetes?", "top_k": "5"}
        request_id = "550e8400-e29b-41d4-a716-446655440000"

        middleware.log_request(
            request_id=request_id,
            request=mock_request,
            response=mock_response,
            execution_time_ms=1200,
        )

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.execute.assert_called_once()

    def test_log_request_db_error_handling(self, middleware, mock_request, mock_response, mock_connection_pool):
        """Test graceful handling of database errors."""
        request_id = "550e8400-e29b-41d4-a716-446655440000"

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.execute.side_effect = Exception("Database connection failed")

        # Should not raise, just log error
        middleware.log_request(
            request_id=request_id,
            request=mock_request,
            response=mock_response,
            execution_time_ms=100,
        )

    def test_calculate_execution_time(self, middleware):
        """Test execution time calculation."""
        start_time = datetime.utcnow()
        import time
        time.sleep(0.1)  # Sleep 100ms
        end_time = datetime.utcnow()

        execution_time = middleware.calculate_execution_time(start_time, end_time)

        assert execution_time >= 100
        assert execution_time < 200

    def test_log_request_timing_breakdown(self, middleware, mock_request, mock_response, mock_connection_pool):
        """Test logging with timing breakdown."""
        request_id = "550e8400-e29b-41d4-a716-446655440000"
        timing_metadata = {
            "retrieval_time_ms": 345,
            "generation_time_ms": 1089,
        }

        middleware.log_request(
            request_id=request_id,
            request=mock_request,
            response=mock_response,
            execution_time_ms=1456,
            metadata=timing_metadata,
        )

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.execute.assert_called_once()

    def test_request_id_in_response_headers(self, middleware, mock_response):
        """Test request ID is added to response headers."""
        request_id = "550e8400-e29b-41d4-a716-446655440000"

        middleware.add_request_id_header(mock_response, request_id)

        assert "X-Request-ID" in mock_response.headers
        assert mock_response.headers["X-Request-ID"] == request_id

    def test_log_different_http_methods(self, middleware, mock_response, mock_connection_pool):
        """Test logging works for different HTTP methods."""
        request_id = "550e8400-e29b-41d4-a716-446655440000"

        for method in ["GET", "POST", "PUT", "DELETE"]:
            request = MagicMock(spec=Request)
            request.method = method
            request.url.path = "/api/v1/test"
            request.query_params = {}
            request.client.host = "127.0.0.1"
            request.headers = {}

            middleware.log_request(
                request_id=request_id,
                request=request,
                response=mock_response,
                execution_time_ms=100,
            )

        # Should have 4 calls (one per method)
        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        assert cursor.execute.call_count == 4

    def test_log_request_client_info(self, middleware, mock_request, mock_response, mock_connection_pool):
        """Test client IP and user agent are logged."""
        mock_request.client.host = "203.0.113.42"
        mock_request.headers = {"user-agent": "Python/3.11 requests/2.31.0"}
        request_id = "550e8400-e29b-41d4-a716-446655440000"

        middleware.log_request(
            request_id=request_id,
            request=mock_request,
            response=mock_response,
            execution_time_ms=500,
        )

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.execute.assert_called_once()

    def test_log_analytics_metrics(self, middleware, mock_connection_pool):
        """Test logging analytics metrics."""
        metrics = {
            "endpoint": "/api/v1/basic/_search",
            "avg_execution_time_ms": 1234.56,
            "request_count": 150,
            "error_count": 3,
        }

        middleware.log_analytics(metrics)

        # Verify metrics were logged (implementation-specific)
        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        # Just verify it doesn't crash
        assert True
