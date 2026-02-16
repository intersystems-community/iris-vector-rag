"""
Unit tests for request/response logging middleware.
"""

import uuid
from unittest.mock import MagicMock

import pytest
from starlette.requests import Request
from starlette.responses import Response

from iris_vector_rag.api.middleware.logging import RequestLoggingMiddleware


def _make_request(headers: dict[str, str] | None = None, method: str = "POST") -> Request:
    header_items = []
    if headers:
        header_items = [(k.lower().encode(), v.encode()) for k, v in headers.items()]

    scope = {
        "type": "http",
        "method": method,
        "path": "/api/v1/basic/_search",
        "headers": header_items,
        "query_string": b"",
        "client": ("127.0.0.1", 12345),
    }
    return Request(scope)


def _make_connection_pool():
    pool = MagicMock()
    conn = MagicMock()
    cursor = MagicMock()
    pool.get_connection.return_value.__enter__.return_value = conn
    conn.cursor.return_value = cursor
    return pool


def _make_middleware(connection_pool):
    async def app(scope, receive, send):
        return None

    return RequestLoggingMiddleware(app, connection_pool)


class TestRequestLoggingMiddleware:
    def test_generate_request_id(self):
        middleware = _make_middleware(_make_connection_pool())
        request_id = middleware.generate_request_id()

        assert isinstance(request_id, uuid.UUID)

    @pytest.mark.asyncio
    async def test_dispatch_logs_success_and_sets_headers(self):
        connection_pool = _make_connection_pool()
        middleware = _make_middleware(connection_pool)
        request = _make_request({"User-Agent": "pytest"})

        async def call_next(_request):
            return Response("ok", status_code=200)

        response = await middleware.dispatch(request, call_next)

        cursor = (
            connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        )
        cursor.execute.assert_called_once()

        assert "X-Request-ID" in response.headers
        assert "X-Execution-Time-Ms" in response.headers

    @pytest.mark.asyncio
    async def test_dispatch_respects_client_request_id(self):
        connection_pool = _make_connection_pool()
        middleware = _make_middleware(connection_pool)
        client_request_id = str(uuid.uuid4())
        request = _make_request({"X-Request-ID": client_request_id})

        async def call_next(_request):
            return Response("ok", status_code=200)

        response = await middleware.dispatch(request, call_next)

        assert response.headers["X-Request-ID"] == client_request_id

    @pytest.mark.asyncio
    async def test_dispatch_logs_exception_and_reraises(self):
        connection_pool = _make_connection_pool()
        middleware = _make_middleware(connection_pool)
        request = _make_request()

        async def call_next(_request):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            await middleware.dispatch(request, call_next)

        cursor = (
            connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        )
        cursor.execute.assert_called_once()
