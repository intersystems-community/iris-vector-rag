"""
Unit tests for authentication middleware components.
"""

import base64
from datetime import datetime, timedelta
from uuid import uuid4
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from iris_vector_rag.api.middleware.auth import ApiKeyAuth
from iris_vector_rag.api.models.auth import ApiKey, Permission, RateLimitTier


def _make_request(auth_header: str | None) -> Request:
    headers = []
    if auth_header is not None:
        headers = [(b"authorization", auth_header.encode())]

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/basic/_search",
        "headers": headers,
    }
    return Request(scope)


class TestParseApiKey:
    """Test API key parsing from base64 credentials."""

    @pytest.fixture
    def auth(self):
        return ApiKeyAuth(MagicMock())

    def test_parse_valid_base64_key(self, auth):
        key_id = str(uuid4())
        key_secret = "test-secret"
        credentials = f"{key_id}:{key_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()

        result = auth.parse_api_key(encoded)

        assert result == (key_id, key_secret)

    def test_parse_invalid_base64(self, auth):
        with pytest.raises(HTTPException) as exc:
            auth.parse_api_key("not-valid-base64!!!")

        assert exc.value.status_code == 401

    def test_parse_missing_colon(self, auth):
        credentials = "key-id-no-colon"
        encoded = base64.b64encode(credentials.encode()).decode()

        with pytest.raises(HTTPException) as exc:
            auth.parse_api_key(encoded)

        assert exc.value.status_code == 401

    def test_parse_invalid_uuid(self, auth):
        credentials = "not-a-uuid:test-secret"
        encoded = base64.b64encode(credentials.encode()).decode()

        with pytest.raises(HTTPException) as exc:
            auth.parse_api_key(encoded)

        assert exc.value.status_code == 401


class TestApiKeyAuthPermissions:
    """Test permission checks."""

    @pytest.fixture
    def auth(self):
        return ApiKeyAuth(MagicMock())

    def test_check_permission_has_permission(self, auth):
        api_key = ApiKey(
            key_id=uuid4(),
            key_secret_hash="hash",
            name="Test",
            permissions=[Permission.READ, Permission.WRITE],
            rate_limit_tier=RateLimitTier.BASIC,
            requests_per_minute=60,
            requests_per_hour=1000,
            created_at=datetime.utcnow(),
            is_active=True,
            owner_email="test@example.com",
        )

        auth.check_permission(api_key, Permission.READ)
        auth.check_permission(api_key, Permission.WRITE)

    def test_check_permission_missing_permission(self, auth):
        api_key = ApiKey(
            key_id=uuid4(),
            key_secret_hash="hash",
            name="Test",
            permissions=[Permission.READ],
            rate_limit_tier=RateLimitTier.BASIC,
            requests_per_minute=60,
            requests_per_hour=1000,
            created_at=datetime.utcnow(),
            is_active=True,
            owner_email="test@example.com",
        )

        with pytest.raises(HTTPException) as exc:
            auth.check_permission(api_key, Permission.ADMIN)

        assert exc.value.status_code == 403

    def test_check_permission_admin_has_all(self, auth):
        api_key = ApiKey(
            key_id=uuid4(),
            key_secret_hash="hash",
            name="Admin",
            permissions=[Permission.ADMIN],
            rate_limit_tier=RateLimitTier.ENTERPRISE,
            requests_per_minute=1000,
            requests_per_hour=50000,
            created_at=datetime.utcnow(),
            is_active=True,
            owner_email="admin@example.com",
        )

        auth.check_permission(api_key, Permission.READ)
        auth.check_permission(api_key, Permission.WRITE)
        auth.check_permission(api_key, Permission.ADMIN)


class TestApiKeyAuthCall:
    """Test end-to-end auth flow without database access."""

    @pytest.fixture
    def auth(self):
        return ApiKeyAuth(MagicMock())

    @pytest.mark.asyncio
    async def test_missing_authorization_header(self, auth):
        request = _make_request(None)

        with pytest.raises(HTTPException) as exc:
            await auth(request)

        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_authorization_header_format(self, auth):
        request = _make_request("Bearer token")

        with pytest.raises(HTTPException) as exc:
            await auth(request)

        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_authorization_header(self, auth):
        key_id = str(uuid4())
        key_secret = "test-secret"
        credentials = f"{key_id}:{key_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()

        api_key = ApiKey(
            key_id=uuid4(),
            key_secret_hash="hash",
            name="Test",
            permissions=[Permission.READ],
            rate_limit_tier=RateLimitTier.BASIC,
            requests_per_minute=60,
            requests_per_hour=1000,
            created_at=datetime.utcnow() - timedelta(days=1),
            is_active=True,
            owner_email="test@example.com",
        )

        auth.verify_api_key = MagicMock(return_value=api_key)

        request = _make_request(f"ApiKey {encoded}")
        result = await auth(request)

        assert result == api_key
        assert request.state.api_key == api_key
