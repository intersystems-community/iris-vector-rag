"""
Unit tests for authentication middleware.

Tests the API key authentication middleware in isolation.
"""

import base64
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from iris_rag.api.middleware.auth import (
    ApiKeyAuthMiddleware,
    extract_api_key,
    validate_api_key_format,
)
from iris_rag.api.models.auth import ApiKey, Permission, RateLimitTier


class TestExtractApiKey:
    """Test API key extraction from Authorization header."""

    def test_extract_valid_base64_key(self):
        """Test extracting valid base64-encoded API key."""
        key_id = "test-key-id"
        key_secret = "test-secret"
        credentials = f"{key_id}:{key_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        header = f"ApiKey {encoded}"

        result = extract_api_key(header)

        assert result == (key_id, key_secret)

    def test_extract_key_missing_prefix(self):
        """Test extraction fails without 'ApiKey' prefix."""
        key_id = "test-key-id"
        key_secret = "test-secret"
        credentials = f"{key_id}:{key_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        header = f"Bearer {encoded}"  # Wrong prefix

        with pytest.raises(HTTPException) as exc:
            extract_api_key(header)

        assert exc.value.status_code == 401
        assert "Invalid authorization header format" in str(exc.value.detail)

    def test_extract_key_invalid_base64(self):
        """Test extraction fails with invalid base64."""
        header = "ApiKey not-valid-base64!!!"

        with pytest.raises(HTTPException) as exc:
            extract_api_key(header)

        assert exc.value.status_code == 401

    def test_extract_key_missing_colon(self):
        """Test extraction fails without colon separator."""
        credentials = "key-id-no-colon"
        encoded = base64.b64encode(credentials.encode()).decode()
        header = f"ApiKey {encoded}"

        with pytest.raises(HTTPException) as exc:
            extract_api_key(header)

        assert exc.value.status_code == 401
        assert "Invalid API key format" in str(exc.value.detail)

    def test_extract_key_empty_header(self):
        """Test extraction fails with empty header."""
        with pytest.raises(HTTPException) as exc:
            extract_api_key("")

        assert exc.value.status_code == 401


class TestValidateApiKeyFormat:
    """Test API key format validation."""

    def test_validate_valid_uuid_format(self):
        """Test validation passes for UUID format."""
        key_id = "550e8400-e29b-41d4-a716-446655440000"
        assert validate_api_key_format(key_id) is True

    def test_validate_valid_alphanumeric_format(self):
        """Test validation passes for alphanumeric format."""
        key_id = "abc123def456"
        assert validate_api_key_format(key_id) is True

    def test_validate_invalid_empty_key(self):
        """Test validation fails for empty key."""
        assert validate_api_key_format("") is False

    def test_validate_invalid_special_chars(self):
        """Test validation fails for special characters."""
        key_id = "key-with-$pecial-chars!"
        assert validate_api_key_format(key_id) is False

    def test_validate_invalid_whitespace(self):
        """Test validation fails with whitespace."""
        key_id = "key with spaces"
        assert validate_api_key_format(key_id) is False


class TestApiKeyAuthMiddleware:
    """Test API key authentication middleware."""

    @pytest.fixture
    def mock_auth_service(self):
        """Create mock authentication service."""
        service = MagicMock()
        return service

    @pytest.fixture
    def middleware(self, mock_auth_service):
        """Create middleware instance."""
        return ApiKeyAuthMiddleware(mock_auth_service)

    def test_authenticate_valid_key(self, middleware, mock_auth_service):
        """Test authentication succeeds with valid API key."""
        key_id = "550e8400-e29b-41d4-a716-446655440000"
        key_secret = "test-secret-123"

        api_key = ApiKey(
            key_id=key_id,
            key_secret_hash="hashed-secret",
            name="Test Key",
            permissions=[Permission.READ],
            rate_limit_tier=RateLimitTier.BASIC,
            requests_per_minute=60,
            requests_per_hour=1000,
            created_at=datetime.utcnow(),
            is_active=True,
            owner_email="test@example.com",
        )

        mock_auth_service.verify_api_key.return_value = api_key

        credentials = f"{key_id}:{key_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        header = f"ApiKey {encoded}"

        result = middleware.authenticate(header)

        assert result == api_key
        mock_auth_service.verify_api_key.assert_called_once_with(key_id, key_secret)

    def test_authenticate_inactive_key(self, middleware, mock_auth_service):
        """Test authentication fails for inactive key."""
        key_id = "550e8400-e29b-41d4-a716-446655440000"
        key_secret = "test-secret-123"

        mock_auth_service.verify_api_key.side_effect = HTTPException(
            status_code=401, detail="API key is inactive"
        )

        credentials = f"{key_id}:{key_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        header = f"ApiKey {encoded}"

        with pytest.raises(HTTPException) as exc:
            middleware.authenticate(header)

        assert exc.value.status_code == 401
        assert "inactive" in str(exc.value.detail).lower()

    def test_authenticate_expired_key(self, middleware, mock_auth_service):
        """Test authentication fails for expired key."""
        key_id = "550e8400-e29b-41d4-a716-446655440000"
        key_secret = "test-secret-123"

        api_key = ApiKey(
            key_id=key_id,
            key_secret_hash="hashed-secret",
            name="Test Key",
            permissions=[Permission.READ],
            rate_limit_tier=RateLimitTier.BASIC,
            requests_per_minute=60,
            requests_per_hour=1000,
            created_at=datetime.utcnow() - timedelta(days=60),
            expires_at=datetime.utcnow() - timedelta(days=1),  # Expired
            is_active=True,
            owner_email="test@example.com",
        )

        mock_auth_service.verify_api_key.return_value = api_key

        credentials = f"{key_id}:{key_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        header = f"ApiKey {encoded}"

        with pytest.raises(HTTPException) as exc:
            middleware.authenticate(header)

        assert exc.value.status_code == 401
        assert "expired" in str(exc.value.detail).lower()

    def test_authenticate_wrong_secret(self, middleware, mock_auth_service):
        """Test authentication fails with wrong secret."""
        key_id = "550e8400-e29b-41d4-a716-446655440000"
        key_secret = "wrong-secret"

        mock_auth_service.verify_api_key.side_effect = HTTPException(
            status_code=401, detail="Invalid API key credentials"
        )

        credentials = f"{key_id}:{key_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        header = f"ApiKey {encoded}"

        with pytest.raises(HTTPException) as exc:
            middleware.authenticate(header)

        assert exc.value.status_code == 401

    def test_check_permission_has_permission(self, middleware):
        """Test permission check succeeds when user has permission."""
        api_key = ApiKey(
            key_id="test-key",
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

        # Should not raise
        middleware.check_permission(api_key, Permission.READ)
        middleware.check_permission(api_key, Permission.WRITE)

    def test_check_permission_missing_permission(self, middleware):
        """Test permission check fails when user lacks permission."""
        api_key = ApiKey(
            key_id="test-key",
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
            middleware.check_permission(api_key, Permission.ADMIN)

        assert exc.value.status_code == 403
        assert "permission" in str(exc.value.detail).lower()

    def test_check_permission_admin_has_all(self, middleware):
        """Test admin permission grants access to everything."""
        api_key = ApiKey(
            key_id="test-key",
            key_secret_hash="hash",
            name="Test",
            permissions=[Permission.ADMIN],
            rate_limit_tier=RateLimitTier.ENTERPRISE,
            requests_per_minute=1000,
            requests_per_hour=50000,
            created_at=datetime.utcnow(),
            is_active=True,
            owner_email="admin@example.com",
        )

        # Admin should have all permissions
        middleware.check_permission(api_key, Permission.READ)
        middleware.check_permission(api_key, Permission.WRITE)
        middleware.check_permission(api_key, Permission.ADMIN)
