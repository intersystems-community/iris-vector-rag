"""
Unit tests for authentication service.

Tests the API key management service in isolation.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import bcrypt
import pytest
from fastapi import HTTPException

from iris_rag.api.models.auth import ApiKey, ApiKeyCreateRequest, Permission, RateLimitTier
from iris_rag.api.services.auth_service import AuthService


class TestAuthService:
    """Test authentication service."""

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
    def auth_service(self, mock_connection_pool):
        """Create auth service instance."""
        return AuthService(mock_connection_pool)

    def test_create_api_key_basic_tier(self, auth_service, mock_connection_pool):
        """Test creating API key with basic tier."""
        request = ApiKeyCreateRequest(
            name="Test Key",
            permissions=[Permission.READ],
            rate_limit_tier=RateLimitTier.BASIC,
            description="Test key",
        )
        owner_email = "test@example.com"

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value

        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.side_effect = ["key-id-uuid", "secret-uuid"]

            response = auth_service.create_api_key(request, owner_email)

        assert response.key_id is not None
        assert response.key_secret is not None
        assert response.name == "Test Key"
        assert response.rate_limit_tier == RateLimitTier.BASIC
        assert Permission.READ in response.permissions

        # Verify database insert
        cursor.execute.assert_called_once()

    def test_create_api_key_premium_tier(self, auth_service, mock_connection_pool):
        """Test creating API key with premium tier."""
        request = ApiKeyCreateRequest(
            name="Premium Key",
            permissions=[Permission.READ, Permission.WRITE],
            rate_limit_tier=RateLimitTier.PREMIUM,
            expires_in_days=90,
        )
        owner_email = "premium@example.com"

        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.side_effect = ["key-id-uuid", "secret-uuid"]

            response = auth_service.create_api_key(request, owner_email)

        assert response.rate_limit_tier == RateLimitTier.PREMIUM
        assert Permission.READ in response.permissions
        assert Permission.WRITE in response.permissions
        assert response.expires_at is not None

    def test_create_api_key_enterprise_tier(self, auth_service, mock_connection_pool):
        """Test creating API key with enterprise tier."""
        request = ApiKeyCreateRequest(
            name="Enterprise Key",
            permissions=[Permission.ADMIN],
            rate_limit_tier=RateLimitTier.ENTERPRISE,
        )
        owner_email = "enterprise@example.com"

        with patch("uuid.uuid4") as mock_uuid:
            mock_uuid.side_effect = ["key-id-uuid", "secret-uuid"]

            response = auth_service.create_api_key(request, owner_email)

        assert response.rate_limit_tier == RateLimitTier.ENTERPRISE
        assert Permission.ADMIN in response.permissions

    def test_verify_api_key_valid(self, auth_service, mock_connection_pool):
        """Test verifying valid API key."""
        key_id = "test-key-id"
        key_secret = "test-secret"
        hashed_secret = bcrypt.hashpw(key_secret.encode(), bcrypt.gensalt(12))

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.fetchone.return_value = (
            key_id,
            hashed_secret.decode(),
            "Test Key",
            "read",
            "basic",
            60,
            1000,
            datetime.utcnow(),
            None,  # expires_at
            1,  # is_active
            "test@example.com",
            "Test description",
        )

        api_key = auth_service.verify_api_key(key_id, key_secret)

        assert api_key.key_id == key_id
        assert api_key.is_active is True
        assert Permission.READ in api_key.permissions

    def test_verify_api_key_wrong_secret(self, auth_service, mock_connection_pool):
        """Test verifying API key with wrong secret."""
        key_id = "test-key-id"
        key_secret = "wrong-secret"
        correct_secret = "correct-secret"
        hashed_secret = bcrypt.hashpw(correct_secret.encode(), bcrypt.gensalt(12))

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.fetchone.return_value = (
            key_id,
            hashed_secret.decode(),
            "Test Key",
            "read",
            "basic",
            60,
            1000,
            datetime.utcnow(),
            None,
            1,
            "test@example.com",
            None,
        )

        with pytest.raises(HTTPException) as exc:
            auth_service.verify_api_key(key_id, key_secret)

        assert exc.value.status_code == 401

    def test_verify_api_key_not_found(self, auth_service, mock_connection_pool):
        """Test verifying non-existent API key."""
        key_id = "nonexistent-key"
        key_secret = "any-secret"

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.fetchone.return_value = None

        with pytest.raises(HTTPException) as exc:
            auth_service.verify_api_key(key_id, key_secret)

        assert exc.value.status_code == 401

    def test_verify_api_key_inactive(self, auth_service, mock_connection_pool):
        """Test verifying inactive API key."""
        key_id = "test-key-id"
        key_secret = "test-secret"
        hashed_secret = bcrypt.hashpw(key_secret.encode(), bcrypt.gensalt(12))

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.fetchone.return_value = (
            key_id,
            hashed_secret.decode(),
            "Test Key",
            "read",
            "basic",
            60,
            1000,
            datetime.utcnow(),
            None,
            0,  # is_active = False
            "test@example.com",
            None,
        )

        with pytest.raises(HTTPException) as exc:
            auth_service.verify_api_key(key_id, key_secret)

        assert exc.value.status_code == 401
        assert "inactive" in str(exc.value.detail).lower()

    def test_list_api_keys_all(self, auth_service, mock_connection_pool):
        """Test listing all API keys."""
        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.fetchall.return_value = [
            (
                "key-1",
                "hash1",
                "Key 1",
                "read",
                "basic",
                60,
                1000,
                datetime.utcnow(),
                None,
                1,
                "user1@example.com",
                "Description 1",
            ),
            (
                "key-2",
                "hash2",
                "Key 2",
                "read,write",
                "premium",
                100,
                5000,
                datetime.utcnow(),
                None,
                1,
                "user2@example.com",
                "Description 2",
            ),
        ]

        keys = auth_service.list_api_keys()

        assert len(keys) == 2
        assert keys[0].key_id == "key-1"
        assert keys[1].key_id == "key-2"

    def test_list_api_keys_by_email(self, auth_service, mock_connection_pool):
        """Test listing API keys filtered by email."""
        owner_email = "user1@example.com"

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.fetchall.return_value = [
            (
                "key-1",
                "hash1",
                "Key 1",
                "read",
                "basic",
                60,
                1000,
                datetime.utcnow(),
                None,
                1,
                owner_email,
                None,
            ),
        ]

        keys = auth_service.list_api_keys(owner_email=owner_email)

        assert len(keys) == 1
        assert keys[0].owner_email == owner_email

    def test_revoke_api_key_success(self, auth_service, mock_connection_pool):
        """Test revoking an API key."""
        key_id = "test-key-id"

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.rowcount = 1

        result = auth_service.revoke_api_key(key_id)

        assert result is True
        cursor.execute.assert_called_once()

    def test_revoke_api_key_not_found(self, auth_service, mock_connection_pool):
        """Test revoking non-existent API key."""
        key_id = "nonexistent-key"

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.rowcount = 0

        result = auth_service.revoke_api_key(key_id)

        assert result is False

    def test_hash_api_key_secret(self, auth_service):
        """Test API key secret hashing."""
        secret = "my-secret-key"

        hashed = auth_service.hash_secret(secret)

        assert hashed != secret
        assert bcrypt.checkpw(secret.encode(), hashed.encode())

    def test_generate_api_key_secret(self, auth_service):
        """Test generating random API key secret."""
        secret = auth_service.generate_secret()

        assert len(secret) >= 32
        assert isinstance(secret, str)

    def test_get_api_key_by_id(self, auth_service, mock_connection_pool):
        """Test retrieving API key by ID."""
        key_id = "test-key-id"

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.fetchone.return_value = (
            key_id,
            "hashed-secret",
            "Test Key",
            "read,write",
            "premium",
            100,
            5000,
            datetime.utcnow(),
            None,
            1,
            "test@example.com",
            "Test description",
        )

        api_key = auth_service.get_api_key(key_id)

        assert api_key.key_id == key_id
        assert Permission.READ in api_key.permissions
        assert Permission.WRITE in api_key.permissions

    def test_update_last_used_timestamp(self, auth_service, mock_connection_pool):
        """Test updating last used timestamp."""
        key_id = "test-key-id"

        auth_service.update_last_used(key_id)

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.execute.assert_called_once()

    def test_bcrypt_cost_factor(self, auth_service):
        """Test bcrypt uses correct cost factor."""
        secret = "test-secret"
        hashed = auth_service.hash_secret(secret)

        # Cost factor 12 should produce hash starting with $2b$12$
        assert hashed.startswith("$2b$12$")
