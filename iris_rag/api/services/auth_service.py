"""
Authentication Service for RAG API.

Implements FR-010 to FR-012: API key lifecycle management.
Provides CRUD operations for API keys with bcrypt hashing.
"""

import logging
from typing import List, Optional
from uuid import UUID, uuid4
from datetime import datetime, timedelta

import bcrypt

from iris_rag.api.models.auth import (
    ApiKey,
    ApiKeyCreateRequest,
    ApiKeyResponse,
    Permission,
    RateLimitTier
)


logger = logging.getLogger(__name__)


class AuthService:
    """
    Service for managing API keys.

    Implements FR-010: API key creation and validation
    Implements FR-011: API key permissions and expiration
    Implements FR-012: API key revocation
    """

    def __init__(self, connection_pool, bcrypt_rounds: int = 12):
        """
        Initialize authentication service.

        Args:
            connection_pool: IRISConnectionPool for database operations
            bcrypt_rounds: bcrypt cost factor for password hashing
        """
        self.connection_pool = connection_pool
        self.bcrypt_rounds = bcrypt_rounds

    def create_api_key(
        self,
        request: ApiKeyCreateRequest,
        owner_email: str
    ) -> ApiKeyResponse:
        """
        Create new API key.

        Args:
            request: API key creation request
            owner_email: Email of key owner

        Returns:
            ApiKeyResponse with key_id and plaintext secret

        Implements FR-010: API key generation with bcrypt hashing
        """
        # Generate UUIDs
        key_id = uuid4()
        key_secret = uuid4().hex  # 32-character hex string

        # Hash secret with bcrypt (FR-010)
        key_secret_hash = bcrypt.hashpw(
            key_secret.encode('utf-8'),
            bcrypt.gensalt(rounds=self.bcrypt_rounds)
        ).decode('utf-8')

        # Calculate expiration
        created_at = datetime.utcnow()
        expires_at = None

        if request.expires_in_days:
            expires_at = created_at + timedelta(days=request.expires_in_days)

        # Determine rate limits from tier
        rate_limits = self._get_rate_limits_for_tier(request.rate_limit_tier)

        # Create API key object
        api_key = ApiKey(
            key_id=key_id,
            key_secret_hash=key_secret_hash,
            name=request.name,
            permissions=request.permissions,
            rate_limit_tier=request.rate_limit_tier,
            requests_per_minute=rate_limits['requests_per_minute'],
            requests_per_hour=rate_limits['requests_per_hour'],
            created_at=created_at,
            expires_at=expires_at,
            is_active=True,
            owner_email=owner_email,
            description=request.description
        )

        # Store in database
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                INSERT INTO api_keys (
                    key_id, key_secret_hash, name, permissions, rate_limit_tier,
                    requests_per_minute, requests_per_hour, created_at,
                    expires_at, is_active, owner_email, description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            cursor.execute(query, (
                str(api_key.key_id),
                api_key.key_secret_hash,
                api_key.name,
                ','.join([p.value for p in api_key.permissions]),
                api_key.rate_limit_tier.value,
                api_key.requests_per_minute,
                api_key.requests_per_hour,
                api_key.created_at,
                api_key.expires_at,
                api_key.is_active,
                api_key.owner_email,
                api_key.description
            ))

            conn.commit()

        logger.info(f"Created API key: {key_id} ({api_key.name})")

        # Return response with plaintext secret (only time it's visible)
        return ApiKeyResponse(
            key_id=key_id,
            key_secret=key_secret,  # Plaintext - only shown once!
            name=api_key.name,
            permissions=api_key.permissions,
            rate_limit_tier=api_key.rate_limit_tier,
            expires_at=api_key.expires_at,
            created_at=api_key.created_at,
            message="IMPORTANT: Save this secret - it will not be shown again!"
        )

    def _get_rate_limits_for_tier(self, tier: RateLimitTier) -> dict:
        """
        Get rate limits for tier.

        Args:
            tier: Rate limit tier

        Returns:
            Dictionary with requests_per_minute and requests_per_hour
        """
        tier_limits = {
            RateLimitTier.BASIC: {
                'requests_per_minute': 60,
                'requests_per_hour': 1000
            },
            RateLimitTier.PREMIUM: {
                'requests_per_minute': 100,
                'requests_per_hour': 5000
            },
            RateLimitTier.ENTERPRISE: {
                'requests_per_minute': 1000,
                'requests_per_hour': 50000
            }
        }

        return tier_limits.get(tier, tier_limits[RateLimitTier.BASIC])

    def get_api_key(self, key_id: UUID) -> Optional[ApiKey]:
        """
        Retrieve API key by ID.

        Args:
            key_id: API key UUID

        Returns:
            ApiKey object or None if not found
        """
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    key_id, key_secret_hash, name, permissions, rate_limit_tier,
                    requests_per_minute, requests_per_hour, created_at,
                    expires_at, is_active, owner_email, description
                FROM api_keys
                WHERE key_id = ?
            """

            cursor.execute(query, (str(key_id),))
            row = cursor.fetchone()

            if not row:
                return None

            return ApiKey(
                key_id=UUID(row[0]),
                key_secret_hash=row[1],
                name=row[2],
                permissions=[Permission(p) for p in row[3].split(',')],
                rate_limit_tier=RateLimitTier(row[4]),
                requests_per_minute=row[5],
                requests_per_hour=row[6],
                created_at=row[7],
                expires_at=row[8],
                is_active=row[9],
                owner_email=row[10],
                description=row[11]
            )

    def list_api_keys(self, owner_email: Optional[str] = None) -> List[ApiKey]:
        """
        List API keys.

        Args:
            owner_email: Optional filter by owner email

        Returns:
            List of ApiKey objects (without secrets)
        """
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            if owner_email:
                query = """
                    SELECT
                        key_id, key_secret_hash, name, permissions, rate_limit_tier,
                        requests_per_minute, requests_per_hour, created_at,
                        expires_at, is_active, owner_email, description
                    FROM api_keys
                    WHERE owner_email = ?
                    ORDER BY created_at DESC
                """
                cursor.execute(query, (owner_email,))
            else:
                query = """
                    SELECT
                        key_id, key_secret_hash, name, permissions, rate_limit_tier,
                        requests_per_minute, requests_per_hour, created_at,
                        expires_at, is_active, owner_email, description
                    FROM api_keys
                    ORDER BY created_at DESC
                """
                cursor.execute(query)

            api_keys = []
            for row in cursor.fetchall():
                api_keys.append(ApiKey(
                    key_id=UUID(row[0]),
                    key_secret_hash=row[1],
                    name=row[2],
                    permissions=[Permission(p) for p in row[3].split(',')],
                    rate_limit_tier=RateLimitTier(row[4]),
                    requests_per_minute=row[5],
                    requests_per_hour=row[6],
                    created_at=row[7],
                    expires_at=row[8],
                    is_active=row[9],
                    owner_email=row[10],
                    description=row[11]
                ))

            return api_keys

    def revoke_api_key(self, key_id: UUID) -> bool:
        """
        Revoke (deactivate) API key.

        Args:
            key_id: API key UUID

        Returns:
            True if revoked, False if not found

        Implements FR-012: API key revocation
        """
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                UPDATE api_keys
                SET is_active = 0
                WHERE key_id = ?
            """

            cursor.execute(query, (str(key_id),))
            conn.commit()

            if cursor.rowcount > 0:
                logger.info(f"Revoked API key: {key_id}")
                return True

            return False

    def activate_api_key(self, key_id: UUID) -> bool:
        """
        Reactivate API key.

        Args:
            key_id: API key UUID

        Returns:
            True if activated, False if not found
        """
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                UPDATE api_keys
                SET is_active = 1
                WHERE key_id = ?
            """

            cursor.execute(query, (str(key_id),))
            conn.commit()

            if cursor.rowcount > 0:
                logger.info(f"Activated API key: {key_id}")
                return True

            return False

    def delete_api_key(self, key_id: UUID) -> bool:
        """
        Permanently delete API key.

        Args:
            key_id: API key UUID

        Returns:
            True if deleted, False if not found

        WARNING: This is permanent and cannot be undone!
        """
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            query = "DELETE FROM api_keys WHERE key_id = ?"

            cursor.execute(query, (str(key_id),))
            conn.commit()

            if cursor.rowcount > 0:
                logger.warning(f"Permanently deleted API key: {key_id}")
                return True

            return False
