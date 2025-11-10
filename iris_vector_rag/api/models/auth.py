"""
Authentication models for RAG API.

Pydantic models for Authentication Token entity (Entity 3 from data-model.md).
Implements API key authentication with permissions and rate limits.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, EmailStr
from enum import Enum
from uuid import UUID
from datetime import datetime


class Permission(str, Enum):
    """API key permissions."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class RateLimitTier(str, Enum):
    """Rate limiting tiers."""

    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class ApiKey(BaseModel):
    """
    Authentication Token (Entity 3 from data-model.md).

    Represents credentials used to authenticate API requests.
    Implements FR-009 to FR-012: API key authentication with permissions.
    """

    key_id: UUID = Field(
        ...,
        description="API key identifier",
    )

    key_secret_hash: str = Field(
        ...,
        description="Hashed API key secret (bcrypt, never store plaintext)",
        examples=["$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYk3H.olfm"],
    )

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Human-readable key name",
        examples=["production-app-key"],
    )

    permissions: List[Permission] = Field(
        ...,
        min_length=1,
        description="Allowed operations (at least one required)",
        examples=[["read", "write"]],
    )

    rate_limit_tier: RateLimitTier = Field(
        ...,
        description="Rate limiting tier",
        examples=["premium"],
    )

    requests_per_minute: int = Field(
        ...,
        ge=1,
        le=10000,
        description="Max requests per minute",
        examples=[100],
    )

    requests_per_hour: int = Field(
        ...,
        ge=1,
        le=100000,
        description="Max requests per hour",
        examples=[5000],
    )

    created_at: datetime = Field(
        ...,
        description="When key was created (ISO8601)",
    )

    expires_at: Optional[datetime] = Field(
        default=None,
        description="Key expiration (optional, ISO8601)",
    )

    last_used_at: Optional[datetime] = Field(
        default=None,
        description="Last request timestamp (ISO8601)",
    )

    is_active: bool = Field(
        default=True,
        description="Whether key is enabled",
    )

    owner_email: Optional[EmailStr] = Field(
        default=None,
        description="Key owner contact email",
    )

    @field_validator("permissions")
    @classmethod
    def permissions_must_not_be_empty(cls, v: List[Permission]) -> List[Permission]:
        """Validate at least one permission."""
        if not v:
            raise ValueError("At least one permission required")
        return v

    @field_validator("expires_at")
    @classmethod
    def expires_at_must_be_future(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Validate expiration is in the future (if provided)."""
        if v is not None and v < datetime.utcnow():
            raise ValueError("Expiration must be in the future")
        return v

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "key_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                "key_secret_hash": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYk3H.olfm",
                "name": "production-app-key",
                "permissions": ["read", "write"],
                "rate_limit_tier": "premium",
                "requests_per_minute": 100,
                "requests_per_hour": 5000,
                "created_at": "2025-01-15T08:00:00.000Z",
                "expires_at": "2026-01-15T08:00:00.000Z",
                "last_used_at": "2025-10-16T12:34:56.789Z",
                "is_active": True,
                "owner_email": "developer@example.com",
            }
        }


class ApiKeyCreateRequest(BaseModel):
    """
    Request to create a new API key.

    Used by API key management CLI.
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Human-readable key name",
    )

    permissions: List[Permission] = Field(
        ...,
        min_length=1,
        description="Allowed operations",
    )

    tier: RateLimitTier = Field(
        default=RateLimitTier.BASIC,
        description="Rate limiting tier",
    )

    expires_in_days: Optional[int] = Field(
        default=365,
        ge=1,
        le=3650,
        description="Days until expiration (1-3650, default: 365)",
    )

    owner_email: Optional[EmailStr] = Field(
        default=None,
        description="Key owner contact email",
    )


class ApiKeyResponse(BaseModel):
    """
    Response when creating API key.

    Includes the plaintext secret (only shown once).
    """

    key_id: UUID = Field(
        ...,
        description="API key identifier",
    )

    secret: str = Field(
        ...,
        description="API key secret (only shown once, save securely!)",
    )

    name: str = Field(
        ...,
        description="Human-readable key name",
    )

    permissions: List[Permission] = Field(
        ...,
        description="Allowed operations",
    )

    rate_limit_tier: RateLimitTier = Field(
        ...,
        description="Rate limiting tier",
    )

    requests_per_minute: int = Field(
        ...,
        description="Max requests per minute",
    )

    requests_per_hour: int = Field(
        ...,
        description="Max requests per hour",
    )

    created_at: datetime = Field(
        ...,
        description="When key was created",
    )

    expires_at: Optional[datetime] = Field(
        default=None,
        description="Key expiration",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "key_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                "secret": "my_secret_key_12345_do_not_share",
                "name": "production-app-key",
                "permissions": ["read", "write"],
                "rate_limit_tier": "premium",
                "requests_per_minute": 100,
                "requests_per_hour": 5000,
                "created_at": "2025-01-15T08:00:00.000Z",
                "expires_at": "2026-01-15T08:00:00.000Z",
            }
        }
