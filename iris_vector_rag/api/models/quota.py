"""
Rate limiting models for RAG API.

Pydantic models for Rate Limit Quota entity (Entity 4 from data-model.md).
Implements adaptive request concurrency and per-key quotas.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from uuid import UUID
from datetime import datetime


class QuotaType(str, Enum):
    """Type of rate limit quota."""

    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    CONCURRENT_REQUESTS = "concurrent_requests"


class RateLimitQuota(BaseModel):
    """
    Rate Limit Quota (Entity 4 from data-model.md).

    Represents allowed request rate for a specific API key.
    Tracks current usage, reset timestamp, and enforcement policy.

    Implements FR-013 to FR-016: Rate limiting with adaptive concurrency.
    """

    quota_id: UUID = Field(
        ...,
        description="Unique quota identifier",
    )

    api_key_id: UUID = Field(
        ...,
        description="Associated API key",
    )

    quota_type: QuotaType = Field(
        ...,
        description="Type of quota",
        examples=["requests_per_minute"],
    )

    limit: int = Field(
        ...,
        gt=0,
        description="Maximum allowed value",
        examples=[100],
    )

    current_usage: int = Field(
        default=0,
        ge=0,
        description="Current usage in time window",
    )

    window_start: datetime = Field(
        ...,
        description="When current window started (ISO8601)",
    )

    window_end: datetime = Field(
        ...,
        description="When current window ends (ISO8601)",
    )

    next_reset_at: datetime = Field(
        ...,
        description="When quota will reset (ISO8601)",
    )

    exceeded_count: int = Field(
        default=0,
        ge=0,
        description="Number of times quota exceeded",
    )

    last_exceeded_at: Optional[datetime] = Field(
        default=None,
        description="Last time quota was exceeded (ISO8601)",
    )

    @field_validator("current_usage")
    @classmethod
    def usage_cannot_be_negative(cls, v: int) -> int:
        """Validate usage is non-negative."""
        if v < 0:
            raise ValueError("Usage cannot be negative")
        return v

    @field_validator("window_end")
    @classmethod
    def window_end_after_start(cls, v: datetime, info) -> datetime:
        """Validate window_end is after window_start."""
        if "window_start" in info.data and v <= info.data["window_start"]:
            raise ValueError("window_end must be after window_start")
        return v

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "quota_id": "f4a3b2c1-5678-4d3e-9f0a-123456789def",
                "api_key_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                "quota_type": "requests_per_minute",
                "limit": 100,
                "current_usage": 73,
                "window_start": "2025-10-16T12:34:00.000Z",
                "window_end": "2025-10-16T12:35:00.000Z",
                "next_reset_at": "2025-10-16T12:35:00.000Z",
                "exceeded_count": 5,
                "last_exceeded_at": "2025-10-16T11:23:45.000Z",
            }
        }


class RateLimitHeaders(BaseModel):
    """
    Rate limit headers for HTTP responses.

    Implements FR-014: Return rate limit information in headers.
    """

    x_ratelimit_limit: int = Field(
        ...,
        alias="X-RateLimit-Limit",
        description="Maximum requests allowed in window",
    )

    x_ratelimit_remaining: int = Field(
        ...,
        alias="X-RateLimit-Remaining",
        description="Remaining requests in current window",
    )

    x_ratelimit_reset: int = Field(
        ...,
        alias="X-RateLimit-Reset",
        description="Unix timestamp when quota resets",
    )

    retry_after: Optional[int] = Field(
        default=None,
        alias="Retry-After",
        description="Seconds to wait before retrying (when rate limited)",
    )

    class Config:
        """Pydantic model configuration."""

        populate_by_name = True
        json_schema_extra = {
            "example": {
                "X-RateLimit-Limit": 100,
                "X-RateLimit-Remaining": 27,
                "X-RateLimit-Reset": 1697461200,
                "Retry-After": 60,
            }
        }


class RateLimitStatus(BaseModel):
    """
    Rate limit status for a specific API key.

    Used to check quota status without making a request.
    """

    api_key_id: UUID = Field(
        ...,
        description="API key identifier",
    )

    requests_per_minute: RateLimitQuota = Field(
        ...,
        description="Per-minute quota status",
    )

    requests_per_hour: RateLimitQuota = Field(
        ...,
        description="Per-hour quota status",
    )

    is_rate_limited: bool = Field(
        ...,
        description="Whether key is currently rate limited",
    )

    class Config:
        """Pydantic model configuration."""

        json_schema_extra = {
            "example": {
                "api_key_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                "requests_per_minute": {
                    "quota_type": "requests_per_minute",
                    "limit": 100,
                    "current_usage": 73,
                    "next_reset_at": "2025-10-16T12:35:00.000Z",
                },
                "requests_per_hour": {
                    "quota_type": "requests_per_hour",
                    "limit": 5000,
                    "current_usage": 2341,
                    "next_reset_at": "2025-10-16T13:00:00.000Z",
                },
                "is_rate_limited": False,
            }
        }
