"""
Rate Limiting Middleware for RAG API.

Implements FR-013 to FR-016: Redis-based rate limiting with sliding window.
Uses Elasticsearch-inspired adaptive concurrency patterns.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import redis
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from iris_rag.api.models.auth import ApiKey
from iris_rag.api.models.quota import RateLimitQuota, RateLimitHeaders, QuotaType
from iris_rag.api.models.errors import ErrorResponse, ErrorType, ErrorInfo, ErrorDetails


logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Redis-based rate limiter with sliding window algorithm.

    Implements FR-015: Per-API-key rate limiting with multiple tiers.
    Uses Redis sorted sets for efficient sliding window tracking.
    """

    def __init__(self, redis_client: redis.Redis):
        """
        Initialize rate limiter.

        Args:
            redis_client: Redis client for quota storage
        """
        self.redis = redis_client

    def _get_redis_key(self, api_key_id: str, quota_type: QuotaType) -> str:
        """
        Generate Redis key for rate limit tracking.

        Args:
            api_key_id: API key UUID
            quota_type: Type of quota (requests_per_minute, etc.)

        Returns:
            Redis key string
        """
        return f"ratelimit:{api_key_id}:{quota_type.value}"

    def check_rate_limit(
        self,
        api_key: ApiKey,
        quota_type: QuotaType = QuotaType.REQUESTS_PER_MINUTE
    ) -> RateLimitQuota:
        """
        Check if request is within rate limits.

        Args:
            api_key: Authenticated API key
            quota_type: Type of quota to check

        Returns:
            RateLimitQuota with current usage

        Raises:
            HTTPException: If rate limit exceeded (FR-015)
        """
        now = datetime.utcnow()
        key_id = str(api_key.key_id)

        # Determine window size and limit based on quota type
        if quota_type == QuotaType.REQUESTS_PER_MINUTE:
            window_seconds = 60
            limit = api_key.requests_per_minute
        elif quota_type == QuotaType.REQUESTS_PER_HOUR:
            window_seconds = 3600
            limit = api_key.requests_per_hour
        else:
            raise ValueError(f"Unsupported quota type: {quota_type}")

        # Calculate window boundaries
        window_start = now - timedelta(seconds=window_seconds)
        window_end = now

        # Redis key for this quota
        redis_key = self._get_redis_key(key_id, quota_type)

        # Use Redis sorted set with timestamps as scores
        # Remove old entries outside window
        self.redis.zremrangebyscore(
            redis_key,
            0,
            window_start.timestamp()
        )

        # Count current requests in window
        current_count = self.redis.zcard(redis_key)

        # Calculate next reset time
        next_reset_at = now + timedelta(seconds=window_seconds)

        # Create quota object
        quota = RateLimitQuota(
            api_key_id=api_key.key_id,
            quota_type=quota_type,
            limit=limit,
            remaining=max(0, limit - current_count),
            window_start=window_start,
            window_end=window_end,
            next_reset_at=next_reset_at,
            current_count=current_count,
            exceeded_count=0
        )

        # Check if limit exceeded (FR-015)
        if current_count >= limit:
            # Increment exceeded counter
            exceeded_key = f"{redis_key}:exceeded"
            self.redis.incr(exceeded_key)
            self.redis.expire(exceeded_key, window_seconds)

            quota.exceeded_count = int(self.redis.get(exceeded_key) or 0)

            # Calculate retry_after in seconds
            retry_after_seconds = int((next_reset_at - now).total_seconds())

            logger.warning(
                f"Rate limit exceeded: {key_id} - {quota_type.value} "
                f"({current_count}/{limit})"
            )

            raise HTTPException(
                status_code=429,
                detail=ErrorResponse(
                    error=ErrorInfo(
                        type=ErrorType.RATE_LIMIT_EXCEEDED,
                        reason="Too many requests",
                        details=ErrorDetails(
                            limit=limit,
                            window=quota_type.value.replace('_', ' '),
                            retry_after_seconds=retry_after_seconds,
                            message=f"Rate limit of {limit} {quota_type.value.replace('_', ' ')} exceeded. "
                                   f"Retry after {retry_after_seconds} seconds."
                        )
                    )
                ).model_dump(),
                headers={
                    "Retry-After": str(retry_after_seconds),
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(next_reset_at.timestamp()))
                }
            )

        # Add current request to window
        self.redis.zadd(
            redis_key,
            {str(now.timestamp()): now.timestamp()}
        )

        # Set expiration to window size + buffer
        self.redis.expire(redis_key, window_seconds + 60)

        # Update remaining count after adding request
        quota.remaining = max(0, limit - (current_count + 1))
        quota.current_count = current_count + 1

        logger.debug(
            f"Rate limit check passed: {key_id} - {quota_type.value} "
            f"({quota.current_count}/{limit})"
        )

        return quota

    def get_rate_limit_headers(self, quota: RateLimitQuota) -> RateLimitHeaders:
        """
        Generate rate limit response headers.

        Args:
            quota: Current quota status

        Returns:
            RateLimitHeaders for HTTP response (FR-016)
        """
        return RateLimitHeaders(
            x_ratelimit_limit=quota.limit,
            x_ratelimit_remaining=quota.remaining,
            x_ratelimit_reset=int(quota.next_reset_at.timestamp())
        )

    def check_concurrent_requests(
        self,
        api_key: ApiKey,
        max_concurrent: int = 10
    ) -> bool:
        """
        Check concurrent request limit.

        Args:
            api_key: Authenticated API key
            max_concurrent: Maximum concurrent requests allowed

        Returns:
            True if within limit, False otherwise
        """
        key_id = str(api_key.key_id)
        concurrent_key = f"concurrent:{key_id}"

        # Increment counter
        current = self.redis.incr(concurrent_key)

        # Set expiration (auto-cleanup if requests don't decrement)
        if current == 1:
            self.redis.expire(concurrent_key, 300)  # 5 minutes max

        if current > max_concurrent:
            # Decrement since we're not processing this request
            self.redis.decr(concurrent_key)

            logger.warning(
                f"Concurrent request limit exceeded: {key_id} ({current}/{max_concurrent})"
            )
            return False

        return True

    def release_concurrent_request(self, api_key: ApiKey):
        """
        Release concurrent request slot.

        Args:
            api_key: Authenticated API key
        """
        key_id = str(api_key.key_id)
        concurrent_key = f"concurrent:{key_id}"
        self.redis.decr(concurrent_key)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    Implements FR-015 to FR-016: Rate limit enforcement with response headers.
    """

    def __init__(
        self,
        app,
        redis_client: redis.Redis,
        max_concurrent_per_key: int = 10
    ):
        """
        Initialize rate limit middleware.

        Args:
            app: FastAPI application
            redis_client: Redis client for quota storage
            max_concurrent_per_key: Max concurrent requests per API key
        """
        super().__init__(app)
        self.rate_limiter = RateLimiter(redis_client)
        self.max_concurrent_per_key = max_concurrent_per_key

        # Endpoints exempt from rate limiting
        self.exempt_endpoints = {
            "/api/v1/health",
            "/api/v1/health/",
            "/docs",
            "/docs/",
            "/redoc",
            "/redoc/",
            "/openapi.json"
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with rate limiting.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain

        Returns:
            HTTP response with rate limit headers
        """
        # Skip rate limiting for exempt endpoints
        if request.url.path in self.exempt_endpoints:
            return await call_next(request)

        # Get authenticated API key (should be set by AuthenticationMiddleware)
        api_key = getattr(request.state, 'api_key', None)

        if not api_key:
            # If no API key, authentication middleware will handle it
            return await call_next(request)

        # Check per-minute rate limit (FR-015)
        quota = self.rate_limiter.check_rate_limit(
            api_key,
            QuotaType.REQUESTS_PER_MINUTE
        )

        # Check concurrent request limit
        if not self.rate_limiter.check_concurrent_requests(
            api_key,
            self.max_concurrent_per_key
        ):
            raise HTTPException(
                status_code=429,
                detail=ErrorResponse(
                    error=ErrorInfo(
                        type=ErrorType.RATE_LIMIT_EXCEEDED,
                        reason="Too many concurrent requests",
                        details=ErrorDetails(
                            limit=self.max_concurrent_per_key,
                            window="concurrent requests",
                            message=f"Maximum {self.max_concurrent_per_key} concurrent requests allowed per API key"
                        )
                    )
                ).model_dump()
            )

        try:
            # Process request
            response = await call_next(request)

            # Add rate limit headers to response (FR-016)
            headers = self.rate_limiter.get_rate_limit_headers(quota)
            response.headers["X-RateLimit-Limit"] = str(headers.x_ratelimit_limit)
            response.headers["X-RateLimit-Remaining"] = str(headers.x_ratelimit_remaining)
            response.headers["X-RateLimit-Reset"] = str(headers.x_ratelimit_reset)

            return response

        finally:
            # Always release concurrent request slot
            self.rate_limiter.release_concurrent_request(api_key)
