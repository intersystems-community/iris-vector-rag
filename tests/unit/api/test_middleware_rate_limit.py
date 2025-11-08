"""
Unit tests for rate limiting middleware.

Tests the Redis-based rate limiting middleware in isolation.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from iris_rag.api.middleware.rate_limit import RateLimitMiddleware
from iris_rag.api.models.auth import ApiKey, Permission, RateLimitTier
from iris_rag.api.models.quota import RateLimitQuota


class TestRateLimitMiddleware:
    """Test rate limiting middleware."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis = MagicMock()
        redis.get.return_value = None
        redis.incr.return_value = 1
        redis.expire.return_value = True
        return redis

    @pytest.fixture
    def middleware(self, mock_redis):
        """Create middleware instance."""
        return RateLimitMiddleware(mock_redis)

    @pytest.fixture
    def basic_api_key(self):
        """Create basic tier API key."""
        return ApiKey(
            key_id="basic-key",
            key_secret_hash="hash",
            name="Basic Key",
            permissions=[Permission.READ],
            rate_limit_tier=RateLimitTier.BASIC,
            requests_per_minute=60,
            requests_per_hour=1000,
            created_at=datetime.utcnow(),
            is_active=True,
            owner_email="basic@example.com",
        )

    @pytest.fixture
    def premium_api_key(self):
        """Create premium tier API key."""
        return ApiKey(
            key_id="premium-key",
            key_secret_hash="hash",
            name="Premium Key",
            permissions=[Permission.READ, Permission.WRITE],
            rate_limit_tier=RateLimitTier.PREMIUM,
            requests_per_minute=100,
            requests_per_hour=5000,
            created_at=datetime.utcnow(),
            is_active=True,
            owner_email="premium@example.com",
        )

    @pytest.fixture
    def enterprise_api_key(self):
        """Create enterprise tier API key."""
        return ApiKey(
            key_id="enterprise-key",
            key_secret_hash="hash",
            name="Enterprise Key",
            permissions=[Permission.ADMIN],
            rate_limit_tier=RateLimitTier.ENTERPRISE,
            requests_per_minute=1000,
            requests_per_hour=50000,
            created_at=datetime.utcnow(),
            is_active=True,
            owner_email="enterprise@example.com",
        )

    def test_check_rate_limit_first_request(self, middleware, basic_api_key, mock_redis):
        """Test rate limit check passes for first request."""
        mock_redis.get.return_value = None
        mock_redis.incr.return_value = 1

        result = middleware.check_rate_limit(basic_api_key)

        assert result.limit == 60
        assert result.remaining == 59
        assert result.reset_at is not None

    def test_check_rate_limit_within_limit(self, middleware, basic_api_key, mock_redis):
        """Test rate limit check passes within limit."""
        mock_redis.get.return_value = "30"  # 30 requests so far
        mock_redis.incr.return_value = 31

        result = middleware.check_rate_limit(basic_api_key)

        assert result.limit == 60
        assert result.remaining == 29
        assert result.used == 31

    def test_check_rate_limit_exceeded(self, middleware, basic_api_key, mock_redis):
        """Test rate limit check fails when exceeded."""
        mock_redis.get.return_value = "60"  # At limit
        mock_redis.incr.return_value = 61

        with pytest.raises(HTTPException) as exc:
            middleware.check_rate_limit(basic_api_key)

        assert exc.value.status_code == 429
        assert "rate limit exceeded" in str(exc.value.detail).lower()

    def test_check_rate_limit_premium_tier(self, middleware, premium_api_key, mock_redis):
        """Test rate limit with premium tier."""
        mock_redis.get.return_value = "50"
        mock_redis.incr.return_value = 51

        result = middleware.check_rate_limit(premium_api_key)

        assert result.limit == 100
        assert result.remaining == 49

    def test_check_rate_limit_enterprise_tier(self, middleware, enterprise_api_key, mock_redis):
        """Test rate limit with enterprise tier."""
        mock_redis.get.return_value = "500"
        mock_redis.incr.return_value = 501

        result = middleware.check_rate_limit(enterprise_api_key)

        assert result.limit == 1000
        assert result.remaining == 499

    def test_get_redis_key_minute_window(self, middleware, basic_api_key):
        """Test Redis key generation for minute window."""
        key = middleware.get_redis_key(basic_api_key.key_id, "minute")

        assert "basic-key" in key
        assert "minute" in key
        # Should include timestamp to create sliding window

    def test_get_redis_key_hour_window(self, middleware, basic_api_key):
        """Test Redis key generation for hour window."""
        key = middleware.get_redis_key(basic_api_key.key_id, "hour")

        assert "basic-key" in key
        assert "hour" in key

    def test_reset_time_calculation(self, middleware, basic_api_key):
        """Test reset time is calculated correctly."""
        result = middleware.calculate_reset_time("minute")

        assert isinstance(result, datetime)
        # Should be within next 60 seconds
        now = datetime.utcnow()
        assert result > now
        assert result <= now + timedelta(seconds=60)

    def test_concurrent_request_tracking(self, middleware, basic_api_key, mock_redis):
        """Test concurrent request limit tracking."""
        mock_redis.get.side_effect = ["5", "10"]  # First call returns 5, second returns 10
        mock_redis.incr.return_value = 6

        result = middleware.check_concurrent_limit(basic_api_key)

        # Basic tier allows 5 concurrent requests
        assert result is True

    def test_concurrent_limit_exceeded(self, middleware, basic_api_key, mock_redis):
        """Test concurrent request limit exceeded."""
        mock_redis.get.return_value = "5"  # At concurrent limit

        with pytest.raises(HTTPException) as exc:
            middleware.check_concurrent_limit(basic_api_key)

        assert exc.value.status_code == 429
        assert "concurrent" in str(exc.value.detail).lower()

    def test_redis_connection_failure(self, middleware, basic_api_key, mock_redis):
        """Test graceful handling of Redis connection failure."""
        mock_redis.get.side_effect = Exception("Redis connection failed")

        # Should allow request to proceed if Redis is unavailable
        result = middleware.check_rate_limit(basic_api_key)

        # Should return permissive quota
        assert result.limit > 0

    def test_sliding_window_algorithm(self, middleware, basic_api_key, mock_redis):
        """Test sliding window rate limiting algorithm."""
        # Simulate requests over time
        mock_redis.get.side_effect = ["10", "20", "30", "40", "50"]
        mock_redis.incr.side_effect = [11, 21, 31, 41, 51]

        for i in range(5):
            result = middleware.check_rate_limit(basic_api_key)
            assert result.used == (i + 1) * 10 + 1

    def test_get_quota_headers(self, middleware):
        """Test generation of rate limit headers."""
        quota = RateLimitQuota(
            limit=60,
            remaining=45,
            used=15,
            window_seconds=60,
            reset_at=datetime.utcnow() + timedelta(seconds=30),
        )

        headers = middleware.get_quota_headers(quota)

        assert headers["X-RateLimit-Limit"] == "60"
        assert headers["X-RateLimit-Remaining"] == "45"
        assert "X-RateLimit-Reset" in headers

    def test_record_rate_limit_hit(self, middleware, basic_api_key, mock_redis):
        """Test recording rate limit hits for analytics."""
        middleware.record_rate_limit_hit(basic_api_key.key_id, "minute")

        # Should increment hit counter
        mock_redis.incr.assert_called()

    def test_different_endpoints_share_quota(self, middleware, basic_api_key, mock_redis):
        """Test that different endpoints share the same quota."""
        mock_redis.get.return_value = "30"
        mock_redis.incr.return_value = 31

        # Same key ID should get same counter regardless of endpoint
        result1 = middleware.check_rate_limit(basic_api_key)
        result2 = middleware.check_rate_limit(basic_api_key)

        assert result1.used == result2.used
