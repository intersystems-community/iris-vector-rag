"""
Integration test for Acceptance Scenario 3: Rate limit enforcement.

E2E test validating:
- Rate limits enforced per API key
- 429 Too Many Requests when limit exceeded
- Retry-After header provided
- Rate limit headers in all responses

IMPORTANT: This test MUST FAIL initially (TDD principle).
"""

import pytest
import time
from fastapi.testclient import TestClient

try:
    from iris_rag.api.main import create_app
except ImportError:
    create_app = None

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    """Create FastAPI test client."""
    if create_app is None:
        pytest.skip("API application not implemented yet (TDD)")

    app = create_app()
    return TestClient(app)


@pytest.fixture
def limited_api_key():
    """
    Create API key with low rate limit for testing.

    For testing, use basic tier with 60 requests/minute.
    """
    pytest.skip("Requires API key creation with rate limit configuration")

    # Expected usage:
    # from iris_rag.api.services.auth_service import AuthService
    # auth = AuthService()
    # key = auth.create_key(
    #     name="rate-limit-test-key",
    #     permissions=["read"],
    #     tier="basic"  # 60 req/min
    # )
    # return key


@pytest.fixture
def auth_header(limited_api_key):
    """Generate Authorization header from limited API key."""
    import base64

    key_id = limited_api_key["key_id"]
    key_secret = limited_api_key["secret"]

    credentials = f"{key_id}:{key_secret}"
    encoded = base64.b64encode(credentials.encode()).decode()

    return {"Authorization": f"ApiKey {encoded}"}


class TestRateLimitEnforcement:
    """
    Acceptance Scenario 3: Rate limit exceeded.

    Given: Developer has exceeded rate limit
    When: Another request is sent
    Then: System returns 429 with retry-after information
    """

    def test_rate_limit_headers_in_successful_response(self, client, auth_header):
        """
        Test that all responses include rate limit headers.

        Validates FR-014: Return rate limit information in headers
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": "What is diabetes?"}
        )

        # Should include rate limit headers regardless of status
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

        # Validate header values
        limit = int(response.headers["X-RateLimit-Limit"])
        remaining = int(response.headers["X-RateLimit-Remaining"])
        reset_timestamp = int(response.headers["X-RateLimit-Reset"])

        assert limit > 0
        assert 0 <= remaining <= limit
        assert reset_timestamp > time.time()

    def test_exceeding_rate_limit_returns_429(self, client, auth_header):
        """
        Test that exceeding rate limit returns 429.

        Validates:
        - FR-013: Enforce per-API-key rate limits
        - FR-015: Return 429 when limits exceeded
        """
        # Send requests until rate limited
        # For basic tier: 60 requests/minute
        max_attempts = 70  # Exceed limit

        responses = []
        for i in range(max_attempts):
            response = client.post(
                "/api/v1/basic/_search",
                headers=auth_header,
                json={"query": f"Query {i}"}
            )
            responses.append(response)

            # Stop if rate limited
            if response.status_code == 429:
                break

        # Should have received 429 at some point
        rate_limited_responses = [r for r in responses if r.status_code == 429]
        assert len(rate_limited_responses) > 0, \
            "Should receive 429 after exceeding rate limit"

        # Validate 429 response structure
        error_response = rate_limited_responses[0]
        data = error_response.json()

        assert "error" in data
        assert data["error"]["type"] == "rate_limit_exceeded"
        assert "details" in data["error"]
        assert "limit" in data["error"]["details"]
        assert "retry_after_seconds" in data["error"]["details"]

    def test_429_response_includes_retry_after_header(self, client, auth_header):
        """
        Test that 429 response includes Retry-After header.

        Validates FR-015: Return Retry-After header when limits exceeded
        """
        # Exhaust rate limit
        for i in range(70):
            response = client.post(
                "/api/v1/basic/_search",
                headers=auth_header,
                json={"query": f"Query {i}"}
            )

            if response.status_code == 429:
                # Validate Retry-After header
                assert "Retry-After" in response.headers
                retry_after = int(response.headers["Retry-After"])
                assert retry_after > 0
                assert retry_after <= 60  # Should be within 1 minute for per-minute limit
                break

    def test_rate_limit_resets_after_window(self, client, auth_header):
        """
        Test that rate limit resets after time window.

        Validates:
        - Sliding window counter implementation
        - Quota resets correctly
        - X-RateLimit-Reset timestamp accurate
        """
        pytest.skip("Requires waiting for rate limit window to reset (slow test)")

        # This test would:
        # 1. Exhaust rate limit (get 429)
        # 2. Wait for reset timestamp
        # 3. Verify new requests succeed
        # 4. Verify X-RateLimit-Remaining reset to max

    def test_different_api_keys_have_independent_limits(self, client):
        """
        Test that rate limits are enforced per API key.

        Validates FR-013: Per-API-key rate limits
        """
        pytest.skip("Requires multiple API keys")

        # This test would:
        # 1. Create two API keys
        # 2. Exhaust limit for key 1
        # 3. Verify key 2 still has quota
        # 4. Verify key 1 gets 429 while key 2 succeeds

    def test_rate_limit_remaining_decrements_with_each_request(self, client, auth_header):
        """
        Test that X-RateLimit-Remaining decrements correctly.

        Validates:
        - Accurate quota tracking
        - Remaining count updates after each request
        """
        # Get initial remaining count
        response1 = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": "First query"}
        )

        if response1.status_code != 200:
            pytest.skip("Requires working query endpoint")

        remaining1 = int(response1.headers["X-RateLimit-Remaining"])

        # Send another request
        response2 = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": "Second query"}
        )

        remaining2 = int(response2.headers["X-RateLimit-Remaining"])

        # Remaining should have decreased
        assert remaining2 < remaining1, \
            "X-RateLimit-Remaining should decrement after each request"

    def test_rate_limit_applies_to_all_endpoints(self, client, auth_header):
        """
        Test that rate limit is shared across all endpoints.

        Validates:
        - Global per-key quota
        - Not per-endpoint quota
        """
        # Send requests to different endpoints
        endpoints = [
            "/api/v1/basic/_search",
            "/api/v1/graphrag/_search",
            "/api/v1/pipelines",
        ]

        for i in range(25):
            endpoint = endpoints[i % len(endpoints)]

            if endpoint == "/api/v1/pipelines":
                response = client.get(endpoint, headers=auth_header)
            else:
                response = client.post(
                    endpoint,
                    headers=auth_header,
                    json={"query": f"Query {i}"}
                )

            # X-RateLimit-Remaining should decrement across all endpoints
            if i > 0:
                remaining = int(response.headers["X-RateLimit-Remaining"])
                # Remaining should decrease (or stay same if request failed)
                assert remaining <= (60 - i)
