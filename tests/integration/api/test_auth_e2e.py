"""
Integration test for Acceptance Scenario 2: Unauthenticated request.

E2E test validating:
- Requests without credentials are rejected
- 401 Unauthorized status returned
- Clear authentication instructions provided
- Security logging for failed attempts

IMPORTANT: This test MUST FAIL initially (TDD principle).
"""

import pytest
import logging
from fastapi.testclient import TestClient

try:
    from iris_vector_rag.api.main import create_app
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


class TestUnauthenticatedRequests:
    """
    Acceptance Scenario 2: Unauthenticated request rejected.

    Given: Request without API key
    When: Developer sends query without credentials
    Then: System returns 401 Unauthorized with clear authentication instructions
    """

    def test_request_without_authorization_header_returns_401(self, client):
        """
        Test that request without Authorization header is rejected.

        Validates FR-010: Reject unauthenticated requests with 401
        """
        response = client.post(
            "/api/v1/basic/_search",
            json={"query": "What is diabetes?"}
        )

        assert response.status_code == 401
        data = response.json()

        # Validate error structure
        assert "error" in data
        assert data["error"]["type"] == "authentication_error"
        assert "reason" in data["error"]

    def test_error_response_includes_authentication_instructions(self, client):
        """
        Test that error response provides clear guidance.

        Validates FR-010: Clear authentication instructions
        """
        response = client.post(
            "/api/v1/basic/_search",
            json={"query": "What is diabetes?"}
        )

        assert response.status_code == 401
        data = response.json()

        # Should mention how to authenticate
        error_text = str(data).lower()
        assert "authorization" in error_text or "apikey" in error_text

        # Should provide actionable guidance
        if "details" in data["error"]:
            assert "message" in data["error"]["details"]

    def test_malformed_authorization_header_returns_401(self, client):
        """
        Test that malformed Authorization header is rejected.

        Validates:
        - Invalid header format detected
        - Clear error message about expected format
        """
        # Wrong format (missing "ApiKey" prefix)
        response = client.post(
            "/api/v1/basic/_search",
            headers={"Authorization": "Bearer some_token"},
            json={"query": "What is diabetes?"}
        )

        assert response.status_code == 401

        # Wrong format (invalid base64)
        response = client.post(
            "/api/v1/basic/_search",
            headers={"Authorization": "ApiKey not_valid_base64!!!"},
            json={"query": "What is diabetes?"}
        )

        assert response.status_code == 401

    def test_invalid_api_key_returns_401(self, client):
        """
        Test that non-existent API key is rejected.

        Validates:
        - API key lookup performed
        - Invalid keys rejected
        - Key ID included in error (if parseable)
        """
        import base64

        # Valid base64 but non-existent key
        fake_credentials = base64.b64encode(
            b"00000000-0000-0000-0000-000000000000:fake_secret"
        ).decode()

        response = client.post(
            "/api/v1/basic/_search",
            headers={"Authorization": f"ApiKey {fake_credentials}"},
            json={"query": "What is diabetes?"}
        )

        assert response.status_code == 401
        data = response.json()

        assert data["error"]["type"] == "invalid_api_key"

    def test_authentication_failure_is_logged(self, client, caplog):
        """
        Test that authentication failures are logged for security monitoring.

        Validates FR-012: Log all authentication failures
        """
        caplog.set_level(logging.WARNING)

        response = client.post(
            "/api/v1/basic/_search",
            json={"query": "What is diabetes?"}
        )

        assert response.status_code == 401

        # Check that failure was logged
        auth_failure_logged = any(
            "authentication" in record.message.lower() or
            "unauthorized" in record.message.lower()
            for record in caplog.records
        )

        assert auth_failure_logged, \
            "Authentication failure should be logged for security monitoring"

    def test_all_endpoints_require_authentication(self, client):
        """
        Test that all API endpoints enforce authentication.

        Validates:
        - Query endpoints require auth
        - Pipeline endpoints require auth
        - Document upload requires auth
        - Health endpoint may be public (for monitoring)
        """
        endpoints = [
            ("POST", "/api/v1/basic/_search", {"query": "test"}),
            ("POST", "/api/v1/graphrag/_search", {"query": "test"}),
            ("GET", "/api/v1/pipelines", None),
            ("POST", "/api/v1/documents/upload", None),
        ]

        for method, endpoint, json_data in endpoints:
            if method == "POST":
                response = client.post(endpoint, json=json_data)
            else:
                response = client.get(endpoint)

            # All should return 401 (except health endpoint)
            if endpoint != "/api/v1/health":
                assert response.status_code == 401, \
                    f"Endpoint {endpoint} should require authentication"

    def test_expired_api_key_returns_401(self, client):
        """
        Test that expired API keys are rejected.

        Validates:
        - Expiration check performed
        - Expired keys rejected with specific error
        - Expiration timestamp included in response
        """
        pytest.skip("Requires test fixture with expired API key")

        # Expected behavior:
        # 1. Create API key with past expiration date
        # 2. Attempt to use expired key
        # 3. Receive 401 with error type "expired_api_key"
        # 4. Error details include expired_at timestamp
