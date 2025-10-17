"""
Contract tests for authentication endpoints.

These tests validate API authentication behavior against auth.yaml OpenAPI spec.
Tests cover FR-009 to FR-012 (authentication and authorization).

IMPORTANT: These tests MUST FAIL initially (TDD principle).
Implementation in Phase 3.3 will make them pass.
"""

import base64
import pytest
from fastapi.testclient import TestClient

# These imports will fail until implementation exists (expected for TDD)
try:
    from iris_rag.api.main import create_app
except ImportError:
    create_app = None  # type: ignore

pytestmark = pytest.mark.contract


@pytest.fixture
def client():
    """Create FastAPI test client."""
    if create_app is None:
        pytest.skip("API application not implemented yet (TDD)")

    app = create_app()
    return TestClient(app)


@pytest.fixture
def valid_api_key():
    """
    Valid API key credentials for testing.

    Format: base64(id:secret) per auth.yaml specification.
    """
    key_id = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
    key_secret = "test_secret_key_12345_do_not_share_in_production"
    credentials = f"{key_id}:{key_secret}"
    return base64.b64encode(credentials.encode()).decode()


@pytest.fixture
def auth_header(valid_api_key):
    """Authorization header with valid API key."""
    return {"Authorization": f"ApiKey {valid_api_key}"}


class TestAuthenticationContract:
    """
    Contract tests for API key authentication (FR-009, FR-010).

    Validates against auth.yaml schemas:
    - ApiKeyCredentials
    - ApiKeyResponse
    - AuthenticationError
    """

    def test_missing_authorization_header_returns_401(self, client):
        """
        FR-010: System MUST reject unauthenticated requests with 401 Unauthorized.

        Contract: AuthenticationError with type='authentication_error'
        """
        response = client.post(
            "/api/v1/basic/_search", json={"query": "test query"}
        )

        assert response.status_code == 401
        data = response.json()

        # Validate against AuthenticationError schema
        assert "error" in data
        assert data["error"]["type"] == "authentication_error"
        assert "reason" in data["error"]
        assert "Missing Authorization header" in data["error"]["reason"]

    def test_invalid_authorization_header_format_returns_401(self, client):
        """
        FR-009: System MUST authenticate using specific header format.

        Contract: AuthenticationError for invalid format
        """
        # Wrong format (missing "ApiKey" prefix)
        response = client.post(
            "/api/v1/basic/_search",
            headers={"Authorization": "Bearer some_token"},
            json={"query": "test query"},
        )

        assert response.status_code == 401
        data = response.json()

        assert data["error"]["type"] == "authentication_error"
        assert "Invalid Authorization header format" in data["error"]["reason"]

    def test_malformed_base64_returns_401(self, client):
        """
        FR-009: System MUST reject malformed base64 encoding.

        Contract: AuthenticationError with type='authentication_error'
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers={"Authorization": "ApiKey not_valid_base64!!!"},
            json={"query": "test query"},
        )

        assert response.status_code == 401
        data = response.json()

        assert data["error"]["type"] == "authentication_error"

    def test_invalid_api_key_returns_401(self, client):
        """
        FR-010: System MUST reject invalid API keys.

        Contract: AuthenticationError with type='invalid_api_key'
        """
        # Valid base64 but non-existent key
        fake_credentials = base64.b64encode(
            b"00000000-0000-0000-0000-000000000000:fake_secret"
        ).decode()

        response = client.post(
            "/api/v1/basic/_search",
            headers={"Authorization": f"ApiKey {fake_credentials}"},
            json={"query": "test query"},
        )

        assert response.status_code == 401
        data = response.json()

        assert data["error"]["type"] == "invalid_api_key"
        assert "key_id" in data["error"].get("details", {})

    def test_expired_api_key_returns_401(self, client):
        """
        FR-010: System MUST reject expired API keys.

        Contract: AuthenticationError with type='expired_api_key'
        """
        # This test will need a test fixture with an expired key
        # For now, document the expected contract
        pytest.skip("Requires expired key test fixture")

        # Expected response structure:
        # {
        #   "error": {
        #     "type": "expired_api_key",
        #     "reason": "API key has expired",
        #     "details": {
        #       "key_id": "uuid",
        #       "expired_at": "2025-01-15T08:00:00.000Z"
        #     }
        #   }
        # }

    def test_valid_api_key_authentication_succeeds(self, client, auth_header):
        """
        FR-009: System MUST authenticate valid API keys.

        Contract: Valid authentication allows request processing
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": "test query", "top_k": 5},
        )

        # Should NOT return 401 (may return other errors if pipeline not ready)
        assert response.status_code != 401

    def test_authentication_failure_logged(self, client, caplog):
        """
        FR-012: System MUST log all authentication failures.

        Contract: Failed auth attempts appear in logs
        """
        import logging

        caplog.set_level(logging.WARNING)

        response = client.post(
            "/api/v1/basic/_search", json={"query": "test query"}
        )

        assert response.status_code == 401

        # Check that authentication failure was logged
        # (implementation will determine exact log format)
        auth_logs = [
            record
            for record in caplog.records
            if "authentication" in record.message.lower()
            or "unauthorized" in record.message.lower()
        ]

        assert len(auth_logs) > 0, "Authentication failure should be logged"


class TestAuthorizationContract:
    """
    Contract tests for permission-based authorization (FR-011).

    Validates against auth.yaml authorization error schemas.
    """

    def test_insufficient_permissions_returns_403(self, client):
        """
        FR-011: System MUST enforce permission-based authorization.

        Contract: AuthenticationError with type='authorization_error' and permission details
        """
        # This requires:
        # 1. API key with only 'read' permission
        # 2. Endpoint requiring 'write' permission
        pytest.skip("Requires test fixture with read-only API key")

        # Expected response structure:
        # {
        #   "error": {
        #     "type": "authorization_error",
        #     "reason": "Insufficient permissions for this operation",
        #     "details": {
        #       "required_permissions": ["write"],
        #       "current_permissions": ["read"]
        #     }
        #   }
        # }


class TestAPIKeyManagementContract:
    """
    Contract tests for API key creation/management endpoints.

    Validates against auth.yaml ApiKeyResponse schema.
    """

    def test_create_api_key_returns_valid_response(self, client):
        """
        Validate API key creation response against ApiKeyResponse schema.

        Contract: ApiKeyResponse with all required fields
        """
        pytest.skip("API key management endpoints not yet specified in contracts")

        # Expected response structure (from auth.yaml):
        # {
        #   "key_id": "uuid",
        #   "name": "string",
        #   "permissions": ["read", "write"],
        #   "rate_limit_tier": "premium",
        #   "requests_per_minute": 100,
        #   "requests_per_hour": 5000,
        #   "created_at": "2025-01-15T08:00:00.000Z",
        #   "expires_at": "2026-01-15T08:00:00.000Z",
        #   "is_active": true
        # }


class TestAuthenticationErrorResponses:
    """
    Contract tests for Elasticsearch-inspired error response format.

    Validates all error responses follow the structured format from auth.yaml.
    """

    def test_error_response_has_required_structure(self, client):
        """
        All authentication errors MUST follow AuthenticationError schema.

        Required fields:
        - error.type (enum)
        - error.reason (string)
        - error.details (object, optional)
        """
        response = client.post(
            "/api/v1/basic/_search", json={"query": "test query"}
        )

        assert response.status_code == 401
        data = response.json()

        # Validate top-level structure
        assert "error" in data
        assert isinstance(data["error"], dict)

        # Validate required fields
        assert "type" in data["error"]
        assert "reason" in data["error"]

        # Validate type is from enum
        valid_types = [
            "authentication_error",
            "authorization_error",
            "invalid_api_key",
            "expired_api_key",
        ]
        assert data["error"]["type"] in valid_types

        # Validate reason is non-empty string
        assert isinstance(data["error"]["reason"], str)
        assert len(data["error"]["reason"]) > 0

    def test_error_response_includes_actionable_details(self, client):
        """
        Error responses SHOULD include actionable guidance in details.

        Contract: details.message provides resolution steps
        """
        response = client.post(
            "/api/v1/basic/_search", json={"query": "test query"}
        )

        assert response.status_code == 401
        data = response.json()

        # Details field is optional but recommended
        if "details" in data["error"]:
            assert isinstance(data["error"]["details"], dict)

            # Should provide actionable guidance
            if "message" in data["error"]["details"]:
                message = data["error"]["details"]["message"]
                assert len(message) > 0
                # Should mention the expected header format
                assert "Authorization" in message or "ApiKey" in message


# Pytest configuration for contract tests
def pytest_configure(config):
    """Register contract marker."""
    config.addinivalue_line(
        "markers", "contract: marks tests as contract validation tests"
    )
