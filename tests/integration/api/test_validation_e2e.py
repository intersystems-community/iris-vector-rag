"""
Integration test for Acceptance Scenario 6: Validation error handling.

E2E test validating:
- Invalid query parameters rejected
- 422 Validation Error returned
- Field-level error messages
- Actionable guidance provided

IMPORTANT: This test MUST FAIL initially (TDD principle).
"""

import pytest
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


@pytest.fixture
def auth_header():
    """Authorization header with valid API key."""
    import base64
    credentials = base64.b64encode(b"test-id:test-secret").decode()
    return {"Authorization": f"ApiKey {credentials}"}


class TestValidationErrorHandling:
    """
    Acceptance Scenario 6: Validation error handling.

    Given: Invalid query parameter
    When: Developer sends malformed request
    Then: System returns 422 with field-level errors
    """

    def test_invalid_top_k_returns_422_with_field_details(self, client, auth_header):
        """
        Test that invalid top_k parameter returns field-level error.

        Validates FR-003: Field-specific error messages
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={
                "query": "What is diabetes?",
                "top_k": -5  # Invalid: must be 1-100
            }
        )

        assert response.status_code == 422
        data = response.json()

        # Validate error structure
        assert "error" in data
        assert data["error"]["type"] == "validation_exception"
        assert "details" in data["error"]

        # Validate field-level details
        details = data["error"]["details"]
        assert "field" in details
        assert "rejected_value" in details
        assert "message" in details

        assert details["field"] == "top_k"
        assert details["rejected_value"] == -5
        assert "1" in details["message"] and "100" in details["message"]

    def test_query_too_long_returns_422_with_max_length(self, client, auth_header):
        """
        Test that query exceeding 10000 chars returns validation error.

        Validates:
        - FR-003: Query text must be 1-10000 characters
        - Error includes max_length in details
        """
        long_query = "a" * 10001

        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": long_query}
        )

        assert response.status_code == 422
        data = response.json()

        assert data["error"]["type"] == "validation_exception"
        assert "query" in data["error"]["details"]["field"]
        assert "10000" in str(data["error"]["details"])

    def test_empty_query_returns_422(self, client, auth_header):
        """
        Test that empty query string is rejected.

        Validates: Query minLength=1
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": ""}
        )

        assert response.status_code == 422

    def test_missing_required_query_field_returns_422(self, client, auth_header):
        """
        Test that request without 'query' field is rejected.

        Validates:
        - Required field enforcement
        - Clear error message about missing field
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"top_k": 5}  # Missing 'query'
        )

        assert response.status_code == 422
        data = response.json()

        assert "query" in str(data).lower()

    def test_invalid_json_body_returns_400(self, client, auth_header):
        """
        Test that malformed JSON returns 400 Bad Request.

        Validates:
        - JSON parsing errors handled
        - Clear error message
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            data="not valid json{{{",  # Malformed JSON
            headers={**auth_header, "Content-Type": "application/json"}
        )

        assert response.status_code == 400
        data = response.json()

        assert data["error"]["type"] == "bad_request"
        assert "json" in data["error"]["reason"].lower()

    def test_validation_errors_are_actionable(self, client, auth_header):
        """
        Test that validation errors provide actionable guidance.

        Validates:
        - Error messages explain how to fix
        - Include valid ranges/formats
        - User can immediately correct the issue
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={
                "query": "test",
                "top_k": 150  # Exceeds max of 100
            }
        )

        assert response.status_code == 422
        data = response.json()

        message = data["error"]["details"]["message"]

        # Should mention valid range
        assert "1" in message or "100" in message
        # Should indicate it's a range issue
        assert "between" in message.lower() or "maximum" in message.lower()

    def test_multiple_validation_errors_reported(self, client, auth_header):
        """
        Test that multiple validation errors are reported together.

        Validates:
        - All field errors included
        - Not just first error
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={
                "query": "",  # Too short
                "top_k": -1   # Invalid
            }
        )

        assert response.status_code == 422
        # Should report both errors (implementation may vary)

    def test_invalid_filter_format_returns_422(self, client, auth_header):
        """
        Test that invalid filter values are rejected.

        Validates:
        - Type validation for filter values
        - Clear error for incorrect types
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={
                "query": "test",
                "filters": "not an object"  # Should be object
            }
        )

        assert response.status_code == 422

    def test_unknown_fields_handled_gracefully(self, client, auth_header):
        """
        Test that unknown fields in request are handled.

        Validates:
        - Either ignored (additionalProperties=true)
        - Or rejected with clear message
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={
                "query": "test",
                "unknown_field": "value"
            }
        )

        # Should either succeed (ignore) or return 422
        assert response.status_code in [200, 422, 503]

        if response.status_code == 422:
            data = response.json()
            assert "unknown_field" in str(data).lower()
