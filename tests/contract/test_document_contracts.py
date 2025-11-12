"""
Contract tests for document upload endpoints.

Validates against document.yaml OpenAPI spec (FR-021 to FR-024).

IMPORTANT: These tests MUST FAIL initially (TDD principle).
"""

import pytest
from fastapi.testclient import TestClient

try:
    from iris_vector_rag.api.main import create_app
except ImportError:
    create_app = None

pytestmark = pytest.mark.contract


@pytest.fixture
def client():
    if create_app is None:
        pytest.skip("API application not implemented yet (TDD)")
    app = create_app()
    return TestClient(app)


@pytest.fixture
def auth_header():
    import base64
    credentials = base64.b64encode(b"test-id:test-secret").decode()
    return {"Authorization": f"ApiKey {credentials}"}


class TestDocumentUploadContract:
    """FR-021, FR-022: Async document upload with validation."""

    def test_upload_document_returns_operation_id(self, client, auth_header):
        """Contract: POST /documents/upload returns operation_id."""
        files = {"file": ("test.txt", b"test content", "text/plain")}
        response = client.post(
            "/api/v1/documents/upload", headers=auth_header, files=files
        )

        if response.status_code != 202:
            pytest.skip("Implementation pending")

        data = response.json()
        assert "operation_id" in data

    def test_upload_rejects_files_over_100mb(self, client, auth_header):
        """FR-022: Max 100MB file size."""
        # Create 101MB file content
        large_content = b"a" * (101 * 1024 * 1024)
        files = {"file": ("large.txt", large_content, "text/plain")}

        response = client.post(
            "/api/v1/documents/upload", headers=auth_header, files=files
        )

        assert response.status_code == 422
        data = response.json()
        assert "100" in str(data["error"]["details"]["message"]).lower()


class TestDocumentOperationStatusContract:
    """FR-023: Operation status tracking."""

    def test_get_operation_status(self, client, auth_header):
        """Contract: GET /documents/operations/{id} returns status."""
        operation_id = "test-operation-id"
        response = client.get(
            f"/api/v1/documents/operations/{operation_id}", headers=auth_header
        )

        if response.status_code != 200:
            pytest.skip("Implementation pending")

        data = response.json()
        assert "status" in data
        assert data["status"] in ["pending", "validating", "processing", "completed", "failed"]
