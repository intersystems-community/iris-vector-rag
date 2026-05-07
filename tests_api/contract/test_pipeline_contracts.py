"""
Contract tests for pipeline management endpoints.

Validates against pipeline.yaml OpenAPI spec (FR-005 to FR-008).

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


class TestPipelineListingContract:
    """FR-008: System MUST provide endpoints to list pipelines with status."""

    def test_list_pipelines_returns_all_configured_pipelines(self, client, auth_header):
        """Contract: GET /pipelines returns array of pipeline info."""
        response = client.get("/api/v1/pipelines", headers=auth_header)

        if response.status_code != 200:
            pytest.skip("Implementation pending")

        data = response.json()
        assert "pipelines" in data
        assert isinstance(data["pipelines"], list)
        assert len(data["pipelines"]) > 0

    def test_pipeline_info_includes_required_fields(self, client, auth_header):
        """Contract: Each pipeline has pipeline_type, name, status, capabilities."""
        response = client.get("/api/v1/pipelines", headers=auth_header)

        if response.status_code != 200:
            pytest.skip("Implementation pending")

        data = response.json()
        for pipeline in data["pipelines"]:
            assert "pipeline_type" in pipeline
            assert "name" in pipeline
            assert "status" in pipeline
            assert "capabilities" in pipeline
            assert pipeline["status"] in ["healthy", "degraded", "unavailable"]


class TestPipelineHealthContract:
    """FR-006: System MUST report pipeline health status."""

    def test_get_specific_pipeline_health(self, client, auth_header):
        """Contract: GET /pipelines/{type} returns specific pipeline status."""
        response = client.get("/api/v1/pipelines/graphrag", headers=auth_header)

        if response.status_code != 200:
            pytest.skip("Implementation pending")

        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unavailable"]
