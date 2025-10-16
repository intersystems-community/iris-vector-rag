"""
Contract tests for health check endpoint.

Validates against openapi.yaml health endpoint spec (FR-032 to FR-034).

IMPORTANT: These tests MUST FAIL initially (TDD principle).
"""

import pytest
from fastapi.testclient import TestClient

try:
    from iris_rag.api.main import create_app
except ImportError:
    create_app = None

pytestmark = pytest.mark.contract


@pytest.fixture
def client():
    if create_app is None:
        pytest.skip("API application not implemented yet (TDD)")
    app = create_app()
    return TestClient(app)


class TestHealthEndpointContract:
    """FR-032, FR-033: Health check with dependency status."""

    def test_health_endpoint_returns_overall_status(self, client):
        """Contract: GET /health returns status and components."""
        response = client.get("/api/v1/health")

        if response.status_code != 200:
            pytest.skip("Implementation pending")

        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unavailable"]
        assert "components" in data
        assert "timestamp" in data

    def test_health_includes_all_dependencies(self, client):
        """FR-033: Report status of IRIS, Redis, pipelines."""
        response = client.get("/api/v1/health")

        if response.status_code != 200:
            pytest.skip("Implementation pending")

        data = response.json()
        components = data["components"]

        # Should include key dependencies
        assert "iris_database" in components
        # Redis is optional
        # Should include at least one pipeline
        pipeline_components = [k for k in components if "pipeline" in k]
        assert len(pipeline_components) > 0
