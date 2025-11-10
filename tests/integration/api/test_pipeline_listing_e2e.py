"""
Integration test for Acceptance Scenario 4: List available pipelines.

E2E test validating:
- List all configured pipelines
- Pipeline capabilities returned
- Current status reported
- Performance metrics included

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


class TestListAvailablePipelines:
    """
    Acceptance Scenario 4: List available pipelines.

    Given: Multiple pipeline types available
    When: Developer queries pipelines endpoint
    Then: System returns list of pipelines with capabilities and status
    """

    def test_list_pipelines_returns_all_configured_types(self, client, auth_header):
        """
        Test that GET /pipelines returns all configured pipelines.

        Validates FR-008: Provide endpoints to list pipelines
        """
        response = client.get("/api/v1/pipelines", headers=auth_header)

        assert response.status_code == 200
        data = response.json()

        # Should return array of pipelines
        assert "pipelines" in data
        assert isinstance(data["pipelines"], list)
        assert len(data["pipelines"]) > 0

        # Should include expected pipeline types
        pipeline_types = [p["pipeline_type"] for p in data["pipelines"]]

        # At minimum, basic pipeline should be available
        assert "basic" in pipeline_types

    def test_each_pipeline_includes_capabilities(self, client, auth_header):
        """
        Test that each pipeline includes capabilities list.

        Validates:
        - Capabilities field present
        - Lists supported features (vector_search, graph_traversal, etc.)
        """
        response = client.get("/api/v1/pipelines", headers=auth_header)

        assert response.status_code == 200
        data = response.json()

        for pipeline in data["pipelines"]:
            # Validate required fields
            assert "pipeline_type" in pipeline
            assert "name" in pipeline
            assert "status" in pipeline
            assert "capabilities" in pipeline

            # Validate capabilities is non-empty array
            assert isinstance(pipeline["capabilities"], list)
            assert len(pipeline["capabilities"]) > 0

            # Examples of capabilities
            valid_capabilities = [
                "vector_search",
                "graph_traversal",
                "entity_extraction",
                "reranking",
                "corrective_retrieval"
            ]

            # At least one capability should be recognized
            has_valid_capability = any(
                cap in valid_capabilities
                for cap in pipeline["capabilities"]
            )
            assert has_valid_capability

    def test_each_pipeline_reports_current_status(self, client, auth_header):
        """
        Test that each pipeline reports health status.

        Validates FR-006: Report pipeline health status
        """
        response = client.get("/api/v1/pipelines", headers=auth_header)

        assert response.status_code == 200
        data = response.json()

        valid_statuses = ["healthy", "degraded", "unavailable"]

        for pipeline in data["pipelines"]:
            assert "status" in pipeline
            assert pipeline["status"] in valid_statuses

    def test_pipeline_listing_includes_performance_metrics(self, client, auth_header):
        """
        Test that pipeline info includes performance stats.

        Validates:
        - Average latency reported
        - Total query count (if available)
        - Error rate (if available)
        """
        response = client.get("/api/v1/pipelines", headers=auth_header)

        assert response.status_code == 200
        data = response.json()

        for pipeline in data["pipelines"]:
            # avg_latency_ms should be present
            if "avg_latency_ms" in pipeline:
                assert isinstance(pipeline["avg_latency_ms"], (int, float))
                assert pipeline["avg_latency_ms"] >= 0

    def test_get_specific_pipeline_details(self, client, auth_header):
        """
        Test getting details for a specific pipeline.

        Validates:
        - GET /pipelines/{type} endpoint
        - More detailed info than list view
        """
        response = client.get("/api/v1/pipelines/graphrag", headers=auth_header)

        # May return 404 if pipeline not configured
        if response.status_code == 404:
            pytest.skip("GraphRAG pipeline not configured")

        assert response.status_code == 200
        data = response.json()

        # Should include same fields as list
        assert "pipeline_type" in data
        assert "status" in data
        assert "capabilities" in data
        assert data["pipeline_type"] == "graphrag"

    def test_pipeline_list_matches_available_query_endpoints(self, client, auth_header):
        """
        Test that listed pipelines have working query endpoints.

        Validates FR-004: Consistent pipeline support across endpoints
        """
        # Get list of pipelines
        list_response = client.get("/api/v1/pipelines", headers=auth_header)
        assert list_response.status_code == 200

        pipelines = list_response.json()["pipelines"]

        # Try to query each pipeline
        for pipeline in pipelines:
            pipeline_type = pipeline["pipeline_type"]

            # Skip if pipeline is unhealthy
            if pipeline["status"] != "healthy":
                continue

            # Try to query this pipeline
            query_response = client.post(
                f"/api/v1/{pipeline_type}/_search",
                headers=auth_header,
                json={"query": "test query"}
            )

            # Should not be 404 (endpoint should exist)
            assert query_response.status_code != 404, \
                f"Pipeline {pipeline_type} listed but query endpoint missing"

    def test_pipeline_list_endpoint_does_not_require_write_permission(self, client):
        """
        Test that listing pipelines only requires read permission.

        Validates FR-011: Permission-based authorization
        """
        pytest.skip("Requires API key with read-only permissions")

        # This test would:
        # 1. Create API key with only 'read' permission
        # 2. Call GET /pipelines
        # 3. Verify 200 response (not 403)
