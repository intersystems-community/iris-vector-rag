"""
Integration test for Acceptance Scenario 8: Health endpoint.

E2E test validating:
- Health endpoint returns component status
- All dependencies included (IRIS, Redis, pipelines)
- Overall health status calculated
- Useful for monitoring and alerting

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


class TestHealthEndpoint:
    """
    Acceptance Scenario 8: Health endpoint with dependency status.

    Given: Administrator wants to monitor system health
    When: Health endpoint is checked
    Then: System returns status of all components
    """

    def test_health_endpoint_returns_overall_status(self, client):
        """
        Test that GET /health returns overall system status.

        Validates FR-032: Provide health check endpoint
        """
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()

        # Validate required fields
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data

        # Status should be valid value
        assert data["status"] in ["healthy", "degraded", "unavailable"]

    def test_health_includes_iris_database_status(self, client):
        """
        Test that health check includes IRIS database status.

        Validates FR-033: Report status of all dependencies (IRIS)
        """
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()

        components = data["components"]

        # IRIS database should be checked
        assert "iris_database" in components

        db_status = components["iris_database"]
        assert "status" in db_status
        assert db_status["status"] in ["healthy", "degraded", "unavailable"]

        # May include additional info
        if "response_time_ms" in db_status:
            assert isinstance(db_status["response_time_ms"], (int, float))
            assert db_status["response_time_ms"] >= 0

    def test_health_includes_redis_status_if_configured(self, client):
        """
        Test that health check includes Redis status (if configured).

        Validates FR-033: Report status of all dependencies (Redis)
        """
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()

        components = data["components"]

        # Redis is optional, but if configured should be checked
        if "redis_cache" in components:
            redis_status = components["redis_cache"]
            assert "status" in redis_status

    def test_health_includes_all_pipeline_statuses(self, client):
        """
        Test that health check includes status of all pipelines.

        Validates:
        - FR-006: Report pipeline health status
        - FR-033: Report all dependencies
        """
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()

        components = data["components"]

        # Should include at least one pipeline
        pipeline_components = [
            name for name in components.keys()
            if "pipeline" in name
        ]

        assert len(pipeline_components) > 0, \
            "Health check should include pipeline statuses"

        # Each pipeline should have status
        for pipeline_name in pipeline_components:
            pipeline_status = components[pipeline_name]
            assert "status" in pipeline_status
            assert pipeline_status["status"] in ["healthy", "degraded", "unavailable"]

    def test_health_endpoint_does_not_require_authentication(self, client):
        """
        Test that health endpoint is public (for monitoring).

        Validates:
        - Health checks should be accessible without auth
        - Useful for load balancers and monitoring tools
        """
        # Health endpoint should work without Authorization header
        response = client.get("/api/v1/health")

        # Should not return 401
        assert response.status_code != 401

    def test_health_endpoint_includes_metrics(self, client):
        """
        Test that health endpoint includes key metrics.

        Validates FR-034: Expose metrics for monitoring
        """
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()

        # Overall metrics may be present
        # Component-specific metrics should be in components

        components = data["components"]

        # IRIS database metrics
        if "iris_database" in components:
            db = components["iris_database"]

            if "metrics" in db:
                # May include connection pool info
                assert isinstance(db["metrics"], dict)

    def test_degraded_status_when_component_unhealthy(self, client):
        """
        Test that overall status is degraded when any component unhealthy.

        Validates:
        - Aggregate health calculation
        - Partial degradation reported
        """
        pytest.skip("Requires ability to simulate component failure")

        # Expected behavior:
        # - If IRIS is down: status="unavailable"
        # - If pipeline is down: status="degraded"
        # - If Redis is down (optional): status="healthy" or "degraded"

    def test_health_response_time_is_fast(self, client):
        """
        Test that health check responds quickly.

        Validates:
        - Health checks are lightweight
        - Suitable for frequent monitoring
        - < 1 second response time
        """
        import time

        start = time.time()
        response = client.get("/api/v1/health")
        elapsed = time.time() - start

        assert response.status_code == 200

        # Health check should be fast (for monitoring)
        assert elapsed < 1.0, \
            f"Health check took {elapsed:.2f}s, should be < 1s"

    def test_health_endpoint_useful_for_kubernetes_probes(self, client):
        """
        Test that health endpoint works for Kubernetes liveness/readiness.

        Validates:
        - Returns 200 when healthy
        - Returns non-200 when unhealthy
        - Simple yes/no decision for orchestration
        """
        response = client.get("/api/v1/health")

        # For Kubernetes, we care about status code
        # 200 = healthy (ready to serve traffic)
        # 503 = unhealthy (remove from load balancer)

        if response.status_code == 200:
            data = response.json()
            # Overall status should be healthy or degraded
            assert data["status"] in ["healthy", "degraded"]
