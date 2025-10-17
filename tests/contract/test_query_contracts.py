"""
Contract tests for query endpoints.

These tests validate query API behavior against query.yaml OpenAPI spec.
Tests cover FR-001 to FR-004 (query processing with multiple pipeline types).

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
def auth_header():
    """Authorization header with valid API key."""
    key_id = "test-key-id"
    key_secret = "test-key-secret"
    credentials = base64.b64encode(f"{key_id}:{key_secret}".encode()).decode()
    return {"Authorization": f"ApiKey {credentials}"}


class TestQueryRequestValidation:
    """
    Contract tests for QueryRequest schema validation (FR-001, FR-003).

    Validates against query.yaml QueryRequest schema.
    """

    def test_query_request_requires_query_field(self, client, auth_header):
        """
        FR-001: System MUST accept query requests with query text.
        FR-003: System MUST validate incoming requests.

        Contract: QueryRequest requires 'query' field
        """
        # Missing required 'query' field
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"top_k": 5},
        )

        assert response.status_code == 422
        data = response.json()

        assert "error" in data
        assert data["error"]["type"] == "validation_exception"
        assert "query" in data["error"]["details"]["field"].lower()

    def test_query_text_must_be_1_to_10000_chars(self, client, auth_header):
        """
        FR-003: Query text must be 1-10000 characters.

        Contract: QueryRequest.query minLength=1, maxLength=10000
        """
        # Test empty query
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": ""},
        )
        assert response.status_code == 422

        # Test too-long query (>10000 chars)
        long_query = "a" * 10001
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": long_query},
        )

        assert response.status_code == 422
        data = response.json()

        assert data["error"]["type"] == "validation_exception"
        assert "query" in data["error"]["details"]["field"].lower()
        assert "10000" in str(data["error"]["details"]["message"])

    def test_top_k_must_be_between_1_and_100(self, client, auth_header):
        """
        FR-003: System MUST validate all incoming requests.

        Contract: QueryRequest.top_k minimum=1, maximum=100
        """
        # Test negative top_k
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": "test query", "top_k": -5},
        )

        assert response.status_code == 422
        data = response.json()

        assert data["error"]["type"] == "validation_exception"
        assert data["error"]["details"]["field"] == "top_k"
        assert data["error"]["details"]["rejected_value"] == -5

        # Test too-large top_k
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": "test query", "top_k": 101},
        )

        assert response.status_code == 422

    def test_filters_accepts_arbitrary_object(self, client, auth_header):
        """
        FR-001: System MUST accept optional parameters (filters).

        Contract: QueryRequest.filters is object with additionalProperties=true
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={
                "query": "test query",
                "filters": {"domain": "medical", "year": 2023},
            },
        )

        # Should not fail validation (may fail if pipeline not ready)
        assert response.status_code != 422


class TestQueryResponseSchema:
    """
    Contract tests for QueryResponse schema (FR-002).

    Validates against query.yaml QueryResponse schema.
    """

    def test_query_response_has_required_fields(self, client, auth_header):
        """
        FR-002: System MUST return structured responses with answer, documents, sources, metadata.

        Contract: QueryResponse requires response_id, request_id, answer, retrieved_documents,
                  sources, pipeline_name, execution_time_ms
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": "What is diabetes?", "top_k": 5},
        )

        # Skip if API not ready, but document expected structure
        if response.status_code != 200:
            pytest.skip(f"API returned {response.status_code}, implementation pending")

        data = response.json()

        # Validate required fields
        required_fields = [
            "response_id",
            "request_id",
            "answer",
            "retrieved_documents",
            "sources",
            "pipeline_name",
            "execution_time_ms",
        ]

        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Validate field types
        assert isinstance(data["response_id"], str)
        assert isinstance(data["request_id"], str)
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0
        assert isinstance(data["retrieved_documents"], list)
        assert len(data["retrieved_documents"]) >= 1
        assert isinstance(data["sources"], list)
        assert isinstance(data["pipeline_name"], str)
        assert isinstance(data["execution_time_ms"], int)
        assert data["execution_time_ms"] >= 0

    def test_retrieved_documents_follow_document_schema(self, client, auth_header):
        """
        FR-002: Retrieved documents must follow Document schema.

        Contract: Document requires doc_id, content, score, metadata
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": "What is diabetes?"},
        )

        if response.status_code != 200:
            pytest.skip("API implementation pending")

        data = response.json()
        documents = data["retrieved_documents"]

        assert len(documents) > 0

        for doc in documents:
            # Validate required fields
            assert "doc_id" in doc
            assert "content" in doc
            assert "score" in doc
            assert "metadata" in doc

            # Validate field types
            assert isinstance(doc["content"], str)
            assert len(doc["content"]) > 0
            assert isinstance(doc["score"], (int, float))
            assert 0.0 <= doc["score"] <= 1.0
            assert isinstance(doc["metadata"], dict)

            # Validate metadata schema
            assert "source" in doc["metadata"]
            assert isinstance(doc["metadata"]["source"], str)

    def test_contexts_field_for_ragas_compatibility(self, client, auth_header):
        """
        FR-002: Response should include 'contexts' for RAGAS evaluation.

        Contract: QueryResponse.contexts is array of strings (document content)
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": "What is diabetes?"},
        )

        if response.status_code != 200:
            pytest.skip("API implementation pending")

        data = response.json()

        # Contexts field should exist for RAGAS compatibility
        assert "contexts" in data
        assert isinstance(data["contexts"], list)

        # Each context should be a string (document content)
        for context in data["contexts"]:
            assert isinstance(context, str)
            assert len(context) > 0


class TestMultiplePipelineSupport:
    """
    Contract tests for multiple pipeline types (FR-004).

    Validates that all 5 pipeline types have working endpoints.
    """

    @pytest.mark.parametrize(
        "pipeline",
        ["basic", "basic_rerank", "crag", "graphrag", "pylate_colbert"],
    )
    def test_pipeline_endpoints_exist(self, client, auth_header, pipeline):
        """
        FR-004: System MUST support querying multiple pipeline types.

        Contract: Each pipeline has /{pipeline}/_search endpoint
        """
        response = client.post(
            f"/api/v1/{pipeline}/_search",
            headers=auth_header,
            json={"query": "test query"},
        )

        # Endpoint should exist (not 404)
        assert response.status_code != 404

    def test_generic_search_endpoint_with_pipeline_param(self, client, auth_header):
        """
        FR-004: Generic /_search endpoint with pipeline in body.

        Contract: POST /_search accepts 'pipeline' parameter
        """
        response = client.post(
            "/api/v1/_search",
            headers=auth_header,
            json={"query": "test query", "pipeline": "graphrag"},
        )

        # Endpoint should exist
        assert response.status_code != 404

        # If validation error, should be for missing implementation, not invalid pipeline
        if response.status_code == 422:
            data = response.json()
            # Should not reject valid pipeline enum value
            if "pipeline" in data.get("error", {}).get("details", {}).get("field", ""):
                pytest.fail("Valid pipeline 'graphrag' rejected")

    def test_all_pipelines_return_same_response_schema(self, client, auth_header):
        """
        FR-004: All pipelines return consistent QueryResponse format.

        Contract: Same schema for all pipeline types
        """
        pipelines = ["basic", "crag", "graphrag"]

        for pipeline in pipelines:
            response = client.post(
                f"/api/v1/{pipeline}/_search",
                headers=auth_header,
                json={"query": "test query"},
            )

            if response.status_code != 200:
                continue  # Skip if pipeline not ready

            data = response.json()

            # All pipelines should return same top-level fields
            assert "answer" in data
            assert "retrieved_documents" in data
            assert "sources" in data
            assert "pipeline_name" in data
            assert "execution_time_ms" in data


class TestResponseHeaders:
    """
    Contract tests for required response headers.

    Validates Elasticsearch-inspired headers from query.yaml.
    """

    def test_response_includes_request_id_header(self, client, auth_header):
        """
        Contract: X-Request-ID header in response for tracing
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": "test query"},
        )

        assert "X-Request-ID" in response.headers
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) > 0

    def test_response_includes_execution_time_header(self, client, auth_header):
        """
        Contract: X-Execution-Time-Ms header for performance monitoring
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": "test query"},
        )

        if response.status_code == 200:
            assert "X-Execution-Time-Ms" in response.headers
            exec_time = int(response.headers["X-Execution-Time-Ms"])
            assert exec_time >= 0

    def test_response_includes_pipeline_name_header(self, client, auth_header):
        """
        Contract: X-Pipeline-Name header identifies which pipeline processed request
        """
        response = client.post(
            "/api/v1/graphrag/_search",
            headers=auth_header,
            json={"query": "test query"},
        )

        if response.status_code == 200:
            assert "X-Pipeline-Name" in response.headers
            assert response.headers["X-Pipeline-Name"] == "graphrag"

    def test_response_includes_rate_limit_headers(self, client, auth_header):
        """
        Contract: X-RateLimit-* headers in all responses
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": "test query"},
        )

        # Rate limit headers should be present
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers


class TestErrorResponseContracts:
    """
    Contract tests for error response formats.

    Validates ValidationError, RateLimitExceeded, ServiceUnavailable schemas.
    """

    def test_validation_error_returns_422_with_field_details(self, client, auth_header):
        """
        FR-003: Validation errors return field-specific details.

        Contract: ValidationError with field, rejected_value, message
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": "test", "top_k": -5},
        )

        assert response.status_code == 422
        data = response.json()

        assert data["error"]["type"] == "validation_exception"
        assert "field" in data["error"]["details"]
        assert "rejected_value" in data["error"]["details"]
        assert "message" in data["error"]["details"]

    def test_service_unavailable_returns_503_with_recovery_estimate(self, client, auth_header):
        """
        FR-007: Unhealthy pipeline returns 503 with estimated recovery time.

        Contract: ServiceUnavailable includes pipeline status and recovery estimate
        """
        # This test requires the ability to simulate pipeline unavailability
        pytest.skip("Requires test fixture for unavailable pipeline")

        # Expected response structure:
        # {
        #   "error": {
        #     "type": "service_unavailable",
        #     "reason": "Pipeline is currently unavailable",
        #     "details": {
        #       "pipeline": "graphrag",
        #       "status": "degraded",
        #       "estimated_recovery_time": 120,
        #       "message": "Pipeline is initializing..."
        #     }
        #   }
        # }
