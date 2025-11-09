"""
Integration test for Acceptance Scenario 1: Query with valid API key.

E2E test validating:
- Valid API key authentication
- Query processing through pipeline
- Response returned within 2 seconds
- Structured response with answer, documents, sources

IMPORTANT: This test MUST FAIL initially (TDD principle).
"""

import pytest
import time
import base64
from fastapi.testclient import TestClient

try:
    from iris_vector_rag.api.main import create_app
except ImportError:
    create_app = None

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    """Create FastAPI test client with initialized app."""
    if create_app is None:
        pytest.skip("API application not implemented yet (TDD)")

    app = create_app()
    return TestClient(app)


@pytest.fixture
def valid_api_key(client):
    """
    Create a valid API key for testing.

    In production, this would be created via API key management CLI.
    For testing, we'll use a test fixture key.
    """
    # This will need to be replaced with actual API key creation
    # once the auth service is implemented
    pytest.skip("Requires API key creation fixture")

    # Expected usage:
    # from iris_vector_rag.api.services.auth_service import AuthService
    # auth_service = AuthService()
    # key = auth_service.create_key(
    #     name="test-key",
    #     permissions=["read", "write"],
    #     tier="premium"
    # )
    # return key


@pytest.fixture
def auth_header(valid_api_key):
    """
    Generate Authorization header from API key.

    Format: Authorization: ApiKey <base64(id:secret)>
    """
    key_id = valid_api_key["key_id"]
    key_secret = valid_api_key["secret"]

    credentials = f"{key_id}:{key_secret}"
    encoded = base64.b64encode(credentials.encode()).decode()

    return {"Authorization": f"ApiKey {encoded}"}


class TestQueryWithValidAPIKey:
    """
    Acceptance Scenario 1: Query with valid API key.

    Given: A valid API key and query text
    When: Developer sends POST request to query endpoint
    Then: System returns answer with documents and sources within 2 seconds
    """

    def test_authenticated_query_returns_answer_with_documents(self, client, auth_header):
        """
        Test that authenticated query returns complete response.

        Validates:
        - Authentication succeeds with valid API key
        - Query is processed by pipeline
        - Response contains answer, documents, sources
        - Response time < 2 seconds (p95 latency target)
        """
        query = "What is diabetes?"

        start_time = time.time()

        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={
                "query": query,
                "top_k": 5
            }
        )

        elapsed_time = time.time() - start_time

        # Validate response status
        assert response.status_code == 200, \
            f"Expected 200, got {response.status_code}: {response.text}"

        # Validate response time (FR: <2s p95 latency)
        assert elapsed_time < 2.0, \
            f"Query took {elapsed_time:.2f}s, exceeds 2s target"

        data = response.json()

        # Validate required response fields (from quickstart.md scenario 1)
        assert "answer" in data
        assert "retrieved_documents" in data
        assert "sources" in data
        assert "execution_time_ms" in data

        # Validate answer is non-empty
        assert isinstance(data["answer"], str)
        assert len(data["answer"]) > 0

        # Validate documents are returned
        assert isinstance(data["retrieved_documents"], list)
        assert len(data["retrieved_documents"]) > 0
        assert len(data["retrieved_documents"]) <= 5  # top_k limit

        # Validate document structure
        for doc in data["retrieved_documents"]:
            assert "content" in doc
            assert "score" in doc
            assert "metadata" in doc
            assert "source" in doc["metadata"]

        # Validate sources are extracted
        assert isinstance(data["sources"], list)

        # Validate execution metadata
        assert isinstance(data["execution_time_ms"], int)
        assert data["execution_time_ms"] >= 0

    def test_query_different_pipeline_types(self, client, auth_header):
        """
        Test that all 5 pipeline types can be queried successfully.

        Validates FR-004: Support multiple pipeline types
        """
        pipelines = ["basic", "basic_rerank", "crag", "graphrag", "pylate_colbert"]
        query = "What is diabetes?"

        for pipeline in pipelines:
            response = client.post(
                f"/api/v1/{pipeline}/_search",
                headers=auth_header,
                json={"query": query, "top_k": 3}
            )

            # Skip if pipeline not available (optional dependency)
            if response.status_code == 503:
                pytest.skip(f"Pipeline {pipeline} not available")

            assert response.status_code == 200, \
                f"Pipeline {pipeline} failed: {response.text}"

            data = response.json()
            assert "answer" in data
            assert "pipeline_name" in data

    def test_query_with_filters(self, client, auth_header):
        """
        Test query with optional filter parameters.

        Validates FR-001: Accept optional parameters
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={
                "query": "What is diabetes?",
                "top_k": 5,
                "filters": {
                    "domain": "medical",
                    "year": 2023
                }
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_query_response_includes_required_headers(self, client, auth_header):
        """
        Test that response includes Elasticsearch-inspired headers.

        Validates:
        - X-Request-ID (for tracing)
        - X-Execution-Time-Ms (for monitoring)
        - X-Pipeline-Name (for debugging)
        - X-RateLimit-* (for quota tracking)
        """
        response = client.post(
            "/api/v1/graphrag/_search",
            headers=auth_header,
            json={"query": "What is diabetes?"}
        )

        assert response.status_code == 200

        # Validate required headers
        assert "X-Request-ID" in response.headers
        assert "X-Execution-Time-Ms" in response.headers
        assert "X-Pipeline-Name" in response.headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers

        # Validate header values
        assert response.headers["X-Pipeline-Name"] == "graphrag"
        assert int(response.headers["X-Execution-Time-Ms"]) >= 0
        assert int(response.headers["X-RateLimit-Limit"]) > 0

    def test_query_response_is_ragas_compatible(self, client, auth_header):
        """
        Test that response format is compatible with RAGAS evaluation.

        Validates:
        - 'contexts' field contains document content as list of strings
        - LangChain Document structure compatibility
        """
        response = client.post(
            "/api/v1/basic/_search",
            headers=auth_header,
            json={"query": "What is diabetes?"}
        )

        assert response.status_code == 200
        data = response.json()

        # RAGAS compatibility: contexts field
        assert "contexts" in data
        assert isinstance(data["contexts"], list)

        # Each context should be document content
        for context in data["contexts"]:
            assert isinstance(context, str)
            assert len(context) > 0

        # Should match number of retrieved documents
        assert len(data["contexts"]) == len(data["retrieved_documents"])

    def test_concurrent_queries_maintain_performance(self, client, auth_header):
        """
        Test that multiple concurrent queries maintain <2s latency.

        Validates:
        - Connection pool handles concurrent requests
        - Performance target maintained under load
        - No connection pool exhaustion
        """
        import concurrent.futures

        def send_query(query_id):
            start = time.time()
            response = client.post(
                "/api/v1/basic/_search",
                headers=auth_header,
                json={"query": f"Query {query_id}"}
            )
            elapsed = time.time() - start
            return response.status_code, elapsed

        # Send 10 concurrent queries
        num_concurrent = 10

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(send_query, i) for i in range(num_concurrent)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Validate all queries succeeded
        for status_code, elapsed in results:
            assert status_code == 200
            # Allow slightly higher latency under concurrent load
            assert elapsed < 3.0, f"Query took {elapsed:.2f}s under concurrent load"

        # Validate at least some queries met the 2s target
        fast_queries = [e for _, e in results if e < 2.0]
        assert len(fast_queries) >= num_concurrent * 0.7, \
            "At least 70% of concurrent queries should meet 2s target"
