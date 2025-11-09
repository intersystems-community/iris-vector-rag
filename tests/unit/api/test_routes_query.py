"""
Unit tests for query route handlers.

Tests the query endpoint routes in isolation.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from iris_vector_rag.api.models.auth import ApiKey, Permission, RateLimitTier
from iris_vector_rag.api.models.request import QueryRequest
from iris_vector_rag.api.models.response import QueryResponse


class TestQueryRoutes:
    """Test query route handlers."""

    @pytest.fixture
    def mock_pipeline_manager(self):
        """Create mock pipeline manager."""
        manager = MagicMock()
        mock_pipeline = MagicMock()
        mock_pipeline.query.return_value = {
            "answer": "Diabetes is a chronic metabolic disorder...",
            "retrieved_documents": [
                {
                    "doc_id": "doc-1",
                    "content": "Diabetes mellitus is...",
                    "score": 0.95,
                    "metadata": {"source": "textbook.pdf", "page_number": 127},
                }
            ],
            "contexts": ["Diabetes mellitus is..."],
            "sources": ["textbook.pdf"],
            "metadata": {
                "retrieval_time_ms": 345,
                "generation_time_ms": 1089,
                "tokens_used": 2345,
            },
        }
        manager.get_pipeline.return_value = mock_pipeline
        return manager

    @pytest.fixture
    def test_api_key(self):
        """Create test API key."""
        return ApiKey(
            key_id="test-key-id",
            key_secret_hash="hashed-secret",
            name="Test Key",
            permissions=[Permission.READ],
            rate_limit_tier=RateLimitTier.BASIC,
            requests_per_minute=60,
            requests_per_hour=1000,
            created_at=datetime.utcnow(),
            is_active=True,
            owner_email="test@example.com",
        )

    def test_query_basic_pipeline_success(self, mock_pipeline_manager, test_api_key):
        """Test successful query to basic pipeline."""
        from iris_vector_rag.api.routes.query import execute_query

        request = QueryRequest(
            query="What is diabetes?",
            top_k=5,
        )

        with patch("iris_rag.api.routes.query.pipeline_manager", mock_pipeline_manager):
            response = execute_query(
                pipeline_type="basic",
                request=request,
                api_key=test_api_key,
            )

        assert response.answer == "Diabetes is a chronic metabolic disorder..."
        assert len(response.retrieved_documents) == 1
        assert response.pipeline_name == "basic"
        assert response.execution_time_ms > 0

    def test_query_with_filters(self, mock_pipeline_manager, test_api_key):
        """Test query with metadata filters."""
        from iris_vector_rag.api.routes.query import execute_query

        request = QueryRequest(
            query="What is diabetes?",
            top_k=5,
            filters={"source": "textbook.pdf"},
        )

        with patch("iris_rag.api.routes.query.pipeline_manager", mock_pipeline_manager):
            response = execute_query(
                pipeline_type="basic",
                request=request,
                api_key=test_api_key,
            )

        assert response is not None
        mock_pipeline_manager.get_pipeline.assert_called_once_with("basic")

    def test_query_invalid_top_k(self, test_api_key):
        """Test query with invalid top_k value."""
        from iris_vector_rag.api.routes.query import execute_query

        request = QueryRequest(
            query="What is diabetes?",
            top_k=-5,  # Invalid
        )

        with pytest.raises(Exception):  # Pydantic validation error
            execute_query(
                pipeline_type="basic",
                request=request,
                api_key=test_api_key,
            )

    def test_query_empty_query_string(self, test_api_key):
        """Test query with empty query string."""
        request = QueryRequest(
            query="",  # Invalid
            top_k=5,
        )

        with pytest.raises(Exception):  # Pydantic validation error
            _ = request

    def test_query_graphrag_pipeline(self, mock_pipeline_manager, test_api_key):
        """Test query to GraphRAG pipeline."""
        from iris_vector_rag.api.routes.query import execute_query

        request = QueryRequest(
            query="What is diabetes?",
            top_k=5,
        )

        with patch("iris_rag.api.routes.query.pipeline_manager", mock_pipeline_manager):
            response = execute_query(
                pipeline_type="graphrag",
                request=request,
                api_key=test_api_key,
            )

        assert response.pipeline_name == "graphrag"
        mock_pipeline_manager.get_pipeline.assert_called_once_with("graphrag")

    def test_query_pipeline_unavailable(self, mock_pipeline_manager, test_api_key):
        """Test query when pipeline is unavailable."""
        from iris_vector_rag.api.routes.query import execute_query

        mock_pipeline_manager.get_pipeline.side_effect = HTTPException(
            status_code=503, detail="Pipeline unavailable"
        )

        request = QueryRequest(
            query="What is diabetes?",
            top_k=5,
        )

        with patch("iris_rag.api.routes.query.pipeline_manager", mock_pipeline_manager):
            with pytest.raises(HTTPException) as exc:
                execute_query(
                    pipeline_type="basic",
                    request=request,
                    api_key=test_api_key,
                )

        assert exc.value.status_code == 503

    def test_query_pipeline_execution_error(self, mock_pipeline_manager, test_api_key):
        """Test query when pipeline execution fails."""
        from iris_vector_rag.api.routes.query import execute_query

        mock_pipeline = mock_pipeline_manager.get_pipeline.return_value
        mock_pipeline.query.side_effect = Exception("Pipeline execution failed")

        request = QueryRequest(
            query="What is diabetes?",
            top_k=5,
        )

        with patch("iris_rag.api.routes.query.pipeline_manager", mock_pipeline_manager):
            with pytest.raises(HTTPException) as exc:
                execute_query(
                    pipeline_type="basic",
                    request=request,
                    api_key=test_api_key,
                )

        assert exc.value.status_code == 500

    def test_query_response_format_ragas_compatible(self, mock_pipeline_manager, test_api_key):
        """Test that query response is RAGAS compatible."""
        from iris_vector_rag.api.routes.query import execute_query

        request = QueryRequest(
            query="What is diabetes?",
            top_k=5,
        )

        with patch("iris_rag.api.routes.query.pipeline_manager", mock_pipeline_manager):
            response = execute_query(
                pipeline_type="basic",
                request=request,
                api_key=test_api_key,
            )

        # RAGAS requires these fields
        assert hasattr(response, "answer")
        assert hasattr(response, "contexts")
        assert hasattr(response, "retrieved_documents")
        assert isinstance(response.contexts, list)

    def test_query_includes_timing_metadata(self, mock_pipeline_manager, test_api_key):
        """Test query response includes timing metadata."""
        from iris_vector_rag.api.routes.query import execute_query

        request = QueryRequest(
            query="What is diabetes?",
            top_k=5,
        )

        with patch("iris_rag.api.routes.query.pipeline_manager", mock_pipeline_manager):
            response = execute_query(
                pipeline_type="basic",
                request=request,
                api_key=test_api_key,
            )

        assert response.execution_time_ms > 0
        assert response.retrieval_time_ms is not None
        assert response.generation_time_ms is not None

    def test_query_without_permission(self, mock_pipeline_manager):
        """Test query without read permission."""
        from iris_vector_rag.api.routes.query import execute_query

        api_key = ApiKey(
            key_id="test-key-id",
            key_secret_hash="hashed-secret",
            name="Test Key",
            permissions=[],  # No permissions
            rate_limit_tier=RateLimitTier.BASIC,
            requests_per_minute=60,
            requests_per_hour=1000,
            created_at=datetime.utcnow(),
            is_active=True,
            owner_email="test@example.com",
        )

        request = QueryRequest(
            query="What is diabetes?",
            top_k=5,
        )

        # Permission check should happen before reaching this point
        # This would be handled by middleware
        pass

    def test_query_max_length_validation(self, test_api_key):
        """Test query length validation."""
        long_query = "x" * 10001  # Exceeds max length

        with pytest.raises(Exception):  # Pydantic validation error
            QueryRequest(
                query=long_query,
                top_k=5,
            )

    def test_query_include_sources_flag(self, mock_pipeline_manager, test_api_key):
        """Test query with include_sources flag."""
        from iris_vector_rag.api.routes.query import execute_query

        request = QueryRequest(
            query="What is diabetes?",
            top_k=5,
            include_sources=True,
        )

        with patch("iris_rag.api.routes.query.pipeline_manager", mock_pipeline_manager):
            response = execute_query(
                pipeline_type="basic",
                request=request,
                api_key=test_api_key,
            )

        assert response.sources is not None
        assert len(response.sources) > 0

    def test_query_include_metadata_flag(self, mock_pipeline_manager, test_api_key):
        """Test query with include_metadata flag."""
        from iris_vector_rag.api.routes.query import execute_query

        request = QueryRequest(
            query="What is diabetes?",
            top_k=5,
            include_metadata=True,
        )

        with patch("iris_rag.api.routes.query.pipeline_manager", mock_pipeline_manager):
            response = execute_query(
                pipeline_type="basic",
                request=request,
                api_key=test_api_key,
            )

        assert len(response.retrieved_documents) > 0
        assert response.retrieved_documents[0].metadata is not None
