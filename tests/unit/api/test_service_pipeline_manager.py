"""
Unit tests for pipeline manager service.

Tests the pipeline lifecycle management service in isolation.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from iris_rag.api.models.pipeline import PipelineInstance, PipelineStatus
from iris_rag.api.services.pipeline_manager import PipelineManager


class TestPipelineManager:
    """Test pipeline manager service."""

    @pytest.fixture
    def mock_connection_pool(self):
        """Create mock connection pool."""
        pool = MagicMock()
        conn = MagicMock()
        cursor = MagicMock()

        pool.get_connection.return_value.__enter__.return_value = conn
        conn.cursor.return_value = cursor

        return pool

    @pytest.fixture
    def pipeline_manager(self, mock_connection_pool):
        """Create pipeline manager instance."""
        return PipelineManager(mock_connection_pool)

    @pytest.fixture
    def mock_pipeline(self):
        """Create mock RAG pipeline."""
        pipeline = MagicMock()
        pipeline.query.return_value = {
            "answer": "Test answer",
            "retrieved_documents": [],
            "contexts": [],
            "sources": [],
            "metadata": {},
        }
        return pipeline

    def test_initialize_pipeline_basic(self, pipeline_manager):
        """Test initializing basic pipeline."""
        with patch("iris_rag.create_pipeline") as mock_create:
            mock_create.return_value = MagicMock()

            pipeline = pipeline_manager.get_pipeline("basic")

            assert pipeline is not None
            mock_create.assert_called_once_with(
                pipeline_type="basic",
                validate_requirements=True,
                auto_setup=False,
            )

    def test_initialize_pipeline_graphrag(self, pipeline_manager):
        """Test initializing GraphRAG pipeline."""
        with patch("iris_rag.create_pipeline") as mock_create:
            mock_create.return_value = MagicMock()

            pipeline = pipeline_manager.get_pipeline("graphrag")

            assert pipeline is not None
            mock_create.assert_called_once()

    def test_get_pipeline_caching(self, pipeline_manager):
        """Test pipeline instances are cached."""
        with patch("iris_rag.create_pipeline") as mock_create:
            mock_create.return_value = MagicMock()

            pipeline1 = pipeline_manager.get_pipeline("basic")
            pipeline2 = pipeline_manager.get_pipeline("basic")

            assert pipeline1 is pipeline2
            mock_create.assert_called_once()  # Should only initialize once

    def test_get_pipeline_invalid_type(self, pipeline_manager):
        """Test getting invalid pipeline type raises error."""
        with pytest.raises(HTTPException) as exc:
            pipeline_manager.get_pipeline("invalid_pipeline")

        assert exc.value.status_code == 404
        assert "not found" in str(exc.value.detail).lower()

    def test_list_all_pipelines(self, pipeline_manager):
        """Test listing all available pipelines."""
        pipelines = pipeline_manager.list_pipelines()

        assert len(pipelines) > 0
        assert any(p.name == "basic" for p in pipelines)
        assert any(p.name == "graphrag" for p in pipelines)

    def test_get_pipeline_status_healthy(self, pipeline_manager, mock_connection_pool):
        """Test getting healthy pipeline status."""
        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.fetchone.return_value = (
            "basic",
            "healthy",
            "1.0.0",
            datetime.utcnow(),
            1000,
            234.56,
            0.01,
            None,
            "vector_search,reranking",
            "Basic RAG pipeline",
        )

        status = pipeline_manager.get_pipeline_status("basic")

        assert status.name == "basic"
        assert status.status == PipelineStatus.HEALTHY
        assert status.total_queries == 1000

    def test_get_pipeline_status_unavailable(self, pipeline_manager, mock_connection_pool):
        """Test getting unavailable pipeline status."""
        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.fetchone.return_value = (
            "graphrag",
            "unavailable",
            "1.0.0",
            datetime.utcnow(),
            0,
            0.0,
            1.0,
            "Pipeline initialization failed",
            "vector_search,graph_traversal",
            "GraphRAG pipeline",
        )

        status = pipeline_manager.get_pipeline_status("graphrag")

        assert status.status == PipelineStatus.UNAVAILABLE
        assert status.error_message is not None

    def test_check_pipeline_health_success(self, pipeline_manager, mock_pipeline):
        """Test checking pipeline health when healthy."""
        with patch.object(pipeline_manager, "get_pipeline", return_value=mock_pipeline):
            is_healthy = pipeline_manager.check_pipeline_health("basic")

        assert is_healthy is True

    def test_check_pipeline_health_failure(self, pipeline_manager):
        """Test checking pipeline health when unhealthy."""
        mock_pipeline = MagicMock()
        mock_pipeline.query.side_effect = Exception("Pipeline error")

        with patch.object(pipeline_manager, "get_pipeline", return_value=mock_pipeline):
            is_healthy = pipeline_manager.check_pipeline_health("basic")

        assert is_healthy is False

    def test_update_pipeline_metrics(self, pipeline_manager, mock_connection_pool):
        """Test updating pipeline metrics."""
        pipeline_name = "basic"
        execution_time_ms = 1234
        error_occurred = False

        pipeline_manager.update_pipeline_metrics(
            pipeline_name, execution_time_ms, error_occurred
        )

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.execute.assert_called()

    def test_get_pipeline_capabilities(self, pipeline_manager):
        """Test getting pipeline capabilities."""
        capabilities = pipeline_manager.get_pipeline_capabilities("basic")

        assert "vector_search" in capabilities
        assert isinstance(capabilities, list)

    def test_warm_up_pipeline(self, pipeline_manager, mock_pipeline):
        """Test warming up pipeline with test query."""
        with patch.object(pipeline_manager, "get_pipeline", return_value=mock_pipeline):
            result = pipeline_manager.warm_up_pipeline("basic")

        assert result is True
        mock_pipeline.query.assert_called_once()

    def test_reload_pipeline(self, pipeline_manager):
        """Test reloading a pipeline."""
        with patch("iris_rag.create_pipeline") as mock_create:
            mock_create.return_value = MagicMock()

            # Get pipeline first time
            pipeline_manager.get_pipeline("basic")

            # Reload pipeline
            pipeline_manager.reload_pipeline("basic")

            # Should have called create_pipeline twice
            assert mock_create.call_count == 2

    def test_get_all_pipeline_statuses(self, pipeline_manager, mock_connection_pool):
        """Test getting all pipeline statuses."""
        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.fetchall.return_value = [
            (
                "basic",
                "healthy",
                "1.0.0",
                datetime.utcnow(),
                1000,
                234.56,
                0.01,
                None,
                "vector_search",
                "Basic RAG",
            ),
            (
                "graphrag",
                "healthy",
                "1.0.0",
                datetime.utcnow(),
                500,
                456.78,
                0.02,
                None,
                "vector_search,graph",
                "GraphRAG",
            ),
        ]

        statuses = pipeline_manager.get_all_pipeline_statuses()

        assert len(statuses) == 2
        assert statuses[0].name == "basic"
        assert statuses[1].name == "graphrag"

    def test_disable_pipeline(self, pipeline_manager, mock_connection_pool):
        """Test disabling a pipeline."""
        pipeline_name = "basic"

        pipeline_manager.disable_pipeline(pipeline_name)

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.execute.assert_called()

    def test_enable_pipeline(self, pipeline_manager, mock_connection_pool):
        """Test enabling a pipeline."""
        pipeline_name = "basic"

        pipeline_manager.enable_pipeline(pipeline_name)

        cursor = mock_connection_pool.get_connection.return_value.__enter__.return_value.cursor.return_value
        cursor.execute.assert_called()
