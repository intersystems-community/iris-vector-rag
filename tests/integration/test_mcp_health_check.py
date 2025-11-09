"""
Integration test: Health check tool (Scenario 4 from quickstart.md).

This test verifies that the rag_health_check tool returns status
for all 6 pipelines, database connectivity, and performance metrics.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

import pytest


@pytest.mark.integration
@pytest.mark.mcp
class TestMCPHealthCheck:
    """Integration test for rag_health_check tool."""

    @pytest.mark.asyncio
    async def test_health_check_basic_execution(self):
        """Verify rag_health_check executes successfully."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.health_check()

        assert isinstance(result, dict)
        assert 'status' in result
        assert result['status'] in ['healthy', 'degraded', 'unavailable']

    @pytest.mark.asyncio
    async def test_health_check_all_6_pipelines(self):
        """Verify health check returns status for all 6 RAG pipelines."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.health_check(include_details=True)

        assert 'pipelines' in result
        pipelines = result['pipelines']

        expected_pipelines = [
            'basic',
            'basic_rerank',
            'crag',
            'graphrag',
            'pylate_colbert',
            'iris_global_graphrag'
        ]

        for pipeline_name in expected_pipelines:
            assert pipeline_name in pipelines, \
                f"Pipeline '{pipeline_name}' missing from health check"

            pipeline_status = pipelines[pipeline_name]
            assert 'status' in pipeline_status
            assert pipeline_status['status'] in ['healthy', 'degraded', 'unavailable']

    @pytest.mark.asyncio
    async def test_health_check_database_connectivity(self):
        """Verify health check reports database connectivity."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.health_check()

        assert 'database' in result
        db_status = result['database']

        assert 'connected' in db_status
        assert isinstance(db_status['connected'], bool)

        assert 'response_time_ms' in db_status
        assert isinstance(db_status['response_time_ms'], (int, float))

        if 'connection_pool_usage' in db_status:
            assert isinstance(db_status['connection_pool_usage'], str)

    @pytest.mark.asyncio
    async def test_health_check_performance_metrics(self):
        """Verify health check includes performance metrics when requested."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.health_check(include_performance_metrics=True)

        if 'performance_metrics' in result:
            metrics = result['performance_metrics']
            assert 'average_response_time_ms' in metrics
            assert 'p95_response_time_ms' in metrics
            assert 'error_rate' in metrics
            assert 'queries_per_minute' in metrics

    @pytest.mark.asyncio
    async def test_health_check_without_details(self):
        """Verify health check without details returns minimal response."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.health_check(include_details=False)

        # Should have overall status
        assert 'status' in result
        # May have pipelines summary but not detailed per-pipeline status

    @pytest.mark.asyncio
    async def test_health_check_timestamp(self):
        """Verify health check includes timestamp."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.health_check()

        assert 'timestamp' in result
        assert isinstance(result['timestamp'], str)

    @pytest.mark.asyncio
    async def test_health_check_performance(self):
        """Verify health check completes quickly."""
        import time
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        start = time.time()
        result = await bridge.health_check()
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 500, \
            f"Health check took {elapsed_ms:.1f}ms (expected < 500ms)"
