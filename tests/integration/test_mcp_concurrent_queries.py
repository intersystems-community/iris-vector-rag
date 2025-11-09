"""
Integration test: Concurrent queries (Scenario 5 from quickstart.md).

This test verifies that the MCP server handles concurrent requests
correctly and enforces the 5-connection limit.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

import pytest
import asyncio


@pytest.mark.integration
@pytest.mark.mcp
class TestMCPConcurrentQueries:
    """Integration test for concurrent MCP query handling."""

    @pytest.mark.asyncio
    async def test_three_concurrent_queries_different_pipelines(self):
        """Verify 3 concurrent queries to different pipelines complete successfully."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Create 3 concurrent query tasks
        tasks = [
            bridge.invoke_technique('basic', 'What is diabetes?', {'top_k': 3}),
            bridge.invoke_technique('crag', 'What are diabetes symptoms?', {'top_k': 3}),
            bridge.invoke_technique('graphrag', 'How is diabetes treated?', {'top_k': 3})
        ]

        # Execute concurrently
        results = await asyncio.gather(*tasks)

        # Verify all succeeded
        for i, result in enumerate(results):
            assert result['success'] is True, f"Query {i} failed: {result.get('error')}"
            assert 'answer' in result['result']

    @pytest.mark.asyncio
    async def test_five_concurrent_queries_max_connections(self):
        """Verify 5 concurrent queries work (at connection limit)."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Create 5 concurrent queries
        tasks = [
            bridge.invoke_technique('basic', f'Query {i}', {'top_k': 2})
            for i in range(5)
        ]

        # Execute concurrently (should succeed - at limit but not exceeding)
        results = await asyncio.gather(*tasks)

        # All should succeed
        for i, result in enumerate(results):
            assert result['success'] is True, f"Query {i} failed at max connections"

    @pytest.mark.asyncio
    async def test_no_resource_conflicts_concurrent(self):
        """Verify concurrent queries don't have resource conflicts."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Run same query multiple times concurrently
        query = 'What is diabetes mellitus?'
        tasks = [
            bridge.invoke_technique('basic', query, {'top_k': 5})
            for _ in range(3)
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed with consistent results
        for result in results:
            assert result['success'] is True
            assert len(result['result']['retrieved_documents']) == 5

    @pytest.mark.asyncio
    async def test_concurrent_queries_performance(self):
        """Verify concurrent queries don't significantly degrade performance."""
        import time
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Measure sequential execution
        start = time.time()
        for i in range(3):
            await bridge.invoke_technique('basic', f'Query {i}', {'top_k': 3})
        sequential_time = time.time() - start

        # Measure concurrent execution
        start = time.time()
        tasks = [
            bridge.invoke_technique('basic', f'Query {i}', {'top_k': 3})
            for i in range(3)
        ]
        await asyncio.gather(*tasks)
        concurrent_time = time.time() - start

        # Concurrent should be faster (or at least not much slower)
        # Allow 20% overhead for coordination
        assert concurrent_time < sequential_time * 1.2, \
            f"Concurrent ({concurrent_time:.2f}s) slower than sequential ({sequential_time:.2f}s)"
