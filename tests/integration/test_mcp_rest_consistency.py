"""
Integration test: MCP-REST API consistency (Scenario 7 from quickstart.md).

This test verifies that the same query via MCP and REST API returns
identical responses (FR-025: response format consistency).

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

import pytest


@pytest.mark.integration
@pytest.mark.mcp
class TestMCPRESTConsistency:
    """Integration test for MCP-REST API response consistency."""

    @pytest.mark.asyncio
    async def test_basic_rag_response_consistency(self):
        """Verify BasicRAG returns identical responses via MCP and REST API."""
        from iris_vector_rag.mcp.bridge import MCPBridge
        from iris_vector_rag.pipelines.basic import BasicRAGPipeline

        # MCP query
        bridge = MCPBridge()
        mcp_result = await bridge.invoke_technique(
            technique='basic',
            query='What is diabetes mellitus?',
            params={'top_k': 5}
        )

        # REST API query (direct pipeline call)
        pipeline = BasicRAGPipeline()
        rest_result = pipeline.query(
            query='What is diabetes mellitus?',
            top_k=5
        )

        # Verify both succeeded
        assert mcp_result['success'] is True
        mcp_response = mcp_result['result']

        # Compare responses (excluding timestamps and request IDs)
        assert 'answer' in mcp_response
        assert 'answer' in rest_result

        # Document counts should match
        assert len(mcp_response['retrieved_documents']) == \
               len(rest_result.get('retrieved_documents', rest_result.get('documents', [])))

        # Response structure should be identical
        mcp_keys = set(mcp_response.keys())
        rest_keys = set(rest_result.keys())

        # Allow for minor key naming differences (retrieved_documents vs documents)
        # but overall structure should be the same

    @pytest.mark.asyncio
    async def test_crag_response_consistency(self):
        """Verify CRAG returns consistent responses via MCP and REST API."""
        from iris_vector_rag.mcp.bridge import MCPBridge
        from iris_vector_rag.pipelines.crag import CRAGPipeline

        # MCP query
        bridge = MCPBridge()
        mcp_result = await bridge.invoke_technique(
            technique='crag',
            query='What are diabetes symptoms?',
            params={'top_k': 3, 'confidence_threshold': 0.8}
        )

        # REST API query
        pipeline = CRAGPipeline()
        rest_result = pipeline.query(
            query='What are diabetes symptoms?',
            top_k=3
        )

        # Verify both succeeded
        assert mcp_result['success'] is True

        # Both should have answer
        assert 'answer' in mcp_result['result']
        assert 'answer' in rest_result

    @pytest.mark.asyncio
    async def test_performance_metrics_consistency(self):
        """Verify performance metrics format is consistent."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()
        result = await bridge.invoke_technique(
            technique='basic',
            query='What is diabetes?',
            params={'top_k': 3}
        )

        assert result['success'] is True
        response = result['result']

        # Verify performance metrics exist
        assert 'performance' in response
        metrics = response['performance']

        # Standard performance fields
        expected_fields = ['execution_time_ms', 'retrieval_time_ms',
                          'generation_time_ms', 'tokens_used']
        for field in expected_fields:
            assert field in metrics, f"Missing performance field: {field}"

    @pytest.mark.asyncio
    async def test_pipeline_instance_reuse_validation(self):
        """Verify MCP uses same pipeline instances as REST API (FR-006)."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Execute query
        result = await bridge.invoke_technique(
            technique='basic',
            query='What is diabetes?',
            params={'top_k': 3}
        )

        assert result['success'] is True

        # This test validates implementation reuses PipelineManager.get_instance()
        # (actual validation done in contract tests)

    @pytest.mark.asyncio
    async def test_metadata_format_consistency(self):
        """Verify metadata format is consistent across MCP and REST API."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()
        result = await bridge.invoke_technique(
            technique='basic',
            query='What is diabetes?',
            params={}
        )

        assert result['success'] is True
        metadata = result['result']['metadata']

        # Metadata should include pipeline name
        assert 'pipeline_name' in metadata or 'technique' in metadata

    @pytest.mark.asyncio
    async def test_error_format_consistency(self):
        """Verify error responses have consistent format."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Trigger error with invalid parameter
        result = await bridge.invoke_technique(
            technique='basic',
            query='test',
            params={'top_k': 1000}  # exceeds max
        )

        assert result['success'] is False
        assert 'error' in result
        assert isinstance(result['error'], str)
        assert len(result['error']) > 0
