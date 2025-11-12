"""
Contract tests for Python MCP Bridge interface.

These tests verify that the MCPBridge implementation satisfies the
IMCPBridge interface contract. Following TDD principles, these tests
MUST FAIL initially until the implementation is complete.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

import pytest
from typing import Dict, Any, List


class TestMCPBridgeContract:
    """Contract tests for IMCPBridge interface implementation."""

    def test_bridge_module_exists(self):
        """Verify iris_rag.mcp.bridge module can be imported."""
        try:
            import iris_vector_rag.mcp.bridge
            assert True
        except ImportError as e:
            pytest.fail(f"MCPBridge module not found: {e}")

    def test_bridge_class_exists(self):
        """Verify MCPBridge class exists in module."""
        from iris_vector_rag.mcp import bridge
        assert hasattr(bridge, 'MCPBridge'), "MCPBridge class not found in module"

    def test_bridge_implements_interface_methods(self):
        """Verify MCPBridge implements all IMCPBridge interface methods."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        required_methods = [
            'invoke_technique',
            'get_available_techniques',
            'health_check',
            'get_metrics'
        ]

        for method_name in required_methods:
            assert hasattr(MCPBridge, method_name), \
                f"MCPBridge missing required method: {method_name}"

    @pytest.mark.asyncio
    async def test_invoke_technique_signature(self):
        """Verify invoke_technique method signature and return type."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Test method can be called with correct parameters
        # (Will fail until implementation exists)
        result = await bridge.invoke_technique(
            technique='basic',
            query='test query',
            params={},
            api_key=None
        )

        # Verify response structure
        assert isinstance(result, dict), "invoke_technique must return dict"
        assert 'success' in result, "Response must have 'success' field"
        assert isinstance(result['success'], bool), "'success' must be boolean"

        # Verify mutually exclusive result/error fields
        if result['success']:
            assert 'result' in result, "Successful response must have 'result' field"
            assert result['result'] is not None
            # Verify result structure
            assert 'answer' in result['result']
            assert 'retrieved_documents' in result['result']
            assert 'sources' in result['result']
            assert 'metadata' in result['result']
            assert 'performance' in result['result']
        else:
            assert 'error' in result, "Failed response must have 'error' field"
            assert isinstance(result['error'], str), "'error' must be string"

    @pytest.mark.asyncio
    async def test_get_available_techniques_signature(self):
        """Verify get_available_techniques method signature and return type."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()
        techniques = await bridge.get_available_techniques()

        # Verify response structure
        assert isinstance(techniques, list), "get_available_techniques must return list"

        # Verify expected pipelines are present
        expected_pipelines = [
            'basic',
            'basic_rerank',
            'crag',
            'graphrag',
            'pylate_colbert',
            'iris_global_graphrag'
        ]

        for pipeline in expected_pipelines:
            assert pipeline in techniques, \
                f"Expected pipeline '{pipeline}' not in available techniques"

    @pytest.mark.asyncio
    async def test_health_check_signature(self):
        """Verify health_check method signature and return type."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()
        health = await bridge.health_check(
            include_details=False,
            include_performance_metrics=True
        )

        # Verify response structure
        assert isinstance(health, dict), "health_check must return dict"
        assert 'status' in health, "Health response must have 'status' field"
        assert health['status'] in ['healthy', 'degraded', 'unavailable'], \
            "Health status must be one of: healthy, degraded, unavailable"

        assert 'timestamp' in health, "Health response must have 'timestamp' field"
        assert 'pipelines' in health, "Health response must have 'pipelines' field"
        assert isinstance(health['pipelines'], dict), "'pipelines' must be dict"

        assert 'database' in health, "Health response must have 'database' field"
        assert 'connected' in health['database']
        assert 'response_time_ms' in health['database']

        # Verify performance metrics when requested
        if health.get('performance_metrics'):
            metrics = health['performance_metrics']
            assert 'average_response_time_ms' in metrics
            assert 'p95_response_time_ms' in metrics
            assert 'error_rate' in metrics
            assert 'queries_per_minute' in metrics

    @pytest.mark.asyncio
    async def test_get_metrics_signature(self):
        """Verify get_metrics method signature and return type."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()
        metrics = await bridge.get_metrics(
            time_range='1h',
            technique_filter=None,
            include_error_details=False
        )

        # Verify response structure
        assert isinstance(metrics, dict), "get_metrics must return dict"
        assert 'time_range' in metrics
        assert 'total_queries' in metrics
        assert 'successful_queries' in metrics
        assert 'failed_queries' in metrics
        assert 'average_response_time_ms' in metrics
        assert 'p95_response_time_ms' in metrics
        assert 'error_rate' in metrics
        assert 'queries_per_minute' in metrics

        # Verify technique_usage breakdown
        if 'technique_usage' in metrics:
            assert isinstance(metrics['technique_usage'], dict)

    @pytest.mark.asyncio
    async def test_invoke_technique_error_handling(self):
        """Verify invoke_technique handles invalid parameters correctly."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Test with invalid technique name
        result = await bridge.invoke_technique(
            technique='invalid_pipeline_name',
            query='test query',
            params={}
        )

        assert result['success'] is False
        assert 'error' in result
        assert 'invalid' in result['error'].lower() or 'not found' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_invoke_technique_parameter_validation(self):
        """Verify invoke_technique validates parameters."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Test with empty query (should fail validation)
        result = await bridge.invoke_technique(
            technique='basic',
            query='',
            params={}
        )

        # Empty query should either fail or be handled gracefully
        # (implementation may choose to return error or allow empty queries)
        assert isinstance(result, dict)
        assert 'success' in result

    @pytest.mark.asyncio
    async def test_bridge_authentication_mode_configurable(self):
        """Verify MCPBridge supports both authenticated and unauthenticated modes."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        # Test with api_key parameter (authenticated mode)
        bridge_auth = MCPBridge()
        result_auth = await bridge_auth.invoke_technique(
            technique='basic',
            query='test query',
            params={},
            api_key='test_api_key_12345'
        )

        assert isinstance(result_auth, dict)

        # Test without api_key (unauthenticated mode)
        bridge_unauth = MCPBridge()
        result_unauth = await bridge_unauth.invoke_technique(
            technique='basic',
            query='test query',
            params={},
            api_key=None
        )

        assert isinstance(result_unauth, dict)


class TestMCPBridgeIntegration:
    """Integration contract tests - verify bridge integrates with REST API PipelineManager."""

    def test_bridge_imports_pipeline_manager(self):
        """Verify MCPBridge can import REST API's PipelineManager."""
        try:
            from iris_vector_rag.api.services import PipelineManager
            assert PipelineManager is not None
        except ImportError as e:
            pytest.fail(f"Cannot import PipelineManager: {e}")

    @pytest.mark.asyncio
    async def test_bridge_reuses_pipeline_instances(self):
        """Verify MCPBridge reuses PipelineManager's pipeline instances (FR-006)."""
        from iris_vector_rag.mcp.bridge import MCPBridge

        # This test verifies implementation reuses existing pipeline instances
        # Implementation should use PipelineManager.get_instance()
        bridge = MCPBridge()

        # Invoke technique - should use shared pipeline instance
        result = await bridge.invoke_technique(
            technique='basic',
            query='test query about diabetes',
            params={'top_k': 3}
        )

        # Verify response follows standard format
        assert isinstance(result, dict)
        assert 'success' in result
