"""
Integration test: List available MCP tools (Scenario 1 from quickstart.md).

This test verifies that the MCP server can list all 8 available tools
(6 RAG pipelines + 2 utility tools) with complete schemas.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

import pytest
from typing import List, Dict, Any


@pytest.mark.integration
@pytest.mark.mcp
class TestMCPListTools:
    """Integration test for MCP tools/list endpoint."""

    @pytest.mark.asyncio
    async def test_list_tools_returns_8_tools(self):
        """Verify MCP server returns all 8 tools."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()
        tools = await bridge.list_tools()

        # Verify count
        assert isinstance(tools, list), "list_tools must return list"
        assert len(tools) == 8, f"Expected 8 tools, got {len(tools)}"

        # Verify all expected tools present
        expected_tools = [
            'rag_basic',
            'rag_basic_rerank',
            'rag_crag',
            'rag_graphrag',
            'rag_pylate_colbert',
            'rag_iris_global_graphrag',
            'rag_health_check',
            'rag_metrics'
        ]

        tool_names = [tool['name'] for tool in tools]
        for expected_name in expected_tools:
            assert expected_name in tool_names, \
                f"Expected tool '{expected_name}' not in list"

    @pytest.mark.asyncio
    async def test_list_tools_includes_schemas(self):
        """Verify each tool includes complete schema with parameter descriptions."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()
        tools = await bridge.list_tools()

        for tool in tools:
            # Verify tool structure
            assert 'name' in tool, f"Tool missing 'name' field: {tool}"
            assert 'description' in tool, f"Tool {tool.get('name')} missing 'description'"
            assert 'inputSchema' in tool, f"Tool {tool.get('name')} missing 'inputSchema'"

            # Verify description is non-empty
            assert len(tool['description']) > 0, \
                f"Tool {tool['name']} has empty description"

            # Verify inputSchema structure
            schema = tool['inputSchema']
            assert 'type' in schema, f"Tool {tool['name']} schema missing 'type'"
            assert schema['type'] == 'object', \
                f"Tool {tool['name']} schema type must be 'object'"

            # Verify properties exist
            assert 'properties' in schema, \
                f"Tool {tool['name']} schema missing 'properties'"

    @pytest.mark.asyncio
    async def test_list_tools_parameter_descriptions(self):
        """Verify all parameters include descriptions."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()
        tools = await bridge.list_tools()

        for tool in tools:
            schema = tool['inputSchema']
            properties = schema.get('properties', {})

            for param_name, param_schema in properties.items():
                assert 'description' in param_schema, \
                    f"Tool {tool['name']} parameter '{param_name}' missing description"
                assert len(param_schema['description']) > 0, \
                    f"Tool {tool['name']} parameter '{param_name}' has empty description"

    @pytest.mark.asyncio
    async def test_rag_tools_require_query_parameter(self):
        """Verify all RAG tools (not utility tools) require 'query' parameter."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()
        tools = await bridge.list_tools()

        rag_tools = [t for t in tools if t['name'].startswith('rag_')
                     and t['name'] not in ['rag_health_check', 'rag_metrics']]

        for tool in rag_tools:
            schema = tool['inputSchema']

            # Verify 'query' parameter exists
            assert 'query' in schema['properties'], \
                f"RAG tool {tool['name']} missing 'query' parameter"

            # Verify 'query' is required
            assert 'required' in schema, \
                f"RAG tool {tool['name']} schema missing 'required' field"
            assert 'query' in schema['required'], \
                f"RAG tool {tool['name']} does not require 'query' parameter"

    @pytest.mark.asyncio
    async def test_utility_tools_no_required_parameters(self):
        """Verify utility tools (health_check, metrics) have no required parameters."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()
        tools = await bridge.list_tools()

        utility_tools = [t for t in tools if t['name'] in ['rag_health_check', 'rag_metrics']]

        for tool in utility_tools:
            schema = tool['inputSchema']

            # Health check and metrics should have optional parameters only
            # (or required list should be empty)
            required_params = schema.get('required', [])
            # Metrics doesn't require any params, health_check doesn't either
            assert len(required_params) == 0, \
                f"Utility tool {tool['name']} should not have required parameters"

    @pytest.mark.asyncio
    async def test_tool_schemas_include_defaults(self):
        """Verify tool parameters include default values where applicable."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()
        tools = await bridge.list_tools()

        # Check basic RAG tool has defaults
        basic_tool = next(t for t in tools if t['name'] == 'rag_basic')
        basic_props = basic_tool['inputSchema']['properties']

        # top_k should have default
        assert 'default' in basic_props['top_k'], \
            "top_k parameter missing default value"
        assert basic_props['top_k']['default'] == 5

        # include_sources should have default
        assert 'default' in basic_props['include_sources']
        assert basic_props['include_sources']['default'] is True

    @pytest.mark.asyncio
    async def test_tool_schemas_include_validation_constraints(self):
        """Verify tool parameters include validation constraints (min, max, enum)."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()
        tools = await bridge.list_tools()

        # Check basic RAG tool has constraints
        basic_tool = next(t for t in tools if t['name'] == 'rag_basic')
        top_k_schema = basic_tool['inputSchema']['properties']['top_k']

        assert 'minimum' in top_k_schema, "top_k missing minimum constraint"
        assert 'maximum' in top_k_schema, "top_k missing maximum constraint"
        assert top_k_schema['minimum'] == 1
        assert top_k_schema['maximum'] == 50

        # Check CRAG tool has enum constraint
        crag_tool = next(t for t in tools if t['name'] == 'rag_crag')
        correction_strategy = crag_tool['inputSchema']['properties']['correction_strategy']

        assert 'enum' in correction_strategy, "correction_strategy missing enum constraint"
        assert 'rewrite' in correction_strategy['enum']
        assert 'web_search' in correction_strategy['enum']

    @pytest.mark.asyncio
    async def test_list_tools_performance(self):
        """Verify list_tools completes within reasonable time."""
        import time
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        start_time = time.time()
        tools = await bridge.list_tools()
        elapsed_ms = (time.time() - start_time) * 1000

        # List tools should be very fast (< 100ms)
        assert elapsed_ms < 100, \
            f"list_tools took {elapsed_ms:.1f}ms (expected < 100ms)"

    @pytest.mark.asyncio
    async def test_list_tools_idempotent(self):
        """Verify list_tools returns consistent results on multiple calls."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Call list_tools three times
        tools1 = await bridge.list_tools()
        tools2 = await bridge.list_tools()
        tools3 = await bridge.list_tools()

        # Verify all return same tool count
        assert len(tools1) == len(tools2) == len(tools3)

        # Verify same tool names
        names1 = sorted([t['name'] for t in tools1])
        names2 = sorted([t['name'] for t in tools2])
        names3 = sorted([t['name'] for t in tools3])

        assert names1 == names2 == names3
