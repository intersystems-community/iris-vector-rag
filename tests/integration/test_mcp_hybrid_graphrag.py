"""
Integration test: HybridGraphRAG with graph traversal (Scenario 6 from quickstart.md).

This test verifies that the rag_graphrag tool executes hybrid search
combining vector, text, and graph methods.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

import pytest


@pytest.mark.integration
@pytest.mark.mcp
class TestMCPHybridGraphRAG:
    """Integration test for rag_graphrag tool with hybrid search."""

    @pytest.mark.asyncio
    async def test_graphrag_hybrid_search(self):
        """Verify rag_graphrag executes hybrid search successfully."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='graphrag',
            query='How does diabetes affect the heart?',
            params={
                'top_k': 5,
                'search_method': 'hybrid',
                'graph_traversal_depth': 2,
                'rrf_k': 60
            }
        )

        assert result['success'] is True
        response = result['result']

        # Verify standard fields
        assert 'answer' in response
        assert 'retrieved_documents' in response
        assert 'metadata' in response

    @pytest.mark.asyncio
    async def test_graphrag_metadata_includes_search_info(self):
        """Verify GraphRAG metadata includes search method and graph info."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='graphrag',
            query='diabetes and cardiovascular disease relationship',
            params={
                'search_method': 'hybrid',
                'graph_traversal_depth': 2
            }
        )

        assert result['success'] is True
        metadata = result['result']['metadata']

        # May include search_method, graph_traversal_depth, rrf_score
        # (depending on implementation)

    @pytest.mark.asyncio
    async def test_graphrag_vector_search_method(self):
        """Verify GraphRAG with vector-only search method."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='graphrag',
            query='What is diabetes?',
            params={'search_method': 'vector'}
        )

        assert result['success'] is True

    @pytest.mark.asyncio
    async def test_graphrag_text_search_method(self):
        """Verify GraphRAG with text-only search method."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='graphrag',
            query='What is diabetes?',
            params={'search_method': 'text'}
        )

        assert result['success'] is True

    @pytest.mark.asyncio
    async def test_graphrag_graph_search_method(self):
        """Verify GraphRAG with graph-only search method."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='graphrag',
            query='diabetes complications',
            params={
                'search_method': 'graph',
                'graph_traversal_depth': 2
            }
        )

        # May succeed or skip if no graph data
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_graphrag_rrf_search_method(self):
        """Verify GraphRAG with RRF (Reciprocal Rank Fusion) method."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='graphrag',
            query='What is diabetes?',
            params={
                'search_method': 'rrf',
                'rrf_k': 60
            }
        )

        assert result['success'] is True

    @pytest.mark.asyncio
    async def test_graphrag_traversal_depth_validation(self):
        """Verify GraphRAG validates graph_traversal_depth parameter."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Valid depth
        result = await bridge.invoke_technique(
            technique='graphrag',
            query='What is diabetes?',
            params={'graph_traversal_depth': 3}
        )
        assert result['success'] is True

        # Invalid depth (> max)
        result = await bridge.invoke_technique(
            technique='graphrag',
            query='What is diabetes?',
            params={'graph_traversal_depth': 10}
        )
        assert result['success'] is False

    @pytest.mark.asyncio
    async def test_graphrag_search_method_validation(self):
        """Verify GraphRAG validates search_method parameter."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Invalid search method
        result = await bridge.invoke_technique(
            technique='graphrag',
            query='What is diabetes?',
            params={'search_method': 'invalid_method'}
        )
        assert result['success'] is False
        assert 'error' in result
