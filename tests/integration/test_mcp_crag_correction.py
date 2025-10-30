"""
Integration test: CRAG with corrective measures (Scenario 3 from quickstart.md).

This test verifies that the rag_crag tool applies corrective measures
(query rewriting, web search) when confidence is low.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

import pytest
from typing import Dict, Any


@pytest.mark.integration
@pytest.mark.mcp
class TestMCPCRAGCorrection:
    """Integration test for rag_crag tool with corrective measures."""

    @pytest.mark.asyncio
    async def test_crag_basic_query_execution(self):
        """Verify rag_crag executes basic query successfully."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='crag',
            query='What is diabetes?',
            params={
                'top_k': 5,
                'confidence_threshold': 0.8
            }
        )

        assert result['success'] is True
        response = result['result']

        # Verify standard fields
        assert 'answer' in response
        assert 'retrieved_documents' in response
        assert 'metadata' in response

    @pytest.mark.asyncio
    async def test_crag_includes_confidence_score(self):
        """Verify CRAG response includes retrieval status in metadata."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='crag',
            query='What is diabetes mellitus?',
            params={'confidence_threshold': 0.8}
        )

        assert result['success'] is True
        metadata = result['result']['metadata']

        # CRAG should include retrieval_status (confident, uncertain, incorrect)
        assert 'retrieval_status' in metadata, \
            "CRAG metadata missing retrieval_status"
        assert metadata['retrieval_status'] in ['confident', 'uncertain', 'incorrect']

    @pytest.mark.asyncio
    async def test_crag_correction_applied_metadata(self):
        """Verify CRAG metadata indicates when correction was applied."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Query with low confidence threshold to potentially trigger correction
        result = await bridge.invoke_technique(
            technique='crag',
            query='What is the pathophysiology of type 2 diabetes?',
            params={
                'confidence_threshold': 0.9,  # High threshold may trigger correction
                'correction_strategy': 'rewrite'
            }
        )

        assert result['success'] is True
        metadata = result['result']['metadata']

        # If correction was applied, metadata should indicate it
        # (may or may not be applied depending on actual confidence)
        if 'correction_applied' in metadata:
            assert isinstance(metadata['correction_applied'], bool)
            if metadata['correction_applied']:
                # Should have additional correction metadata
                assert 'rewritten_query' in metadata or \
                       'correction_strategy' in metadata

    @pytest.mark.asyncio
    async def test_crag_rewrite_correction_strategy(self):
        """Verify CRAG with rewrite correction strategy."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='crag',
            query='diabetes complications',
            params={
                'confidence_threshold': 0.8,
                'correction_strategy': 'rewrite'
            }
        )

        assert result['success'] is True
        response = result['result']

        # Verify response has answer
        assert 'answer' in response
        assert len(response['answer']) > 0

        # If rewriting was triggered, metadata should show it
        if 'rewritten_query' in response['metadata']:
            assert isinstance(response['metadata']['rewritten_query'], str)
            assert len(response['metadata']['rewritten_query']) > 0

    @pytest.mark.asyncio
    async def test_crag_web_search_correction_strategy(self):
        """Verify CRAG with web_search correction strategy."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='crag',
            query='What are the latest diabetes treatments in 2025?',
            params={
                'confidence_threshold': 0.9,  # May trigger web search
                'correction_strategy': 'web_search',
                'enable_web_search': True
            }
        )

        # May succeed or fail depending on web search availability
        assert isinstance(result, dict)
        assert 'success' in result

        if result['success']:
            metadata = result['result']['metadata']
            # If web search was used, should be indicated
            if 'web_search_used' in metadata:
                assert isinstance(metadata['web_search_used'], bool)

    @pytest.mark.asyncio
    async def test_crag_both_correction_strategy(self):
        """Verify CRAG with 'both' correction strategy."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='crag',
            query='diabetes treatment',
            params={
                'confidence_threshold': 0.8,
                'correction_strategy': 'both'
            }
        )

        assert result['success'] is True
        response = result['result']
        assert 'answer' in response

    @pytest.mark.asyncio
    async def test_crag_confidence_threshold_validation(self):
        """Verify CRAG validates confidence_threshold parameter."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Valid confidence threshold
        result = await bridge.invoke_technique(
            technique='crag',
            query='What is diabetes?',
            params={'confidence_threshold': 0.75}
        )
        assert result['success'] is True

        # Invalid confidence threshold (> 1.0)
        result = await bridge.invoke_technique(
            technique='crag',
            query='What is diabetes?',
            params={'confidence_threshold': 1.5}
        )
        assert result['success'] is False
        assert 'error' in result

        # Invalid confidence threshold (< 0.0)
        result = await bridge.invoke_technique(
            technique='crag',
            query='What is diabetes?',
            params={'confidence_threshold': -0.1}
        )
        assert result['success'] is False

    @pytest.mark.asyncio
    async def test_crag_correction_strategy_validation(self):
        """Verify CRAG validates correction_strategy parameter."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Valid strategies
        valid_strategies = ['rewrite', 'web_search', 'both']
        for strategy in valid_strategies:
            result = await bridge.invoke_technique(
                technique='crag',
                query='What is diabetes?',
                params={'correction_strategy': strategy}
            )
            # Should succeed (though web_search may fail if disabled)
            assert isinstance(result, dict)

        # Invalid strategy
        result = await bridge.invoke_technique(
            technique='crag',
            query='What is diabetes?',
            params={'correction_strategy': 'invalid_strategy'}
        )
        assert result['success'] is False
        assert 'error' in result

    @pytest.mark.asyncio
    async def test_crag_default_parameters(self):
        """Verify CRAG uses default parameters when not specified."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Only provide query
        result = await bridge.invoke_technique(
            technique='crag',
            query='What is diabetes?',
            params={}
        )

        assert result['success'] is True
        response = result['result']

        # Should use defaults: confidence_threshold=0.8, top_k=5, correction_strategy='rewrite'
        assert 'answer' in response
        assert len(response['retrieved_documents']) == 5  # default top_k

    @pytest.mark.asyncio
    async def test_crag_with_low_confidence_threshold(self):
        """Verify CRAG with very low confidence threshold (likely no correction)."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='crag',
            query='What is diabetes?',
            params={
                'confidence_threshold': 0.1,  # Very low threshold
                'correction_strategy': 'rewrite'
            }
        )

        assert result['success'] is True
        metadata = result['result']['metadata']

        # With very low threshold, correction likely not applied
        if 'correction_applied' in metadata:
            # Could be True or False, just verify it's boolean
            assert isinstance(metadata['correction_applied'], bool)

    @pytest.mark.asyncio
    async def test_crag_with_high_confidence_threshold(self):
        """Verify CRAG with very high confidence threshold (likely correction)."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='crag',
            query='What is diabetes?',
            params={
                'confidence_threshold': 0.99,  # Very high threshold
                'correction_strategy': 'rewrite'
            }
        )

        assert result['success'] is True
        metadata = result['result']['metadata']

        # With very high threshold, correction more likely applied
        # (but depends on actual retrieval quality)
        if 'correction_applied' in metadata:
            assert isinstance(metadata['correction_applied'], bool)

    @pytest.mark.asyncio
    async def test_crag_performance_metrics(self):
        """Verify CRAG includes performance metrics."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='crag',
            query='What is diabetes?',
            params={'confidence_threshold': 0.8}
        )

        assert result['success'] is True
        response = result['result']

        # Verify performance metrics
        assert 'performance' in response
        metrics = response['performance']
        assert 'execution_time_ms' in metrics
        assert 'retrieval_time_ms' in metrics
        assert metrics['execution_time_ms'] > 0

    @pytest.mark.asyncio
    async def test_crag_response_format_consistency(self):
        """Verify CRAG response format matches standard RAG format."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='crag',
            query='What is diabetes?',
            params={}
        )

        assert result['success'] is True
        response = result['result']

        # Standard fields
        assert 'answer' in response
        assert 'retrieved_documents' in response
        assert 'sources' in response
        assert 'metadata' in response
        assert 'performance' in response

        # CRAG-specific metadata
        # (may include confidence_score, correction_applied, rewritten_query)
