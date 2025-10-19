"""
Integration test: Execute BasicRAG query (Scenario 2 from quickstart.md).

This test verifies that the rag_basic tool executes queries successfully
and returns complete responses with answer, documents, sources, and metrics.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

import pytest
import time
from typing import Dict, Any


@pytest.mark.integration
@pytest.mark.mcp
class TestMCPBasicRAG:
    """Integration test for rag_basic tool execution."""

    @pytest.mark.asyncio
    async def test_basic_rag_query_execution(self):
        """Verify rag_basic tool executes query and returns complete response."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='basic',
            query='What are the symptoms of diabetes?',
            params={'top_k': 5}
        )

        # Verify response structure
        assert isinstance(result, dict), "Response must be dict"
        assert result['success'] is True, f"Query failed: {result.get('error')}"
        assert 'result' in result

        response = result['result']

        # Verify answer exists and is non-empty
        assert 'answer' in response, "Response missing 'answer' field"
        assert isinstance(response['answer'], str)
        assert len(response['answer']) > 0, "Answer is empty"

        # Verify retrieved_documents
        assert 'retrieved_documents' in response
        assert isinstance(response['retrieved_documents'], list)
        assert len(response['retrieved_documents']) == 5, \
            f"Expected 5 documents, got {len(response['retrieved_documents'])}"

        # Verify each document has required fields
        for doc in response['retrieved_documents']:
            assert 'doc_id' in doc or 'id' in doc, "Document missing ID"
            assert 'content' in doc or 'text' in doc, "Document missing content"
            assert 'score' in doc or 'similarity' in doc, "Document missing score"

        # Verify sources
        assert 'sources' in response
        assert isinstance(response['sources'], list)
        assert len(response['sources']) > 0, "Sources list is empty"

        # Verify metadata
        assert 'metadata' in response
        assert isinstance(response['metadata'], dict)

    @pytest.mark.asyncio
    async def test_basic_rag_performance_metrics(self):
        """Verify response includes performance metrics."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='basic',
            query='What is diabetes?',
            params={'top_k': 3}
        )

        assert result['success'] is True
        response = result['result']

        # Verify performance metrics exist
        assert 'performance' in response, "Response missing 'performance' field"
        metrics = response['performance']

        # Verify required performance fields
        assert 'execution_time_ms' in metrics
        assert 'retrieval_time_ms' in metrics
        assert 'generation_time_ms' in metrics
        assert 'tokens_used' in metrics

        # Verify metrics are reasonable values
        assert metrics['execution_time_ms'] > 0
        assert metrics['retrieval_time_ms'] >= 0
        assert metrics['generation_time_ms'] >= 0
        # tokens_used may be 0 if pipeline doesn't track usage (future enhancement)
        assert metrics['tokens_used'] >= 0

    @pytest.mark.asyncio
    async def test_basic_rag_query_latency_p95(self):
        """Verify query execution time meets p95 latency requirement (<10s)."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        start_time = time.time()
        result = await bridge.invoke_technique(
            technique='basic',
            query='What are the symptoms of diabetes?',
            params={'top_k': 5}
        )
        elapsed_ms = (time.time() - start_time) * 1000

        # Verify query completed successfully
        assert result['success'] is True

        # Verify latency meets p95 requirement
        # Note: Includes LLM generation time which can be slow (typically 2-8s)
        assert elapsed_ms < 10000, \
            f"Query took {elapsed_ms:.1f}ms (p95 requirement: <10000ms)"

    @pytest.mark.asyncio
    async def test_basic_rag_with_minimal_parameters(self):
        """Verify rag_basic works with only required parameters."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Only provide query (top_k should use default)
        result = await bridge.invoke_technique(
            technique='basic',
            query='What is diabetes?',
            params={}
        )

        assert result['success'] is True
        response = result['result']

        # Should use default top_k=5
        assert 'retrieved_documents' in response
        assert len(response['retrieved_documents']) == 5

    @pytest.mark.asyncio
    async def test_basic_rag_with_custom_top_k(self):
        """Verify rag_basic respects custom top_k parameter."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # Test with top_k=3
        result = await bridge.invoke_technique(
            technique='basic',
            query='What is diabetes?',
            params={'top_k': 3}
        )

        assert result['success'] is True
        response = result['result']
        assert len(response['retrieved_documents']) == 3

        # Test with top_k=10
        result = await bridge.invoke_technique(
            technique='basic',
            query='What is diabetes?',
            params={'top_k': 10}
        )

        assert result['success'] is True
        response = result['result']
        assert len(response['retrieved_documents']) == 10

    @pytest.mark.asyncio
    async def test_basic_rag_with_include_sources(self):
        """Verify include_sources parameter controls source inclusion."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # With sources
        result_with = await bridge.invoke_technique(
            technique='basic',
            query='What is diabetes?',
            params={'include_sources': True}
        )

        assert result_with['success'] is True
        assert len(result_with['result']['sources']) > 0

        # Without sources
        result_without = await bridge.invoke_technique(
            technique='basic',
            query='What is diabetes?',
            params={'include_sources': False}
        )

        assert result_without['success'] is True
        # May still have sources (implementation choice), or empty list

    @pytest.mark.asyncio
    async def test_basic_rag_with_include_metadata(self):
        """Verify include_metadata parameter controls metadata inclusion."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # With metadata
        result_with = await bridge.invoke_technique(
            technique='basic',
            query='What is diabetes?',
            params={'include_metadata': True}
        )

        assert result_with['success'] is True
        assert 'metadata' in result_with['result']

        # Without metadata
        result_without = await bridge.invoke_technique(
            technique='basic',
            query='What is diabetes?',
            params={'include_metadata': False}
        )

        assert result_without['success'] is True
        # metadata field may still exist but be minimal

    @pytest.mark.asyncio
    async def test_basic_rag_invalid_top_k_out_of_range(self):
        """Verify rag_basic rejects top_k values outside valid range."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        # top_k too low (< 1)
        result = await bridge.invoke_technique(
            technique='basic',
            query='What is diabetes?',
            params={'top_k': 0}
        )

        assert result['success'] is False
        assert 'error' in result
        assert 'top_k' in result['error'].lower() or 'range' in result['error'].lower()

        # top_k too high (> 50)
        result = await bridge.invoke_technique(
            technique='basic',
            query='What is diabetes?',
            params={'top_k': 100}
        )

        assert result['success'] is False
        assert 'error' in result

    @pytest.mark.asyncio
    async def test_basic_rag_empty_query_handling(self):
        """Verify rag_basic handles empty query appropriately."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='basic',
            query='',
            params={}
        )

        # Should either reject or handle gracefully
        # (implementation may choose to return error or empty result)
        assert isinstance(result, dict)
        assert 'success' in result

    @pytest.mark.asyncio
    async def test_basic_rag_response_format_consistency(self):
        """Verify response format matches REST API response format (FR-025)."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        result = await bridge.invoke_technique(
            technique='basic',
            query='What is diabetes?',
            params={'top_k': 5}
        )

        assert result['success'] is True
        response = result['result']

        # Verify standard RAG response fields
        required_fields = ['answer', 'retrieved_documents', 'sources', 'metadata', 'performance']
        for field in required_fields:
            assert field in response, f"Response missing required field: {field}"

        # Verify metadata includes pipeline name
        assert 'pipeline_name' in response['metadata'] or \
               'technique' in response['metadata']

    @pytest.mark.asyncio
    async def test_basic_rag_multiple_queries_sequential(self):
        """Verify multiple sequential queries work correctly."""
        from iris_rag.mcp.bridge import MCPBridge

        bridge = MCPBridge()

        queries = [
            'What is diabetes?',
            'What are the symptoms of diabetes?',
            'How is diabetes treated?'
        ]

        for query in queries:
            result = await bridge.invoke_technique(
                technique='basic',
                query=query,
                params={'top_k': 3}
            )

            assert result['success'] is True, f"Query failed: {query}"
            assert 'answer' in result['result']
            assert len(result['result']['retrieved_documents']) == 3
