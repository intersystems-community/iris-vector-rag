"""
Comprehensive Integration Tests for RAG Templates Bridge Adapter

Tests the unified interface between rag-templates RAG ecosystem and kg-ticket-resolver,
validating performance SLOs, circuit breaker patterns, and error handling.
"""

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from adapters.rag_templates_bridge import (
    RAGTemplatesBridge, 
    RAGResponse, 
    RAGTechnique, 
    CircuitBreakerState,
    PerformanceMetrics
)
from iris_rag.core.exceptions import PipelineNotFoundError


class TestRAGTemplatesBridge:
    """Test suite for RAG Templates Bridge adapter."""

    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager with test settings."""
        config_manager = Mock()
        config_manager.get.return_value = {
            "default_technique": "basic",
            "fallback_technique": "basic",
            "query_timeout": 5,
            "circuit_breaker": {
                "failure_threshold": 3,
                "recovery_timeout": 10,
                "half_open_max_calls": 2
            },
            "performance": {
                "target_latency_p95_ms": 500,
                "target_memory_api_latency_ms": 200
            }
        }
        return config_manager

    @pytest.fixture
    def mock_connection_manager(self):
        """Mock connection manager."""
        conn_manager = Mock()
        conn_manager.get_connection.return_value = Mock()
        return conn_manager

    @pytest.fixture
    def mock_pipeline(self):
        """Mock RAG pipeline with configurable responses."""
        pipeline = Mock()
        pipeline.config_manager = Mock()
        
        # Default successful response
        pipeline.query.return_value = {
            "answer": "Test answer from pipeline",
            "sources": [{"id": "doc1", "content": "test content", "score": 0.85}],
            "confidence_score": 0.8,
            "metadata": {"tokens_used": 150}
        }
        
        pipeline.load_documents = Mock()
        return pipeline

    @pytest.fixture
    async def bridge(self, mock_config_manager, mock_connection_manager):
        """Create bridge instance with mocked dependencies."""
        with patch('adapters.rag_templates_bridge.ConfigurationManager', return_value=mock_config_manager), \
             patch('adapters.rag_templates_bridge.ConnectionManager', return_value=mock_connection_manager):
            
            bridge = RAGTemplatesBridge()
            
            # Mock all pipelines for testing
            mock_basic = Mock()
            mock_basic.config_manager = Mock()
            mock_basic.query.return_value = {"answer": "Basic response", "sources": [], "confidence_score": 0.7}
            
            mock_crag = Mock()
            mock_crag.config_manager = Mock()
            mock_crag.query.return_value = {"answer": "CRAG response", "sources": [], "confidence_score": 0.8}
            
            mock_graph = Mock()
            mock_graph.config_manager = Mock()
            mock_graph.query.return_value = {"answer": "Graph response", "sources": [], "confidence_score": 0.9}
            
            mock_rerank = Mock()
            mock_rerank.config_manager = Mock()
            mock_rerank.query.return_value = {"answer": "Reranked response", "sources": [], "confidence_score": 0.85}
            
            bridge._pipelines = {
                RAGTechnique.BASIC: mock_basic,
                RAGTechnique.CRAG: mock_crag,
                RAGTechnique.GRAPH: mock_graph,
                RAGTechnique.RERANKING: mock_rerank
            }
            
            return bridge

    async def test_all_rag_techniques_accessible(self, bridge):
        """Test that all RAG techniques are accessible through unified interface."""
        test_query = "What is machine learning?"
        
        # Test each technique
        techniques = [
            RAGTechnique.BASIC,
            RAGTechnique.CRAG, 
            RAGTechnique.GRAPH,
            RAGTechnique.RERANKING
        ]
        
        for technique in techniques:
            response = await bridge.query(test_query, technique=technique.value)
            
            assert isinstance(response, RAGResponse)
            assert response.technique_used == technique.value
            assert response.answer != ""
            assert response.confidence_score > 0
            assert response.processing_time_ms > 0
            assert response.error is None

    async def test_performance_slo_compliance(self, bridge):
        """Test that performance meets SLO requirements (<500ms p95)."""
        test_query = "Performance test query"
        response_times = []
        
        # Execute multiple queries to gather performance data
        for _ in range(20):
            start_time = time.time()
            response = await bridge.query(test_query)
            end_time = time.time()
            
            response_times.append((end_time - start_time) * 1000)
            assert response.processing_time_ms < 500  # Individual query SLO
        
        # Check p95 latency
        p95_latency = sorted(response_times)[int(len(response_times) * 0.95)]
        assert p95_latency < 500, f"P95 latency {p95_latency}ms exceeds 500ms SLO"
        
        # Validate metrics collection
        metrics = await bridge.get_performance_metrics()
        assert "slo_compliance" in metrics
        assert metrics["slo_compliance"]["p95_target_ms"] == 500

    async def test_circuit_breaker_functionality(self, bridge):
        """Test circuit breaker pattern and automatic fallback."""
        test_query = "Circuit breaker test"
        
        # Configure CRAG pipeline to fail consistently
        bridge._pipelines[RAGTechnique.CRAG].query.side_effect = Exception("Pipeline failure")
        
        # Execute queries to trigger circuit breaker
        for _ in range(4):  # threshold is 3, so 4th should trigger fallback
            response = await bridge.query(test_query, technique="crag")
            if response.error:
                continue  # Expected failures
        
        # Check circuit breaker state
        cb_state = bridge._circuit_breakers[RAGTechnique.CRAG]["state"]
        assert cb_state == CircuitBreakerState.OPEN
        
        # Next query should use fallback
        response = await bridge.query(test_query, technique="crag")
        assert response.technique_used == bridge.fallback_technique.value

    async def test_graceful_error_handling(self, bridge):
        """Test graceful degradation and error handling."""
        test_query = "Error handling test"
        
        # Simulate pipeline failure
        bridge._pipelines[RAGTechnique.BASIC].query.side_effect = Exception("Test error")
        
        response = await bridge.query(test_query, technique="basic")
        
        assert isinstance(response, RAGResponse)
        assert response.error is not None
        assert response.answer == ""
        assert response.confidence_score == 0.0
        assert "Test error" in response.error

    async def test_configuration_validation(self, bridge):
        """Test configuration loading and validation."""
        # Test available techniques
        techniques = await bridge.get_available_techniques()
        assert len(techniques) > 0
        assert "basic" in techniques
        
        # Test health check includes configuration
        health = await bridge.health_check()
        assert "components" in health
        assert "pipelines" in health["components"]
        
        # Validate performance targets are loaded
        metrics = await bridge.get_performance_metrics()
        assert "slo_compliance" in metrics
        assert "p95_target_ms" in metrics["slo_compliance"]

    async def test_kg_ticket_resolver_integration_patterns(self, bridge):
        """Test integration patterns with kg-ticket-resolver."""
        # Simulate kg-ticket-resolver query patterns
        test_contexts = [
            {
                "query": "How to resolve authentication issues?",
                "user_context": {"project_id": "proj_123", "user_role": "developer"},
                "query_id": "ticket_456"
            },
            {
                "query": "Database connection troubleshooting",
                "user_context": {"project_id": "proj_789", "user_role": "admin"},
                "query_id": "ticket_789"
            }
        ]
        
        for context in test_contexts:
            response = await bridge.query(
                query_text=context["query"],
                user_context=context["user_context"],
                query_id=context["query_id"]
            )
            
            # Validate kg-ticket-resolver expected response format
            assert isinstance(response, RAGResponse)
            assert response.metadata["query_id"] == context["query_id"]
            assert response.metadata["user_context"] == context["user_context"]
            assert "timestamp" in response.metadata

    async def test_incremental_indexing_support(self, bridge):
        """Test incremental indexing functionality."""
        test_documents = [
            {
                "content": "Test document 1 content",
                "metadata": {"doc_id": "doc1", "project": "test_project"}
            },
            {
                "content": "Test document 2 content", 
                "metadata": {"doc_id": "doc2", "project": "test_project"}
            }
        ]
        
        # Test incremental indexing
        result = await bridge.index_documents(
            documents=test_documents,
            technique="basic",
            incremental=True
        )
        
        assert result["status"] == "success"
        assert result["documents_indexed"] == 2
        assert result["technique_used"] == "basic"
        assert result["incremental"] is True
        assert result["processing_time_ms"] > 0

    async def test_health_check_comprehensive(self, bridge):
        """Test comprehensive health check functionality."""
        health = await bridge.health_check()
        
        # Validate health check structure
        required_keys = ["status", "timestamp", "components", "metrics"]
        for key in required_keys:
            assert key in health
        
        # Check component health
        assert "pipelines" in health["components"]
        assert "circuit_breakers" in health["components"]
        assert "dependencies" in health["components"]
        
        # Validate pipeline health status
        for technique in bridge._pipelines.keys():
            assert technique.value in health["components"]["pipelines"]
            pipeline_health = health["components"]["pipelines"][technique.value]
            assert "status" in pipeline_health

    async def test_concurrent_query_handling(self, bridge):
        """Test handling of concurrent queries."""
        test_query = "Concurrent test query"
        num_concurrent = 5
        
        # Execute concurrent queries
        tasks = [
            bridge.query(test_query, technique="basic")
            for _ in range(num_concurrent)
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # Validate all responses
        assert len(responses) == num_concurrent
        for response in responses:
            assert isinstance(response, RAGResponse)
            assert response.technique_used == "basic"
            assert response.error is None

    async def test_technique_auto_selection(self, bridge):
        """Test automatic technique selection when not specified."""
        test_query = "Auto selection test"
        
        response = await bridge.query(test_query)  # No technique specified
        
        assert response.technique_used == bridge.default_technique.value
        assert isinstance(response, RAGResponse)
        assert response.error is None

    async def test_metrics_collection_accuracy(self, bridge):
        """Test accuracy of performance metrics collection."""
        # Reset metrics
        bridge.metrics = PerformanceMetrics()
        
        # Execute known number of queries
        successful_queries = 3
        failed_queries = 2
        
        for i in range(successful_queries):
            await bridge.query(f"Success query {i}")
        
        # Simulate failures
        bridge._pipelines[RAGTechnique.BASIC].query.side_effect = Exception("Test failure")
        for i in range(failed_queries):
            await bridge.query(f"Failure query {i}")
        
        metrics = bridge.get_metrics()
        
        assert metrics["total_queries"] == successful_queries + failed_queries
        assert metrics["successful_queries"] == successful_queries
        assert metrics["failed_queries"] == failed_queries

    async def test_circuit_breaker_recovery(self, bridge):
        """Test circuit breaker recovery mechanism."""
        technique = RAGTechnique.BASIC
        
        # Force circuit breaker open
        bridge._circuit_breakers[technique]["state"] = CircuitBreakerState.OPEN
        bridge._circuit_breakers[technique]["last_failure_time"] = time.time() - 61  # Past recovery timeout
        
        # Should move to half-open on next check
        assert bridge._check_circuit_breaker(technique) is True
        assert bridge._circuit_breakers[technique]["state"] == CircuitBreakerState.HALF_OPEN
        
        # Successful calls should close the circuit
        for _ in range(bridge.cb_config.half_open_max_calls):
            bridge._record_success(technique)
        
        assert bridge._circuit_breakers[technique]["state"] == CircuitBreakerState.CLOSED


class TestPerformanceBenchmarks:
    """Performance-focused tests for SLO validation."""

    async def test_cold_start_performance(self):
        """Test bridge initialization performance (<50ms)."""
        with patch('adapters.rag_templates_bridge.ConfigurationManager'), \
             patch('adapters.rag_templates_bridge.ConnectionManager'):
            
            start_time = time.time()
            bridge = RAGTemplatesBridge()
            init_time = (time.time() - start_time) * 1000
            
            assert init_time < 50, f"Cold start took {init_time}ms, exceeds 50ms target"

    async def test_error_recovery_performance(self, bridge):
        """Test error recovery within 1s target."""
        # Simulate failure and measure recovery time
        start_time = time.time()
        
        # Force circuit breaker trigger
        bridge._record_failure(RAGTechnique.BASIC)
        bridge._record_failure(RAGTechnique.BASIC)
        bridge._record_failure(RAGTechnique.BASIC)
        bridge._record_failure(RAGTechnique.BASIC)  # Should trigger circuit breaker
        
        # Attempt query (should fallback)
        response = await bridge.query("Recovery test")
        
        recovery_time = (time.time() - start_time) * 1000
        
        assert recovery_time < 1000, f"Error recovery took {recovery_time}ms, exceeds 1s target"
        assert response.technique_used == bridge.fallback_technique.value


@pytest.mark.asyncio
class TestRAGBridgeIntegration:
    """End-to-end integration tests."""

    async def test_full_workflow_integration(self):
        """Test complete workflow from query to response."""
        # This would integrate with actual pipelines in a full test environment
        # For now, validates the interface contract
        
        with patch('adapters.rag_templates_bridge.ConfigurationManager'), \
             patch('adapters.rag_templates_bridge.ConnectionManager'):
            
            bridge = RAGTemplatesBridge()
            
            # Validate bridge provides required interface
            assert hasattr(bridge, 'query')
            assert hasattr(bridge, 'health_check')
            assert hasattr(bridge, 'get_available_techniques')
            assert hasattr(bridge, 'get_performance_metrics')
            assert hasattr(bridge, 'index_documents')
            
            # Validate response format compliance
            mock_pipeline = Mock()
            mock_pipeline.config_manager = Mock()
            mock_pipeline.query.return_value = {
                "answer": "Test",
                "sources": [],
                "confidence_score": 0.8
            }
            
            bridge._pipelines[RAGTechnique.BASIC] = mock_pipeline
            
            response = await bridge.query("Test query")
            
            # Validate RAGResponse format for kg-ticket-resolver
            required_fields = [
                'answer', 'sources', 'confidence_score', 
                'technique_used', 'processing_time_ms', 'metadata'
            ]
            
            for field in required_fields:
                assert hasattr(response, field), f"Missing required field: {field}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])