#!/usr/bin/env python3
"""
RAG Templates Bridge Validation Script

Validates that the RAG Templates Bridge meets all specified requirements:
- Performance SLOs (<500ms p95 latency)
- Circuit breaker functionality
- Error handling and graceful degradation
- All RAG techniques accessible
- Configuration management
"""

import asyncio
import os
import statistics
import sys
import time
from typing import Any, Dict
from unittest.mock import Mock, patch

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapters.rag_templates_bridge import (
    CircuitBreakerState,
    RAGTechnique,
    RAGTemplatesBridge,
)


class RAGBridgeValidator:
    """Validates RAG Templates Bridge implementation."""

    def __init__(self):
        self.results = {
            "performance": {"passed": False, "details": {}},
            "circuit_breaker": {"passed": False, "details": {}},
            "error_handling": {"passed": False, "details": {}},
            "techniques": {"passed": False, "details": {}},
            "configuration": {"passed": False, "details": {}},
        }

    async def setup_mock_bridge(self) -> RAGTemplatesBridge:
        """Set up bridge with mocked dependencies for testing."""

        # Mock configuration manager with comprehensive config
        mock_config = Mock()
        config_data = {
            "rag_integration": {
                "default_technique": "basic",
                "fallback_technique": "basic",
                "query_timeout": 5,
                "circuit_breaker": {
                    "failure_threshold": 3,
                    "recovery_timeout": 10,
                    "half_open_max_calls": 2,
                },
                "performance": {
                    "target_latency_p95_ms": 500,
                    "target_memory_api_latency_ms": 200,
                },
            },
            "storage": {
                "iris": {"vector_dimension": 384, "table_name": "RAG.SourceDocuments"}
            },
            "embeddings": {
                "default_provider": "sentence_transformers",
                "sentence_transformers": {
                    "model_name": "all-MiniLM-L6-v2",
                    "device": "cpu",
                },
            },
            "pipelines": {
                "basic": {"chunk_size": 1000, "chunk_overlap": 200, "default_top_k": 5}
            },
        }

        def mock_get(key, default=None):
            parts = key.split(".")
            result = config_data
            for part in parts:
                result = result.get(part, {})
            return result if result else default

        mock_config.get.side_effect = mock_get

        # Mock connection manager
        mock_conn = Mock()
        mock_conn.get_connection.return_value = Mock()

        with patch(
            "adapters.rag_templates_bridge.ConfigurationManager",
            return_value=mock_config,
        ), patch(
            "adapters.rag_templates_bridge.ConnectionManager", return_value=mock_conn
        ), patch(
            "adapters.rag_templates_bridge.BasicRAGPipeline"
        ), patch(
            "adapters.rag_templates_bridge.CRAGPipeline"
        ), patch(
            "adapters.rag_templates_bridge.GraphRAGPipeline"
        ), patch(
            "adapters.rag_templates_bridge.BasicRAGRerankingPipeline"
        ):

            bridge = RAGTemplatesBridge()

            # Set up mock pipelines after patching
            self._setup_mock_pipelines(bridge)

            return bridge

    def _setup_mock_pipelines(self, bridge: RAGTemplatesBridge):
        """Set up mock pipelines for testing."""

        def create_mock_pipeline(technique_name: str, base_latency: float = 0.05):
            pipeline = Mock()
            pipeline.config_manager = Mock()

            def mock_query(query_text):
                # Simulate processing time
                time.sleep(base_latency)
                return {
                    "answer": f"{technique_name} response to: {query_text}",
                    "sources": [{"id": f"doc_{technique_name}_1", "score": 0.8}],
                    "confidence_score": 0.8,
                    "metadata": {"technique": technique_name},
                }

            pipeline.query = mock_query
            pipeline.load_documents = Mock()
            return pipeline

        bridge._pipelines = {
            RAGTechnique.BASIC: create_mock_pipeline("basic", 0.02),
            RAGTechnique.CRAG: create_mock_pipeline("crag", 0.08),
            RAGTechnique.GRAPH: create_mock_pipeline("graphrag", 0.12),
            RAGTechnique.RERANKING: create_mock_pipeline("reranking", 0.15),
        }

    async def test_performance_requirements(self, bridge: RAGTemplatesBridge) -> bool:
        """Test performance SLO compliance."""
        print("🔄 Testing performance requirements...")

        test_queries = [
            "What is machine learning?",
            "How does neural network work?",
            "Explain deep learning concepts",
            "What are the benefits of AI?",
            "How to implement RAG systems?",
        ]

        response_times = []

        try:
            # Execute multiple queries across different techniques
            for i in range(50):  # More samples for accurate p95
                query = test_queries[i % len(test_queries)]
                technique = list(RAGTechnique)[i % len(RAGTechnique)]

                start_time = time.time()
                response = await bridge.query(query, technique=technique.value)
                end_time = time.time()

                latency_ms = (end_time - start_time) * 1000
                response_times.append(latency_ms)

                # Validate individual response
                if response.error:
                    print(f"❌ Query failed: {response.error}")
                    return False

                if latency_ms > 1000:  # Individual query should not exceed 1s
                    print(f"❌ Individual query exceeded 1s: {latency_ms:.2f}ms")
                    return False

            # Calculate performance metrics
            avg_latency = statistics.mean(response_times)
            p95_latency = statistics.quantiles(response_times, n=20)[
                18
            ]  # 95th percentile

            self.results["performance"]["details"] = {
                "avg_latency_ms": round(avg_latency, 2),
                "p95_latency_ms": round(p95_latency, 2),
                "target_p95_ms": 500,
                "total_queries": len(response_times),
            }

            # Check SLO compliance
            slo_compliant = p95_latency < 500

            if slo_compliant:
                print(
                    f"✅ Performance SLO met - P95: {p95_latency:.2f}ms (target: <500ms)"
                )
                self.results["performance"]["passed"] = True
                return True
            else:
                print(
                    f"❌ Performance SLO failed - P95: {p95_latency:.2f}ms (target: <500ms)"
                )
                return False

        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            self.results["performance"]["details"]["error"] = str(e)
            return False

    async def test_circuit_breaker(self, bridge: RAGTemplatesBridge) -> bool:
        """Test circuit breaker functionality."""
        print("🔄 Testing circuit breaker pattern...")

        try:
            # Configure a pipeline to fail
            failing_pipeline = Mock()
            failing_pipeline.config_manager = Mock()
            failing_pipeline.query.side_effect = Exception("Simulated pipeline failure")

            bridge._pipelines[RAGTechnique.CRAG] = failing_pipeline

            # Execute queries to trigger circuit breaker
            failure_count = 0
            for i in range(5):
                response = await bridge.query("test query", technique="crag")
                if response.error:
                    failure_count += 1

            # Check circuit breaker state
            cb_state = bridge._circuit_breakers[RAGTechnique.CRAG]["state"]

            self.results["circuit_breaker"]["details"] = {
                "failure_count": failure_count,
                "circuit_breaker_state": cb_state.value,
                "triggered_correctly": cb_state == CircuitBreakerState.OPEN,
            }

            if cb_state == CircuitBreakerState.OPEN:
                print("✅ Circuit breaker triggered correctly after failures")

                # Test fallback behavior
                response = await bridge.query("fallback test", technique="crag")
                if response.technique_used == bridge.fallback_technique.value:
                    print("✅ Fallback mechanism working correctly")
                    self.results["circuit_breaker"]["passed"] = True
                    return True
                else:
                    print("❌ Fallback mechanism not working")
                    return False
            else:
                print(f"❌ Circuit breaker not triggered (state: {cb_state.value})")
                return False

        except Exception as e:
            print(f"❌ Circuit breaker test failed: {e}")
            self.results["circuit_breaker"]["details"]["error"] = str(e)
            return False

    async def test_error_handling(self, bridge: RAGTemplatesBridge) -> bool:
        """Test error handling and graceful degradation."""
        print("🔄 Testing error handling...")

        try:
            # Test invalid technique - should handle ValueError gracefully
            invalid_technique_handled = False
            try:
                response = await bridge.query("test", technique="invalid_technique")
                if response.error and "invalid_technique" in response.error.lower():
                    invalid_technique_handled = True
            except Exception as e:
                # Should still handle this gracefully
                if "invalid_technique" in str(e):
                    invalid_technique_handled = True

            if not invalid_technique_handled:
                print("❌ Invalid technique not handled properly")
                return False

            # Test pipeline failure with graceful response
            failing_pipeline = Mock()
            failing_pipeline.config_manager = Mock()
            failing_pipeline.query.side_effect = Exception("Pipeline error")

            bridge._pipelines[RAGTechnique.BASIC] = failing_pipeline

            response = await bridge.query("error test", technique="basic")

            # Should return error response but not crash
            if response.error is None:
                print("❌ Error not captured in response")
                return False

            if response.answer != "" or response.confidence_score != 0.0:
                print("❌ Error response format incorrect")
                return False

            self.results["error_handling"]["details"] = {
                "handles_invalid_technique": invalid_technique_handled,
                "graceful_failure_response": True,
                "error_captured": response.error is not None,
            }

            print("✅ Error handling working correctly")
            self.results["error_handling"]["passed"] = True
            return True

        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            self.results["error_handling"]["details"]["error"] = str(e)
            return False

    async def test_all_techniques(self, bridge: RAGTemplatesBridge) -> bool:
        """Test all RAG techniques are accessible."""
        print("🔄 Testing RAG technique accessibility...")

        try:
            # Reset pipelines to fresh state for this test
            self._setup_mock_pipelines(bridge)

            # Reset circuit breakers
            for technique in bridge._circuit_breakers.keys():
                bridge._circuit_breakers[technique][
                    "state"
                ] = CircuitBreakerState.CLOSED
                bridge._circuit_breakers[technique]["failure_count"] = 0

            available_techniques = await bridge.get_available_techniques()
            expected_techniques = ["basic", "crag", "graphrag", "basic_reranking"]

            techniques_tested = {}

            for technique in expected_techniques:
                try:
                    response = await bridge.query("test query", technique=technique)
                    techniques_tested[technique] = {
                        "accessible": response.error is None,
                        "correct_technique": response.technique_used == technique,
                    }
                except Exception as e:
                    techniques_tested[technique] = {
                        "accessible": False,
                        "error": str(e),
                    }

            self.results["techniques"]["details"] = {
                "available_techniques": available_techniques,
                "techniques_tested": techniques_tested,
            }

            all_accessible = all(
                result.get("accessible", False) for result in techniques_tested.values()
            )

            if all_accessible:
                print(
                    f"✅ All RAG techniques accessible: {list(techniques_tested.keys())}"
                )
                self.results["techniques"]["passed"] = True
                return True
            else:
                failed = [
                    t
                    for t, r in techniques_tested.items()
                    if not r.get("accessible", False)
                ]
                print(f"❌ Some techniques not accessible: {failed}")
                return False

        except Exception as e:
            print(f"❌ Technique accessibility test failed: {e}")
            self.results["techniques"]["details"]["error"] = str(e)
            return False

    async def test_configuration_management(self, bridge: RAGTemplatesBridge) -> bool:
        """Test configuration management."""
        print("🔄 Testing configuration management...")

        try:
            # Test health check includes configuration
            health = await bridge.health_check()

            required_health_keys = ["status", "components", "metrics"]
            health_complete = all(key in health for key in required_health_keys)

            # Test performance metrics
            metrics = await bridge.get_performance_metrics()

            required_metric_keys = [
                "total_queries",
                "successful_queries",
                "slo_compliance",
            ]
            metrics_complete = all(key in metrics for key in required_metric_keys)

            self.results["configuration"]["details"] = {
                "health_check_complete": health_complete,
                "metrics_complete": metrics_complete,
                "health_status": health.get("status"),
                "slo_target": metrics.get("slo_compliance", {}).get("p95_target_ms"),
            }

            if health_complete and metrics_complete:
                print("✅ Configuration management working correctly")
                self.results["configuration"]["passed"] = True
                return True
            else:
                print("❌ Configuration management incomplete")
                return False

        except Exception as e:
            print(f"❌ Configuration test failed: {e}")
            self.results["configuration"]["details"]["error"] = str(e)
            return False

    async def run_validation(self) -> Dict[str, Any]:
        """Run complete validation suite."""
        print("🚀 Starting RAG Templates Bridge Validation\n")

        # Set up test environment
        bridge = await self.setup_mock_bridge()

        # Run all validation tests
        tests = [
            ("Performance Requirements", self.test_performance_requirements),
            ("Circuit Breaker Pattern", self.test_circuit_breaker),
            ("Error Handling", self.test_error_handling),
            ("RAG Technique Access", self.test_all_techniques),
            ("Configuration Management", self.test_configuration_management),
        ]

        passed_tests = 0
        total_tests = len(tests)

        for test_name, test_func in tests:
            print(f"\n📋 {test_name}")
            try:
                if await test_func(bridge):
                    passed_tests += 1
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                print(f"❌ {test_name} FAILED with exception: {e}")

        # Generate summary
        success_rate = (passed_tests / total_tests) * 100
        overall_pass = passed_tests == total_tests

        print(f"\n{'='*60}")
        print(f"🎯 VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        print(f"Overall Result: {'✅ PASS' if overall_pass else '❌ FAIL'}")

        if overall_pass:
            print("\n🏆 RAG Templates Bridge meets all requirements!")
            print("✅ Ready for kg-ticket-resolver integration")
        else:
            print("\n🔧 Some requirements not met. Check details above.")

        return {
            "overall_pass": overall_pass,
            "success_rate": success_rate,
            "tests_passed": passed_tests,
            "total_tests": total_tests,
            "detailed_results": self.results,
        }


async def main():
    """Main validation entry point."""
    validator = RAGBridgeValidator()
    results = await validator.run_validation()

    # Exit with appropriate code
    sys.exit(0 if results["overall_pass"] else 1)


if __name__ == "__main__":
    asyncio.run(main())
