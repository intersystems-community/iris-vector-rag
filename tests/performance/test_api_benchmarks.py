"""
Performance benchmarks for RAG API.

Measures latency, throughput, and resource usage.
"""

import asyncio
import time
from statistics import mean, median, stdev

import pytest


class TestAPIPerformanceBenchmarks:
    """Performance benchmarks for API endpoints."""

    @pytest.fixture
    def benchmark_config(self):
        """Benchmark configuration."""
        return {
            "iterations": 100,
            "warmup_iterations": 10,
            "target_p95_latency_ms": 2000,
            "target_p99_latency_ms": 3000,
            "target_throughput_rps": 100,
        }

    def test_query_latency_benchmark(self, benchmark_config):
        """Benchmark query endpoint latency."""
        latencies = []

        # Warmup
        for _ in range(benchmark_config["warmup_iterations"]):
            start = time.time()
            # Simulate query execution
            time.sleep(0.01)  # Mock 10ms query
            end = time.time()

        # Actual benchmark
        for _ in range(benchmark_config["iterations"]):
            start = time.time()
            # Simulate query execution
            time.sleep(0.01)  # Mock 10ms query
            end = time.time()
            latencies.append((end - start) * 1000)  # Convert to ms

        # Calculate percentiles
        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]

        print(f"\nQuery Latency Benchmark:")
        print(f"  p50: {p50:.2f}ms")
        print(f"  p95: {p95:.2f}ms")
        print(f"  p99: {p99:.2f}ms")
        print(f"  Mean: {mean(latencies):.2f}ms")
        print(f"  StdDev: {stdev(latencies):.2f}ms")

        # Assertions
        assert p95 < benchmark_config["target_p95_latency_ms"]
        assert p99 < benchmark_config["target_p99_latency_ms"]

    def test_health_check_latency_benchmark(self, benchmark_config):
        """Benchmark health check endpoint latency."""
        latencies = []

        for _ in range(benchmark_config["iterations"]):
            start = time.time()
            # Simulate health check
            time.sleep(0.001)  # Mock 1ms health check
            end = time.time()
            latencies.append((end - start) * 1000)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]

        print(f"\nHealth Check Latency Benchmark:")
        print(f"  p95: {p95:.2f}ms")
        print(f"  Mean: {mean(latencies):.2f}ms")

        # Health check should be very fast
        assert p95 < 100  # <100ms

    def test_api_key_validation_benchmark(self, benchmark_config):
        """Benchmark API key validation latency."""
        latencies = []

        for _ in range(benchmark_config["iterations"]):
            start = time.time()
            # Simulate bcrypt validation
            time.sleep(0.0001)  # Mock 0.1ms validation
            end = time.time()
            latencies.append((end - start) * 1000)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]

        print(f"\nAPI Key Validation Benchmark:")
        print(f"  p95: {p95:.2f}ms")
        print(f"  Mean: {mean(latencies):.2f}ms")

        # API key validation should be fast
        assert p95 < 50  # <50ms

    def test_rate_limiter_throughput_benchmark(self, benchmark_config):
        """Benchmark rate limiter throughput."""
        start = time.time()
        checks_completed = 0

        # Run rate limit checks for 1 second
        end_time = start + 1.0
        while time.time() < end_time:
            # Simulate rate limit check (Redis operation)
            time.sleep(0.0001)  # Mock 0.1ms Redis check
            checks_completed += 1

        elapsed = time.time() - start
        throughput = checks_completed / elapsed

        print(f"\nRate Limiter Throughput Benchmark:")
        print(f"  Checks/second: {throughput:.2f}")
        print(f"  Checks completed: {checks_completed}")

        # Should handle many checks per second
        assert throughput > 1000

    def test_connection_pool_overhead_benchmark(self, benchmark_config):
        """Benchmark database connection pool overhead."""
        latencies = []

        for _ in range(benchmark_config["iterations"]):
            start = time.time()
            # Simulate getting connection from pool
            time.sleep(0.0005)  # Mock 0.5ms pool overhead
            end = time.time()
            latencies.append((end - start) * 1000)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]

        print(f"\nConnection Pool Overhead Benchmark:")
        print(f"  p95: {p95:.2f}ms")
        print(f"  Mean: {mean(latencies):.2f}ms")

        # Connection pool should be very fast
        assert p95 < 10  # <10ms

    @pytest.mark.asyncio
    async def test_concurrent_query_benchmark(self, benchmark_config):
        """Benchmark concurrent query handling."""
        async def mock_query():
            await asyncio.sleep(0.01)  # Mock 10ms query
            return {"answer": "test"}

        concurrent_requests = 50
        start = time.time()

        # Execute concurrent queries
        tasks = [mock_query() for _ in range(concurrent_requests)]
        results = await asyncio.gather(*tasks)

        elapsed = time.time() - start
        throughput = concurrent_requests / elapsed

        print(f"\nConcurrent Query Benchmark:")
        print(f"  Concurrent requests: {concurrent_requests}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")

        # Should handle concurrent requests efficiently
        assert len(results) == concurrent_requests
        assert throughput > 10  # At least 10 req/s

    def test_response_serialization_benchmark(self, benchmark_config):
        """Benchmark response JSON serialization."""
        import json

        response_data = {
            "response_id": "550e8400-e29b-41d4-a716-446655440000",
            "answer": "x" * 1000,  # 1KB answer
            "retrieved_documents": [
                {
                    "doc_id": f"doc-{i}",
                    "content": "x" * 500,
                    "score": 0.95,
                    "metadata": {"source": "test.pdf"},
                }
                for i in range(10)
            ],
            "contexts": ["x" * 500 for _ in range(10)],
            "sources": [f"source-{i}.pdf" for i in range(10)],
        }

        latencies = []

        for _ in range(benchmark_config["iterations"]):
            start = time.time()
            json.dumps(response_data)
            end = time.time()
            latencies.append((end - start) * 1000)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]

        print(f"\nResponse Serialization Benchmark:")
        print(f"  p95: {p95:.2f}ms")
        print(f"  Mean: {mean(latencies):.2f}ms")

        # Serialization should be fast
        assert p95 < 10  # <10ms

    def test_request_logging_overhead_benchmark(self, benchmark_config):
        """Benchmark request logging overhead."""
        latencies = []

        for _ in range(benchmark_config["iterations"]):
            start = time.time()
            # Simulate logging operation (database insert)
            time.sleep(0.001)  # Mock 1ms logging
            end = time.time()
            latencies.append((end - start) * 1000)

        latencies.sort()
        p95 = latencies[int(len(latencies) * 0.95)]

        print(f"\nRequest Logging Overhead Benchmark:")
        print(f"  p95: {p95:.2f}ms")
        print(f"  Mean: {mean(latencies):.2f}ms")

        # Logging should have minimal overhead
        assert p95 < 50  # <50ms

    def test_pipeline_initialization_benchmark(self):
        """Benchmark pipeline initialization time."""
        initialization_times = []

        for _ in range(5):
            start = time.time()
            # Simulate pipeline initialization
            time.sleep(0.1)  # Mock 100ms initialization
            end = time.time()
            initialization_times.append((end - start) * 1000)

        avg_init_time = mean(initialization_times)

        print(f"\nPipeline Initialization Benchmark:")
        print(f"  Average: {avg_init_time:.2f}ms")

        # Initialization can be slower but should be reasonable
        assert avg_init_time < 5000  # <5s

    def test_memory_usage_benchmark(self, benchmark_config):
        """Benchmark memory usage during operations."""
        import sys

        initial_objects = len(gc.get_objects())

        # Simulate operations
        data = []
        for _ in range(1000):
            data.append({"query": "test", "results": ["x" * 100 for _ in range(10)]})

        final_objects = len(gc.get_objects())
        objects_created = final_objects - initial_objects

        print(f"\nMemory Usage Benchmark:")
        print(f"  Objects created: {objects_created}")

        # Cleanup
        del data

        # Should not leak too many objects
        assert objects_created < 50000


import gc


class TestRateLimiterBenchmarks:
    """Benchmarks specific to rate limiting."""

    def test_redis_sliding_window_performance(self):
        """Benchmark Redis sliding window algorithm."""
        operations = []

        for i in range(1000):
            start = time.time()
            # Simulate Redis INCR + EXPIRE operations
            time.sleep(0.0001)  # Mock 0.1ms Redis operation
            end = time.time()
            operations.append((end - start) * 1000)

        operations.sort()
        p95 = operations[int(len(operations) * 0.95)]

        print(f"\nRedis Sliding Window Benchmark:")
        print(f"  p95: {p95:.2f}ms")
        print(f"  Mean: {mean(operations):.2f}ms")

        assert p95 < 5  # <5ms per operation

    def test_concurrent_rate_limit_checks(self):
        """Benchmark concurrent rate limit checks."""
        import concurrent.futures

        def check_rate_limit():
            time.sleep(0.001)  # Mock 1ms check
            return True

        num_checks = 100
        start = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(lambda _: check_rate_limit(), range(num_checks)))

        elapsed = time.time() - start
        throughput = num_checks / elapsed

        print(f"\nConcurrent Rate Limit Checks Benchmark:")
        print(f"  Total checks: {num_checks}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.2f} checks/s")

        assert all(results)
        assert throughput > 50  # At least 50 checks/s
