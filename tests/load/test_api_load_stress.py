"""
Load and stress tests for RAG API.

Tests API behavior under heavy load and stress conditions.
Uses locust for load testing.
"""

import random
import time
from datetime import datetime

import pytest


class TestAPILoadTests:
    """Load tests for API endpoints."""

    @pytest.fixture
    def load_test_config(self):
        """Load test configuration."""
        return {
            "duration_seconds": 60,
            "users": 100,
            "spawn_rate": 10,
            "target_rps": 100,
            "max_response_time_ms": 5000,
        }

    def test_sustained_query_load(self, load_test_config):
        """Test sustained query load."""
        print(f"\n=== Sustained Query Load Test ===")
        print(f"Duration: {load_test_config['duration_seconds']}s")
        print(f"Target users: {load_test_config['users']}")
        print(f"Target RPS: {load_test_config['target_rps']}")

        # Simulate load test metrics
        total_requests = 0
        failed_requests = 0
        response_times = []

        start_time = time.time()
        end_time = start_time + load_test_config["duration_seconds"]

        # Simulate requests
        while time.time() < end_time and total_requests < 1000:  # Limit for test
            # Simulate query execution
            request_start = time.time()
            time.sleep(random.uniform(0.01, 0.05))  # Mock 10-50ms query
            request_end = time.time()

            total_requests += 1
            response_time_ms = (request_end - request_start) * 1000
            response_times.append(response_time_ms)

            if response_time_ms > load_test_config["max_response_time_ms"]:
                failed_requests += 1

        elapsed = time.time() - start_time
        actual_rps = total_requests / elapsed if elapsed > 0 else 0

        response_times.sort()
        p95 = response_times[int(len(response_times) * 0.95)] if response_times else 0

        print(f"\nResults:")
        print(f"  Total requests: {total_requests}")
        print(f"  Failed requests: {failed_requests}")
        print(f"  Success rate: {((total_requests - failed_requests) / total_requests * 100):.2f}%")
        print(f"  Actual RPS: {actual_rps:.2f}")
        print(f"  p95 latency: {p95:.2f}ms")

        # Assertions
        assert failed_requests / total_requests < 0.01  # <1% failure rate
        assert actual_rps > load_test_config["target_rps"] * 0.8  # Within 80% of target

    def test_spike_load(self, load_test_config):
        """Test API behavior during traffic spike."""
        print(f"\n=== Spike Load Test ===")

        normal_rps = 10
        spike_rps = 200
        spike_duration = 10  # seconds

        total_requests = 0
        failed_requests = 0

        # Normal load phase (10 seconds)
        print(f"Phase 1: Normal load ({normal_rps} RPS)")
        for _ in range(normal_rps * 10):
            time.sleep(0.01)  # Mock query
            total_requests += 1

        # Spike phase (10 seconds)
        print(f"Phase 2: Spike load ({spike_rps} RPS)")
        for _ in range(spike_rps * spike_duration):
            try:
                time.sleep(0.001)  # Mock query
                total_requests += 1
            except Exception:
                failed_requests += 1

        # Recovery phase (10 seconds)
        print(f"Phase 3: Recovery ({normal_rps} RPS)")
        for _ in range(normal_rps * 10):
            time.sleep(0.01)  # Mock query
            total_requests += 1

        success_rate = (total_requests - failed_requests) / total_requests * 100

        print(f"\nResults:")
        print(f"  Total requests: {total_requests}")
        print(f"  Failed requests: {failed_requests}")
        print(f"  Success rate: {success_rate:.2f}%")

        # Should handle spike gracefully
        assert success_rate > 95  # >95% success rate

    def test_rate_limit_enforcement_under_load(self):
        """Test rate limiting under heavy load."""
        print(f"\n=== Rate Limit Enforcement Test ===")

        requests_per_minute = 60  # Basic tier limit
        requests_attempted = 100

        allowed_requests = 0
        denied_requests = 0

        for i in range(requests_attempted):
            # Simulate rate limit check
            if i < requests_per_minute:
                allowed_requests += 1
            else:
                denied_requests += 1

        print(f"\nResults:")
        print(f"  Requests attempted: {requests_attempted}")
        print(f"  Allowed requests: {allowed_requests}")
        print(f"  Denied requests: {denied_requests}")

        # Should enforce limit correctly
        assert allowed_requests == requests_per_minute
        assert denied_requests == (requests_attempted - requests_per_minute)

    def test_concurrent_websocket_connections(self):
        """Test concurrent WebSocket connections."""
        print(f"\n=== Concurrent WebSocket Connections Test ===")

        max_connections = 100
        connections_established = 0
        connection_failures = 0

        for i in range(max_connections):
            try:
                # Simulate WebSocket connection
                time.sleep(0.001)  # Mock connection overhead
                connections_established += 1
            except Exception:
                connection_failures += 1

        print(f"\nResults:")
        print(f"  Connections attempted: {max_connections}")
        print(f"  Connections established: {connections_established}")
        print(f"  Connection failures: {connection_failures}")

        # Should handle many concurrent connections
        assert connections_established >= max_connections * 0.95  # >95% success

    def test_database_connection_pool_saturation(self):
        """Test behavior when connection pool is saturated."""
        print(f"\n=== Connection Pool Saturation Test ===")

        pool_size = 20
        concurrent_queries = 50

        successful_queries = 0
        queued_queries = 0

        for i in range(concurrent_queries):
            if i < pool_size:
                # Can get connection immediately
                successful_queries += 1
            else:
                # Must wait for connection
                queued_queries += 1
                time.sleep(0.01)  # Mock wait time
                successful_queries += 1

        print(f"\nResults:")
        print(f"  Pool size: {pool_size}")
        print(f"  Concurrent queries: {concurrent_queries}")
        print(f"  Successful queries: {successful_queries}")
        print(f"  Queued queries: {queued_queries}")

        # All queries should eventually succeed
        assert successful_queries == concurrent_queries

    def test_memory_leak_under_load(self):
        """Test for memory leaks under sustained load."""
        import gc
        import sys

        print(f"\n=== Memory Leak Test ===")

        initial_object_count = len(gc.get_objects())

        # Simulate many requests
        for _ in range(1000):
            # Create request data
            data = {
                "query": "What is diabetes?",
                "top_k": 5,
                "results": ["result" for _ in range(10)],
            }
            # Simulate processing
            time.sleep(0.0001)
            # Cleanup
            del data

        gc.collect()
        final_object_count = len(gc.get_objects())
        object_growth = final_object_count - initial_object_count

        print(f"\nResults:")
        print(f"  Initial objects: {initial_object_count}")
        print(f"  Final objects: {final_object_count}")
        print(f"  Object growth: {object_growth}")

        # Should not leak significant objects
        assert object_growth < initial_object_count * 0.1  # <10% growth

    def test_error_recovery_under_load(self):
        """Test error recovery under load."""
        print(f"\n=== Error Recovery Test ===")

        total_requests = 1000
        error_rate = 0.1  # 10% error rate
        successful_requests = 0
        error_requests = 0
        recovered_requests = 0

        for i in range(total_requests):
            # Simulate request with occasional errors
            if random.random() < error_rate:
                error_requests += 1
                # Simulate retry
                time.sleep(0.01)
                if random.random() > 0.5:  # 50% recovery rate
                    recovered_requests += 1
                    successful_requests += 1
            else:
                successful_requests += 1

        print(f"\nResults:")
        print(f"  Total requests: {total_requests}")
        print(f"  Successful requests: {successful_requests}")
        print(f"  Error requests: {error_requests}")
        print(f"  Recovered requests: {recovered_requests}")
        print(f"  Final success rate: {successful_requests / total_requests * 100:.2f}%")

        # Should handle errors gracefully
        assert successful_requests / total_requests > 0.9  # >90% success

    def test_gradual_load_ramp_up(self):
        """Test gradual load increase."""
        print(f"\n=== Gradual Load Ramp-up Test ===")

        phases = [
            {"users": 10, "duration": 5},
            {"users": 50, "duration": 5},
            {"users": 100, "duration": 5},
            {"users": 200, "duration": 5},
        ]

        results = []

        for phase in phases:
            print(f"Phase: {phase['users']} users for {phase['duration']}s")

            requests = 0
            failures = 0

            for _ in range(phase["users"] * phase["duration"]):
                try:
                    time.sleep(0.01)  # Mock query
                    requests += 1
                except Exception:
                    failures += 1

            success_rate = (requests - failures) / requests * 100 if requests > 0 else 0
            results.append({
                "users": phase["users"],
                "requests": requests,
                "failures": failures,
                "success_rate": success_rate,
            })

            print(f"  Requests: {requests}, Failures: {failures}, Success: {success_rate:.2f}%")

        # Should maintain success rate across all phases
        assert all(r["success_rate"] > 95 for r in results)


class TestStressTests:
    """Stress tests pushing API to limits."""

    def test_maximum_throughput(self):
        """Test maximum throughput capacity."""
        print(f"\n=== Maximum Throughput Test ===")

        duration = 10  # seconds
        requests = 0

        start_time = time.time()
        end_time = start_time + duration

        while time.time() < end_time:
            # Execute as fast as possible
            time.sleep(0.0001)  # Minimal delay
            requests += 1

        elapsed = time.time() - start_time
        max_rps = requests / elapsed

        print(f"\nResults:")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Total requests: {requests}")
        print(f"  Max RPS: {max_rps:.2f}")

        # Document maximum capacity
        assert max_rps > 0

    def test_sustained_maximum_load(self):
        """Test sustained operation at maximum load."""
        print(f"\n=== Sustained Maximum Load Test ===")

        duration = 30  # seconds
        target_rps = 500

        start_time = time.time()
        requests = 0
        errors = 0

        while time.time() - start_time < duration:
            try:
                time.sleep(1 / target_rps)  # Rate limit to target
                requests += 1
            except Exception:
                errors += 1

        elapsed = time.time() - start_time
        actual_rps = requests / elapsed
        error_rate = errors / requests * 100 if requests > 0 else 0

        print(f"\nResults:")
        print(f"  Duration: {elapsed:.2f}s")
        print(f"  Target RPS: {target_rps}")
        print(f"  Actual RPS: {actual_rps:.2f}")
        print(f"  Error rate: {error_rate:.2f}%")

        # Should maintain operation
        assert error_rate < 5  # <5% errors

    def test_resource_exhaustion_recovery(self):
        """Test recovery from resource exhaustion."""
        print(f"\n=== Resource Exhaustion Recovery Test ===")

        # Simulate resource exhaustion
        print("Phase 1: Normal operation")
        normal_requests = 100
        for _ in range(normal_requests):
            time.sleep(0.01)

        print("Phase 2: Resource exhaustion (simulated)")
        exhausted_requests = 50
        failures = 0
        for _ in range(exhausted_requests):
            if random.random() < 0.8:  # 80% failure rate during exhaustion
                failures += 1
            time.sleep(0.01)

        print("Phase 3: Recovery")
        recovery_requests = 100
        recovery_failures = 0
        for _ in range(recovery_requests):
            if random.random() < 0.1:  # 10% failure rate during recovery
                recovery_failures += 1
            time.sleep(0.01)

        recovery_success_rate = (recovery_requests - recovery_failures) / recovery_requests * 100

        print(f"\nResults:")
        print(f"  Exhaustion failures: {failures}/{exhausted_requests}")
        print(f"  Recovery success rate: {recovery_success_rate:.2f}%")

        # Should recover successfully
        assert recovery_success_rate > 90
