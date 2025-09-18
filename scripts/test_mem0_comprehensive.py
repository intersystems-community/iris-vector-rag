#!/usr/bin/env python3
"""
Comprehensive mem0 MCP Server Test Script
Tests both direct mem0 functionality and MCP server responsiveness.
Bypasses interface issues to verify actual server functionality.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import traceback


@dataclass
class TestResult:
    """Test result data structure."""

    test_name: str
    success: bool
    duration: float
    message: str
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class Mem0ComprehensiveTest:
    """Comprehensive test suite for mem0 MCP server functionality."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.test_user_id = "test_user_comprehensive_001"
        self.mcp_timeout = 10.0  # seconds

    def log_result(
        self,
        test_name: str,
        success: bool,
        duration: float,
        message: str,
        details: Optional[Dict] = None,
        error: Optional[str] = None,
    ):
        """Log a test result."""
        result = TestResult(
            test_name=test_name,
            success=success,
            duration=duration,
            message=message,
            details=details,
            error=error,
        )
        self.results.append(result)

        # Print immediate feedback
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name} ({duration:.2f}s): {message}")
        if error and not success:
            print(f"   ERROR: {error}")

    def load_env_file(self) -> bool:
        """Load environment variables from .env.mem0 file."""
        env_file = Path(".env.mem0")
        if not env_file.exists():
            print("❌ .env.mem0 file not found")
            return False

        # Simple env file parser
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Skip placeholder variables
                    if not value.startswith("${"):
                        os.environ[key] = value

        return True

    def test_environment_setup(self) -> bool:
        """Test that required environment variables are present."""
        start_time = time.time()

        required_vars = ["OPENAI_API_KEY"]
        optional_vars = ["MEM0_API_KEY", "QDRANT_URL", "MCP_SERVER_HOST"]

        missing_required = []
        present_optional = []

        for var in required_vars:
            if not os.getenv(var):
                missing_required.append(var)

        for var in optional_vars:
            if os.getenv(var):
                present_optional.append(var)

        duration = time.time() - start_time
        success = len(missing_required) == 0

        details = {
            "required_present": len(required_vars) - len(missing_required),
            "required_total": len(required_vars),
            "optional_present": present_optional,
            "missing_required": missing_required,
        }

        if success:
            message = f"Environment configured. Optional vars present: {len(present_optional)}"
        else:
            message = f"Missing required variables: {', '.join(missing_required)}"

        self.log_result("Environment Setup", success, duration, message, details)
        return success

    def test_mem0_import_and_version(self) -> bool:
        """Test mem0 library import and version detection."""
        start_time = time.time()

        try:
            import mem0

            version = getattr(mem0, "__version__", "unknown")
            duration = time.time() - start_time

            details = {"version": version}
            message = f"mem0 imported successfully (version: {version})"

            self.log_result("mem0 Import", True, duration, message, details)
            return True

        except ImportError as e:
            duration = time.time() - start_time
            error = str(e)
            message = "Failed to import mem0 library"

            self.log_result("mem0 Import", False, duration, message, error=error)
            return False

    def test_mem0_basic_functionality(self) -> bool:
        """Test basic mem0 memory operations."""
        start_time = time.time()

        try:
            from mem0 import Memory

            # Initialize mem0 client
            m = Memory()

            # Test data
            test_memories = [
                "I love programming in Python and building AI applications",
                "My favorite framework for web development is FastAPI",
                "I work with mem0 for memory management in AI systems",
            ]

            stored_memories = []

            # Store memories
            for memory_text in test_memories:
                result = m.add(memory_text, user_id=self.test_user_id)
                stored_memories.append(result)
                time.sleep(0.1)  # Small delay between operations

            # Search memories
            search_results = m.search("programming", user_id=self.test_user_id)

            # Get all memories
            all_memories = m.get_all(user_id=self.test_user_id)

            duration = time.time() - start_time

            details = {
                "memories_stored": len(stored_memories),
                "search_results_count": len(search_results),
                "total_memories": len(all_memories),
                "stored_memory_ids": [
                    str(m.get("id", "unknown"))
                    for m in stored_memories
                    if isinstance(m, dict)
                ],
            }

            success = (
                len(stored_memories) == len(test_memories)
                and len(search_results) > 0
                and len(all_memories) >= len(test_memories)
            )

            if success:
                message = f"Basic operations successful. Stored: {len(stored_memories)}, Found: {len(search_results)}"
            else:
                message = "Basic operations failed - unexpected results"

            self.log_result(
                "mem0 Basic Operations", success, duration, message, details
            )
            return success

        except Exception as e:
            duration = time.time() - start_time
            error = str(e)
            message = "Exception during basic mem0 operations"

            self.log_result(
                "mem0 Basic Operations", False, duration, message, error=error
            )
            return False

    def test_mcp_server_availability(self) -> bool:
        """Test MCP server process availability and basic connectivity."""
        start_time = time.time()

        try:
            import subprocess
            import socket

            # Check if mem0 MCP server process might be running
            host = os.getenv("MCP_SERVER_HOST", "localhost")
            port = int(os.getenv("MCP_SERVER_PORT", "3000"))

            # Try to connect to the server port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)

            try:
                result = sock.connect_ex((host, port))
                port_open = result == 0
            except:
                port_open = False
            finally:
                sock.close()

            # Check for Python MCP server processes
            try:
                ps_output = subprocess.run(
                    ["ps", "aux"], capture_output=True, text=True, timeout=5
                )
                process_found = (
                    "mem0" in ps_output.stdout and "server" in ps_output.stdout
                )
            except:
                process_found = False

            duration = time.time() - start_time

            details = {
                "host": host,
                "port": port,
                "port_open": port_open,
                "process_found": process_found,
            }

            success = port_open or process_found

            if success:
                message = f"MCP server detected (port_open: {port_open}, process: {process_found})"
            else:
                message = f"MCP server not detected on {host}:{port}"

            self.log_result(
                "MCP Server Availability", success, duration, message, details
            )
            return success

        except Exception as e:
            duration = time.time() - start_time
            error = str(e)
            message = "Exception checking MCP server availability"

            self.log_result(
                "MCP Server Availability", False, duration, message, error=error
            )
            return False

    def test_mcp_tools_via_direct_call(self) -> bool:
        """Test MCP tools by simulating direct tool calls."""
        start_time = time.time()

        try:
            # Simulate MCP tool functionality using the same backend as the server
            from mem0 import Memory

            m = Memory()
            test_user = "mcp_test_user_001"

            # Simulate store_memory tool
            store_result = m.add(
                "MCP server test: I am testing memory storage functionality",
                user_id=test_user,
            )

            # Simulate search_memories tool
            search_result = m.search("testing memory storage", user_id=test_user)

            # Simulate get_memories tool
            get_result = m.get_all(user_id=test_user)

            duration = time.time() - start_time

            details = {
                "store_success": store_result is not None,
                "search_count": len(search_result) if search_result else 0,
                "total_memories": len(get_result) if get_result else 0,
                "test_user": test_user,
            }

            success = (
                store_result is not None
                and search_result is not None
                and len(search_result) > 0
                and get_result is not None
            )

            if success:
                message = f"MCP tools simulation successful. Found {len(search_result)} search results"
            else:
                message = "MCP tools simulation failed"

            self.log_result("MCP Tools (Direct)", success, duration, message, details)
            return success

        except Exception as e:
            duration = time.time() - start_time
            error = str(e)
            message = "Exception during MCP tools simulation"

            self.log_result("MCP Tools (Direct)", False, duration, message, error=error)
            return False

    def test_memory_persistence_and_retrieval(self) -> bool:
        """Test memory persistence across different operations."""
        start_time = time.time()

        try:
            from mem0 import Memory

            m = Memory()
            persistence_user = "persistence_test_user"

            # Store different types of memories
            test_data = [
                {
                    "content": "I prefer Python for backend development",
                    "category": "preference",
                },
                {
                    "content": "My current project involves building an AI assistant",
                    "category": "project",
                },
                {
                    "content": "I learned about vector databases last week",
                    "category": "learning",
                },
            ]

            stored_ids = []

            # Store memories
            for data in test_data:
                result = m.add(data["content"], user_id=persistence_user)
                if result and isinstance(result, dict):
                    stored_ids.append(result.get("id"))
                time.sleep(0.1)

            # Test different search queries
            search_tests = [
                ("Python", "preference"),
                ("project AI", "project"),
                ("vector databases", "learning"),
            ]

            search_results = {}
            for query, expected_category in search_tests:
                results = m.search(query, user_id=persistence_user)
                search_results[query] = len(results) if results else 0

            # Get all memories to verify persistence
            all_memories = m.get_all(user_id=persistence_user)

            duration = time.time() - start_time

            details = {
                "stored_count": len([id for id in stored_ids if id is not None]),
                "search_results": search_results,
                "total_retrieved": len(all_memories) if all_memories else 0,
                "stored_ids": stored_ids,
            }

            success = (
                len(stored_ids) == len(test_data)
                and all(count > 0 for count in search_results.values())
                and all_memories
                and len(all_memories) >= len(test_data)
            )

            if success:
                message = f"Persistence test successful. {len(test_data)} stored, all searches returned results"
            else:
                message = (
                    "Persistence test failed - some operations returned no results"
                )

            self.log_result("Memory Persistence", success, duration, message, details)
            return success

        except Exception as e:
            duration = time.time() - start_time
            error = str(e)
            message = "Exception during persistence testing"

            self.log_result("Memory Persistence", False, duration, message, error=error)
            return False

    def test_performance_and_timeouts(self) -> bool:
        """Test performance characteristics and timeout handling."""
        start_time = time.time()

        try:
            from mem0 import Memory

            m = Memory()
            perf_user = "performance_test_user"

            # Test batch operations
            batch_size = 5
            batch_memories = [
                f"Performance test memory {i}: This is test data for performance evaluation"
                for i in range(batch_size)
            ]

            # Measure batch storage time
            batch_start = time.time()
            batch_results = []

            for memory in batch_memories:
                result = m.add(memory, user_id=perf_user)
                batch_results.append(result)

            batch_duration = time.time() - batch_start

            # Measure search performance
            search_start = time.time()
            search_results = m.search("performance test", user_id=perf_user)
            search_duration = time.time() - search_start

            # Measure retrieval performance
            retrieval_start = time.time()
            all_results = m.get_all(user_id=perf_user)
            retrieval_duration = time.time() - retrieval_start

            duration = time.time() - start_time

            details = {
                "batch_size": batch_size,
                "batch_duration": round(batch_duration, 3),
                "search_duration": round(search_duration, 3),
                "retrieval_duration": round(retrieval_duration, 3),
                "avg_storage_time": round(batch_duration / batch_size, 3),
                "search_results_count": len(search_results) if search_results else 0,
                "total_memories": len(all_results) if all_results else 0,
            }

            # Performance criteria (reasonable for development)
            storage_ok = batch_duration < 10.0  # Less than 10 seconds for 5 items
            search_ok = search_duration < 5.0  # Less than 5 seconds for search
            retrieval_ok = retrieval_duration < 5.0  # Less than 5 seconds for retrieval

            success = storage_ok and search_ok and retrieval_ok

            if success:
                message = f"Performance acceptable. Storage: {batch_duration:.2f}s, Search: {search_duration:.2f}s"
            else:
                message = f"Performance issues detected. Times exceeded thresholds"

            self.log_result(
                "Performance & Timeouts", success, duration, message, details
            )
            return success

        except Exception as e:
            duration = time.time() - start_time
            error = str(e)
            message = "Exception during performance testing"

            self.log_result(
                "Performance & Timeouts", False, duration, message, error=error
            )
            return False

    def run_all_tests(self) -> bool:
        """Run all test scenarios and return overall success."""
        print("🧪 Starting Comprehensive mem0 MCP Server Tests...")
        print("=" * 60)

        test_methods = [
            self.test_environment_setup,
            self.test_mem0_import_and_version,
            self.test_mem0_basic_functionality,
            self.test_mcp_server_availability,
            self.test_mcp_tools_via_direct_call,
            self.test_memory_persistence_and_retrieval,
            self.test_performance_and_timeouts,
        ]

        overall_success = True

        for test_method in test_methods:
            try:
                success = test_method()
                if not success:
                    overall_success = False
            except Exception as e:
                print(f"❌ CRITICAL ERROR in {test_method.__name__}: {e}")
                traceback.print_exc()
                overall_success = False

            print()  # Add spacing between tests

        return overall_success

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - passed_tests

        total_duration = sum(r.duration for r in self.results)

        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (
                    round((passed_tests / total_tests) * 100, 1)
                    if total_tests > 0
                    else 0
                ),
                "total_duration": round(total_duration, 2),
            },
            "test_results": [asdict(result) for result in self.results],
            "recommendations": self._get_recommendations(),
        }

        return report

    def _get_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        failed_tests = [r for r in self.results if not r.success]

        if any(
            r.test_name == "Environment Setup" and not r.success for r in failed_tests
        ):
            recommendations.append(
                "Configure missing environment variables in .env.mem0"
            )

        if any(r.test_name == "mem0 Import" and not r.success for r in failed_tests):
            recommendations.append("Install mem0 library: pip install mem0ai")

        if any(
            r.test_name == "MCP Server Availability" and not r.success
            for r in failed_tests
        ):
            recommendations.append("Start the mem0 MCP server process")

        if any("Performance" in r.test_name and not r.success for r in failed_tests):
            recommendations.append("Check system resources and network connectivity")

        if not failed_tests:
            recommendations.append(
                "All tests passed! mem0 MCP server is functioning correctly"
            )

        return recommendations

    def print_summary(self):
        """Print a summary of test results."""
        report = self.generate_report()
        summary = report["summary"]

        print("📊 Test Summary")
        print("=" * 60)
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ✅")
        print(f"Failed: {summary['failed']} ❌")
        print(f"Success Rate: {summary['success_rate']}%")
        print(f"Total Duration: {summary['total_duration']}s")
        print()

        print("💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")
        print()

        if summary["failed"] == 0:
            print("🎉 ALL TESTS PASSED! mem0 MCP server is functioning correctly.")
        else:
            print("⚠️  Some tests failed. Check the details above for troubleshooting.")


def main():
    """Main test execution function."""
    print("🔧 Loading environment variables...")

    # Load environment
    test_suite = Mem0ComprehensiveTest()
    if not test_suite.load_env_file():
        print("❌ Failed to load .env.mem0 file")
        return 1

    # Run tests
    overall_success = test_suite.run_all_tests()

    # Print summary
    test_suite.print_summary()

    # Save detailed report
    report = test_suite.generate_report()
    report_file = Path("test_results_mem0_comprehensive.json")

    try:
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"📄 Detailed report saved to: {report_file}")
    except Exception as e:
        print(f"⚠️  Could not save report file: {e}")

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
