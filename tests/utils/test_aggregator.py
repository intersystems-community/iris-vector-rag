"""Test result aggregation utility.

Feature: 025-fixes-for-testing
Task: T013

Aggregates test results from pytest JSON output into TestCase entities
for reporting and analysis.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class TestStatus(Enum):
    """Test case status."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestCase:
    """TestCase entity from data-model.md.

    Attributes:
        name: Full test name (e.g., tests/e2e/test_basic_pipeline_e2e.py::test_query)
        status: Test status (passed, failed, skipped, error)
        execution_time: Test execution time in seconds
        coverage_lines: Number of code lines covered by this test
        test_suite: Test suite category (unit, e2e, integration, contract)
        failure_message: Error message if failed
    """

    name: str
    status: TestStatus
    execution_time: float = 0.0
    coverage_lines: int = 0
    test_suite: str = "unknown"
    failure_message: Optional[str] = None


@dataclass
class TestSummary:
    """Aggregated test summary statistics.

    Attributes:
        total: Total number of tests
        passed: Number of passed tests
        failed: Number of failed tests
        skipped: Number of skipped tests
        errors: Number of errored tests
        total_time: Total execution time in seconds
        test_cases: List of all test cases
        by_suite: Test counts grouped by suite (unit, e2e, etc.)
    """

    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    total_time: float = 0.0
    test_cases: List[TestCase] = field(default_factory=list)
    by_suite: Dict[str, Dict[str, int]] = field(default_factory=dict)


class TestAggregator:
    """Aggregates test results from pytest JSON output."""

    def __init__(self):
        """Initialize test aggregator."""
        self.test_cases: List[TestCase] = []

    def parse_pytest_json(self, json_path: Path) -> TestSummary:
        """Parse pytest JSON report and create TestCase entities.

        Args:
            json_path: Path to pytest JSON report (pytest --json-report)

        Returns:
            TestSummary with aggregated statistics
        """
        if not json_path.exists():
            raise FileNotFoundError(f"pytest JSON report not found: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        summary = TestSummary()

        # Parse tests from JSON report
        tests = data.get("tests", [])

        for test in tests:
            test_case = self._parse_test_case(test)
            self.test_cases.append(test_case)
            summary.test_cases.append(test_case)

            # Update summary statistics
            summary.total += 1
            summary.total_time += test_case.execution_time

            if test_case.status == TestStatus.PASSED:
                summary.passed += 1
            elif test_case.status == TestStatus.FAILED:
                summary.failed += 1
            elif test_case.status == TestStatus.SKIPPED:
                summary.skipped += 1
            elif test_case.status == TestStatus.ERROR:
                summary.errors += 1

            # Aggregate by test suite
            suite = test_case.test_suite
            if suite not in summary.by_suite:
                summary.by_suite[suite] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "errors": 0,
                }

            summary.by_suite[suite]["total"] += 1
            summary.by_suite[suite][test_case.status.value] += 1

        return summary

    def _parse_test_case(self, test_data: dict) -> TestCase:
        """Parse individual test case from pytest JSON.

        Args:
            test_data: Test data from pytest JSON

        Returns:
            TestCase entity
        """
        # Extract test name
        nodeid = test_data.get("nodeid", "")
        name = nodeid

        # Determine test suite from path
        test_suite = "unknown"
        if "/unit/" in nodeid:
            test_suite = "unit"
        elif "/e2e/" in nodeid:
            test_suite = "e2e"
        elif "/integration/" in nodeid:
            test_suite = "integration"
        elif "/contract/" in nodeid:
            test_suite = "contract"

        # Extract status
        outcome = test_data.get("outcome", "")
        status = TestStatus.PASSED
        if outcome == "passed":
            status = TestStatus.PASSED
        elif outcome == "failed":
            status = TestStatus.FAILED
        elif outcome == "skipped":
            status = TestStatus.SKIPPED
        elif outcome == "error":
            status = TestStatus.ERROR

        # Extract execution time
        call = test_data.get("call", {})
        execution_time = call.get("duration", 0.0)

        # Extract failure message
        failure_message = None
        if status in [TestStatus.FAILED, TestStatus.ERROR]:
            call_info = test_data.get("call", {})
            longrepr = call_info.get("longrepr", "")
            if longrepr:
                failure_message = str(longrepr)

        return TestCase(
            name=name,
            status=status,
            execution_time=execution_time,
            test_suite=test_suite,
            failure_message=failure_message,
        )

    def get_slowest_tests(self, n: int = 10) -> List[TestCase]:
        """Get N slowest test cases.

        Args:
            n: Number of slowest tests to return

        Returns:
            List of slowest test cases
        """
        return sorted(self.test_cases, key=lambda t: t.execution_time, reverse=True)[:n]

    def get_failing_tests(self) -> List[TestCase]:
        """Get all failing test cases.

        Returns:
            List of failed and errored test cases
        """
        return [
            tc
            for tc in self.test_cases
            if tc.status in [TestStatus.FAILED, TestStatus.ERROR]
        ]

    def print_summary(self, summary: TestSummary):
        """Print test summary to console.

        Args:
            summary: TestSummary to print
        """
        print(f"\n{'=' * 60}")
        print("Test Summary")
        print(f"{'=' * 60}")
        print(f"Total: {summary.total}")
        print(f"Passed: {summary.passed}")
        print(f"Failed: {summary.failed}")
        print(f"Skipped: {summary.skipped}")
        print(f"Errors: {summary.errors}")
        print(f"Total Time: {summary.total_time:.2f}s")
        print()

        print("By Test Suite:")
        for suite, stats in summary.by_suite.items():
            print(
                f"  {suite}: {stats['passed']}/{stats['total']} passed "
                f"({stats['failed']} failed, {stats['skipped']} skipped, "
                f"{stats['errors']} errors)"
            )

        print(f"{'=' * 60}\n")
