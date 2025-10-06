"""Pytest plugin for TDD compliance support.

This plugin provides helpers and markers for contract tests to support
TDD validation workflows.
"""

import pytest
from pathlib import Path
from typing import List, Set


# Track contract test files
_contract_tests: Set[str] = set()


def pytest_configure(config):
    """Register markers and configure plugin."""
    config.addinivalue_line(
        "markers",
        "contract: mark test as a contract test that must fail before implementation"
    )
    config.addinivalue_line(
        "markers",
        "tdd_compliant: mark test as following TDD workflow"
    )
    config.addinivalue_line(
        "markers",
        "implementation_commit(commit_sha): mark the commit where implementation was added"
    )


def pytest_collection_modifyitems(session, config, items):
    """Track contract tests during collection."""
    global _contract_tests

    for item in items:
        # Check if test is in contract directory
        test_path = Path(item.fspath)
        if "contract" in test_path.parts:
            _contract_tests.add(str(test_path))

        # Check if explicitly marked as contract
        if item.get_closest_marker("contract"):
            _contract_tests.add(str(test_path))


def is_contract_test(test_path: str) -> bool:
    """Check if a test file is a contract test."""
    path = Path(test_path)

    # Check common patterns
    if "contract" in path.name:
        return True

    if "contract" in path.parts:
        return True

    # Check if tracked
    if str(path) in _contract_tests:
        return True

    return False


def get_contract_tests() -> List[str]:
    """Get list of all contract test files."""
    return list(_contract_tests)


class ContractTestHelper:
    """Helper class for contract test validation."""

    @staticmethod
    def mark_as_contract(func):
        """Decorator to mark a test as a contract test."""
        return pytest.mark.contract(func)

    @staticmethod
    def mark_implementation_commit(commit_sha: str):
        """Mark the commit where implementation was added."""
        return pytest.mark.implementation_commit(commit_sha)

    @staticmethod
    def assert_will_fail(test_func):
        """Decorator to verify test fails without implementation."""
        @pytest.mark.contract
        def wrapper(*args, **kwargs):
            # This is a placeholder - actual validation happens in standalone script
            return test_func(*args, **kwargs)
        return wrapper


# Export helper instance
contract_test = ContractTestHelper()


def pytest_report_header(config):
    """Add TDD compliance info to test report header."""
    contract_count = len(_contract_tests)
    if contract_count > 0:
        return [
            f"TDD Contract Tests: {contract_count} files tracked",
            "Run scripts/validate_tdd_compliance.py to verify TDD workflow"
        ]


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add TDD reminders to terminal summary."""
    if _contract_tests and config.option.verbose > 0:
        terminalreporter.section("TDD Compliance Reminder")
        terminalreporter.write_line(
            "Contract tests detected. Remember to:",
            yellow=True
        )
        terminalreporter.write_line(
            "  1. Write failing contract tests first",
            yellow=True
        )
        terminalreporter.write_line(
            "  2. Commit the failing tests",
            yellow=True
        )
        terminalreporter.write_line(
            "  3. Implement features to make tests pass",
            yellow=True
        )
        terminalreporter.write_line(
            "  4. Run validate_tdd_compliance.py in CI",
            yellow=True
        )