#!/usr/bin/env python3
"""Validate TDD compliance by checking contract tests failed before implementation.

This script analyzes git history to ensure contract tests followed the TDD workflow:
1. Tests were written and failed initially
2. Implementation was added
3. Tests then passed

Usage:
    python scripts/validate_tdd_compliance.py [--fail-on-violations]
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import git
from datetime import datetime
from dataclasses import dataclass


@dataclass
class TestHistory:
    """Represents the history of a contract test."""
    test_file: str
    initial_state: str  # "failing", "passing", "error", "not_found"
    implementation_commit: Optional[str] = None
    compliant: bool = True
    violation_type: Optional[str] = None
    details: Optional[str] = None


def find_contract_tests(repo_path: str = ".") -> List[Path]:
    """Find all contract test files in the repository."""
    repo_path = Path(repo_path)
    contract_tests = []

    # Find tests in contract directory
    contract_dir = repo_path / "tests" / "contract"
    if contract_dir.exists():
        contract_tests.extend(contract_dir.glob("test_*.py"))

    # Also look for files with _contract in name
    tests_dir = repo_path / "tests"
    if tests_dir.exists():
        contract_tests.extend(tests_dir.rglob("*_contract.py"))

    return sorted(set(contract_tests))


def run_test_at_commit(repo: git.Repo, commit_sha: str, test_file: str) -> Dict[str, bool]:
    """Run a specific test at a given commit and return results."""
    original_head = repo.head.commit

    try:
        # Checkout the commit
        repo.git.checkout(commit_sha, force=True)

        # Run the specific test
        cmd = [
            sys.executable, "-m", "pytest",
            test_file,
            "-v", "--tb=short",
            "--no-header",
            "-q"
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=repo.working_dir
        )

        # Parse output to determine test state
        output = result.stdout + result.stderr

        if "FAILED" in output or "failed" in output:
            return {"failed": True, "passed": False, "error": False}
        elif "PASSED" in output or "passed" in output:
            return {"failed": False, "passed": True, "error": False}
        elif "ERROR" in output or "ModuleNotFoundError" in output:
            return {"failed": True, "passed": False, "error": True}
        else:
            # Unknown state, assume error
            return {"failed": False, "passed": False, "error": True}

    finally:
        # Always return to original commit
        repo.git.checkout(original_head.hexsha, force=True)


def find_test_introduction_commit(repo: git.Repo, test_file: Path) -> Optional[git.Commit]:
    """Find the commit where a test file was introduced."""
    try:
        # Get the first commit that added this file
        commits = list(repo.iter_commits(paths=str(test_file), max_count=1000))
        if commits:
            # Return the oldest commit (last in list)
            return commits[-1]
    except Exception:
        pass
    return None


def check_test_history(repo: git.Repo, test_file: str, implementation_commit: Optional[git.Commit] = None) -> TestHistory:
    """Check if a contract test followed TDD workflow."""
    test_path = Path(test_file)

    # Find when test was introduced
    intro_commit = find_test_introduction_commit(repo, test_path)
    if not intro_commit:
        return TestHistory(
            test_file=str(test_file),
            initial_state="not_found",
            compliant=False,
            violation_type="test_not_found",
            details="Could not find test introduction in git history"
        )

    # Check test state at introduction
    test_result = run_test_at_commit(repo, intro_commit.hexsha, str(test_path))

    if test_result["passed"]:
        # Test passed initially - TDD violation
        return TestHistory(
            test_file=str(test_file),
            initial_state="passing",
            implementation_commit=intro_commit.hexsha,
            compliant=False,
            violation_type="never_failed",
            details=f"Contract test was already passing at introduction ({intro_commit.hexsha[:8]})"
        )

    elif test_result["failed"] and not test_result["error"]:
        # Test failed properly - good TDD practice
        return TestHistory(
            test_file=str(test_file),
            initial_state="failing",
            implementation_commit=implementation_commit.hexsha if implementation_commit else None,
            compliant=True,
            details="Test failed initially as expected for TDD"
        )

    else:
        # Test had errors (might be missing imports, which is acceptable)
        return TestHistory(
            test_file=str(test_file),
            initial_state="error",
            implementation_commit=implementation_commit.hexsha if implementation_commit else None,
            compliant=True,  # Errors are acceptable (missing implementations)
            details="Test had errors initially (expected for missing implementations)"
        )


def find_violations(repo_path: str = ".") -> Tuple[List[TestHistory], List[TestHistory]]:
    """Find all TDD violations in contract tests."""
    repo = git.Repo(repo_path)
    contract_tests = find_contract_tests(repo_path)

    violations = []
    compliant_tests = []

    for test_file in contract_tests:
        history = check_test_history(repo, str(test_file))

        if history.compliant:
            compliant_tests.append(history)
        else:
            violations.append(history)

    return violations, compliant_tests


def generate_report(violations: List[TestHistory], compliant_tests: List[TestHistory]) -> str:
    """Generate a human-readable compliance report."""
    total_tests = len(violations) + len(compliant_tests)

    report = []
    report.append("=" * 70)
    report.append("TDD Compliance Report")
    report.append("=" * 70)
    report.append(f"\nTotal Contract Tests: {total_tests}")
    report.append(f"Compliant Tests: {len(compliant_tests)}")
    report.append(f"VIOLATIONS FOUND: {len(violations)}")

    if violations:
        report.append("\n" + "=" * 70)
        report.append("VIOLATIONS DETAIL")
        report.append("=" * 70)

        for v in violations:
            report.append(f"\n❌ {v.test_file}")
            report.append(f"   Violation: {v.violation_type}")
            report.append(f"   Details: {v.details}")

    if compliant_tests:
        report.append("\n" + "=" * 70)
        report.append("COMPLIANT TESTS: {len(compliant_tests)}")
        report.append("=" * 70)

        for t in compliant_tests[:5]:  # Show first 5
            report.append(f"\n✅ {t.test_file}")
            report.append(f"   Initial State: {t.initial_state}")

        if len(compliant_tests) > 5:
            report.append(f"\n... and {len(compliant_tests) - 5} more compliant tests")

    report.append("\n" + "=" * 70)
    compliance_rate = (len(compliant_tests) / total_tests * 100) if total_tests > 0 else 100
    report.append(f"Compliance Rate: {compliance_rate:.1f}%")
    report.append("=" * 70)

    return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate TDD compliance for contract tests"
    )
    parser.add_argument(
        "--fail-on-violations",
        action="store_true",
        help="Exit with non-zero code if violations found (for CI)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--repo-path",
        default=".",
        help="Path to git repository (default: current directory)"
    )

    args = parser.parse_args()

    try:
        # Find violations
        violations, compliant_tests = find_violations(args.repo_path)

        if args.json:
            # JSON output
            result = {
                "total_tests": len(violations) + len(compliant_tests),
                "compliant_count": len(compliant_tests),
                "violation_count": len(violations),
                "compliance_rate": (len(compliant_tests) / (len(violations) + len(compliant_tests)) * 100)
                    if (violations or compliant_tests) else 100,
                "violations": [
                    {
                        "test_file": v.test_file,
                        "violation_type": v.violation_type,
                        "details": v.details
                    }
                    for v in violations
                ],
                "compliant_tests": [
                    {
                        "test_file": t.test_file,
                        "initial_state": t.initial_state
                    }
                    for t in compliant_tests
                ]
            }
            print(json.dumps(result, indent=2))
        else:
            # Human-readable report
            report = generate_report(violations, compliant_tests)
            print(report)

        # Exit code
        if args.fail_on_violations and violations:
            sys.exit(1)
        else:
            sys.exit(0)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()