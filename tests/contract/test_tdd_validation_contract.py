"""Contract tests for TDD compliance validation per TDD-001.

These tests define the expected behavior of the TDD validation system.
They must fail initially and pass once the validation is implemented.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import git
from datetime import datetime


class TestTDDValidationContract:
    """Contract tests for TDD-001 TDD compliance validation behavior."""

    def test_TDD001_contract_test_discovery(self):
        """Verify system can discover contract test files."""
        # This should fail until validator implements find_contract_tests
        from scripts.validate_tdd_compliance import find_contract_tests
        from pathlib import Path

        # Mock file system with contract tests
        with patch('pathlib.Path.glob') as mock_glob, \
             patch('pathlib.Path.rglob') as mock_rglob:
            mock_glob.return_value = [
                Path("tests/contract/test_foo_contract.py"),
                Path("tests/contract/test_bar_contract.py"),
            ]
            mock_rglob.return_value = [
                Path("tests/test_other_contract.py")
            ]

            contract_tests = find_contract_tests(".")

            assert len(contract_tests) >= 2
            assert all("contract" in str(test) for test in contract_tests)

    def test_TDD001_violation_detection(self):
        """Verify system detects when contract tests never failed."""
        from scripts.validate_tdd_compliance import check_test_history
        from pathlib import Path

        # Mock git repo with test that was always passing
        mock_repo = Mock(spec=git.Repo)
        mock_commit = Mock()
        mock_commit.hexsha = "abc123"
        mock_commit.committed_datetime = datetime.now()
        mock_commit.message = "Add implementation"

        # Mock the helper functions
        with patch('scripts.validate_tdd_compliance.find_test_introduction_commit') as mock_find, \
             patch('scripts.validate_tdd_compliance.run_test_at_commit') as mock_run:
            mock_find.return_value = mock_commit
            # Test passed initially (TDD violation)
            mock_run.return_value = {"failed": False, "passed": True, "error": False}

            result = check_test_history(
                repo=mock_repo,
                test_file="tests/contract/test_feature_contract.py",
                implementation_commit=mock_commit
            )

            # Should detect violation when test never failed
            assert result.compliant is False
            assert result.violation_type == "never_failed"
            assert result.test_file == "tests/contract/test_feature_contract.py"

    def test_TDD001_compliant_workflow_detection(self):
        """Verify system recognizes proper TDD workflow."""
        from scripts.validate_tdd_compliance import check_test_history

        # Mock git history showing test failed then passed
        mock_repo = Mock(spec=git.Repo)

        # Create mock commits
        commit1 = Mock(hexsha="aaa111", message="Add failing contract test")
        commit2 = Mock(hexsha="bbb222", message="Implement feature")

        # Mock helper functions
        with patch('scripts.validate_tdd_compliance.find_test_introduction_commit') as mock_find, \
             patch('scripts.validate_tdd_compliance.run_test_at_commit') as mock_run:
            mock_find.return_value = commit1
            # Test failed initially (proper TDD)
            mock_run.return_value = {"failed": True, "passed": False, "error": False}

            result = check_test_history(
                repo=mock_repo,
                test_file="tests/contract/test_good_contract.py",
                implementation_commit=commit2
            )

            assert result.compliant is True
            assert result.initial_state == "failing"
            assert result.implementation_commit == "bbb222"

    def test_TDD001_ci_integration(self):
        """Verify script can run in CI mode and fail build on violations."""
        # This should fail until script implements main with CI mode
        from scripts.validate_tdd_compliance import main

        # Mock violations found
        with patch('scripts.validate_tdd_compliance.find_violations') as mock_find:
            mock_find.return_value = [
                {"test": "test_bad_contract.py", "reason": "never failed"}
            ]

            # Run in CI mode
            with patch('sys.argv', ['validate_tdd_compliance.py', '--fail-on-violations']):
                with pytest.raises(SystemExit) as exc_info:
                    main()

                # Should exit with non-zero code
                assert exc_info.value.code != 0

    def test_TDD001_report_generation(self):
        """Verify system generates readable compliance report."""
        from scripts.validate_tdd_compliance import generate_report, TestHistory

        violations = [
            TestHistory(
                test_file="tests/contract/test_feature_contract.py",
                initial_state="passing",
                implementation_commit="abc123",
                compliant=False,
                violation_type="never_failed",
                details="Contract test was already passing before implementation"
            )
        ]

        compliant_tests = [
            TestHistory(
                test_file="tests/contract/test_good_contract.py",
                initial_state="failing",
                implementation_commit="def456",
                compliant=True
            )
        ]

        report = generate_report(violations, compliant_tests)

        # Report should contain both violations and compliant tests
        assert "TDD Compliance Report" in report
        assert "VIOLATIONS FOUND: 1" in report
        assert "test_feature_contract.py" in report
        assert "never_failed" in report
        assert "COMPLIANT TESTS:" in report  # Note: dynamic number
        assert "test_good_contract.py" in report