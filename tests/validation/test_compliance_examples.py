"""Example test file demonstrating all compliance features.

This file shows both good and bad practices for:
- Error messages
- Test coverage
- TDD compliance
"""

import pytest
from tests.fixtures.low_coverage_module import barely_tested_function


class TestGoodErrorMessages:
    """Examples of tests with well-structured error messages."""

    @pytest.mark.good_errors
    def test_user_authentication_with_good_error(self):
        """Example of a test with a good three-part error message."""
        username = "testuser"
        password = "wrongpass"
        expected_status = 200
        actual_status = 401

        assert actual_status == expected_status, (
            f"Test test_user_authentication failed: Authentication error.\n"
            f"Expected status {expected_status} for user '{username}' but got {actual_status} (Unauthorized).\n"
            f"Check credentials in test config and ensure the auth service is running correctly."
        )

    @pytest.mark.good_errors
    def test_database_connection_with_context(self):
        """Example showing good context in error messages."""
        db_host = "localhost"
        db_port = 5432
        connection_timeout = 30

        # Simulate connection failure
        connected = False

        assert connected, (
            f"Test test_database_connection failed: Connection timeout.\n"
            f"Could not connect to PostgreSQL at {db_host}:{db_port} after {connection_timeout}s.\n"
            f"Verify PostgreSQL is running, check firewall rules, and ensure connection parameters are correct."
        )


class TestBadErrorMessages:
    """Examples of tests with poor error messages (for demonstration)."""

    def test_with_minimal_error(self):
        """Bad example: Too brief error message."""
        result = False
        assert result, "Test failed"  # Bad: No context, no why, no action

    def test_with_no_action(self):
        """Bad example: Missing suggested action."""
        expected = 42
        actual = 41
        assert actual == expected, (
            f"Value mismatch. Expected {expected} but got {actual}."
            # Bad: Has what and why, but no action
        )

    def test_with_vague_message(self):
        """Bad example: Vague error with no specifics."""
        assert False, "Something went wrong with the thing"  # Bad: Vague what, no why, no action


class TestCoverageExamples:
    """Tests that use the low coverage module to trigger warnings."""

    def test_barely_tested_function_single_path(self):
        """This only tests one path, resulting in ~25% coverage."""
        result = barely_tested_function(1)
        assert result == 1
        # Not testing x=2,3,4, or else cases - will trigger coverage warning

    @pytest.mark.skip(reason="Intentionally skipped to show low coverage")
    def test_other_paths(self):
        """If this ran, it would improve coverage."""
        assert barely_tested_function(2) == 4
        assert barely_tested_function(3) == 9


class TestContractExamples:
    """Examples related to contract testing."""

    @pytest.mark.contract
    def test_example_contract(self):
        """Example of a contract test properly marked."""
        # In real TDD, this would have been written to fail first
        # Then implementation would be added to make it pass
        feature_implemented = True  # Simulating implemented feature
        assert feature_implemented, (
            "Test test_example_contract failed: Feature not implemented.\n"
            "Expected feature flag to be True but got False.\n"
            "Implement the feature in the main module before this test can pass."
        )


def helper_function_with_branches(value):
    """Helper function with multiple branches for coverage testing."""
    if value < 0:
        return "negative"
    elif value == 0:
        return "zero"
    elif value < 10:
        return "small"
    elif value < 100:
        return "medium"
    else:
        return "large"


class TestPartialCoverage:
    """Tests that only cover some branches."""

    def test_only_positive_small_values(self):
        """Only tests one branch, triggering coverage warnings."""
        assert helper_function_with_branches(5) == "small"
        # Not testing negative, zero, medium, or large branches


# Intentionally untested class to demonstrate coverage warnings
class CompletelyUntestedExample:
    """This class is never tested and should trigger warnings."""

    def untested_method(self):
        """This method is never called in tests."""
        return "This should trigger a coverage warning"

    def another_untested_method(self, x, y):
        """Also never called."""
        return x + y


if __name__ == "__main__":
    # Quick manual test
    print("Running manual validation...")
    try:
        test_user_authentication_with_good_error(None)
    except AssertionError as e:
        print("Good error example:")
        print(str(e))
        print()

    try:
        test_with_minimal_error(None)
    except AssertionError as e:
        print("Bad error example:")
        print(str(e))