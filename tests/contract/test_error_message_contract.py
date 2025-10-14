"""Contract tests for error message validation per ERR-001.

These tests define the expected behavior of the error message validator.
They must fail initially and pass once the plugin is implemented.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch


class TestErrorMessageContract:
    """Contract tests for ERR-001 error message validation behavior."""

    def test_ERR001_three_part_detection(self):
        """Verify three-part structure is detected correctly."""
        good_message = """Test failed: Database connection error.
Expected connection to localhost:5432 but got timeout.
Check PostgreSQL is running and accepting connections."""

        bad_message = "Test failed: Something went wrong"

        # This should fail until plugin implements validate_error_message
        from tests.plugins.error_message_validator import validate_error_message

        # Good message should pass validation
        result = validate_error_message(good_message)
        assert result.has_what is True
        assert result.has_why is True
        assert result.has_action is True
        assert result.is_valid is True

        # Bad message should fail validation
        result = validate_error_message(bad_message)
        assert result.has_what is True  # Has "what"
        assert result.has_why is False  # Missing "why"
        assert result.has_action is False  # Missing "action"
        assert result.is_valid is False

    def test_ERR001_missing_components_feedback(self):
        """Verify feedback identifies which components are missing."""
        incomplete_message = """Test failed: Configuration error.
The config file is missing required fields."""
        # Missing action component

        # This should fail until plugin implements get_improvement_suggestions
        from tests.plugins.error_message_validator import get_improvement_suggestions

        suggestions = get_improvement_suggestions(incomplete_message)

        # Should identify missing action (case-insensitive check)
        suggestions_lower = suggestions.lower()
        assert "what to do" in suggestions_lower or "fix it" in suggestions_lower or "check" in suggestions_lower

    def test_ERR001_context_validation(self):
        """Verify error messages include relevant context."""
        message_with_context = """Test test_user_login failed: Authentication error.
Expected successful login for user 'testuser' but got 401 Unauthorized.
Verify credentials in test_data.json and ensure test API is running."""

        message_without_context = """error"""

        # This should fail until plugin implements has_sufficient_context
        from tests.plugins.error_message_validator import has_sufficient_context

        assert has_sufficient_context(message_with_context) is True
        assert has_sufficient_context(message_without_context) is False

    def test_ERR001_pytest_hook_integration(self):
        """Verify plugin integrates with pytest_exception_interact hook."""
        # Mock pytest components
        mock_node = Mock()
        mock_node.name = "test_example"
        mock_node.config = Mock()
        mock_node.config.option = Mock()
        mock_node.config.option.verbose = 0

        mock_call = Mock()
        mock_call.when = "call"  # Must be "call" to trigger validation
        mock_call.excinfo = Mock()
        mock_call.excinfo.value = AssertionError("fail")  # Very minimal message

        mock_report = Mock()
        mock_report.longrepr = Mock()
        mock_report.longrepr.reprcrash = Mock(message="fail")

        # This should fail until plugin implements pytest_exception_interact
        from tests.plugins.error_message_validator import pytest_exception_interact

        # Should not raise but should log validation warnings
        with patch('tests.plugins.error_message_validator.logger') as mock_logger:
            pytest_exception_interact(mock_node, mock_call, mock_report)

            # Should have logged a warning about poor error message
            mock_logger.warning.assert_called()

    def test_ERR001_configurable_patterns(self):
        """Verify validation patterns can be configured."""
        # This should fail until plugin implements configure_validation
        from tests.plugins.error_message_validator import configure_validation

        # Configure custom patterns
        config = {
            "what_pattern": r"(?:failed|error|exception):",
            "why_pattern": r"(?:expected|got|because|due to)",
            "action_pattern": r"(?:check|verify|ensure|try|fix)"
        }

        configure_validation(config)

        # Test with custom patterns
        custom_message = """Operation failed: Network timeout.
Request took longer than 30s due to high latency.
Try increasing timeout or check network connection."""

        from tests.plugins.error_message_validator import validate_error_message
        result = validate_error_message(custom_message)
        assert result.is_valid is True