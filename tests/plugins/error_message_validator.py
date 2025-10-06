"""Pytest plugin for validating error message quality.

This plugin ensures test failure messages follow the three-part structure:
1. What failed
2. Why it failed
3. Suggested action
"""

import re
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from unittest.mock import patch


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of error message validation."""
    has_what: bool
    has_why: bool
    has_action: bool
    message: str

    @property
    def is_valid(self) -> bool:
        """Check if message meets all criteria."""
        return self.has_what and self.has_why and self.has_action


# Default validation patterns
DEFAULT_PATTERNS = {
    "what_pattern": r"(?:test.*failed|failed|error|exception|assertion)[\s:]+",
    "why_pattern": r"(?:expected|got|but|because|due to|caused by|reason)",
    "action_pattern": r"(?:check|verify|ensure|try|fix|make sure|consider|should|must)"
}

# Global configuration
_validation_config = DEFAULT_PATTERNS.copy()


def configure_validation(config: Dict[str, str]) -> None:
    """Configure validation patterns."""
    global _validation_config
    _validation_config.update(config)


def validate_error_message(message: str) -> ValidationResult:
    """Validate that error message has required components."""
    message_lower = message.lower()

    # Check for "what failed"
    has_what = bool(re.search(_validation_config["what_pattern"], message_lower))

    # Check for "why it failed"
    has_why = bool(re.search(_validation_config["why_pattern"], message_lower))

    # Check for "suggested action"
    has_action = bool(re.search(_validation_config["action_pattern"], message_lower))

    return ValidationResult(
        has_what=has_what,
        has_why=has_why,
        has_action=has_action,
        message=message
    )


def get_improvement_suggestions(message: str) -> str:
    """Get suggestions for improving error message."""
    result = validate_error_message(message)
    suggestions = []

    if not result.has_what:
        suggestions.append("Add what failed (e.g., 'Test X failed: ...')")

    if not result.has_why:
        suggestions.append("Explain why it failed (e.g., 'Expected X but got Y')")

    if not result.has_action:
        suggestions.append("Add what to do to fix it (e.g., 'Check that...' or 'Ensure...')")

    if not suggestions:
        return "Error message structure looks good!"

    return "Error message needs improvement:\n" + "\n".join(f"  - {s}" for s in suggestions)


def has_sufficient_context(message: str) -> bool:
    """Check if message provides enough context."""
    # Look for specific details like values, names, or paths
    context_patterns = [
        r"'[^']+'",  # Quoted values
        r'"[^"]+"',  # Double quoted values
        r"\b\d+\b",  # Numbers
        r"\b[A-Z][a-zA-Z]+(?:[A-Z][a-zA-Z]+)*\b",  # CamelCase names
        r"\b\w+\.\w+\b",  # File names or dotted paths
        r"/[\w/]+",  # File paths
    ]

    # Count context elements
    context_count = 0
    for pattern in context_patterns:
        matches = re.findall(pattern, message)
        context_count += len(matches)

    # Require at least 2 pieces of context
    return context_count >= 2


def format_error_suggestion(node_name: str, original_error: str, suggestions: str) -> str:
    """Format error validation feedback."""
    max_length = 1000  # From clarification

    formatted = f"""
════════════════════════════════════════════════════════════════
ERROR MESSAGE VALIDATION WARNING

Test: {node_name}
Original Error: {original_error[:200]}{'...' if len(original_error) > 200 else ''}

{suggestions}

Example of a good error message:
  "Test test_user_login failed: Authentication error.
   Expected successful login for user 'testuser' but got 401 Unauthorized.
   Check credentials in test_data.json and ensure test API is running."
════════════════════════════════════════════════════════════════
"""

    # Truncate if too long
    if len(formatted) > max_length:
        formatted = formatted[:max_length - 3] + "..."

    return formatted


def pytest_exception_interact(node, call, report):
    """Validate error messages when tests fail."""
    try:
        # Only check test failures (not setup/teardown)
        if call.when != "call":
            return

        # Get the error message
        error_message = ""
        if hasattr(report.longrepr, "reprcrash"):
            error_message = report.longrepr.reprcrash.message
        elif hasattr(call.excinfo, "value"):
            error_message = str(call.excinfo.value)

        if not error_message:
            return

        # Validate the error message
        result = validate_error_message(error_message)

        if not result.is_valid:
            # Get improvement suggestions
            suggestions = get_improvement_suggestions(error_message)

            # Log validation warning
            warning_msg = format_error_suggestion(
                node.name,
                error_message,
                suggestions
            )

            logger.warning(warning_msg)

            # Also print to terminal if verbose
            if hasattr(node.config.option, "verbose") and node.config.option.verbose > 0:
                print(warning_msg)

    except Exception as e:
        # Don't fail tests if validation fails
        logger.debug(f"Error message validation failed: {e}")


def pytest_configure(config):
    """Register plugin with pytest."""
    config.addinivalue_line(
        "markers",
        "good_errors: mark test as having exemplary error messages"
    )


def pytest_runtest_protocol(item, nextitem):
    """Check if test is marked for good errors and skip validation."""
    if item.get_closest_marker("good_errors"):
        # Temporarily disable validation for this test
        with patch.object(logger, 'warning'):
            return None  # Continue with default protocol
    return None  # Continue with default protocol