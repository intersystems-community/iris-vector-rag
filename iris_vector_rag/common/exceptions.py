"""
Custom exceptions for IRIS connection management.

This module defines the exception hierarchy for the unified IRIS connection module,
providing clear, actionable error messages for connection failures.
"""


class ValidationError(ValueError):
    """
    Raised when connection parameters fail validation before connection attempt.

    This exception is raised during parameter validation (before any connection
    attempt), providing immediate feedback on invalid configuration.

    Attributes:
        parameter_name: Name of the parameter that failed validation
        invalid_value: The value that was rejected
        valid_range: Description of acceptable values
        message: Full error message with actionable guidance

    Example:
        >>> raise ValidationError(
        ...     parameter_name="port",
        ...     invalid_value=99999,
        ...     valid_range="1-65535",
        ...     message="Invalid port 99999: must be between 1-65535"
        ... )
    """

    def __init__(
        self, parameter_name: str, invalid_value, valid_range: str, message: str
    ):
        self.parameter_name = parameter_name
        self.invalid_value = invalid_value
        self.valid_range = valid_range
        super().__init__(message)


class ConnectionLimitError(RuntimeError):
    """
    Raised when IRIS Community Edition connection limit is reached.

    This exception is raised when attempting to create more connections than
    allowed by the detected IRIS edition (typically 1 for Community Edition).

    Attributes:
        current_limit: Maximum connections allowed for detected edition
        suggested_actions: List of actionable suggestions for resolving the issue
        message: Full error message with context and suggestions

    Example:
        >>> raise ConnectionLimitError(
        ...     current_limit=1,
        ...     suggested_actions=[
        ...         "Use connection queuing with IRISConnectionPool",
        ...         "Run tests serially with pytest -n 0",
        ...         "Set IRIS_BACKEND_MODE=community explicitly"
        ...     ],
        ...     message="IRIS Community Edition connection limit (1) reached. Consider: ..."
        ... )
    """

    def __init__(self, current_limit: int, suggested_actions: list[str], message: str):
        self.current_limit = current_limit
        self.suggested_actions = suggested_actions
        super().__init__(message)

    def __str__(self) -> str:
        """Format error message with suggested actions."""
        base_message = super().__str__()
        if self.suggested_actions:
            actions = "\n  - ".join(self.suggested_actions)
            return f"{base_message}\n\nSuggested actions:\n  - {actions}"
        return base_message
