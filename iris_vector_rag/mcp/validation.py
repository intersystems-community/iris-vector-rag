"""
MCP Parameter Validation Module.

Provides the validation exceptions. Range and type checks live in one place,
``tool_schemas.validate_params``, driven by the JSON schema file.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

from typing import Any


class ValidationError(Exception):
    """Raised when parameter validation fails."""

    def __init__(self, field: str, value: Any, message: str):
        self.field = field
        self.value = value
        self.message = message
        super().__init__(f"Validation error on field '{field}': {message}")
