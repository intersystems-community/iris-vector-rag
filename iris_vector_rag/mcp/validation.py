"""
MCP Parameter Validation Module.

Provides validation exceptions and helper functions for validating
MCP tool parameters against schemas.

Feature: Complete MCP Tools Implementation
Branch: 043-complete-mcp-tools
"""

from typing import Any, Optional


class ValidationError(Exception):
    """Raised when parameter validation fails."""

    def __init__(self, field: str, value: Any, message: str):
        self.field = field
        self.value = value
        self.message = message
        super().__init__(f"Validation error on field '{field}': {message}")


class MCPError(Exception):
    """Base exception for MCP-related errors."""

    def __init__(self, code: str, message: str, data: Optional[dict] = None):
        self.code = code
        self.message = message
        self.data = data or {}
        super().__init__(message)


def validate_query_length(query: str, min_length: int = 1, max_length: int = 8000) -> None:
    """
    Validate query string length.

    Args:
        query: Query string to validate
        min_length: Minimum allowed length
        max_length: Maximum allowed length

    Raises:
        ValidationError: If query length is invalid
    """
    if not isinstance(query, str):
        raise ValidationError('query', query, "Query must be a string")

    if len(query) < min_length:
        raise ValidationError('query', query,
                             f"Query must be at least {min_length} character(s)")

    if len(query) > max_length:
        raise ValidationError('query', query,
                             f"Query must be at most {max_length} characters")


def validate_top_k(top_k: int, min_value: int = 1, max_value: int = 50) -> None:
    """
    Validate top_k parameter.

    Args:
        top_k: Number of documents to retrieve
        min_value: Minimum allowed value
        max_value: Maximum allowed value

    Raises:
        ValidationError: If top_k is invalid
    """
    if not isinstance(top_k, int):
        raise ValidationError('top_k', top_k, "top_k must be an integer")

    if top_k < min_value:
        raise ValidationError('top_k', top_k,
                             f"top_k must be at least {min_value}")

    if top_k > max_value:
        raise ValidationError('top_k', top_k,
                             f"top_k must be at most {max_value}")


def validate_confidence_threshold(threshold: float) -> None:
    """
    Validate confidence threshold (0.0-1.0).

    Args:
        threshold: Confidence threshold value

    Raises:
        ValidationError: If threshold is invalid
    """
    if not isinstance(threshold, (int, float)):
        raise ValidationError('confidence_threshold', threshold,
                             "confidence_threshold must be a number")

    if threshold < 0.0 or threshold > 1.0:
        raise ValidationError('confidence_threshold', threshold,
                             "confidence_threshold must be between 0.0 and 1.0")


def validate_enum_value(field: str, value: Any, allowed_values: list) -> None:
    """
    Validate that value is in allowed enum values.

    Args:
        field: Field name
        value: Value to validate
        allowed_values: List of allowed values

    Raises:
        ValidationError: If value is not in allowed values
    """
    if value not in allowed_values:
        allowed_str = ', '.join(str(v) for v in allowed_values)
        raise ValidationError(field, value,
                             f"{field} must be one of: {allowed_str}")


def validate_interaction_threshold(threshold: float) -> None:
    """
    Validate interaction threshold (0.0-1.0) for PyLateColBERT.

    Args:
        threshold: Interaction threshold value

    Raises:
        ValidationError: If threshold is invalid
    """
    if not isinstance(threshold, (int, float)):
        raise ValidationError('interaction_threshold', threshold,
                             "interaction_threshold must be a number")

    if threshold < 0.0 or threshold > 1.0:
        raise ValidationError('interaction_threshold', threshold,
                             "interaction_threshold must be between 0.0 and 1.0")


def validate_graph_traversal_depth(depth: int) -> None:
    """
    Validate graph traversal depth (1-5).

    Args:
        depth: Traversal depth

    Raises:
        ValidationError: If depth is invalid
    """
    if not isinstance(depth, int):
        raise ValidationError('graph_traversal_depth', depth,
                             "graph_traversal_depth must be an integer")

    if depth < 1 or depth > 5:
        raise ValidationError('graph_traversal_depth', depth,
                             "graph_traversal_depth must be between 1 and 5")


def validate_rrf_k(k_value: int) -> None:
    """
    Validate RRF k parameter (1-100).

    Args:
        k_value: RRF k parameter

    Raises:
        ValidationError: If k value is invalid
    """
    if not isinstance(k_value, int):
        raise ValidationError('rrf_k', k_value,
                             "rrf_k must be an integer")

    if k_value < 1 or k_value > 100:
        raise ValidationError('rrf_k', k_value,
                             "rrf_k must be between 1 and 100")


def validate_compression_ratio(ratio: float) -> None:
    """
    Validate compression ratio (0.1-1.0) for PyLateColBERT.

    Args:
        ratio: Compression ratio

    Raises:
        ValidationError: If ratio is invalid
    """
    if not isinstance(ratio, (int, float)):
        raise ValidationError('compression_ratio', ratio,
                             "compression_ratio must be a number")

    if ratio < 0.1 or ratio > 1.0:
        raise ValidationError('compression_ratio', ratio,
                             "compression_ratio must be between 0.1 and 1.0")
