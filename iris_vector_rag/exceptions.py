"""
Exception classes for iris-vector-rag enterprise features.

This module defines custom exceptions for security, configuration, and
operational errors in the RAG framework.

Classes:
    VectorStoreConfigurationError: Raised when vector store configuration is invalid
    PermissionDeniedError: Raised when user lacks required permissions for an operation

Example:
    >>> from iris_vector_rag.exceptions import VectorStoreConfigurationError
    >>>
    >>> # Configuration validation
    >>> if invalid_filter_key:
    ...     raise VectorStoreConfigurationError(
    ...         "Invalid metadata filter key",
    ...         rejected_keys=["invalid_key"],
    ...         allowed_keys=["tenant_id", "department"]
    ...     )
    >>>
    >>> # RBAC enforcement
    >>> from iris_vector_rag.exceptions import PermissionDeniedError
    >>>
    >>> if not has_permission(user, "read", collection):
    ...     raise PermissionDeniedError(
    ...         user=user,
    ...         resource=collection_id,
    ...         operation="read"
    ...     )
"""

from typing import Any, Dict, List, Optional


class VectorStoreConfigurationError(Exception):
    """
    Exception raised when vector store configuration is invalid.

    This exception is raised when attempting to use metadata filter keys
    that are not in the allowed list, or when other configuration errors occur.

    Attributes:
        message: Human-readable error message
        rejected_keys: List of rejected metadata filter keys (if applicable)
        allowed_keys: List of allowed metadata filter keys (if applicable)
        details: Additional error context

    Example:
        >>> raise VectorStoreConfigurationError(
        ...     "Invalid metadata filter keys",
        ...     rejected_keys=["invalid_field"],
        ...     allowed_keys=["tenant_id", "department", "security_level"]
        ... )
    """

    def __init__(
        self,
        message: str,
        rejected_keys: Optional[List[str]] = None,
        allowed_keys: Optional[List[str]] = None,
        **details: Any
    ):
        """
        Initialize VectorStoreConfigurationError.

        Args:
            message: Error message describing the configuration issue
            rejected_keys: Optional list of rejected metadata filter keys
            allowed_keys: Optional list of allowed metadata filter keys
            **details: Additional error context (e.g., field_name, rejected_value)
        """
        super().__init__(message)
        self.message = message
        self.rejected_keys = rejected_keys or []
        self.allowed_keys = allowed_keys or []
        self.details = details

    def __str__(self) -> str:
        """Format error message with context."""
        parts = [self.message]

        if self.rejected_keys:
            parts.append(f"Rejected keys: {', '.join(self.rejected_keys)}")

        if self.allowed_keys:
            parts.append(f"Allowed keys: {', '.join(self.allowed_keys)}")

        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {detail_str}")

        return " | ".join(parts)


class PermissionDeniedError(Exception):
    """
    Exception raised when user lacks required permissions for an operation.

    This exception is raised by RBAC policy enforcement when a user attempts
    to access a resource or perform an operation they are not authorized for.

    Attributes:
        user: User identifier (username, email, or user ID)
        resource: Resource identifier (collection ID, document ID, etc.)
        operation: Operation name (read, write, delete, etc.)
        reason: Optional additional context explaining why permission was denied
        details: Additional error context

    Example:
        >>> raise PermissionDeniedError(
        ...     user="john.doe@example.com",
        ...     resource="medical_records_collection",
        ...     operation="read",
        ...     reason="User does not have 'medical' clearance level"
        ... )
    """

    def __init__(
        self,
        user: str,
        resource: str,
        operation: str,
        reason: Optional[str] = None,
        **details: Any
    ):
        """
        Initialize PermissionDeniedError.

        Args:
            user: User identifier (username, email, or user ID)
            resource: Resource identifier being accessed
            operation: Operation being attempted (read, write, delete)
            reason: Optional explanation for permission denial
            **details: Additional error context (e.g., required_role, user_role)
        """
        message = (
            f"Permission denied: User '{user}' cannot perform '{operation}' "
            f"on resource '{resource}'"
        )
        if reason:
            message += f" - {reason}"

        super().__init__(message)
        self.user = user
        self.resource = resource
        self.operation = operation
        self.reason = reason
        self.details = details

    def __str__(self) -> str:
        """Format error message with full context."""
        parts = [
            f"User: {self.user}",
            f"Resource: {self.resource}",
            f"Operation: {self.operation}",
        ]

        if self.reason:
            parts.append(f"Reason: {self.reason}")

        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            parts.append(f"Details: {detail_str}")

        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for JSON serialization.

        Returns:
            Dictionary with user, resource, operation, reason, and details

        Example:
            >>> error = PermissionDeniedError(
            ...     user="john", resource="docs", operation="read"
            ... )
            >>> error.to_dict()
            {'user': 'john', 'resource': 'docs', 'operation': 'read', ...}
        """
        return {
            "user": self.user,
            "resource": self.resource,
            "operation": self.operation,
            "reason": self.reason,
            **self.details,
        }
