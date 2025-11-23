"""
Metadata Filter Manager for custom metadata filtering in multi-tenant deployments.

This module provides validation and management of metadata filter keys, enabling
administrators to configure custom fields beyond the default 17 fields while
preventing SQL injection and configuration errors.

Classes:
    MetadataFilterManager: Validates and manages allowed metadata filter keys
    ValidationResult: Result of metadata filter key validation

Example:
    >>> config = {
    ...     "storage": {
    ...         "iris": {
    ...             "custom_filter_keys": ["tenant_id", "security_level"]
    ...         }
    ...     }
    ... }
    >>> manager = MetadataFilterManager(config)
    >>> allowed = manager.get_allowed_filter_keys()
    >>> # Returns: ["source", "doc_id", ..., "tenant_id", "security_level"]
    >>>
    >>> result = manager.validate_filter_keys({"tenant_id": "acme"})
    >>> # Returns: ValidationResult(is_valid=True, rejected_keys=[])
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Any

from iris_vector_rag.storage.constants import DEFAULT_FILTER_KEYS
from iris_vector_rag.exceptions import VectorStoreConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of metadata filter key validation."""
    is_valid: bool
    rejected_keys: List[str]
    allowed_keys: List[str]
    error_message: str = ""


class MetadataFilterManager:
    """
    Manages metadata filter keys for multi-tenant RAG deployments.

    Validates custom metadata fields, prevents SQL injection, and ensures
    backward compatibility with default 17 fields.

    Attributes:
        default_keys: Set of 17 default metadata filter keys
        custom_keys: Set of administrator-configured custom keys
        all_keys: Combined set of default + custom keys
    """

    # Regex for valid field names (alphanumeric + underscores, no SQL keywords)
    VALID_FIELD_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]*$")

    # SQL keywords to reject (case-insensitive check)
    SQL_KEYWORDS = {
        "SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "TABLE",
        "FROM", "WHERE", "AND", "OR", "JOIN", "UNION", "ALTER",
        "CREATE", "TRUNCATE", "EXEC", "EXECUTE", "--", ";", "/*", "*/"
    }

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize MetadataFilterManager with custom configuration.

        Args:
            config: Configuration dictionary with structure:
                {
                    "storage": {
                        "iris": {
                            "custom_filter_keys": ["tenant_id", "security_level"]
                        }
                    }
                }

        Raises:
            VectorStoreConfigurationError: If custom fields conflict with defaults
                or contain invalid characters
        """
        self.default_keys: Set[str] = set(DEFAULT_FILTER_KEYS)
        self.custom_keys: Set[str] = set()

        if config:
            custom_list = (
                config.get("storage", {})
                .get("iris", {})
                .get("custom_filter_keys", [])
            )

            if custom_list:
                self._validate_and_register_custom_keys(custom_list)

        self.all_keys: Set[str] = self.default_keys | self.custom_keys

        logger.info(
            f"MetadataFilterManager initialized: "
            f"{len(self.default_keys)} default keys, "
            f"{len(self.custom_keys)} custom keys"
        )

    def _validate_and_register_custom_keys(self, custom_keys: List[str]) -> None:
        """
        Validate and register custom metadata filter keys.

        Args:
            custom_keys: List of custom field names to validate

        Raises:
            VectorStoreConfigurationError: If validation fails
        """
        for key in custom_keys:
            # Check for duplicate with default fields
            if key in self.default_keys:
                raise VectorStoreConfigurationError(
                    f"Custom filter key '{key}' conflicts with default field. "
                    f"Default fields: {sorted(self.default_keys)}"
                )

            # Validate field name pattern
            if not self.VALID_FIELD_NAME_PATTERN.match(key):
                raise VectorStoreConfigurationError(
                    f"Invalid field name '{key}': must start with letter and "
                    f"contain only alphanumeric characters and underscores"
                )

            # Check for SQL keywords (case-insensitive)
            if any(keyword in key.upper() for keyword in self.SQL_KEYWORDS):
                raise VectorStoreConfigurationError(
                    f"Invalid field name '{key}': contains SQL keyword or special "
                    f"characters that could enable SQL injection"
                )

            self.custom_keys.add(key)

    def get_allowed_filter_keys(self) -> List[str]:
        """
        Get list of all allowed metadata filter keys.

        Returns:
            Sorted list of default + custom filter keys
        """
        return sorted(self.all_keys)

    def validate_filter_keys(self, metadata_filter: Dict[str, Any]) -> ValidationResult:
        """
        Validate that all keys in metadata_filter are allowed.

        Args:
            metadata_filter: Dictionary of filter key-value pairs

        Returns:
            ValidationResult with validation status and rejected keys

        Example:
            >>> result = manager.validate_filter_keys({"tenant_id": "acme"})
            >>> if not result.is_valid:
            ...     raise VectorStoreConfigurationError(result.error_message)
        """
        if not metadata_filter:
            return ValidationResult(
                is_valid=True,
                rejected_keys=[],
                allowed_keys=self.get_allowed_filter_keys()
            )

        provided_keys = set(metadata_filter.keys())
        rejected_keys = provided_keys - self.all_keys

        if rejected_keys:
            error_msg = (
                f"Invalid metadata filter keys: {sorted(rejected_keys)}. "
                f"Allowed keys: {sorted(self.all_keys)}"
            )
            return ValidationResult(
                is_valid=False,
                rejected_keys=list(rejected_keys),
                allowed_keys=self.get_allowed_filter_keys(),
                error_message=error_msg
            )

        return ValidationResult(
            is_valid=True,
            rejected_keys=[],
            allowed_keys=self.get_allowed_filter_keys()
        )
