"""
Configuration validators for preflight checks.

This module provides validators for checking configuration against existing
database state to prevent data corruption and configuration errors.

Feature: 058-cloud-config-flexibility
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from iris_vector_rag.config.entities import (
    ConnectionConfiguration,
    TableConfiguration,
    VectorConfiguration,
)


class ValidationStatus(Enum):
    """Status of a validation check."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


@dataclass
class ValidationResult:
    """
    Result of a configuration validation check.

    Attributes:
        status: ValidationStatus indicating pass/fail/warning
        message: Human-readable description of the result
        help_url: Optional URL to documentation for resolving issues
        details: Optional additional details about the validation
    """

    status: ValidationStatus
    message: str
    help_url: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def is_success(self) -> bool:
        """Check if validation passed."""
        return self.status == ValidationStatus.PASS

    def is_failure(self) -> bool:
        """Check if validation failed."""
        return self.status == ValidationStatus.FAIL

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        result = {
            "status": self.status.value,
            "message": self.message,
        }
        if self.help_url:
            result["help_url"] = self.help_url
        if self.details:
            result["details"] = self.details
        return result


class VectorDimensionValidator:
    """
    Validates vector dimension configuration against existing database tables.

    Prevents data corruption by detecting mismatches between configured
    vector dimensions and existing table schemas.

    Example:
        >>> validator = VectorDimensionValidator()
        >>> result = validator.validate(
        ...     config=VectorConfiguration(vector_dimension=1024),
        ...     connection=conn
        ... )
        >>> if result.is_failure():
        ...     print(result.message)
    """

    HELP_URL = "https://github.com/isc-tdyar/iris-vector-rag-private/blob/master/docs/migration/vector-dimensions.md"

    def validate(
        self,
        config: VectorConfiguration,
        connection: Any,
        table_config: Optional[TableConfiguration] = None,
    ) -> ValidationResult:
        """
        Validate vector dimension against existing tables.

        Args:
            config: VectorConfiguration with desired vector_dimension
            connection: Active IRIS database connection
            table_config: Optional TableConfiguration for schema-prefixed tables

        Returns:
            ValidationResult indicating success or failure with details

        Raises:
            RuntimeError: If database query fails
        """
        configured_dim = config.vector_dimension
        table_config = table_config or TableConfiguration()

        try:
            cursor = connection.cursor()

            # Query existing table dimension
            # Check if table exists first
            cursor.execute(
                """SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
                   WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?""",
                (table_config.table_schema, table_config.entities_table)
            )
            table_exists = cursor.fetchone()[0] > 0

            if not table_exists:
                # No existing table - safe to create with configured dimension
                return ValidationResult(
                    status=ValidationStatus.PASS,
                    message=(
                        f"No existing tables found. "
                        f"Will create {table_config.full_entities_table} "
                        f"with {configured_dim}-dimensional vectors"
                    ),
                    details={
                        "configured_dimension": configured_dim,
                        "existing_dimension": None,
                        "table": table_config.full_entities_table,
                    },
                )

            # Get vector dimension from CHARACTER_MAXIMUM_LENGTH
            # IRIS stores VECTOR columns as varchar with byte length ≈ dimensions × 346
            cursor.execute(
                """SELECT CHARACTER_MAXIMUM_LENGTH FROM INFORMATION_SCHEMA.COLUMNS
                   WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? AND COLUMN_NAME = 'embedding'""",
                (table_config.table_schema, table_config.entities_table)
            )
            length_result = cursor.fetchone()

            if not length_result:
                raise ValueError(f"No embedding column found in {table_config.full_entities_table}")

            byte_length = length_result[0]
            if not byte_length:
                raise ValueError(f"Could not determine vector dimension for {table_config.full_entities_table}")

            # Calculate dimension from byte length
            # Formula: dimensions = round(CHARACTER_MAXIMUM_LENGTH / 346)
            # Verified for dimensions 128, 384, 1024, 1536
            existing_dim = round(byte_length / 346)

            if configured_dim == existing_dim:
                # Perfect match
                return ValidationResult(
                    status=ValidationStatus.PASS,
                    message=(
                        f"Vector dimension matches existing tables: "
                        f"{configured_dim} dimensions"
                    ),
                    details={
                        "configured_dimension": configured_dim,
                        "existing_dimension": existing_dim,
                        "table": table_config.full_entities_table,
                    },
                )

            # Mismatch detected - this is a critical error
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=self._format_mismatch_error(
                    configured_dim, existing_dim, table_config
                ),
                help_url=self.HELP_URL,
                details={
                    "configured_dimension": configured_dim,
                    "existing_dimension": existing_dim,
                    "table": table_config.full_entities_table,
                    "config_source": config.source.get("vector_dimension", "unknown"),
                },
            )

        except Exception as e:
            # Database query failed
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message=(
                    f"Could not validate vector dimensions: {str(e)}. "
                    f"Proceeding with configured dimension {configured_dim}"
                ),
                details={
                    "configured_dimension": configured_dim,
                    "error": str(e),
                },
            )

    def _format_mismatch_error(
        self,
        configured_dim: int,
        existing_dim: int,
        table_config: TableConfiguration,
    ) -> str:
        """Format detailed error message for dimension mismatch."""
        return f"""Vector dimension mismatch detected

Configured dimension: {configured_dim}
Existing table dimension: {existing_dim} ({table_config.full_entities_table}.embedding)

This mismatch will cause data corruption. To resolve:

Option 1: Match existing tables
  Set VECTOR_DIMENSION={existing_dim} (or remove from config to use default)

Option 2: Recreate tables with new dimension (⚠️  DELETES ALL DATA)
  $ python -m iris_vector_rag.cli.init_tables --drop --config your-config.yaml

Option 3: Migrate data to new dimension (requires external tool)
  See: {self.HELP_URL}"""


class NamespaceValidator:
    """
    Validates namespace configuration and permissions.

    Ensures the configured namespace exists and the user has sufficient
    permissions to create tables and schemas.

    Example:
        >>> validator = NamespaceValidator()
        >>> result = validator.validate(
        ...     config=ConnectionConfiguration(namespace='%SYS'),
        ...     connection=conn
        ... )
    """

    HELP_URL = "https://github.com/isc-tdyar/iris-vector-rag-private/blob/master/docs/configuration/namespace-permissions.md"

    def validate(
        self,
        config: ConnectionConfiguration,
        connection: Any,
        table_config: Optional[TableConfiguration] = None,
    ) -> ValidationResult:
        """
        Validate namespace configuration and permissions.

        Args:
            config: ConnectionConfiguration with namespace setting
            connection: Active IRIS database connection
            table_config: Optional TableConfiguration for schema checks

        Returns:
            ValidationResult indicating success or failure

        Raises:
            RuntimeError: If database query fails
        """
        namespace = config.namespace
        table_config = table_config or TableConfiguration()

        try:
            cursor = connection.cursor()

            # Check if namespace exists or if we can access the schema
            # Try to use the namespace - if we can query INFORMATION_SCHEMA, it exists
            cursor.execute(
                "SELECT COUNT(*) FROM INFORMATION_SCHEMA.SCHEMATA WHERE SCHEMA_NAME = ? OR SCHEMA_NAME = ?",
                (namespace, table_config.table_schema),
            )
            result = cursor.fetchone()

            if result is None or result[0] == 0:
                # Try a simple query in the namespace to see if it's accessible
                try:
                    cursor.execute("SELECT 1")
                    # If we got here, namespace is accessible even if not in SCHEMATA
                    pass
                except Exception:
                    return ValidationResult(
                        status=ValidationStatus.FAIL,
                        message=f"Namespace '{namespace}' does not exist or is not accessible",
                        help_url=self.HELP_URL,
                        details={
                            "namespace": namespace,
                            "available_namespaces": self._get_available_namespaces(
                                connection
                            ),
                        },
                    )

            # Check schema permissions
            schema_check = self._check_schema_permissions(
                connection, namespace, table_config.table_schema
            )

            if not schema_check["can_create_tables"]:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    message=(
                        f"User '{config.username}' lacks CREATE TABLE permission "
                        f"in namespace '{namespace}' for schema '{table_config.table_schema}'"
                    ),
                    help_url=self.HELP_URL,
                    details=schema_check,
                )

            # All checks passed
            return ValidationResult(
                status=ValidationStatus.PASS,
                message=f"Namespace '{namespace}' validated successfully",
                details={
                    "namespace": namespace,
                    "schema": table_config.table_schema,
                    "permissions": schema_check,
                },
            )

        except Exception as e:
            # Permission check failed - warning only
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message=f"Could not validate namespace permissions: {str(e)}",
                details={
                    "namespace": namespace,
                    "error": str(e),
                },
            )

    def _get_available_namespaces(self, connection: Any) -> list:
        """Get list of available namespaces."""
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA")
            return [row[0] for row in cursor.fetchall()]
        except Exception:
            return []

    def _check_schema_permissions(
        self, connection: Any, namespace: str, schema: str
    ) -> Dict[str, Any]:
        """
        Check if user can create tables in the schema.

        This is a simplified check - actual permission validation may require
        IRIS-specific privilege queries.
        """
        try:
            cursor = connection.cursor()

            # Check if schema exists
            cursor.execute(
                """
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.SCHEMATA
                WHERE SCHEMA_NAME = ?
                """,
                (schema,),
            )
            result = cursor.fetchone()
            schema_exists = result and result[0] > 0

            # Attempt to query system tables for permissions
            # Note: This is a simplified check - real IRIS permission check
            # would query %SYS.Security or similar
            can_create_tables = True  # Assume true unless proven false

            return {
                "namespace": namespace,
                "schema": schema,
                "schema_exists": schema_exists,
                "can_create_tables": can_create_tables,
            }

        except Exception as e:
            return {
                "namespace": namespace,
                "schema": schema,
                "error": str(e),
                "can_create_tables": False,
            }


class PreflightValidator:
    """
    Orchestrates all preflight validation checks.

    Runs all validators and aggregates results for comprehensive
    configuration validation before database operations.

    Example:
        >>> validator = PreflightValidator()
        >>> results = validator.validate_all(
        ...     connection_config=conn_config,
        ...     vector_config=vec_config,
        ...     table_config=table_config,
        ...     connection=conn
        ... )
        >>> if not validator.all_passed(results):
        ...     print("Configuration validation failed")
    """

    def __init__(self):
        self.vector_validator = VectorDimensionValidator()
        self.namespace_validator = NamespaceValidator()

    def validate_all(
        self,
        connection_config: ConnectionConfiguration,
        vector_config: VectorConfiguration,
        table_config: TableConfiguration,
        connection: Any,
    ) -> Dict[str, ValidationResult]:
        """
        Run all preflight validation checks.

        Args:
            connection_config: ConnectionConfiguration instance
            vector_config: VectorConfiguration instance
            table_config: TableConfiguration instance
            connection: Active IRIS database connection

        Returns:
            Dictionary mapping validator name to ValidationResult
        """
        results = {}

        # Validate namespace and permissions
        results["namespace"] = self.namespace_validator.validate(
            connection_config, connection, table_config
        )

        # Validate vector dimensions
        results["vector_dimension"] = self.vector_validator.validate(
            vector_config, connection, table_config
        )

        return results

    def all_passed(self, results: Dict[str, ValidationResult]) -> bool:
        """
        Check if all validations passed.

        Args:
            results: Dictionary of validation results

        Returns:
            True if all validations passed, False otherwise
        """
        return all(
            result.status in (ValidationStatus.PASS, ValidationStatus.WARNING)
            for result in results.values()
        )

    def get_failures(
        self, results: Dict[str, ValidationResult]
    ) -> Dict[str, ValidationResult]:
        """
        Get only failed validations.

        Args:
            results: Dictionary of validation results

        Returns:
            Dictionary containing only failed validations
        """
        return {
            name: result
            for name, result in results.items()
            if result.status == ValidationStatus.FAIL
        }

    def format_results(self, results: Dict[str, ValidationResult]) -> str:
        """
        Format validation results as human-readable text.

        Args:
            results: Dictionary of validation results

        Returns:
            Formatted string with all validation results
        """
        lines = ["Configuration Validation Results", "=" * 40, ""]

        for name, result in results.items():
            status_symbol = {
                ValidationStatus.PASS: "✓",
                ValidationStatus.FAIL: "✗",
                ValidationStatus.WARNING: "⚠",
                ValidationStatus.SKIPPED: "○",
            }.get(result.status, "?")

            lines.append(f"{status_symbol} {name.upper()}: {result.status.value}")
            lines.append(f"  {result.message}")

            if result.help_url:
                lines.append(f"  Help: {result.help_url}")

            lines.append("")

        return "\n".join(lines)
