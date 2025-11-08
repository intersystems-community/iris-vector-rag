"""
Data models for schema validation and management.

Defines core entities for test infrastructure resilience.
Implements T011-T012 from Feature 028.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class ColumnType(str, Enum):
    """Database column types supported by IRIS."""
    VARCHAR = "VARCHAR"
    INT = "INT"
    BIGINT = "BIGINT"
    DATETIME = "DATETIME"
    JSON = "JSON"
    VECTOR = "VECTOR"
    CLOB = "CLOB"


class MismatchType(str, Enum):
    """Types of schema mismatches that can be detected."""
    MISSING_TABLE = "missing_table"
    MISSING_COLUMN = "missing_column"
    TYPE_MISMATCH = "type_mismatch"
    EXTRA_TABLE = "extra_table"
    EXTRA_COLUMN = "extra_column"


@dataclass
class ColumnDefinition:
    """
    Definition of a database column.

    Attributes:
        name: Column name
        column_type: Data type (VARCHAR, INT, JSON, etc.)
        nullable: Whether column allows NULL values
        max_length: Maximum length for VARCHAR columns
        default_value: Default value if any
    """
    name: str
    column_type: ColumnType
    nullable: bool = True
    max_length: Optional[int] = None
    default_value: Optional[str] = None

    def __repr__(self) -> str:
        """String representation for debugging."""
        type_str = f"{self.column_type.value}"
        if self.max_length:
            type_str = f"{type_str}({self.max_length})"
        nullable_str = "NULL" if self.nullable else "NOT NULL"
        return f"{self.name} {type_str} {nullable_str}"


@dataclass
class SchemaDefinition:
    """
    Complete definition of a database table schema.

    Attributes:
        table_name: Name of the table
        schema_name: Database schema (e.g., 'RAG')
        columns: List of column definitions
        primary_key: Name of primary key column
    """
    table_name: str
    schema_name: str
    columns: List[ColumnDefinition]
    primary_key: Optional[str] = None

    def get_column(self, column_name: str) -> Optional[ColumnDefinition]:
        """
        Retrieve column definition by name.

        Args:
            column_name: Name of column to find

        Returns:
            ColumnDefinition if found, None otherwise
        """
        for col in self.columns:
            if col.name.lower() == column_name.lower():
                return col
        return None

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"{self.schema_name}.{self.table_name} ({len(self.columns)} columns)"


@dataclass
class SchemaMismatch:
    """
    Represents a detected schema mismatch.

    Attributes:
        table_name: Table where mismatch occurred
        mismatch_type: Type of mismatch (missing_table, type_mismatch, etc.)
        column_name: Column name (if column-level mismatch)
        expected_value: What was expected
        actual_value: What was found
        severity: Severity level (error, warning)
    """
    table_name: str
    mismatch_type: MismatchType
    column_name: Optional[str] = None
    expected_value: Optional[str] = None
    actual_value: Optional[str] = None
    severity: str = "error"

    def __repr__(self) -> str:
        """String representation for debugging."""
        if self.column_name:
            return (
                f"{self.mismatch_type.value}: {self.table_name}.{self.column_name} "
                f"(expected: {self.expected_value}, found: {self.actual_value})"
            )
        else:
            return f"{self.mismatch_type.value}: {self.table_name}"


@dataclass
class SchemaValidationResult:
    """
    Result of schema validation operation.

    Attributes:
        is_valid: Whether schema is valid (no errors)
        mismatches: List of detected mismatches
        validated_tables: Tables that were validated
        validation_time_ms: Time taken for validation
        message: Summary message
    """
    is_valid: bool
    mismatches: List[SchemaMismatch] = field(default_factory=list)
    validated_tables: List[str] = field(default_factory=list)
    validation_time_ms: int = 0
    message: str = ""

    @property
    def error_count(self) -> int:
        """Count of error-level mismatches."""
        return sum(1 for m in self.mismatches if m.severity == "error")

    @property
    def missing_tables(self) -> List[str]:
        """List of tables that are missing from the database."""
        return [
            m.table_name
            for m in self.mismatches
            if m.mismatch_type == MismatchType.MISSING_TABLE
        ]

    @property
    def warning_count(self) -> int:
        """Count of warning-level mismatches."""
        return sum(1 for m in self.mismatches if m.severity == "warning")

    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "VALID" if self.is_valid else "INVALID"
        return (
            f"SchemaValidationResult({status}, "
            f"{self.error_count} errors, {self.warning_count} warnings, "
            f"{self.validation_time_ms}ms)"
        )


# Expected RAG schema definitions
def get_expected_rag_schema() -> List[SchemaDefinition]:
    """
    Get expected RAG schema definitions.

    Returns:
        List of SchemaDefinition for all RAG tables
    """
    return [
        SchemaDefinition(
            table_name="SourceDocuments",
            schema_name="RAG",
            columns=[
                ColumnDefinition("id", ColumnType.VARCHAR, nullable=False, max_length=255),
                ColumnDefinition("source", ColumnType.VARCHAR, nullable=False, max_length=500),
                ColumnDefinition("content", ColumnType.CLOB, nullable=False),
                ColumnDefinition("metadata", ColumnType.JSON, nullable=True),
                ColumnDefinition("created_at", ColumnType.DATETIME, nullable=True),
                ColumnDefinition("test_run_id", ColumnType.VARCHAR, nullable=True, max_length=255),
            ],
            primary_key="id"
        ),
        SchemaDefinition(
            table_name="DocumentChunks",
            schema_name="RAG",
            columns=[
                ColumnDefinition("id", ColumnType.VARCHAR, nullable=False, max_length=255),
                ColumnDefinition("document_id", ColumnType.VARCHAR, nullable=False, max_length=255),
                ColumnDefinition("chunk_text", ColumnType.CLOB, nullable=False),
                ColumnDefinition("chunk_index", ColumnType.INT, nullable=False),
                ColumnDefinition("embedding", ColumnType.VECTOR, nullable=True),
                ColumnDefinition("created_at", ColumnType.DATETIME, nullable=True),
                ColumnDefinition("test_run_id", ColumnType.VARCHAR, nullable=True, max_length=255),
            ],
            primary_key="id"
        ),
        SchemaDefinition(
            table_name="Entities",
            schema_name="RAG",
            columns=[
                ColumnDefinition("id", ColumnType.VARCHAR, nullable=False, max_length=255),
                ColumnDefinition("entity_name", ColumnType.VARCHAR, nullable=False, max_length=500),
                ColumnDefinition("entity_type", ColumnType.VARCHAR, nullable=False, max_length=100),
                ColumnDefinition("properties", ColumnType.JSON, nullable=True),
                ColumnDefinition("created_at", ColumnType.DATETIME, nullable=True),
                ColumnDefinition("test_run_id", ColumnType.VARCHAR, nullable=True, max_length=255),
            ],
            primary_key="id"
        ),
        SchemaDefinition(
            table_name="Relationships",
            schema_name="RAG",
            columns=[
                ColumnDefinition("id", ColumnType.VARCHAR, nullable=False, max_length=255),
                ColumnDefinition("source_entity_id", ColumnType.VARCHAR, nullable=False, max_length=255),
                ColumnDefinition("target_entity_id", ColumnType.VARCHAR, nullable=False, max_length=255),
                ColumnDefinition("relationship_type", ColumnType.VARCHAR, nullable=False, max_length=100),
                ColumnDefinition("properties", ColumnType.JSON, nullable=True),
                ColumnDefinition("created_at", ColumnType.DATETIME, nullable=True),
                ColumnDefinition("test_run_id", ColumnType.VARCHAR, nullable=True, max_length=255),
            ],
            primary_key="id"
        ),
    ]
