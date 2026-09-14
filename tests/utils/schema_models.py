"""
Data models for schema validation and management.

Defines core entities for test infrastructure resilience.
Implements T011-T012 from Feature 028.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from iris_vector_rag.storage.table_shape import (
    CANONICAL_CHUNK_SHAPE,
    CANONICAL_DOCUMENT_SHAPE,
    ColumnFact,
    TableShape,
    create_table_statement,
)


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
        create_sql: The exact CREATE TABLE the library itself emits, when the table
            is one the library owns. Set for the document and chunk tables so the
            test infrastructure creates what the code under test creates rather than
            rebuilding the statement from `columns` and hoping the two agree.
    """

    table_name: str
    schema_name: str
    columns: List[ColumnDefinition]
    primary_key: Optional[str] = None
    create_sql: Optional[str] = None

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

    @property
    def issue(self) -> str:
        """Human-readable mismatch summary for contract assertions."""
        return self.mismatch_type.value.replace("_", " ")


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


def _column_type_of(column: ColumnFact) -> ColumnType:
    """The `ColumnType` this test vocabulary uses for a catalog column fact."""
    if column.vector is not None:
        return ColumnType.VECTOR
    if column.is_long_text:
        return ColumnType.CLOB
    if column.sql_type == "timestamp":
        return ColumnType.DATETIME
    if column.sql_type == "integer":
        return ColumnType.INT
    if column.sql_type == "bigint":
        return ColumnType.BIGINT
    return ColumnType.VARCHAR


def _definition_from_shape(shape: TableShape) -> SchemaDefinition:
    """Describe a library-owned table from the shape the library creates.

    The document and chunk tables used to be described here a second time, by hand,
    and the hand-written version had `id` where the library writes `doc_id` and no
    embedding column at all. `SchemaValidator` therefore rejected a healthy table and
    `SchemaResetter` replaced it with one that could not answer a vector search
    (feature 082, FR-006).
    """
    ordered = sorted(shape.columns.values(), key=lambda column: column.position)
    keys = [column.name for column in ordered if column.is_key]
    return SchemaDefinition(
        table_name=shape.table,
        schema_name=shape.schema,
        columns=[
            ColumnDefinition(
                column.name,
                _column_type_of(column),
                nullable=column.nullable,
                max_length=column.max_length,
            )
            for column in ordered
        ],
        primary_key=keys[0] if len(keys) == 1 else None,
        create_sql=create_table_statement(shape),
    )


# Expected RAG schema definitions
def get_expected_rag_schema() -> List[SchemaDefinition]:
    """
    Get expected RAG schema definitions.

    Returns:
        List of SchemaDefinition for all RAG tables
    """
    return [
        _definition_from_shape(CANONICAL_DOCUMENT_SHAPE),
        _definition_from_shape(CANONICAL_CHUNK_SHAPE),
        SchemaDefinition(
            table_name="Entities",
            schema_name="RAG",
            columns=[
                ColumnDefinition(
                    "id", ColumnType.VARCHAR, nullable=False, max_length=255
                ),
                ColumnDefinition(
                    "entity_name", ColumnType.VARCHAR, nullable=False, max_length=500
                ),
                ColumnDefinition(
                    "entity_type", ColumnType.VARCHAR, nullable=False, max_length=100
                ),
                ColumnDefinition("properties", ColumnType.JSON, nullable=True),
                ColumnDefinition("created_at", ColumnType.DATETIME, nullable=True),
                ColumnDefinition(
                    "test_run_id", ColumnType.VARCHAR, nullable=True, max_length=255
                ),
            ],
            primary_key="id",
        ),
        SchemaDefinition(
            table_name="Relationships",
            schema_name="RAG",
            columns=[
                ColumnDefinition(
                    "id", ColumnType.VARCHAR, nullable=False, max_length=255
                ),
                ColumnDefinition(
                    "source_entity_id",
                    ColumnType.VARCHAR,
                    nullable=False,
                    max_length=255,
                ),
                ColumnDefinition(
                    "target_entity_id",
                    ColumnType.VARCHAR,
                    nullable=False,
                    max_length=255,
                ),
                ColumnDefinition(
                    "relationship_type",
                    ColumnType.VARCHAR,
                    nullable=False,
                    max_length=100,
                ),
                ColumnDefinition("properties", ColumnType.JSON, nullable=True),
                ColumnDefinition("created_at", ColumnType.DATETIME, nullable=True),
                ColumnDefinition(
                    "test_run_id", ColumnType.VARCHAR, nullable=True, max_length=255
                ),
            ],
            primary_key="id",
        ),
    ]
