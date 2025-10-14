"""
Schema validation for test infrastructure.

Validates IRIS database schema against expected structure.
Implements T014 from Feature 028.
"""

import time
from typing import List, Optional

from tests.utils.schema_models import (
    SchemaDefinition,
    SchemaValidationResult,
    SchemaMismatch,
    MismatchType,
    ColumnType,
    get_expected_rag_schema
)


class SchemaValidator:
    """
    Validates database schema against expected definitions.

    Detects missing tables, missing columns, and type mismatches.
    Performance target: <2 seconds (NFR-003).
    """

    def __init__(self):
        """Initialize validator with expected schema definitions."""
        self.expected_schemas = get_expected_rag_schema()

    def validate_schema(self, table_name: Optional[str] = None) -> SchemaValidationResult:
        """
        Validate current database schema against expected structure.

        Args:
            table_name: Optional specific table to validate. If None, validates all tables.

        Returns:
            SchemaValidationResult with validation outcome

        Raises:
            Exception: If database connection fails
        """
        start_time = time.time()
        mismatches: List[SchemaMismatch] = []

        try:
            from common.iris_connection_manager import get_iris_connection

            conn = get_iris_connection()
            cursor = conn.cursor()

            # Get list of existing tables in RAG schema
            cursor.execute("""
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'RAG'
            """)
            existing_tables = {row[0] for row in cursor.fetchall()}

            validated_tables = []

            # Check each expected table
            for schema_def in self.expected_schemas:
                validated_tables.append(schema_def.table_name)

                # Check if table exists
                if schema_def.table_name not in existing_tables:
                    mismatches.append(SchemaMismatch(
                        table_name=schema_def.table_name,
                        mismatch_type=MismatchType.MISSING_TABLE,
                        severity="error"
                    ))
                    continue  # Skip column checks if table doesn't exist

                # Validate columns for this table
                column_mismatches = self._validate_table_columns(
                    cursor,
                    schema_def
                )
                mismatches.extend(column_mismatches)

            # Calculate validation time
            validation_time_ms = int((time.time() - start_time) * 1000)

            # Build result
            is_valid = len([m for m in mismatches if m.severity == "error"]) == 0

            message = "Schema validation passed" if is_valid else \
                      f"Schema validation failed: {len(mismatches)} mismatch(es) found"

            return SchemaValidationResult(
                is_valid=is_valid,
                mismatches=mismatches,
                validated_tables=validated_tables,
                validation_time_ms=validation_time_ms,
                message=message
            )

        except Exception as e:
            validation_time_ms = int((time.time() - start_time) * 1000)
            return SchemaValidationResult(
                is_valid=False,
                mismatches=[],
                validated_tables=[],
                validation_time_ms=validation_time_ms,
                message=f"Schema validation error: {str(e)}"
            )

    def _validate_table_columns(
        self,
        cursor,
        schema_def: SchemaDefinition
    ) -> List[SchemaMismatch]:
        """
        Validate columns for a specific table.

        Args:
            cursor: Database cursor
            schema_def: Expected schema definition

        Returns:
            List of detected column mismatches
        """
        mismatches: List[SchemaMismatch] = []

        # Query actual columns for this table
        cursor.execute("""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            ORDER BY ORDINAL_POSITION
        """, [schema_def.schema_name, schema_def.table_name])

        actual_columns = {}
        for row in cursor.fetchall():
            col_name, data_type, is_nullable, max_length = row
            actual_columns[col_name] = {
                'data_type': data_type,
                'is_nullable': is_nullable == 'YES',
                'max_length': max_length
            }

        # Check each expected column
        for expected_col in schema_def.columns:
            if expected_col.name not in actual_columns:
                mismatches.append(SchemaMismatch(
                    table_name=schema_def.table_name,
                    mismatch_type=MismatchType.MISSING_COLUMN,
                    column_name=expected_col.name,
                    expected_value=str(expected_col.column_type.value),
                    actual_value="<missing>",
                    severity="error"
                ))
                continue

            # Check column type
            actual_col = actual_columns[expected_col.name]
            actual_type = self._normalize_type(actual_col['data_type'])
            expected_type = expected_col.column_type.value

            if not self._types_compatible(expected_type, actual_type):
                mismatches.append(SchemaMismatch(
                    table_name=schema_def.table_name,
                    mismatch_type=MismatchType.TYPE_MISMATCH,
                    column_name=expected_col.name,
                    expected_value=expected_type,
                    actual_value=actual_type,
                    severity="error"
                ))

        return mismatches

    def _normalize_type(self, iris_type: str) -> str:
        """
        Normalize IRIS data type to ColumnType enum value.

        Args:
            iris_type: IRIS data type string

        Returns:
            Normalized type string matching ColumnType enum
        """
        # Map IRIS types to ColumnType enum values
        type_mapping = {
            'VARCHAR': 'VARCHAR',
            'INTEGER': 'INT',
            'INT': 'INT',
            'BIGINT': 'BIGINT',
            'TIMESTAMP': 'DATETIME',
            'DATETIME': 'DATETIME',
            'LONGVARCHAR': 'CLOB',
            'CLOB': 'CLOB',
            'VARBINARY': 'VECTOR',  # IRIS vector type may show as VARBINARY
            'VECTOR': 'VECTOR',
        }

        # Handle JSON type (may be stored as VARCHAR or custom type)
        upper_type = iris_type.upper()
        if 'JSON' in upper_type:
            return 'JSON'

        return type_mapping.get(upper_type, upper_type)

    def _types_compatible(self, expected_type: str, actual_type: str) -> bool:
        """
        Check if two column types are compatible.

        JSON columns may be stored as CLOB/LONGVARCHAR in IRIS.

        Args:
            expected_type: Expected column type
            actual_type: Actual column type from database

        Returns:
            True if types are compatible, False otherwise
        """
        # Exact match
        if expected_type == actual_type:
            return True

        # JSON stored as CLOB is acceptable
        if expected_type == 'JSON' and actual_type in ('CLOB', 'LONGVARCHAR'):
            return True

        return False

    def get_schema_summary(self) -> dict:
        """
        Get a summary of expected schema structure.

        Returns:
            Dictionary summarizing schema (table count, column count)
        """
        total_columns = sum(len(s.columns) for s in self.expected_schemas)

        return {
            'table_count': len(self.expected_schemas),
            'total_columns': total_columns,
            'tables': [s.table_name for s in self.expected_schemas]
        }
