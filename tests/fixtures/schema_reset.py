"""
Schema reset functionality for test infrastructure.

Provides idempotent schema reset operations.
Implements T015 from Feature 028.
"""

import time
from typing import List

from tests.utils.schema_models import get_expected_rag_schema, SchemaDefinition, ColumnType


class SchemaResetter:
    """
    Resets database schema to expected structure.

    Idempotent operations - safe to call multiple times.
    Performance target: <5 seconds (NFR-001).
    """

    def __init__(self):
        """Initialize resetter with expected schema definitions."""
        self.expected_schemas = get_expected_rag_schema()

    def reset_schema(self) -> None:
        """
        Reset database schema to expected structure.

        Drops and recreates all RAG tables.
        Idempotent - safe to call multiple times.

        Raises:
            Exception: If database connection fails or DDL operations fail
        """
        from common.iris_connection_manager import get_iris_connection

        conn = get_iris_connection()
        cursor = conn.cursor()

        try:
            # Drop existing tables in correct order to handle foreign key constraints
            # EntityRelationships must be dropped before Entities
            table_drop_order = [
                "EntityRelationships",  # Has FK to Entities
                "DocumentChunks",       # Independent
                "Entities",             # Referenced by EntityRelationships
                "SourceDocuments",      # Independent
            ]

            for table_name in table_drop_order:
                drop_sql = f"DROP TABLE IF EXISTS RAG.{table_name}"
                try:
                    cursor.execute(drop_sql)
                except Exception as drop_error:
                    # Ignore errors for tables that don't exist
                    if "does not exist" in str(drop_error).lower() or "not found" in str(drop_error).lower():
                        pass
                    else:
                        # Log but continue - we'll try to create tables anyway
                        print(f"Warning: Could not drop table {table_name}: {drop_error}")

            # Create tables in dependency order
            # Entities before EntityRelationships (FK dependency)
            table_create_order = [
                "SourceDocuments",
                "DocumentChunks",
                "Entities",
                "EntityRelationships",  # Depends on Entities
            ]

            for table_name in table_create_order:
                schema_def = next((s for s in self.expected_schemas if s.table_name == table_name), None)
                if schema_def:
                    create_sql = self._build_create_table_sql(schema_def)
                    cursor.execute(create_sql)

            conn.commit()

        except Exception as e:
            conn.rollback()
            raise Exception(f"Schema reset failed: {str(e)}")

    def _build_create_table_sql(self, schema_def: SchemaDefinition) -> str:
        """
        Build CREATE TABLE SQL statement.

        Args:
            schema_def: Schema definition to convert to SQL

        Returns:
            CREATE TABLE SQL statement
        """
        columns_sql = []

        for col in schema_def.columns:
            col_def = f"{col.name} {self._map_column_type(col.column_type, col.max_length)}"

            if not col.nullable:
                col_def += " NOT NULL"

            if col.default_value:
                col_def += f" DEFAULT {col.default_value}"

            columns_sql.append(col_def)

        # Add primary key constraint
        if schema_def.primary_key:
            columns_sql.append(f"PRIMARY KEY ({schema_def.primary_key})")

        columns_str = ",\n    ".join(columns_sql)

        return f"""
CREATE TABLE {schema_def.schema_name}.{schema_def.table_name} (
    {columns_str}
)
"""

    def _map_column_type(self, column_type: ColumnType, max_length: int = None) -> str:
        """
        Map ColumnType enum to IRIS SQL type.

        Args:
            column_type: Column type enum value
            max_length: Maximum length for VARCHAR columns

        Returns:
            IRIS SQL type string
        """
        type_mapping = {
            ColumnType.VARCHAR: f"VARCHAR({max_length})" if max_length else "VARCHAR(255)",
            ColumnType.INT: "INT",
            ColumnType.BIGINT: "BIGINT",
            ColumnType.DATETIME: "TIMESTAMP",
            ColumnType.JSON: "VARCHAR(MAX)",  # IRIS stores JSON as VARCHAR
            ColumnType.CLOB: "LONGVARCHAR",
            ColumnType.VECTOR: "VARBINARY(MAX)",  # IRIS vector type
        }

        return type_mapping.get(column_type, "VARCHAR(255)")

    def reset_schema_timed(self) -> float:
        """
        Reset schema and return execution time.

        Returns:
            Execution time in seconds

        Raises:
            Exception: If reset fails or exceeds 5 second limit
        """
        start = time.time()
        self.reset_schema()
        duration = time.time() - start

        if duration >= 5.0:
            raise Exception(
                f"Schema reset took {duration:.2f}s, exceeds 5s limit (NFR-001)"
            )

        return duration

    def validate_tables_exist(self) -> bool:
        """
        Verify all expected tables exist after reset.

        Returns:
            True if all tables exist, False otherwise
        """
        from common.iris_connection_manager import get_iris_connection

        conn = get_iris_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT TABLE_NAME
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = 'RAG'
        """)

        existing_tables = {row[0] for row in cursor.fetchall()}
        expected_tables = {s.table_name for s in self.expected_schemas}

        return expected_tables.issubset(existing_tables)

    def get_table_count(self) -> int:
        """
        Get count of RAG tables in database.

        Returns:
            Number of tables in RAG schema
        """
        from common.iris_connection_manager import get_iris_connection

        conn = get_iris_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT COUNT(*)
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = 'RAG'
        """)

        return cursor.fetchone()[0]
