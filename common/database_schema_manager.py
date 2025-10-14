#!/usr/bin/env python3
"""
Database Schema Manager for RAG Templates
Provides centralized, config-driven table and column name resolution.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TableConfig:
    """Configuration for a database table."""

    name: str
    alias: Optional[str] = None
    description: str = ""
    columns: Dict[str, str] = field(default_factory=dict)
    indexes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class SchemaConfig:
    """Complete database schema configuration."""

    name: str
    description: str = ""
    tables: Dict[str, TableConfig] = field(default_factory=dict)
    column_mappings: Dict[str, Any] = field(default_factory=dict)
    data_types: Dict[str, str] = field(default_factory=dict)


class DatabaseSchemaManager:
    """
    Centralized manager for database schema configuration.
    Eliminates hardcoded table/column names throughout the codebase.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize with schema configuration."""
        if config_path is None:
            # Default to config/database_schema.yaml in project root
            project_root = Path(__file__).parent.parent
            config_path = project_root / "config" / "database_schema.yaml"

        self.config_path = Path(config_path)
        self.schema_config = self._load_schema_config()

        # Cache for performance
        self._table_cache = {}
        self._column_cache = {}

        logger.info(f"✅ Database schema manager initialized from {config_path}")

    def _load_schema_config(self) -> SchemaConfig:
        """Load schema configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                config_data = yaml.safe_load(f)

            # Parse schema info
            schema_info = config_data.get("schema", {})
            schema_name = schema_info.get("name", "RAG")
            schema_desc = schema_info.get("description", "")

            # Parse tables
            tables = {}
            tables_data = config_data.get("tables", {})
            for table_key, table_data in tables_data.items():
                tables[table_key] = TableConfig(
                    name=table_data.get("name", table_key),
                    alias=table_data.get("alias"),
                    description=table_data.get("description", ""),
                    columns=table_data.get("columns", {}),
                    indexes=table_data.get("indexes", []),
                )

            return SchemaConfig(
                name=schema_name,
                description=schema_desc,
                tables=tables,
                column_mappings=config_data.get("column_mappings", {}),
                data_types=config_data.get("data_types", {}),
            )

        except Exception as e:
            logger.error(f"Failed to load schema config from {self.config_path}: {e}")
            raise

    def get_schema_name(self) -> str:
        """Get the schema name (e.g., 'RAG')."""
        return self.schema_config.name

    def get_table_name(self, table_key: str, fully_qualified: bool = True) -> str:
        """
        Get the actual table name for a logical table key.

        Args:
            table_key: Logical name (e.g., 'source_documents')
            fully_qualified: Whether to include schema prefix (e.g., 'RAG.SourceDocuments')

        Returns:
            Actual table name
        """
        if table_key in self._table_cache:
            table_name = self._table_cache[table_key]
        else:
            if table_key not in self.schema_config.tables:
                raise ValueError(f"Unknown table key: {table_key}")

            table_config = self.schema_config.tables[table_key]
            table_name = table_config.name
            self._table_cache[table_key] = table_name

        if fully_qualified:
            return f"{self.schema_config.name}.{table_name}"
        return table_name

    def get_column_name(self, table_key: str, column_key: str) -> str:
        """
        Get the actual column name for a logical column key.

        Args:
            table_key: Logical table name
            column_key: Logical column name

        Returns:
            Actual column name
        """
        cache_key = f"{table_key}.{column_key}"
        if cache_key in self._column_cache:
            return self._column_cache[cache_key]

        if table_key not in self.schema_config.tables:
            raise ValueError(f"Unknown table key: {table_key}")

        table_config = self.schema_config.tables[table_key]
        if column_key not in table_config.columns:
            raise ValueError(
                f"Unknown column key '{column_key}' for table '{table_key}'"
            )

        column_name = table_config.columns[column_key]
        self._column_cache[cache_key] = column_name
        return column_name

    def get_qualified_column(self, table_key: str, column_key: str) -> str:
        """Get fully qualified column name (table.column)."""
        table_name = self.get_table_name(table_key, fully_qualified=False)
        column_name = self.get_column_name(table_key, column_key)
        return f"{table_name}.{column_name}"

    def standardize_column_name(self, column_name: str) -> str:
        """
        Standardize a column name using the mappings.

        Args:
            column_name: Original column name

        Returns:
            Standardized column name
        """
        mappings = self.schema_config.column_mappings

        # Check each mapping category
        for mapping_key, mapping_data in mappings.items():
            if isinstance(mapping_data, dict) and "standard" in mapping_data:
                variants = mapping_data.get("variants", [])
                if column_name in variants:
                    return mapping_data["standard"]

        # Return original if no mapping found
        return column_name

    def get_data_type(self, type_key: str) -> str:
        """Get data type definition."""
        return self.schema_config.data_types.get(type_key, "VARCHAR(255)")

    def build_create_table_sql(self, table_key: str) -> str:
        """Generate CREATE TABLE SQL for a table."""
        if table_key not in self.schema_config.tables:
            raise ValueError(f"Unknown table key: {table_key}")

        table_config = self.schema_config.tables[table_key]
        table_name = self.get_table_name(table_key, fully_qualified=True)

        # Build column definitions
        column_defs = []
        for col_key, col_name in table_config.columns.items():
            data_type = self.get_data_type(col_key)
            column_defs.append(f"    {col_name} {data_type}")

        sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n"
        sql += ",\n".join(column_defs)
        sql += "\n);"

        return sql

    def get_all_tables(self) -> Dict[str, str]:
        """Get mapping of all table keys to actual names."""
        return {
            key: self.get_table_name(key, fully_qualified=True)
            for key in self.schema_config.tables.keys()
        }

    def validate_table_exists(self, table_key: str) -> bool:
        """Check if a table configuration exists."""
        return table_key in self.schema_config.tables

    def get_table_info(self, table_key: str) -> Dict[str, Any]:
        """Get complete information about a table."""
        if table_key not in self.schema_config.tables:
            raise ValueError(f"Unknown table key: {table_key}")

        table_config = self.schema_config.tables[table_key]
        return {
            "key": table_key,
            "name": table_config.name,
            "alias": table_config.alias,
            "description": table_config.description,
            "fully_qualified_name": self.get_table_name(
                table_key, fully_qualified=True
            ),
            "columns": table_config.columns,
            "indexes": table_config.indexes,
        }


# Global schema manager instance
_schema_manager = None


def get_schema_manager() -> DatabaseSchemaManager:
    """Get the global schema manager instance."""
    global _schema_manager
    if _schema_manager is None:
        _schema_manager = DatabaseSchemaManager()
    return _schema_manager


# Convenience functions for common operations
def get_table(table_key: str, fully_qualified: bool = True) -> str:
    """Convenience function to get table name."""
    return get_schema_manager().get_table_name(table_key, fully_qualified)


def get_column(table_key: str, column_key: str) -> str:
    """Convenience function to get column name."""
    return get_schema_manager().get_column_name(table_key, column_key)


def get_qualified_column(table_key: str, column_key: str) -> str:
    """Convenience function to get qualified column name."""
    return get_schema_manager().get_qualified_column(table_key, column_key)


# Example usage and testing
if __name__ == "__main__":
    # Test the schema manager
    logging.basicConfig(level=logging.INFO)

    manager = DatabaseSchemaManager()

    print("🔍 Testing Database Schema Manager")
    print("=" * 50)

    # Test table names
    print("Table Names:")
    for table_key in ["source_documents", "document_chunks", "document_entities"]:
        name = manager.get_table_name(table_key)
        print(f"  {table_key} → {name}")

    print("\nColumn Names:")
    for table_key in ["source_documents", "document_chunks"]:
        for col_key in ["id", "content"]:
            try:
                col_name = manager.get_column_name(table_key, col_key)
                print(f"  {table_key}.{col_key} → {col_name}")
            except ValueError as e:
                print(f"  {table_key}.{col_key} → ERROR: {e}")

    print("\nCreate Table SQL:")
    print(manager.build_create_table_sql("source_documents"))
