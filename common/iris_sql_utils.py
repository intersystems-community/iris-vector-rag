"""
IRIS SQL Utilities - Proper IRIS SQL Syntax Support

This module provides utilities for working with IRIS-specific SQL syntax,
particularly for vector operations and index creation.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Import the DBAPI connector for self-managed connections
try:
    from common.iris_dbapi_connector import get_iris_dbapi_connection
except ImportError:
    logger.warning(
        "Could not import get_iris_dbapi_connection. Self-managed connections will not be available."
    )
    get_iris_dbapi_connection = None


class IRISSQLUtils:
    """Utilities for IRIS-specific SQL operations."""

    @staticmethod
    def create_vector_index(
        connection: Any,
        table_name: str,
        column_name: str,
        index_name: Optional[str] = None,
        metric: str = "COSINE",
        index_type: str = "HNSW",
    ) -> bool:
        """
        Create a vector index using proper IRIS syntax.

        Uses the correct IRIS HNSW index syntax from db_init_complete.sql:
        CREATE INDEX IF NOT EXISTS idx_hnsw_source_embedding
        ON RAG.SourceDocuments (embedding)
        AS HNSW(M=16, efConstruction=200, Distance='COSINE');

        Args:
            connection: Database connection
            table_name: Name of the table
            column_name: Name of the vector column
            index_name: Optional custom index name
            metric: Vector similarity metric (COSINE, DOT, EUCLIDEAN)
            index_type: Index type (HNSW, FLAT) - currently only HNSW supported

        Returns:
            True if successful, False otherwise
        """
        if not index_name:
            # Clean table name for index naming
            clean_table = table_name.replace(".", "_")
            index_name = f"{clean_table}_{column_name}_vector_idx"

        # IRIS vector index syntax using AS HNSW (from db_init_complete.sql)
        # Note: IRIS doesn't support "IF NOT EXISTS" in CREATE INDEX statements
        sql = f"""
        CREATE INDEX {index_name}
        ON {table_name} ({column_name})
        AS HNSW(M=16, efConstruction=200, Distance='{metric}')
        """

        # Check if index already exists first
        if IRISSQLUtils.check_index_exists(connection, index_name, table_name):
            logger.debug(f"Vector index {index_name} already exists")
            return True

        try:
            cursor = connection.cursor()
            cursor.execute(sql)
            connection.commit()
            cursor.close()
            logger.info(
                f"✓ Created vector index {index_name} on {table_name}.{column_name}"
            )
            return True
        except Exception as e:
            error_msg = str(e).lower()
            # Handle IRIS-specific error codes and messages
            if (
                "already exists" in error_msg
                or "duplicate" in error_msg
                or "sqlcode: <-324>" in error_msg
                or "index with this name already defined" in error_msg
            ):
                logger.debug(f"Vector index {index_name} already exists")
                return True
            else:
                logger.error(f"Failed to create vector index {index_name}: {e}")
                return False

    @staticmethod
    def create_simple_index(
        connection: Any,
        table_name: str,
        column_name: str,
        index_name: Optional[str] = None,
    ) -> bool:
        """
        Create a simple (non-vector) index using IRIS syntax.

        Args:
            connection: Database connection
            table_name: Name of the table
            column_name: Name of the column
            index_name: Optional custom index name

        Returns:
            True if successful, False otherwise
        """
        if not index_name:
            clean_table = table_name.replace(".", "_")
            index_name = f"{clean_table}_{column_name}_idx"

        # Simple index syntax for IRIS
        sql = f"CREATE INDEX {index_name} ON {table_name} ({column_name})"

        # Check if index already exists first
        if IRISSQLUtils.check_index_exists(connection, index_name, table_name):
            logger.debug(f"Index {index_name} already exists")
            return True

        try:
            cursor = connection.cursor()
            cursor.execute(sql)
            connection.commit()
            cursor.close()
            logger.info(f"✓ Created index {index_name} on {table_name}.{column_name}")
            return True
        except Exception as e:
            error_msg = str(e).lower()
            # Handle IRIS-specific error codes and messages
            if (
                "already exists" in error_msg
                or "duplicate" in error_msg
                or "sqlcode: <-324>" in error_msg
                or "index with this name already defined" in error_msg
            ):
                logger.debug(f"Index {index_name} already exists")
                return True
            else:
                logger.error(f"Failed to create index {index_name}: {e}")
                return False

    @staticmethod
    def check_table_exists(connection: Any, table_name: str) -> bool:
        """
        Check if a table exists using IRIS-compatible syntax.
        Uses a self-managed database connection to avoid "DataRow is inaccessible" errors.

        Args:
            connection: Database connection (kept for compatibility, but not used)
            table_name: Name of the table (schema.table format)

        Returns:
            True if table exists, False otherwise
        """
        # Use self-managed connection to avoid interference with other operations
        if get_iris_dbapi_connection is None:
            logger.warning(
                "Self-managed connections not available, falling back to provided connection"
            )
            return IRISSQLUtils._check_table_exists_with_connection(
                connection, table_name
            )

        temp_connection = None
        cursor = None
        try:
            temp_connection = get_iris_dbapi_connection()
            if not temp_connection:
                logger.error(
                    "Failed to obtain temporary connection for table existence check"
                )
                return False

            # Split schema and table name
            if "." in table_name:
                schema, table = table_name.split(".", 1)
            else:
                schema = "USER"  # Default schema
                table = table_name

            cursor = temp_connection.cursor()
            # Use IRIS system tables to check existence
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            """,
                [schema, table],
            )

            result = cursor.fetchone()
            return result[0] > 0

        except Exception as e:
            logger.error(f"Failed to check table existence for {table_name}: {e}")
            return False
        finally:
            # Ensure cleanup of temporary connection and cursor
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if temp_connection:
                try:
                    temp_connection.close()
                except:
                    pass

    @staticmethod
    def _check_table_exists_with_connection(connection: Any, table_name: str) -> bool:
        """
        Fallback method using provided connection for table existence check.
        """
        try:
            # Split schema and table name
            if "." in table_name:
                schema, table = table_name.split(".", 1)
            else:
                schema = "USER"  # Default schema
                table = table_name

            cursor = connection.cursor()
            # Use IRIS system tables to check existence
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
            """,
                [schema, table],
            )

            result = cursor.fetchone()
            cursor.close()
            return result[0] > 0

        except Exception as e:
            logger.error(f"Failed to check table existence for {table_name}: {e}")
            return False

    @staticmethod
    def check_index_exists(
        connection: Any, index_name: str, table_name: Optional[str] = None
    ) -> bool:
        """
        Check if an index exists using IRIS-compatible syntax.
        Uses a self-managed database connection to avoid "DataRow is inaccessible" errors.

        Args:
            connection: Database connection (kept for compatibility, but not used)
            index_name: Name of the index
            table_name: Optional table name to filter by

        Returns:
            True if index exists, False otherwise
        """
        # Use self-managed connection to avoid interference with other operations
        if get_iris_dbapi_connection is None:
            logger.warning(
                "Self-managed connections not available, falling back to provided connection"
            )
            return IRISSQLUtils._check_index_exists_with_connection(
                connection, index_name, table_name
            )

        temp_connection = None
        cursor = None
        try:
            temp_connection = get_iris_dbapi_connection()
            if not temp_connection:
                logger.error(
                    "Failed to obtain temporary connection for index existence check"
                )
                return False

            cursor = temp_connection.cursor()

            # IRIS doesn't have INFORMATION_SCHEMA.STATISTICS, use system tables
            if table_name:
                # Split schema and table name
                if "." in table_name:
                    schema, table = table_name.split(".", 1)
                else:
                    schema = "USER"
                    table = table_name

                # Check for index on specific table
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM %Dictionary.IndexDefinition
                    WHERE parent = ? AND name = ?
                """,
                    [f"{schema}.{table}", index_name],
                )
            else:
                # Check for index globally
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM %Dictionary.IndexDefinition
                    WHERE name = ?
                """,
                    [index_name],
                )

            result = cursor.fetchone()
            return result[0] > 0

        except Exception as e:
            logger.debug(f"Could not check index existence for {index_name}: {e}")
            # Fallback: assume index doesn't exist
            return False
        finally:
            # Ensure cleanup of temporary connection and cursor
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if temp_connection:
                try:
                    temp_connection.close()
                except:
                    pass

    @staticmethod
    def _check_index_exists_with_connection(
        connection: Any, index_name: str, table_name: Optional[str] = None
    ) -> bool:
        """
        Fallback method using provided connection for index existence check.
        """
        try:
            cursor = connection.cursor()

            # IRIS doesn't have INFORMATION_SCHEMA.STATISTICS, use system tables
            if table_name:
                # Split schema and table name
                if "." in table_name:
                    schema, table = table_name.split(".", 1)
                else:
                    schema = "USER"
                    table = table_name

                # Check for index on specific table
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM %Dictionary.IndexDefinition
                    WHERE parent = ? AND name = ?
                """,
                    [f"{schema}.{table}", index_name],
                )
            else:
                # Check for index globally
                cursor.execute(
                    """
                    SELECT COUNT(*)
                    FROM %Dictionary.IndexDefinition
                    WHERE name = ?
                """,
                    [index_name],
                )

            result = cursor.fetchone()
            cursor.close()
            return result[0] > 0

        except Exception as e:
            logger.debug(f"Could not check index existence for {index_name}: {e}")
            # Fallback: assume index doesn't exist
            return False

    @staticmethod
    def get_table_columns(connection: Any, table_name: str) -> List[Dict[str, Any]]:
        """
        Get column information for a table using IRIS-compatible syntax.
        Uses a self-managed database connection to avoid "DataRow is inaccessible" errors.

        Args:
            connection: Database connection (kept for compatibility, but not used)
            table_name: Name of the table (schema.table format)

        Returns:
            List of dictionaries with column information
        """
        # Use self-managed connection to avoid interference with other operations
        if get_iris_dbapi_connection is None:
            logger.warning(
                "Self-managed connections not available, falling back to provided connection"
            )
            return IRISSQLUtils._get_table_columns_with_connection(
                connection, table_name
            )

        temp_connection = None
        cursor = None
        try:
            temp_connection = get_iris_dbapi_connection()
            if not temp_connection:
                logger.error(
                    "Failed to obtain temporary connection for table columns check"
                )
                return []

            # Split schema and table name
            if "." in table_name:
                schema, table = table_name.split(".", 1)
            else:
                schema = "USER"
                table = table_name

            cursor = temp_connection.cursor()
            cursor.execute(
                """
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
            """,
                [schema, table],
            )

            columns = []
            for row in cursor.fetchall():
                columns.append(
                    {
                        "name": row[0],
                        "type": row[1],
                        "nullable": row[2],
                        "default": row[3],
                    }
                )

            return columns

        except Exception as e:
            logger.error(f"Failed to get columns for {table_name}: {e}")
            return []
        finally:
            # Ensure cleanup of temporary connection and cursor
            if cursor:
                try:
                    cursor.close()
                except:
                    pass
            if temp_connection:
                try:
                    temp_connection.close()
                except:
                    pass

    @staticmethod
    def _get_table_columns_with_connection(
        connection: Any, table_name: str
    ) -> List[Dict[str, Any]]:
        """
        Fallback method using provided connection for table columns check.
        """
        try:
            # Split schema and table name
            if "." in table_name:
                schema, table = table_name.split(".", 1)
            else:
                schema = "USER"
                table = table_name

            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?
                ORDER BY ORDINAL_POSITION
            """,
                [schema, table],
            )

            columns = []
            for row in cursor.fetchall():
                columns.append(
                    {
                        "name": row[0],
                        "type": row[1],
                        "nullable": row[2],
                        "default": row[3],
                    }
                )

            cursor.close()
            return columns

        except Exception as e:
            logger.error(f"Failed to get columns for {table_name}: {e}")
            return []

    @staticmethod
    def optimize_vector_table(
        connection: Any,
        table_name: str,
        vector_column: str = "embedding",
        vector_config: dict = None,
    ) -> bool:
        """
        Optimize a table for vector operations by creating appropriate indices.

        Args:
            connection: Database connection
            table_name: Name of the table
            vector_column: Name of the vector column
            vector_config: Vector index configuration with HNSW parameters

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Optimizing table {table_name} for vector operations...")

            # Get vector index configuration with defaults
            if vector_config is None:
                vector_config = {
                    "type": "HNSW",
                    "M": 16,
                    "efConstruction": 200,
                    "Distance": "COSINE",
                }

            # Create vector index for similarity search with configurable parameters
            vector_success = IRISSQLUtils.create_vector_index_with_config(
                connection, table_name, vector_column, vector_config
            )

            # Create simple indices on commonly queried columns
            common_columns = ["doc_id", "id", "source_doc_id"]
            for col in common_columns:
                # Check if column exists first
                columns = IRISSQLUtils.get_table_columns(connection, table_name)
                column_names = [c["name"].lower() for c in columns]

                if col.lower() in column_names:
                    IRISSQLUtils.create_simple_index(connection, table_name, col)

            return vector_success

        except Exception as e:
            logger.error(f"Failed to optimize table {table_name}: {e}")
            return False

    @staticmethod
    def create_vector_index_with_config(
        connection: Any,
        table_name: str,
        column_name: str,
        vector_config: dict,
        index_name: Optional[str] = None,
    ) -> bool:
        """
        Create a vector index using configurable HNSW parameters.

        Args:
            connection: Database connection
            table_name: Name of the table
            column_name: Name of the vector column
            vector_config: Vector index configuration with HNSW parameters
            index_name: Optional custom index name

        Returns:
            True if successful, False otherwise
        """
        if not index_name:
            # Clean table name for index naming
            clean_table = table_name.replace(".", "_")
            index_name = f"{clean_table}_{column_name}_vector_idx"

        try:
            # Build HNSW parameters string
            hnsw_params = []
            if "M" in vector_config:
                hnsw_params.append(f"M={vector_config['M']}")
            if "efConstruction" in vector_config:
                hnsw_params.append(f"efConstruction={vector_config['efConstruction']}")
            if "Distance" in vector_config:
                hnsw_params.append(f"Distance='{vector_config['Distance']}'")

            params_str = ", ".join(hnsw_params)

            # Create HNSW vector index with proper parameters
            # Note: IRIS doesn't support "IF NOT EXISTS" in CREATE INDEX statements
            if params_str:
                sql = f"""
                CREATE INDEX {index_name}
                ON {table_name} ({column_name})
                AS HNSW({params_str})
                """
            else:
                sql = f"""
                CREATE INDEX {index_name}
                ON {table_name} ({column_name})
                AS HNSW(M=16, efConstruction=200, Distance='COSINE')
                """

            # Check if index already exists first
            if IRISSQLUtils.check_index_exists(connection, index_name, table_name):
                logger.debug(f"Vector index {index_name} already exists")
                return True

            cursor = connection.cursor()
            cursor.execute(sql)
            connection.commit()
            cursor.close()
            logger.info(
                f"✓ Created vector index {index_name} on {table_name}.{column_name} with parameters: {params_str}"
            )
            return True

        except Exception as e:
            error_msg = str(e).lower()
            # Handle IRIS-specific error codes and messages
            if (
                "already exists" in error_msg
                or "duplicate" in error_msg
                or "sqlcode: <-324>" in error_msg
                or "index with this name already defined" in error_msg
            ):
                logger.debug(f"Vector index {index_name} already exists")
                return True
            else:
                logger.error(f"Failed to create vector index {index_name}: {e}")
                return False


def test_iris_sql_utils():
    """Test the IRIS SQL utilities with self-managed connections."""
    print("=== Testing IRIS SQL Utilities ===")

    try:
        # Test with self-managed connections (pass None as connection since functions will create their own)
        utils = IRISSQLUtils()

        # Test table existence check
        exists = utils.check_table_exists(None, "RAG.SourceDocuments")
        print(f"✓ Table existence check: {exists}")

        # Test column information
        columns = utils.get_table_columns(None, "RAG.SourceDocuments")
        print(f"✓ Found {len(columns)} columns")

        # Test index existence check
        index_exists = utils.check_index_exists(
            None, "RAG_SourceDocuments_embedding_vector_idx", "RAG.SourceDocuments"
        )
        print(f"✓ Index existence check: {index_exists}")

        # For index creation, we still need a connection since it performs DDL operations
        if get_iris_dbapi_connection:
            connection = get_iris_dbapi_connection()
            if connection:
                # Test vector index creation (this should work with proper syntax)
                print("Testing vector index creation...")
                success = utils.create_vector_index(
                    connection,
                    "RAG.SourceDocuments",
                    "embedding",
                    "test_vector_idx",
                    "COSINE",
                    "HNSW",
                )
                print(f"✓ Vector index creation: {success}")

                # Clean up test index
                if success:
                    try:
                        cursor = connection.cursor()
                        cursor.execute("DROP INDEX test_vector_idx")
                        cursor.close()
                        print("✓ Test index cleaned up")
                    except:
                        pass

                connection.close()
            else:
                print("⚠️ Could not get database connection for index creation test")
        else:
            print("⚠️ Self-managed connections not available for index creation test")

        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_iris_sql_utils()
