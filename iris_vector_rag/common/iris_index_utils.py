"""
IRIS index creation utilities with proper error handling.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def create_index_if_not_exists(
    cursor,
    index_name: str,
    table_name: str,
    columns: str,
    index_type: Optional[str] = None,
):
    """
    Create an index if it doesn't already exist.

    Args:
        cursor: Database cursor
        index_name: Name of the index
        table_name: Name of the table
        columns: Column specification for the index
        index_type: Optional index type (e.g., "AS HNSW(M=16, efConstruction=200, Distance='COSINE')")
    """
    try:
        # Check if index already exists
        check_sql = """
        SELECT COUNT(*) FROM INFORMATION_SCHEMA.INDEXES 
        WHERE INDEX_NAME = ? AND TABLE_NAME = ?
        """
        cursor.execute(
            check_sql, (index_name, table_name.split(".")[-1])
        )  # Remove schema prefix for check
        result = cursor.fetchone()

        if result and result[0] > 0:
            logger.debug(f"Index {index_name} already exists on {table_name}")
            return True

        # Create the index
        if index_type:
            create_sql = (
                f"CREATE INDEX {index_name} ON {table_name} ({columns}) {index_type}"
            )
        else:
            create_sql = f"CREATE INDEX {index_name} ON {table_name} ({columns})"

        logger.info(f"Creating index: {create_sql}")
        cursor.execute(create_sql)
        logger.info(f"Successfully created index {index_name}")
        return True

    except Exception as e:
        error_str = str(e).lower()

        # Check if error is due to index already existing
        if any(
            indicator in error_str
            for indicator in [
                "already exists",
                "duplicate",
                "index exists",
                "name already used",
            ]
        ):
            logger.debug(f"Index {index_name} already exists (caught exception): {e}")
            return True
        else:
            logger.error(f"Failed to create index {index_name}: {e}")
            return False


def create_indexes_from_sql_file(cursor, sql_file_path: str) -> List[str]:
    """
    Create indexes from SQL file with proper error handling.

    Args:
        cursor: Database cursor
        sql_file_path: Path to SQL file containing index creation statements

    Returns:
        List of failed index creation statements
    """
    failed_statements = []

    try:
        with open(sql_file_path, "r") as f:
            sql_content = f.read()

        # Split into individual statements
        statements = [stmt.strip() for stmt in sql_content.split(";") if stmt.strip()]

        for statement in statements:
            if statement.upper().startswith("CREATE INDEX"):
                try:
                    # Replace "CREATE INDEX IF NOT EXISTS" with "CREATE INDEX"
                    statement = statement.replace("IF NOT EXISTS", "").replace(
                        "if not exists", ""
                    )

                    logger.debug(f"Executing: {statement}")
                    cursor.execute(statement)
                    logger.debug(f"Successfully executed: {statement[:50]}...")

                except Exception as e:
                    error_str = str(e).lower()

                    # Check if error is due to index already existing
                    if any(
                        indicator in error_str
                        for indicator in [
                            "already exists",
                            "duplicate",
                            "index exists",
                            "name already used",
                        ]
                    ):
                        logger.debug(
                            f"Index already exists (ignored): {statement[:50]}..."
                        )
                    else:
                        logger.warning(
                            f"Failed to execute statement: {statement[:50]}... Error: {e}"
                        )
                        failed_statements.append(statement)
            else:
                # Execute non-index statements normally
                if statement and not statement.startswith("--"):
                    try:
                        cursor.execute(statement)
                    except Exception as e:
                        logger.warning(
                            f"Failed to execute statement: {statement[:50]}... Error: {e}"
                        )
                        failed_statements.append(statement)

    except Exception as e:
        logger.error(f"Failed to read SQL file {sql_file_path}: {e}")
        failed_statements.append(f"Failed to read file: {sql_file_path}")

    return failed_statements


def ensure_schema_indexes(cursor, schema_name: str = "RAG") -> bool:
    """
    Ensure all required indexes exist for the RAG schema.

    Args:
        cursor: Database cursor
        schema_name: Name of the schema

    Returns:
        True if all indexes were created successfully, False otherwise
    """
    indexes = [
        # SourceDocuments indexes
        ("idx_source_docs_id", f"{schema_name}.SourceDocuments", "doc_id"),
        (
            "idx_hnsw_source_embedding",
            f"{schema_name}.SourceDocuments",
            "embedding",
            "AS HNSW(M=16, efConstruction=200, Distance='COSINE')",
        ),
        ("idx_source_docs_created", f"{schema_name}.SourceDocuments", "created_at"),
        # DocumentChunks indexes
        ("idx_chunks_doc_id", f"{schema_name}.DocumentChunks", "doc_id"),
        ("idx_chunks_type", f"{schema_name}.DocumentChunks", "chunk_type"),
        (
            "idx_hnsw_chunk_embedding",
            f"{schema_name}.DocumentChunks",
            "chunk_embedding",
            "AS HNSW(M=16, efConstruction=200, Distance='COSINE')",
        ),
    ]

    success_count = 0
    total_count = len(indexes)

    for index_spec in indexes:
        if len(index_spec) == 3:
            index_name, table_name, columns = index_spec
            index_type = None
        else:
            index_name, table_name, columns, index_type = index_spec

        if create_index_if_not_exists(
            cursor, index_name, table_name, columns, index_type
        ):
            success_count += 1

    logger.info(f"Successfully created/verified {success_count}/{total_count} indexes")
    return success_count == total_count
