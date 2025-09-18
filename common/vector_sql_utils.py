"""
Vector SQL Utilities for IRIS Database

This module encapsulates workarounds for InterSystems IRIS SQL vector operations limitations.
It provides helper functions that RAG pipelines can use to safely construct SQL queries
with vector operations.

Key IRIS SQL Limitations Addressed:

1. TO_VECTOR() Function Rejects Parameter Markers:
   The TO_VECTOR() function does not accept parameter markers (?, :param, or :%qpar),
   which are standard in SQL for safe query parameterization.

2. TOP/FETCH FIRST Clauses Cannot Be Parameterized:
   The TOP and FETCH FIRST clauses, essential for limiting results in vector similarity
   searches, do not accept parameter markers.

3. Client Drivers Rewrite Literals:
   Python, JDBC, and other client drivers replace embedded literals with :%qpar(n)
   even when no parameter list is supplied, creating misleading parse errors.

These limitations force developers to use string interpolation instead of parameterized
queries, which introduces potential security risks. This module provides functions to
validate inputs and safely construct SQL queries using string interpolation.

SAFE VECTOR UTILITIES:
- build_safe_vector_dot_sql(): Safe single-parameter query builder
- execute_safe_vector_search(): Safe execution with single vector parameter

QUARANTINED UTILITIES (pending IRIS fix):
- format_vector_search_sql(): Deprecated - see docs/reports/IRIS_VECTOR_SQL_PARAMETERIZATION_REPRO.md
- format_vector_search_sql_with_params(): Deprecated - see docs/reports/IRIS_VECTOR_SQL_PARAMETERIZATION_REPRO.md
- execute_vector_search_with_params(): Deprecated - see docs/reports/IRIS_VECTOR_SQL_PARAMETERIZATION_REPRO.md

For more details on these limitations, see docs/IRIS_SQL_VECTOR_OPERATIONS.md
"""

import logging
import re
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


def validate_vector_string(vector_string: str) -> bool:
    """
    Validates that a vector string contains a valid vector format.
    Allows negative numbers and scientific notation while preventing SQL injection.

    Args:
        vector_string: The vector string to validate, typically in format "[0.1,-0.2,...]"

    Returns:
        bool: True if valid vector format, False otherwise

    Example:
        >>> validate_vector_string("[0.1,-0.2,3.5e-4]")
        True
        >>> validate_vector_string("'; DROP TABLE users; --")
        False
    """
    # Check basic structure
    stripped = vector_string.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return False

    # Extract content between brackets
    content = stripped[1:-1]
    if not content.strip():
        return False

    # Validate each number
    parts = content.split(",")
    for part in parts:
        try:
            float(part.strip())
        except ValueError:
            return False

    # Check for SQL injection patterns
    if re.search(
        r"(DROP|DELETE|INSERT|UPDATE|SELECT|;|--)", vector_string, re.IGNORECASE
    ):
        return False

    return True


def validate_top_k(top_k: Any) -> bool:
    """
    Validates that top_k is a positive integer.
    This is important for security when using string interpolation.

    Args:
        top_k: The value to validate

    Returns:
        bool: True if top_k is a positive integer, False otherwise

    Example:
        >>> validate_top_k(10)
        True
        >>> validate_top_k(0)
        False
        >>> validate_top_k("10; DROP TABLE users; --")
        False
    """
    if not isinstance(top_k, int):
        return False
    return top_k > 0


def format_vector_search_sql(
    table_name: str,
    vector_column: str,
    vector_string: str,
    embedding_dim: int,
    top_k: int,
    id_column: str = "doc_id",
    content_column: str = "text_content",
    additional_where: str = None,
) -> str:
    """
    @deprecated - Quarantined pending IRIS fix
    See docs/reports/IRIS_VECTOR_SQL_PARAMETERIZATION_REPRO.md

    Constructs a SQL query for vector search using careful string concatenation
    to avoid IRIS driver auto-parameterization issues.

    IRIS-specific workaround: The IRIS driver converts embedded literals to :%qpar(n)
    parameter markers automatically, but TO_VECTOR() and TOP clauses cannot accept
    parameter markers. This function constructs SQL to avoid triggering the
    auto-parameterization.

    Args:
        table_name: The name of the table to search
        vector_column: The name of the column containing vector embeddings
        vector_string: The vector string to search for, in format "[0.1,0.2,...]"
        embedding_dim: The dimension of the embedding vectors
        top_k: The number of results to return
        id_column: The name of the ID column (default: "doc_id")
        content_column: The name of the content column (default: "text_content")
                        Set to None if you don't want to include content in results
        additional_where: Additional WHERE clause conditions (default: None)

    Returns:
        str: The formatted SQL query string

    Raises:
        ValueError: If any of the inputs fail validation

    Example:
        >>> format_vector_search_sql(
        ...     "SourceDocuments",
        ...     "embedding",
        ...     "[0.1,0.2,0.3]",
        ...     768,
        ...     10,
        ...     "doc_id",
        ...     "text_content"
        ... )
        'SELECT TOP 10 doc_id, text_content,
            VECTOR_COSINE(embedding, TO_VECTOR('[0.1,0.2,0.3]', 'FLOAT', 768)) AS score
         FROM SourceDocuments
         WHERE embedding IS NOT NULL
         ORDER BY score DESC'
    """
    # Validate table_name to prevent SQL injection (allow schema.table format)
    if not re.match(r"^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)?$", table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    # Validate column names to prevent SQL injection
    for col in [vector_column, id_column]:
        if not re.match(r"^[a-zA-Z0-9_]+$", col):
            raise ValueError(f"Invalid column name: {col}")

    if content_column and not re.match(r"^[a-zA-Z0-9_]+$", content_column):
        raise ValueError(f"Invalid content column name: {content_column}")

    # Validate vector_string
    if not validate_vector_string(vector_string):
        raise ValueError(f"Invalid vector string: {vector_string}")

    # Validate embedding_dim
    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise ValueError(f"Invalid embedding dimension: {embedding_dim}")

    # Validate top_k
    if not validate_top_k(top_k):
        raise ValueError(f"Invalid top_k value: {top_k}")

    # Convert values to strings to avoid auto-parameterization
    top_k_str = str(top_k)
    embedding_dim_str = str(embedding_dim)

    # Construct SQL using string concatenation to avoid f-string literal detection
    select_parts = ["SELECT TOP ", top_k_str, " ", id_column]
    if content_column:
        select_parts.extend([", ", content_column])

    # Construct TO_VECTOR call carefully to avoid parameter detection
    vector_func_parts = [
        ", VECTOR_COSINE(",
        vector_column,
        ", TO_VECTOR('",
        vector_string,
        "', 'FLOAT', ",
        embedding_dim_str,
        ")) AS score",
    ]
    select_parts.extend(vector_func_parts)
    select_clause = "".join(select_parts)

    # Construct the WHERE clause
    where_parts = ["WHERE ", vector_column, " IS NOT NULL"]
    if additional_where:
        where_parts.extend([" AND (", additional_where, ")"])
    where_clause = "".join(where_parts)

    # Construct the full SQL query using string concatenation
    sql_parts = [
        select_clause,
        " FROM ",
        table_name,
        " ",
        where_clause,
        " ORDER BY score DESC",
    ]

    return "".join(sql_parts)


def format_vector_search_sql_with_params(
    table_name: str,
    vector_column: str,
    embedding_dim: int,
    top_k: int,
    id_column: str = "doc_id",
    content_column: str = "text_content",
    additional_where: str = None,
) -> str:
    """
    @deprecated - Quarantined pending IRIS fix
    See docs/reports/IRIS_VECTOR_SQL_PARAMETERIZATION_REPRO.md

    Constructs a SQL query for vector search using parameter placeholders.
    Uses string concatenation to avoid IRIS driver auto-parameterization issues.

    IRIS-specific workaround: Even when using ? placeholders, the IRIS driver
    can still auto-parameterize literals in f-strings. This function uses
    string concatenation to avoid triggering the auto-parameterization.

    Args:
        table_name: The name of the table to search
        vector_column: The name of the column containing vector embeddings
        embedding_dim: The dimension of the embedding vectors (for documentation)
        top_k: The number of results to return
        id_column: The name of the ID column (default: "doc_id")
        content_column: The name of the content column (default: "text_content")
        additional_where: Additional WHERE clause conditions (default: None)

    Returns:
        str: The formatted SQL query string with ? placeholder
    """
    # Validate inputs (reuse existing validation)
    if not re.match(r"^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)?$", table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    for col in [vector_column, id_column]:
        if not re.match(r"^[a-zA-Z0-9_]+$", col):
            raise ValueError(f"Invalid column name: {col}")

    if content_column and not re.match(r"^[a-zA-Z0-9_]+$", content_column):
        raise ValueError(f"Invalid content column name: {content_column}")

    if not validate_top_k(top_k):
        raise ValueError(f"Invalid top_k value: {top_k}")

    # Convert values to strings to avoid auto-parameterization
    top_k_str = str(top_k)
    embedding_dim_str = str(embedding_dim)

    # Construct SQL using string concatenation to avoid f-string literal detection
    select_parts = ["SELECT TOP ", top_k_str, " ", id_column]
    if content_column:
        select_parts.extend([", ", content_column])

    # Construct TO_VECTOR call with ? placeholder but avoid embedding dimension parameterization
    vector_func_parts = [
        ", VECTOR_COSINE(",
        vector_column,
        ", TO_VECTOR(?, 'FLOAT', ",
        embedding_dim_str,
        ")) AS score",
    ]
    select_parts.extend(vector_func_parts)
    select_clause = "".join(select_parts)

    # Construct the WHERE clause
    where_parts = ["WHERE ", vector_column, " IS NOT NULL"]
    if additional_where:
        where_parts.extend([" AND (", additional_where, ")"])
    where_clause = "".join(where_parts)

    # Construct the full SQL query using string concatenation
    sql_parts = [
        select_clause,
        " FROM ",
        table_name,
        " ",
        where_clause,
        " ORDER BY score DESC",
    ]

    return "".join(sql_parts)


def execute_vector_search_with_params(
    cursor: Any, sql: str, vector_string: str, table_name: str = "RAG.SourceDocuments"
) -> List[Tuple]:
    """
    @deprecated - Quarantined pending IRIS fix
    See docs/reports/IRIS_VECTOR_SQL_PARAMETERIZATION_REPRO.md

    Executes a vector search SQL query using parameters.

    Args:
        cursor: A database cursor object
        sql: The SQL query with ? placeholder
        vector_string: The vector string to use as parameter
        table_name: The table name for diagnostic queries (optional, defaults to RAG.SourceDocuments)

    Returns:
        List[Tuple]: The query results
    """
    results = []
    try:
        # Use the provided table name directly instead of parsing from SQL
        logger.debug(f"Using table name: {table_name}")

        count_sql = f"SELECT COUNT(*) FROM {table_name} WHERE embedding IS NOT NULL"
        logger.debug(f"Executing count SQL: {count_sql}")
        try:
            cursor.execute(count_sql)
            embedding_result = cursor.fetchone()
            # Handle both real results and mock objects
            if embedding_result:
                try:
                    embedding_count = (
                        embedding_result[0]
                        if hasattr(embedding_result, "__getitem__")
                        else 0
                    )
                except (TypeError, IndexError):
                    # Handle Mock objects or other non-subscriptable results
                    embedding_count = 0
            else:
                embedding_count = 0
            logger.debug(
                f"Table {table_name} has {embedding_count} rows with embeddings"
            )
        except Exception as count_error:
            logger.error(f"Error executing count SQL: {count_error}")
            logger.error(f"Count SQL was: {count_sql}")
            # Skip count check and proceed with vector search
            embedding_count = 0

        # Also check total rows
        total_sql = f"SELECT COUNT(*) FROM {table_name}"
        logger.debug(f"Executing total SQL: {total_sql}")
        try:
            cursor.execute(total_sql)
            total_result = cursor.fetchone()
            # Handle both real results and mock objects
            if total_result:
                try:
                    total_count = (
                        total_result[0] if hasattr(total_result, "__getitem__") else 0
                    )
                except (TypeError, IndexError):
                    # Handle Mock objects or other non-subscriptable results
                    total_count = 0
            else:
                total_count = 0
            logger.debug(f"Table {table_name} has {total_count} total rows")
        except Exception as total_error:
            logger.error(f"Error executing total count SQL: {total_error}")
            logger.error(f"Total SQL was: {total_sql}")
            # Skip total count check and proceed with vector search
            total_count = 0

        logger.debug(f"Executing vector search SQL: {sql}")
        logger.debug(f"Vector string parameter: {vector_string[:100]}...")

        # Execute the SQL with parameter binding
        cursor.execute(sql, [vector_string])

        # Try to fetch results with better error handling
        try:
            fetched_rows = cursor.fetchall()
            if fetched_rows:
                results = fetched_rows
                # Handle Mock objects that don't have len()
                try:
                    result_count = len(results)
                    logger.debug(f"Found {result_count} results.")
                except (TypeError, AttributeError):
                    # Handle Mock objects or other non-sequence types
                    logger.debug("Found results (count unavailable due to mock object)")
            else:
                logger.debug("No results returned from vector search")
        except StopIteration as e:
            logger.error(f"StopIteration error during fetchall(): {e}")
            logger.error(
                "This usually indicates the cursor is empty or in an invalid state"
            )
            # Return empty results instead of raising
            results = []
        except Exception as fetch_error:
            logger.error(f"Error during fetchall(): {fetch_error}")
            raise
    except Exception as e:
        logger.error(f"Error during vector search: {e}")
        logger.error(f"SQL was: {sql}")
        logger.error(f"Vector parameter was: {vector_string[:100]}...")
        raise
    return results


# =============================================================================
# SAFE VECTOR UTILITIES (PROVEN PATTERN)
# =============================================================================


def build_safe_vector_dot_sql(
    table: str,
    vector_column: str,
    id_column: str = "doc_id",
    extra_columns: list[str] | None = None,
    top_k: int = 5,
    additional_where: str | None = None,
) -> str:
    """
    Build safe vector search SQL using single parameter pattern.

    This is the ONLY proven pattern for IRIS vector queries that works reliably.
    Uses VECTOR_DOT_PRODUCT with TO_VECTOR(?) for single parameter binding.

    Args:
        table: Table name (e.g., "RAG.SourceDocuments")
        vector_column: Vector column name (e.g., "embedding")
        id_column: ID column name (default: "doc_id")
        extra_columns: Additional columns to select (optional)
        top_k: Number of results to return (default: 5)
        additional_where: Additional WHERE conditions (optional)

    Returns:
        str: Safe SQL query with single ? parameter for vector

    Example:
        sql = build_safe_vector_dot_sql("RAG.SourceDocuments", "embedding", "doc_id", ["title"], 5)
        results = execute_safe_vector_search(cursor, sql, [0.1, 0.2, 0.3])
    """
    # Validate identifiers to prevent SQL injection
    import re

    # Validate table name (allow schema.table format)
    if not re.match(r"^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)?$", table):
        raise ValueError(f"Invalid table name: {table}")

    # Validate column names
    for col in [vector_column, id_column]:
        if not re.match(r"^[a-zA-Z0-9_]+$", col):
            raise ValueError(f"Invalid column name: {col}")

    if extra_columns:
        for col in extra_columns:
            if not re.match(r"^[a-zA-Z0-9_]+$", col):
                raise ValueError(f"Invalid extra column name: {col}")

    # Validate top_k
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"Invalid top_k value: {top_k}")

    # Build SELECT clause
    select_parts = [f"SELECT TOP {top_k} {id_column}"]
    if extra_columns:
        select_parts.extend([f", {col}" for col in extra_columns])
    select_parts.append(f", VECTOR_DOT_PRODUCT({vector_column}, TO_VECTOR(?)) AS score")

    # Build FROM clause
    from_clause = f" FROM {table}"

    # Build WHERE clause
    where_parts = [f" WHERE {vector_column} IS NOT NULL"]
    if additional_where:
        where_parts.append(f" AND ({additional_where})")

    # Build ORDER BY clause
    order_clause = " ORDER BY score DESC"

    # Combine all parts
    sql = "".join(select_parts) + from_clause + "".join(where_parts) + order_clause
    return sql


def execute_safe_vector_search(
    cursor, sql: str, vector_list: list[float]
) -> list[tuple]:
    """
    Execute safe vector search with single parameter.

    This function safely executes vector queries using the single parameter pattern
    that is proven to work with IRIS vector operations.

    Args:
        cursor: Database cursor
        sql: SQL query from build_safe_vector_dot_sql()
        vector_list: Vector as list of floats (e.g., [0.1, 0.2, 0.3])

    Returns:
        list[tuple]: Query results

    Example:
        sql = build_safe_vector_dot_sql("RAG.SourceDocuments", "embedding")
        results = execute_safe_vector_search(cursor, sql, [0.1, 0.2, 0.3])
    """
    # Convert vector list to comma-separated string
    vector_str = ",".join(map(str, vector_list))

    # Execute with single parameter
    cursor.execute(sql, (vector_str,))
    return cursor.fetchall()


def execute_vector_search(cursor: Any, sql: str) -> List[Tuple]:
    """
    @deprecated - Quarantined pending IRIS fix
    See docs/reports/IRIS_VECTOR_SQL_PARAMETERIZATION_REPRO.md

    Executes a vector search SQL query using the provided cursor.
    Handles common errors and returns the results.

    Args:
        cursor: A database cursor object
        sql: The SQL query to execute

    Returns:
        List[Tuple]: The query results

    Raises:
        Exception: If the query execution fails

    Example:
        >>> cursor = connection.cursor()
        >>> sql = format_vector_search_sql(...)
        >>> results = execute_vector_search(cursor, sql)
    """
    results = []
    try:
        logger.debug(f"Executing vector search SQL: {sql[:100]}...")
        cursor.execute(sql)  # No parameters passed as all are interpolated
        fetched_rows = cursor.fetchall()
        if fetched_rows:
            results = fetched_rows
        logger.debug(f"Found {len(results)} results.")
    except Exception as e:
        logger.error(f"Error during vector search: {e}")
        # Re-raise the exception so the calling pipeline can handle it
        raise
    return results
