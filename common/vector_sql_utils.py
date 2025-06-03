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

For more details on these limitations, see docs/IRIS_SQL_VECTOR_OPERATIONS.md
"""

import logging
import re
from typing import Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def validate_vector_string(vector_string: str) -> bool:
    """
    Validates that a vector string contains only valid characters.
    This is important for security when using string interpolation.

    Args:
        vector_string: The vector string to validate, typically in format "[0.1,0.2,...]"

    Returns:
        bool: True if the vector string contains only valid characters, False otherwise

    Example:
        >>> validate_vector_string("[0.1,0.2,0.3]")
        True
        >>> validate_vector_string("'; DROP TABLE users; --")
        False
    """
    # Only allow digits, dots, commas, and square brackets
    allowed_chars = set("0123456789.[],")
    return all(c in allowed_chars for c in vector_string)


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
    additional_where: str = None
) -> str:
    """
    Constructs a SQL query for vector search using string interpolation.
    Validates inputs to prevent SQL injection.

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
            VECTOR_COSINE(embedding, TO_VECTOR('[0.1,0.2,0.3]', 'DOUBLE', 768)) AS score
         FROM SourceDocuments
         WHERE embedding IS NOT NULL
         ORDER BY score DESC'
    """
    # Validate table_name to prevent SQL injection
    if not re.match(r'^[a-zA-Z0-9_]+$', table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    # Validate column names to prevent SQL injection
    for col in [vector_column, id_column]:
        if not re.match(r'^[a-zA-Z0-9_]+$', col):
            raise ValueError(f"Invalid column name: {col}")

    if content_column and not re.match(r'^[a-zA-Z0-9_]+$', content_column):
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

    # Construct the SELECT clause
    select_clause = f"SELECT TOP {top_k} {id_column}"
    if content_column:
        select_clause += f", {content_column}"
    select_clause += f", VECTOR_COSINE({vector_column}, TO_VECTOR('{vector_string}', 'FLOAT', {embedding_dim})) AS score"

    # Construct the WHERE clause
    where_clause = f"WHERE {vector_column} IS NOT NULL"
    if additional_where:
        where_clause += f" AND ({additional_where})"

    # Construct the full SQL query
    sql = f"""
        {select_clause}
        FROM {table_name}
        {where_clause}
        ORDER BY score DESC
    """

    return sql


def execute_vector_search(
    cursor: Any,
    sql: str
) -> List[Tuple]:
    """
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