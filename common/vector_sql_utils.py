"""
Vector SQL Utilities for IRIS Database

This module provides driver-aware SQL generation for InterSystems IRIS vector operations.
It automatically detects the database driver type (DBAPI vs JDBC) and generates
appropriate SQL syntax to work around driver-specific limitations.

Key IRIS SQL Limitations Addressed:

1. TO_VECTOR() Function Rejects Parameter Markers:
   The TO_VECTOR() function does not accept parameter markers (?, :param, or :%qpar),
   which are standard in SQL for safe query parameterization.

2. TOP/FETCH FIRST Clauses Cannot Be Parameterized:
   The TOP and FETCH FIRST clauses, essential for limiting results in vector similarity
   searches, do not accept parameter markers.

3. JDBC Driver Auto-Parameterization Bug:
   The JDBC driver incorrectly converts fully interpolated SQL strings into parameterized
   queries by replacing literal values with :%qpar(n) parameter markers, causing failures.

4. Driver-Specific Capabilities:
   - DBAPI: Supports vector operations with proper parameter handling
   - JDBC: Limited vector support, requires string interpolation workarounds

This module provides functions to validate inputs and generate driver-appropriate SQL
queries for vector operations.

For more details on these limitations, see docs/architecture/IRIS_DRIVER_COMPATIBILITY.md
"""

import logging
import re
from typing import Any, List, Tuple, Optional

# Import driver detection capabilities
from common.db_driver_utils import (
    get_driver_type,
    get_driver_capabilities,
    DriverType
)

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
    if not (stripped.startswith('[') and stripped.endswith(']')):
        return False
    
    # Extract content between brackets
    content = stripped[1:-1]
    if not content.strip():
        return False
    
    # Validate each number
    parts = content.split(',')
    for part in parts:
        try:
            float(part.strip())
        except ValueError:
            return False
    
    # Check for SQL injection patterns
    if re.search(r'(DROP|DELETE|INSERT|UPDATE|SELECT|;|--)', vector_string, re.IGNORECASE):
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


def get_driver_aware_vector_search_sql(
    table_name: str,
    vector_column: str,
    embedding_dim: int,
    top_k: int,
    id_column: str = "doc_id",
    content_column: str = "text_content",
    additional_where: str = None,
    driver_type: Optional[DriverType] = None
) -> Tuple[str, bool]:
    """
    Generates driver-appropriate SQL for vector search operations.
    
    Args:
        table_name: The name of the table to search
        vector_column: The name of the column containing vector embeddings
        embedding_dim: The dimension of the embedding vectors
        top_k: The number of results to return
        id_column: The name of the ID column (default: "doc_id")
        content_column: The name of the content column (default: "text_content")
        additional_where: Additional WHERE clause conditions (default: None)
        driver_type: Override driver type detection (default: None for auto-detect)
        
    Returns:
        Tuple[str, bool]: (SQL query, uses_parameters)
            - SQL query string (with ? placeholder for DBAPI, interpolated for JDBC)
            - Boolean indicating if the query uses parameter markers (True) or string interpolation (False)
    """
    # Auto-detect driver type if not provided
    if driver_type is None:
        driver_type = get_driver_type()
    
    capabilities = get_driver_capabilities(driver_type)
    
    # Validate inputs (reuse existing validation)
    if not re.match(r'^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)?$', table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
    for col in [vector_column, id_column]:
        if not re.match(r'^[a-zA-Z0-9_]+$', col):
            raise ValueError(f"Invalid column name: {col}")
    
    if content_column and not re.match(r'^[a-zA-Z0-9_]+$', content_column):
        raise ValueError(f"Invalid content column name: {content_column}")
    
    if not validate_top_k(top_k):
        raise ValueError(f"Invalid top_k value: {top_k}")
    
    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise ValueError(f"Invalid embedding dimension: {embedding_dim}")
    
    # Construct the SELECT clause
    select_clause = f"SELECT TOP {top_k} {id_column}"
    if content_column:
        select_clause += f", {content_column}"
    
    # Choose vector function syntax based on driver capabilities
    if capabilities.get('supports_vector_operations', False):
        # DBAPI: Use parameter marker
        select_clause += f", VECTOR_COSINE({vector_column}, TO_VECTOR(?, FLOAT, {embedding_dim})) AS score"
        uses_parameters = True
        logger.debug(f"Using DBAPI-compatible SQL with parameter markers for {driver_type}")
    else:
        # JDBC: Use string interpolation placeholder (will be filled later)
        select_clause += f", VECTOR_COSINE({vector_column}, TO_VECTOR('{{vector_string}}', FLOAT, {embedding_dim})) AS score"
        uses_parameters = False
        logger.debug(f"Using JDBC-compatible SQL with string interpolation for {driver_type}")
    
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
    """.strip()
    
    return sql, uses_parameters


def execute_driver_aware_vector_search(
    cursor: Any,
    table_name: str,
    vector_column: str,
    vector_string: str,
    embedding_dim: int,
    top_k: int,
    id_column: str = "doc_id",
    content_column: str = "text_content",
    additional_where: str = None,
    driver_type: Optional[DriverType] = None
) -> List[Tuple]:
    """
    Executes a vector search using driver-appropriate SQL generation and execution.
    
    Args:
        cursor: A database cursor object
        table_name: The name of the table to search
        vector_column: The name of the column containing vector embeddings
        vector_string: The vector string to search for, in format "[0.1,0.2,...]"
        embedding_dim: The dimension of the embedding vectors
        top_k: The number of results to return
        id_column: The name of the ID column (default: "doc_id")
        content_column: The name of the content column (default: "text_content")
        additional_where: Additional WHERE clause conditions (default: None)
        driver_type: Override driver type detection (default: None for auto-detect)
        
    Returns:
        List[Tuple]: The query results
        
    Raises:
        ValueError: If inputs fail validation
        Exception: If query execution fails
    """
    # Validate vector string
    if not validate_vector_string(vector_string):
        raise ValueError(f"Invalid vector string: {vector_string}")
    
    # Auto-detect driver type if not provided
    if driver_type is None:
        driver_type = get_driver_type()
    
    # Get driver-appropriate SQL
    sql, uses_parameters = get_driver_aware_vector_search_sql(
        table_name=table_name,
        vector_column=vector_column,
        embedding_dim=embedding_dim,
        top_k=top_k,
        id_column=id_column,
        content_column=content_column,
        additional_where=additional_where,
        driver_type=driver_type
    )
    
    results = []
    try:
        if uses_parameters:
            # DBAPI: Use parameterized query
            logger.debug(f"Executing DBAPI vector search with parameters")
            cursor.execute(sql, [vector_string])
        else:
            # JDBC: Use string interpolation
            logger.debug(f"Executing JDBC vector search with string interpolation")
            interpolated_sql = sql.format(vector_string=vector_string)
            cursor.execute(interpolated_sql)
        
        fetched_rows = cursor.fetchall()
        if fetched_rows:
            results = fetched_rows
        logger.debug(f"Found {len(results)} results using {driver_type} driver")
        
    except Exception as e:
        logger.error(f"Error during driver-aware vector search with {driver_type}: {e}")
        raise
    
    return results


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
    This function is maintained for backward compatibility but now uses
    driver-aware SQL generation internally.

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
    # Validate vector_string
    if not validate_vector_string(vector_string):
        raise ValueError(f"Invalid vector string: {vector_string}")

    # Get driver-aware SQL
    sql, uses_parameters = get_driver_aware_vector_search_sql(
        table_name=table_name,
        vector_column=vector_column,
        embedding_dim=embedding_dim,
        top_k=top_k,
        id_column=id_column,
        content_column=content_column,
        additional_where=additional_where
    )
    
    # If the SQL uses parameters, interpolate the vector string
    if uses_parameters:
        # For DBAPI, we need to convert to string interpolation for backward compatibility
        sql = sql.replace("TO_VECTOR(?, FLOAT,", f"TO_VECTOR('{vector_string}', FLOAT,")
    else:
        # For JDBC, interpolate the vector string placeholder
        sql = sql.format(vector_string=vector_string)
    
    return sql


def format_vector_search_sql_with_params(
    table_name: str,
    vector_column: str,
    embedding_dim: int,
    top_k: int,
    id_column: str = "doc_id",
    content_column: str = "text_content",
    additional_where: str = None
) -> str:
    """
    Constructs a SQL query for vector search using parameter placeholders.
    This version works with both DBAPI and JDBC by using TO_VECTOR(?, FLOAT).
    
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
    if not re.match(r'^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)?$', table_name):
        raise ValueError(f"Invalid table name: {table_name}")
    
    for col in [vector_column, id_column]:
        if not re.match(r'^[a-zA-Z0-9_]+$', col):
            raise ValueError(f"Invalid column name: {col}")
    
    if content_column and not re.match(r'^[a-zA-Z0-9_]+$', content_column):
        raise ValueError(f"Invalid content column name: {content_column}")
    
    if not validate_top_k(top_k):
        raise ValueError(f"Invalid top_k value: {top_k}")
    
    # Construct the SELECT clause
    select_clause = f"SELECT TOP {top_k} {id_column}"
    if content_column:
        select_clause += f", {content_column}"
    select_clause += f", VECTOR_COSINE({vector_column}, TO_VECTOR(?, FLOAT)) AS score"
    
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
    
    return sql.strip()


def execute_vector_search_with_params(
    cursor: Any,
    sql: str,
    vector_string: str
) -> List[Tuple]:
    """
    Executes a vector search SQL query using parameters.
    
    Args:
        cursor: A database cursor object
        sql: The SQL query with ? placeholder
        vector_string: The vector string to use as parameter
        
    Returns:
        List[Tuple]: The query results
    """
    results = []
    try:
        logger.debug(f"Executing vector search SQL with params")
        cursor.execute(sql, [vector_string])
        fetched_rows = cursor.fetchall()
        if fetched_rows:
            results = fetched_rows
        logger.debug(f"Found {len(results)} results.")
    except Exception as e:
        logger.error(f"Error during vector search: {e}")
        raise
    return results


def execute_vector_search(
    cursor: Any,
    sql: str
) -> List[Tuple]:
    """
    Executes a vector search SQL query using the provided cursor.
    This function is maintained for backward compatibility but now uses
    driver-aware execution internally.

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
        driver_type = get_driver_type()
        capabilities = get_driver_capabilities()
        
        logger.debug(f"Executing vector search SQL with {driver_type} driver: {sql[:100]}...")
        
        if capabilities.get('supports_vector_operations', False):
            # DBAPI: Direct execution should work
            cursor.execute(sql)
        else:
            # JDBC: Handle auto-parameterization bug
            try:
                cursor.execute(sql)
            except Exception as e:
                error_msg = str(e)
                if driver_type == DriverType.JDBC and ":%qpar" in error_msg and "SELECT TOP" in sql:
                    logger.warning("JDBC driver auto-parameterization detected, trying FETCH FIRST alternative...")
                    top_match = re.search(r'SELECT TOP (\d+)', sql)
                    if top_match:
                        top_value = top_match.group(1)
                        alt_sql = sql.replace(f"SELECT TOP {top_value}", "SELECT") + f" FETCH FIRST {top_value} ROWS ONLY"
                    else:
                        # Fallback if we can't parse the TOP value, though this is unlikely
                        alt_sql = sql.replace("SELECT TOP ", "SELECT ") + " FETCH FIRST 1 ROWS ONLY"
                    
                    logger.debug(f"Trying alternative SQL for JDBC: {alt_sql[:100]}...")
                    cursor.execute(alt_sql)
                else:
                    # For DBAPI or other errors, re-raise the original exception
                    raise
        
        fetched_rows = cursor.fetchall()
        if fetched_rows:
            results = fetched_rows
        logger.debug(f"Found {len(results)} results using {driver_type} driver.")
        
    except Exception as e:
        logger.error(f"Error during vector search with {get_driver_type()}: {e}")
        # Re-raise the exception so the calling pipeline can handle it
        raise
    return results