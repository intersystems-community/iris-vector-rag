"""
Vector SQL Utilities for IRIS Database

This module encapsulates workarounds for InterSystems IRIS SQL vector operations limitations.
It provides helper functions that RAG pipelines can use to safely construct SQL queries
with vector operations.
"""

import logging
import re
from typing import Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


def validate_vector_string(vector_string: str) -> bool:
    """
    Validates that a vector string contains a valid vector format.
    Allows negative numbers and scientific notation while preventing SQL injection.
    """
    stripped = vector_string.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return False

    content = stripped[1:-1]
    if not content.strip():
        return False

    parts = content.split(",")
    for part in parts:
        try:
            float(part.strip())
        except ValueError:
            return False

    if re.search(
        r"(DROP|DELETE|INSERT|UPDATE|SELECT|;|--)", vector_string, re.IGNORECASE
    ):
        return False

    return True


def validate_top_k(top_k: Any) -> bool:
    """Validates that top_k is a positive integer."""
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
    content_column: Optional[str] = "text_content",
    additional_where: Optional[str] = None,
    vector_data_type: str = "FLOAT"
) -> str:
    """Constructs a SQL query for vector search (deprecated - use build_safe_vector_dot_sql)."""
    if not re.match(r"^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)?$", table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    for col in [vector_column, id_column]:
        if not re.match(r"^[a-zA-Z0-9_]+$", col):
            raise ValueError(f"Invalid column name: {col}")

    if content_column and not re.match(r"^[a-zA-Z0-9_]+$", content_column):
        raise ValueError(f"Invalid content column name: {content_column}")

    if not validate_vector_string(vector_string):
        raise ValueError(f"Invalid vector string: {vector_string}")

    if not isinstance(embedding_dim, int) or embedding_dim <= 0:
        raise ValueError(f"Invalid embedding dimension: {embedding_dim}")

    if not validate_top_k(top_k):
        raise ValueError(f"Invalid top_k value: {top_k}")

    top_k_str = str(top_k)
    embedding_dim_str = str(embedding_dim)

    select_parts = ["SELECT TOP ", top_k_str, " ", id_column]
    if content_column:
        select_parts.extend([", ", content_column])

    vector_func_parts = [
        ", VECTOR_COSINE(",
        vector_column,
        ", TO_VECTOR('",
        vector_string,
        "', ",
        vector_data_type,
        ", ",
        embedding_dim_str,
        ")) AS score",
    ]
    select_parts.extend(vector_func_parts)
    select_clause = "".join(select_parts)

    where_parts = ["WHERE ", vector_column, " IS NOT NULL"]
    if additional_where:
        where_parts.extend([" AND (", additional_where, ")"])
    where_clause = "".join(where_parts)

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
    additional_where: Optional[str] = None,
    vector_data_type: str = "FLOAT"
) -> str:
    """Constructs a SQL query for vector search with parameters (deprecated)."""
    if not re.match(r"^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)?$", table_name):
        raise ValueError(f"Invalid table name: {table_name}")

    for col in [vector_column, id_column]:
        if not re.match(r"^[a-zA-Z0-9_]+$", col):
            raise ValueError(f"Invalid column name: {col}")

    if content_column and not re.match(r"^[a-zA-Z0-9_]+$", content_column):
        raise ValueError(f"Invalid content column name: {content_column}")

    if not validate_top_k(top_k):
        raise ValueError(f"Invalid top_k value: {top_k}")

    top_k_str = str(top_k)
    embedding_dim_str = str(embedding_dim)

    select_parts = ["SELECT TOP ", top_k_str, " ", id_column]
    if content_column:
        select_parts.extend([", ", content_column])

    vector_func_parts = [
        ", VECTOR_COSINE(",
        vector_column,
        ", TO_VECTOR(?, ",
        vector_data_type,
        ", ",
        embedding_dim_str,
        ")) AS score",
    ]
    select_parts.extend(vector_func_parts)
    select_clause = "".join(select_parts)

    where_parts = ["WHERE ", vector_column, " IS NOT NULL"]
    if additional_where:
        where_parts.extend([" AND (", additional_where, ")"])
    where_clause = "".join(where_parts)

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
    """Executes a vector search SQL query using parameters."""
    results = []
    try:
        cursor.execute(sql, [vector_string])
        results = cursor.fetchall()
    except Exception as e:
        logger.error(f"Error during vector search: {e}")
        raise
    return results


# =============================================================================
# SAFE VECTOR UTILITIES (PROVEN PATTERN)
# =============================================================================


def build_safe_vector_dot_sql(
    table: str,
    vector_column: str,
    vector_string: str,
    vector_dimension: int,
    id_column: str = "doc_id",
    extra_columns: Optional[List[str]] = None,
    top_k: int = 5,
    additional_where: Optional[str] = None,
    vector_data_type: str = "FLOAT"
) -> str:
    """
    Build safe vector search SQL with embedded vector string.
    """
    if not re.match(r"^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)?$", table):
        raise ValueError(f"Invalid table name: {table}")

    for col in [vector_column, id_column]:
        if not re.match(r"^[a-zA-Z0-9_]+$", col):
            raise ValueError(f"Invalid column name: {col}")

    if extra_columns:
        for col in extra_columns:
            if not re.match(r"^[a-zA-Z0-9_]+$", col):
                raise ValueError(f"Invalid extra column name: {col}")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"Invalid top_k value: {top_k}")

    if not isinstance(vector_dimension, int) or vector_dimension <= 0:
        raise ValueError(f"Invalid vector_dimension: {vector_dimension}")

    if not validate_vector_string(vector_string):
        raise ValueError(f"Invalid vector string: {vector_string}")

    select_parts = [f"SELECT TOP {top_k} {id_column}"]
    if extra_columns:
        select_parts.extend([f", {col}" for col in extra_columns])
    
    select_parts.append(f", VECTOR_DOT_PRODUCT({vector_column}, TO_VECTOR('{vector_string}', {vector_data_type}, {vector_dimension})) AS score")

    from_clause = f" FROM {table}"

    where_parts = [f" WHERE {vector_column} IS NOT NULL"]
    if additional_where:
        where_parts.append(f" AND ({additional_where})")

    order_clause = " ORDER BY score DESC"

    sql = "".join(select_parts) + from_clause + "".join(where_parts) + order_clause
    return sql


def execute_safe_vector_search(
    cursor, sql: str
) -> List[Tuple]:
    """Execute safe vector search with embedded vector string."""
    cursor.execute(sql)
    return cursor.fetchall()


def execute_vector_search(cursor: Any, sql: str) -> List[Tuple]:
    """Executes a vector search SQL query."""
    results = []
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
    except Exception as e:
        logger.error(f"Error during vector search: {e}")
        raise
    return results
