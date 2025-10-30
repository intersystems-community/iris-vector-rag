import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def insert_vector(
    cursor: Any,
    table_name: str,
    vector_column_name: str,
    vector_data: List[float],
    target_dimension: int,
    key_columns: Dict[str, Any],
    additional_data: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Inserts a record with a vector embedding into a specified table.
    The vector is truncated to the target_dimension and inserted using TO_VECTOR(?).

    Args:
        cursor: Database cursor.
        table_name: Name of the table to insert into (e.g., "RAG.DocumentTokenEmbeddings").
        vector_column_name: Name of the column that stores the vector.
        vector_data: The raw embedding vector (list of floats).
        target_dimension: The dimension to truncate/pad the vector to.
        key_columns: Dictionary of primary key or identifying column names and their values.
                     (e.g., {"doc_id": "id1", "token_index": 0})
        additional_data: Optional dictionary of other column names and their values.
                         (e.g., {"token_text": "example"})

    Returns:
        True if insertion was successful, False otherwise.
    """
    # Validate cursor handle
    if cursor is None:
        logger.error(
            f"DB Vector Util: Cannot insert vector into table '{table_name}': cursor is NULL"
        )
        return False

    if not isinstance(vector_data, list) or not all(
        isinstance(x, (float, int)) for x in vector_data
    ):
        logger.error(
            f"DB Vector Util: Invalid vector_data format for table '{table_name}'. "
            f"Expected list of floats/ints. Got type: {type(vector_data)}. Skipping insertion."
        )
        return False

    # Truncate or pad the vector to the target dimension
    processed_vector = vector_data[:target_dimension]
    if len(processed_vector) < target_dimension:
        logger.warning(
            f"DB Vector Util: Original vector length ({len(vector_data)}) for table '{table_name}', column '{vector_column_name}' "
            f"is less than target dimension ({target_dimension}). Padding with zeros."
        )
        processed_vector.extend([0.0] * (target_dimension - len(processed_vector)))

    # Format as bracketed comma-separated string for IRIS TO_VECTOR() function
    embedding_str = "[" + ",".join(map(str, processed_vector)) + "]"

    all_columns_dict = {}
    all_columns_dict.update(key_columns)
    if additional_data:
        all_columns_dict.update(additional_data)

    # Separate vector column from other data for SQL construction
    other_column_names = [col for col in all_columns_dict.keys()]
    other_column_values = [all_columns_dict[col] for col in other_column_names]

    column_names_sql = ", ".join(other_column_names + [vector_column_name])

    # IMPORTANT: TO_VECTOR() does NOT accept parameter markers (?, :param, etc.)
    # Must embed the vector string directly in SQL
    # Use FLOAT to match how test data was created (see tests/fixtures/embedding_generator.py:247)
    placeholders_list = ["?" for _ in other_column_names] + [
        f"TO_VECTOR('{embedding_str}', FLOAT, {target_dimension})"
    ]
    placeholders_sql = ", ".join(placeholders_list)

    # Use MERGE for upsert functionality to handle duplicates
    # Build MERGE statement for IRIS
    key_conditions = " AND ".join(
        [f"target.{col} = source.{col}" for col in key_columns.keys()]
    )
    update_assignments = ", ".join(
        [
            f"{col} = source.{col}"
            for col in other_column_names
            if col not in key_columns
        ]
    )

    # Separate approach: try INSERT first, if it fails due to constraint, try UPDATE
    sql_query = (
        f"INSERT INTO {table_name} ({column_names_sql}) VALUES ({placeholders_sql})"
    )
    # Don't pass embedding_str as a parameter - it's embedded in SQL above
    params = other_column_values

    try:
        logger.debug(f"DB Vector Util: Executing INSERT: {sql_query}")
        logger.debug(f"DB Vector Util: Parameters: {params}")
        logger.debug(
            f"DB Vector Util: Embedding string length: {len(embedding_str)} chars"
        )
        logger.debug(f"DB Vector Util: Vector dimension: {target_dimension}")
        cursor.execute(sql_query, params)
        return True
    except Exception as e:
        # Check for connection handle issues
        error_str = str(e).lower()
        if "_handle is null" in error_str or "handle is null" in error_str:
            logger.error(
                f"DB Vector Util: Database connection handle is NULL during vector insertion: {e}"
            )
            return False

        # Check if it's a unique constraint violation
        if "UNIQUE" in str(e) or "constraint failed" in str(e):
            logger.debug(
                f"DB Vector Util: INSERT failed due to duplicate key, attempting UPDATE..."
            )

            # Build UPDATE statement
            set_clauses = []
            update_params = []

            # Add non-key columns to SET clause
            for col in other_column_names:
                if col not in key_columns:
                    set_clauses.append(f"{col} = ?")
                    update_params.append(all_columns_dict[col])

            # Add vector column to SET clause - TO_VECTOR doesn't accept parameters
            # Use FLOAT to match how test data was created (see tests/fixtures/embedding_generator.py:247)
            set_clauses.append(
                f"{vector_column_name} = TO_VECTOR('{embedding_str}', FLOAT, {target_dimension})"
            )
            # Don't append embedding_str to params - it's embedded in SQL above

            # Add key columns to WHERE clause
            where_clauses = []
            for col, val in key_columns.items():
                where_clauses.append(f"{col} = ?")
                update_params.append(val)

            if set_clauses and where_clauses:
                update_sql = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE {' AND '.join(where_clauses)}"

                try:
                    logger.debug(f"DB Vector Util: Executing UPDATE: {update_sql}")
                    cursor.execute(update_sql, update_params)
                    return True
                except Exception as update_error:
                    # Check for connection handle issues in UPDATE
                    update_error_str = str(update_error).lower()
                    if (
                        "_handle is null" in update_error_str
                        or "handle is null" in update_error_str
                    ):
                        logger.error(
                            f"DB Vector Util: Database connection handle is NULL during UPDATE: {update_error}"
                        )
                    else:
                        logger.error(
                            f"DB Vector Util: UPDATE also failed: {update_error}"
                        )
                    return False
            else:
                logger.error(f"DB Vector Util: Could not build UPDATE statement")
                return False
        else:
            logger.error(
                f"DB Vector Util: Error inserting vector into table '{table_name}', column '{vector_column_name}': {e}"
            )
            logger.error(f"DB Vector Util: Key columns: {key_columns}")
            logger.error(
                f"DB Vector Util: Failing embedding string (first 100 chars): {embedding_str[:100] if 'embedding_str' in locals() else 'NOT_SET'}"
            )
            return False
