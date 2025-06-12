import logging
from typing import List, Any, Dict, Optional

logger = logging.getLogger(__name__)

def insert_vector(
    cursor: Any,
    table_name: str,
    vector_column_name: str,
    vector_data: List[float],
    target_dimension: int,
    key_columns: Dict[str, Any],
    additional_data: Optional[Dict[str, Any]] = None
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
    if not isinstance(vector_data, list) or not all(isinstance(x, (float, int)) for x in vector_data):
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
    
    placeholders_list = ["?" for _ in other_column_names] + ["TO_VECTOR(?, FLOAT)"]
    placeholders_sql = ", ".join(placeholders_list)

    sql_query = f"INSERT INTO {table_name} ({column_names_sql}) VALUES ({placeholders_sql})"
    
    params = other_column_values + [embedding_str]
    
    try:
        logger.debug(f"DB Vector Util: Executing SQL: {sql_query}")
        logger.debug(f"DB Vector Util: With params (vector string truncated for log): {other_column_values + [embedding_str[:100] + '...'] if len(embedding_str) > 100 else params}")
        cursor.execute(sql_query, params)
        return True
    except Exception as e:
        logger.error(
            f"DB Vector Util: Error inserting vector into table '{table_name}', column '{vector_column_name}': {e}"
        )
        logger.error(f"DB Vector Util: Key columns: {key_columns}")
        logger.error(f"DB Vector Util: Failing embedding string (first 100 chars): {embedding_str[:100] if 'embedding_str' in locals() else 'NOT_SET'}")
        # Consider re-raising or logging traceback for more detailed debugging if needed
        # import traceback
        # logger.error(f"Traceback: {traceback.format_exc()}")
        return False