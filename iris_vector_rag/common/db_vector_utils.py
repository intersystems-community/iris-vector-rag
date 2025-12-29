"""
Standardized DB Vector Utilities for IRIS.
Uses proper parameter binding for security and performance.
"""

import os
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
    Inserts a record with a vector embedding using parameter binding.
    """
    if cursor is None: return False

    # Process vector
    processed_vector = vector_data[:target_dimension]
    if len(processed_vector) < target_dimension:
        processed_vector.extend([0.0] * (target_dimension - len(processed_vector)))
    embedding_str = "[" + ",".join(map(str, processed_vector)) + "]"

    # Prepare data
    all_data = {**key_columns, **(additional_data or {})}
    columns = list(all_data.keys())
    values = [all_data[c] for c in columns]
    
    # Get vector type
    vector_data_type = os.environ.get("IRIS_VECTOR_DATA_TYPE") or "FLOAT"

    # Build SQL
    column_names = columns + [vector_column_name]
    column_sql = ", ".join(column_names)
    
    placeholders = ["?" for _ in columns]
    placeholders.append(f"TO_VECTOR(?, {vector_data_type}, {target_dimension})")
    placeholders_sql = ", ".join(placeholders)
    
    sql = f"INSERT INTO {table_name} ({column_sql}) VALUES ({placeholders_sql})"
    params = values + [embedding_str]

    try:
        cursor.execute(sql, tuple(params))
        return True
    except Exception as e:
        if "UNIQUE" in str(e) or "constraint failed" in str(e):
            # Attempt update
            set_clauses = [f"{c} = ?" for c in columns if c not in key_columns]
            update_params = [all_data[c] for c in columns if c not in key_columns]
            
            set_clauses.append(f"{vector_column_name} = TO_VECTOR(?, {vector_data_type}, {target_dimension})")
            update_params.append(embedding_str)
            
            where_clauses = [f"{c} = ?" for c in key_columns]
            for c in key_columns:
                update_params.append(all_data[c])
            
            update_sql = f"UPDATE {table_name} SET {', '.join(set_clauses)} WHERE {' AND '.join(where_clauses)}"
            try:
                cursor.execute(update_sql, tuple(update_params))
                return True
            except Exception as ue:
                logger.error(f"Update failed: {ue}")
                return False
        else:
            logger.error(f"Insert failed: {e}")
            return False
