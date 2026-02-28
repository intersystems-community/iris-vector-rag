"""
Standardized DB Vector Utilities for IRIS.
Uses proper parameter binding for security and performance.
"""

import os
import logging
import math
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _normalize_vector_data(
    vector_data: Any,
    target_dimension: int,
) -> Optional[List[float]]:
    if vector_data is None:
        return None

    normalized: Optional[List[float]] = None

    try:
        import numpy as np

        if isinstance(vector_data, np.ndarray):
            normalized = (
                vector_data.astype("float32", copy=False).ravel().tolist()
            )
    except Exception:
        pass

    if normalized is None:
        try:
            import torch

            if isinstance(vector_data, torch.Tensor):
                normalized = (
                    vector_data.detach()
                    .to(dtype=torch.float32)
                    .cpu()
                    .contiguous()
                    .flatten()
                    .tolist()
                )
        except Exception:
            pass

    if normalized is None:
        try:
            if isinstance(vector_data, Sequence):
                normalized = [float(value) for value in vector_data]
            else:
                normalized = [float(vector_data)]
        except Exception:
            return None

    if not normalized:
        return None

    non_finite = 0
    for idx, value in enumerate(normalized):
        if not math.isfinite(value):
            normalized[idx] = 0.0
            non_finite += 1

    if non_finite and os.environ.get("IRIS_VECTOR_DEBUG"):
        logger.warning("Vector contained %s non-finite values; coerced to 0.0", non_finite)

    processed_vector = normalized[:target_dimension]
    if len(processed_vector) < target_dimension:
        processed_vector.extend([0.0] * (target_dimension - len(processed_vector)))

    return processed_vector


def insert_vector(
    cursor: Any,
    table_name: str,
    vector_column_name: str,
    vector_data: List[float],
    target_dimension: int,
    key_columns: Dict[str, Any],
    additional_data: Optional[Dict[str, Any]] = None,
    vector_data_type: Optional[str] = None,
) -> bool:
    """
    Inserts a record with a vector embedding using parameter binding.
    """
    if cursor is None:
        return False

    # Process vector
    processed_vector = _normalize_vector_data(vector_data, target_dimension)
    if processed_vector is None:
        logger.error("Insert failed: unable to normalize vector input")
        return False
    if os.environ.get("IRIS_VECTOR_DEBUG"):
        logger.info(
            "Vector insert: dim=%s target=%s first=%s last=%s",
            len(processed_vector),
            target_dimension,
            processed_vector[0],
            processed_vector[-1],
        )
    embedding_str = "[" + ",".join(map(str, processed_vector)) + "]"

    # Prepare data
    all_data = {**key_columns, **(additional_data or {})}
    columns = list(all_data.keys())
    values = [all_data[c] for c in columns]
    
    # Get vector type
    vector_data_type = vector_data_type or os.environ.get("IRIS_VECTOR_DATA_TYPE") or "FLOAT"

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
