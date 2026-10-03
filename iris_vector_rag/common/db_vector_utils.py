"""
Standardized DB Vector Utilities for IRIS.

Re-exports iris_vector_graph.dbapi_utils, which every supported IVG (4.x) ships and
iris-vector-graph is a hard dependency of this package. Two functions are wrapped so a
vector of the wrong width is refused: ivg's own `normalize_vector` zero-pads or truncates
to the target, which stores a vector that means something else (ST-23).
"""

from typing import Any, Dict, List, Optional

from iris_vector_graph.dbapi_utils import create_hnsw_index
from iris_vector_graph.dbapi_utils import create_hnsw_index as create_vector_index
from iris_vector_graph.dbapi_utils import create_ivfflat_index
from iris_vector_graph.dbapi_utils import insert_vector as _ivg_insert_vector
from iris_vector_graph.dbapi_utils import normalize_vector as _ivg_normalize_vector
from iris_vector_graph.dbapi_utils import vector_similarity_search

from iris_vector_rag.exceptions import DimensionMismatchError


def _width_of(vector_data: Any) -> Optional[int]:
    """How many elements `vector_data` holds, for the shapes ivg accepts."""
    numel = getattr(vector_data, "numel", None)  # torch tensor
    if callable(numel):
        return int(numel())
    size = getattr(vector_data, "size", None)  # numpy array
    if isinstance(size, int):
        return size
    try:
        return len(vector_data)
    except TypeError:
        return None  # a scalar: ivg wraps it in a one-element list


def normalize_vector(vector_data: Any, target_dimension: int) -> Optional[List[float]]:
    """A float list of exactly `target_dimension` elements, or `DimensionMismatchError`.

    Converts numpy arrays, torch tensors and sequences (and replaces NaN and Inf with
    0.0, as ivg does). Never pads and never truncates.
    """
    if vector_data is None:
        return None
    width = _width_of(vector_data)
    if width is not None and width != target_dimension:
        raise DimensionMismatchError(width, target_dimension)
    return _ivg_normalize_vector(vector_data, target_dimension)


def insert_vector(
    cursor: Any,
    table_name: str,
    vector_column: str,
    vector_data: Any,
    dimension: int,
    key_columns: Dict[str, Any],
    additional_columns: Optional[Dict[str, Any]] = None,
    dtype: str = "FLOAT",
    upsert: bool = True,
) -> bool:
    """ivg's `insert_vector`, after checking the vector is `dimension` wide.

    Raises `DimensionMismatchError` before any SQL is sent. Returns False only for a
    failure ivg has already logged (the row was not written).
    """
    normalize_vector(vector_data, dimension)
    return _ivg_insert_vector(
        cursor=cursor,
        table_name=table_name,
        vector_column=vector_column,
        vector_data=vector_data,
        dimension=dimension,
        key_columns=key_columns,
        additional_columns=additional_columns,
        dtype=dtype,
        upsert=upsert,
    )


__all__ = [
    "create_hnsw_index",
    "create_ivfflat_index",
    "create_vector_index",
    "insert_vector",
    "normalize_vector",
    "vector_similarity_search",
]
