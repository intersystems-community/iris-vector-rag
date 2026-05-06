"""
Standardized DB Vector Utilities for IRIS.

Re-exports from iris_vector_graph.dbapi_utils for backward compatibility.
New code should import directly from iris_vector_graph.
"""

from iris_vector_graph.dbapi_utils import (
    normalize_vector as _normalize_vector_data,
    insert_vector,
    create_hnsw_index,
    create_hnsw_index as create_vector_index,
    create_ivfflat_index,
    vector_similarity_search,
)
