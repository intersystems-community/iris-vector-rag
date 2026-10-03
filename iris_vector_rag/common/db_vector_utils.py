"""
Standardized DB Vector Utilities for IRIS.

Re-exports iris_vector_graph.dbapi_utils, which every supported IVG (4.x) ships and
iris-vector-graph is a hard dependency of this package.
"""

from iris_vector_graph.dbapi_utils import create_hnsw_index
from iris_vector_graph.dbapi_utils import create_hnsw_index as create_vector_index
from iris_vector_graph.dbapi_utils import create_ivfflat_index, insert_vector
from iris_vector_graph.dbapi_utils import normalize_vector as _normalize_vector_data
from iris_vector_graph.dbapi_utils import vector_similarity_search
