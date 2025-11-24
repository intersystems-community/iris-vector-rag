# common module

# Feature 051: Simplified IRIS Connection API
from iris_vector_rag.common.iris_connection import (
    get_iris_connection,
    detect_iris_edition,
    IRISConnectionPool,
)

__all__ = [
    "get_iris_connection",
    "detect_iris_edition",
    "IRISConnectionPool",
]
