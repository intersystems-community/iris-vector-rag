# common module

from iris_vector_rag.common.connection_pool import IRISConnectionPool
from iris_vector_rag.common.iris_connection import (
    detect_iris_edition,
    get_iris_connection,
)

__all__ = [
    "get_iris_connection",
    "detect_iris_edition",
    "IRISConnectionPool",
]
