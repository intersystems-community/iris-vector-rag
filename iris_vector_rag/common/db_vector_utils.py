"""
Standardized DB Vector Utilities for IRIS.

Tries to import from iris_vector_graph.dbapi_utils (IVG >=1.99.0 / 2.x).
Falls back to pure-SQL implementations for IVG 1.x (<1.99.0) where that
module was absent.
"""

import json
import logging
import math
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    from iris_vector_graph.dbapi_utils import create_hnsw_index
    from iris_vector_graph.dbapi_utils import create_hnsw_index as create_vector_index
    from iris_vector_graph.dbapi_utils import create_ivfflat_index, insert_vector
    from iris_vector_graph.dbapi_utils import normalize_vector as _normalize_vector_data
    from iris_vector_graph.dbapi_utils import vector_similarity_search
except ImportError:
    # IVG 1.x (<1.99.0) — dbapi_utils did not exist; provide pure-SQL fallbacks.

    def _normalize_vector_data(v: Sequence[float]) -> List[float]:
        mag = math.sqrt(sum(x * x for x in v))
        if mag == 0.0:
            return list(v)
        return [x / mag for x in v]

    def insert_vector(
        cursor: Any,
        table_name: str,
        id_column: str,
        vector_column: str,
        id_value: Any,
        vector: Sequence[float],
        dtype: str = "FLOAT",
    ) -> None:
        vec_str = json.dumps(list(vector))
        sql = (
            f"INSERT INTO {table_name} ({id_column}, {vector_column}) "
            f"VALUES (?, TO_VECTOR(?, {dtype}, {len(vector)}))"
        )
        cursor.execute(sql, [id_value, vec_str])

    def create_hnsw_index(
        cursor: Any,
        table_name: str,
        vector_column: str,
        index_name: Optional[str] = None,
        m: int = 16,
        ef_construction: int = 200,
        distance: str = "COSINE",
    ) -> None:
        if index_name is None:
            index_name = f"idx_hnsw_{table_name.replace('.', '_')}_{vector_column}"
        sql = (
            f"CREATE INDEX {index_name} ON {table_name}({vector_column}) "
            f"AS HNSW(M={m}, efConstruction={ef_construction}, Distance='{distance}')"
        )
        try:
            cursor.execute(sql)
        except Exception as exc:
            msg = str(exc)
            if any(
                kw in msg.lower()
                for kw in ("already exists", "already defined", "-324", "-201")
            ):
                logger.debug("HNSW index %s already exists, skipping", index_name)
            else:
                raise

    create_vector_index = create_hnsw_index

    def create_ivfflat_index(
        cursor: Any,
        table_name: str,
        vector_column: str,
        index_name: Optional[str] = None,
        nlist: int = 100,
        distance: str = "COSINE",
    ) -> None:
        if index_name is None:
            index_name = f"idx_ivf_{table_name.replace('.', '_')}_{vector_column}"
        sql = (
            f"CREATE INDEX {index_name} ON {table_name}({vector_column}) "
            f"AS IVFFlat(nlist={nlist}, Distance='{distance}')"
        )
        try:
            cursor.execute(sql)
        except Exception as exc:
            msg = str(exc)
            if any(
                kw in msg.lower()
                for kw in ("already exists", "already defined", "-324", "-201")
            ):
                logger.debug("IVFFlat index %s already exists, skipping", index_name)
            else:
                raise

    def vector_similarity_search(
        cursor: Any,
        table_name: str,
        vector_column: str,
        query_vector: Sequence[float],
        top_k: int = 10,
        id_column: str = "id",
        return_columns: Optional[List[str]] = None,
        metric: str = "COSINE",
        dtype: str = "FLOAT",
    ) -> List[Dict[str, Any]]:
        dim = len(query_vector)
        vec_str = json.dumps(list(query_vector))
        extra_cols = ""
        if return_columns:
            extra_cols = ", " + ", ".join(return_columns)
        sql = (
            f"SELECT TOP {top_k} {id_column}{extra_cols}, "
            f"VECTOR_{metric.upper()}({vector_column}, TO_VECTOR(?, {dtype}, {dim})) AS score "
            f"FROM {table_name} "
            f"WHERE {vector_column} IS NOT NULL "
            f"ORDER BY score DESC"
        )
        cursor.execute(sql, [vec_str])
        rows = cursor.fetchall()
        col_names = [id_column] + (return_columns or []) + ["score"]
        return [dict(zip(col_names, row)) for row in rows]
