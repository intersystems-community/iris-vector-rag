import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TOKEN_DIM = 128
_DEFAULT_INDEX = "colbert_plaid"


class PLAIDNotBuiltError(RuntimeError):
    pass


def _ensure_native_conn(conn):
    import os

    import intersystems_iris

    if isinstance(conn, intersystems_iris.IRISConnection):
        return conn
    return intersystems_iris.createConnection(
        hostname=getattr(
            conn, "hostname", os.environ.get("IRIS_HOSTNAME", "localhost")
        ),
        port=getattr(conn, "port", int(os.environ.get("IRIS_PORT", "1972"))),
        namespace=getattr(conn, "namespace", os.environ.get("IRIS_NAMESPACE", "USER")),
        username=os.environ.get("IRIS_USERNAME", "_SYSTEM"),
        password=os.environ.get("IRIS_PASSWORD", "SYS"),
    )


def _make_engine(conn):
    from iris_vector_graph import IRISGraphEngine

    return IRISGraphEngine(_ensure_native_conn(conn))


class PLAIDBuilder:
    def __init__(
        self, conn, token_dim: int = TOKEN_DIM, index_name: str = _DEFAULT_INDEX, _engine=None
    ):
        self._conn = conn
        self._token_dim = token_dim
        self._index_name = index_name
        self._engine = _engine if _engine is not None else _make_engine(conn)

    @staticmethod
    def recommended_k(n_tokens: int) -> int:
        raw = max(16, n_tokens // 400)
        exp = round(math.log2(raw))
        return max(16, min(4096, 2**exp))

    def build(self, n_clusters: Optional[int] = None) -> Dict:
        docs = self._fetch_docs_with_tokens()
        if not docs:
            raise ValueError("No token vectors found — run ingest before build()")

        n_tokens = sum(len(d["tokens"]) for d in docs)
        k = n_clusters if n_clusters is not None else self.recommended_k(n_tokens)
        if k <= 0:
            raise ValueError(f"n_clusters must be > 0, got {k}")
        k = min(k, max(1, n_tokens // 4))

        logger.info(
            f"PLAID build: n_tokens={n_tokens}, k={k}, index={self._index_name}"
        )
        t0 = time.perf_counter()

        self._engine.plaid_drop(self._index_name)
        result = self._engine.plaid_build(
            self._index_name, docs, n_clusters=k, dim=self._token_dim
        )

        elapsed = time.perf_counter() - t0
        logger.info(f"PLAID build complete in {elapsed:.1f}s")
        return {
            "n_clusters": k,
            "n_tokens": n_tokens,
            "elapsed_s": round(elapsed, 2),
            **result,
        }

    def _fetch_docs_with_tokens(self) -> List[Dict]:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "SELECT doc_id, tok_pos, tok_vec "
                "FROM RAG.DocumentTokenEmbeddings "
                "ORDER BY doc_id, tok_pos"
            )
            rows = cur.fetchall()
        finally:
            cur.close()

        docs_map: Dict[str, List] = {}
        for doc_id, _tok_pos, raw in rows:
            v = (
                [float(x) for x in raw.strip("[]").split(",")]
                if isinstance(raw, str)
                else list(raw)
            )
            docs_map.setdefault(doc_id, []).append(v)

        return [{"id": doc_id, "tokens": tokens} for doc_id, tokens in docs_map.items()]


class PLAIDSearcher:
    def __init__(
        self, conn, token_dim: int = TOKEN_DIM, index_name: str = _DEFAULT_INDEX, _engine=None
    ):
        self._conn = conn
        self._token_dim = token_dim
        self._index_name = index_name
        self._engine = _engine if _engine is not None else _make_engine(conn)

    def _centroids_built(self) -> bool:
        try:
            return self._engine.plaid_info(self._index_name).get("nCentroids", 0) > 0
        except Exception:
            return False

    def _total_doc_count(self) -> int:
        cur = self._conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM RAG.ColBERTDocuments")
            return cur.fetchone()[0]
        except Exception:
            return 0
        finally:
            cur.close()

    def search(
        self,
        query_token_vecs: np.ndarray,
        top_k: int = 10,
        n_probe: int = 4,
    ) -> Tuple[List[Tuple[str, float]], Dict]:
        if not self._centroids_built():
            raise PLAIDNotBuiltError(
                "PLAID centroid index not built. Call PLAIDBuilder.build() first."
            )

        q_vecs = np.array(query_token_vecs, dtype=np.float32)
        t0 = time.perf_counter()
        raw = self._engine.plaid_search(
            self._index_name, q_vecs.tolist(), k=top_k, nprobe=n_probe
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        results = [(r["id"], float(r["score"])) for r in raw]
        total_docs = self._total_doc_count()
        meta = {
            "n_candidates": len(results),
            "total_docs": total_docs,
            "pruning_ratio": len(results) / max(total_docs, 1),
            "stage1_ms": 0.0,
            "stage15_ms": 0.0,
            "stage2_ms": round(elapsed_ms, 2),
            "total_ms": round(elapsed_ms, 2),
        }
        return results, meta

    def search_via_sp(
        self,
        conn,
        query_token_vecs: np.ndarray,
        top_k: int = 10,
        n_probe: int = 4,
    ) -> Tuple[List[Tuple[str, float]], Dict]:
        return self.search(query_token_vecs, top_k=top_k, n_probe=n_probe)
