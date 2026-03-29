import json
import time
from typing import Dict, List, Tuple

import numpy as np


def _ensure_native_conn(conn):
    import os

    import intersystems_iris

    if isinstance(conn, intersystems_iris.IRISConnection):
        return conn
    hostname = getattr(conn, "hostname", os.environ.get("IRIS_HOSTNAME", "localhost"))
    port = getattr(conn, "port", int(os.environ.get("IRIS_PORT", "1972")))
    namespace = getattr(conn, "namespace", os.environ.get("IRIS_NAMESPACE", "USER"))
    username = os.environ.get("IRIS_USERNAME", "_SYSTEM")
    password = os.environ.get("IRIS_PASSWORD", "SYS")
    return intersystems_iris.createConnection(
        hostname=hostname,
        port=port,
        namespace=namespace,
        username=username,
        password=password,
    )


def _to_dict(result) -> dict:
    return result if isinstance(result, dict) else json.loads(str(result))


class VecIndexNotAvailableError(RuntimeError):
    pass


class VecIndexSearcher:
    def __init__(
        self,
        conn,
        index_name: str = "colbert_tokens",
        token_dim: int = 128,
        num_trees: int = 4,
        leaf_size: int = 50,
    ):
        from iris_vector_graph import IRISGraphEngine

        self.conn = conn
        self.index_name = index_name
        self.token_dim = token_dim
        self.num_trees = num_trees
        self.leaf_size = leaf_size
        self._engine = IRISGraphEngine(_ensure_native_conn(conn))
        self._iris = self._engine._iris_obj()

    def build(self) -> dict:
        return _to_dict(self._engine.vec_build(self.index_name))

    def info(self) -> dict:
        try:
            return _to_dict(self._engine.vec_info(self.index_name))
        except Exception:
            return {"name": self.index_name, "count": 0}

    def drop(self) -> None:
        self._engine.vec_drop(self.index_name)

    def _fast_insert(self, doc_id: str, tok_pos: int, vec: np.ndarray) -> None:
        key = f"{doc_id}:{tok_pos}"
        self._engine.vec_insert(self.index_name, key, vec.tolist())

    def _ensure_index_created(self) -> None:
        info = self.info()
        if info.get("count", -1) == -1 or "dim" not in info:
            self._engine.vec_create_index(
                self.index_name,
                self.token_dim,
                "dot",
                self.num_trees,
                self.leaf_size,
            )

    def search(
        self,
        query_token_vecs: np.ndarray,
        top_k: int = 10,
        nprobe: int = 2,
        k_per_token: int = 50,
    ) -> Tuple[List[Tuple[str, float]], Dict]:
        info = self.info()
        if info.get("count", 0) == 0:
            raise VecIndexNotAvailableError(
                f"VecIndex '{self.index_name}' is empty or not built. "
                "Run ColBERTIngestor with use_vecindex=True then vec_build(), "
                "and ensure Graph.KG.VecIndex is loaded via scripts/deploy_vecindex.sh"
            )

        q_vecs = np.atleast_2d(query_token_vecs).astype(np.float32)

        t_search = time.perf_counter()
        per_doc_max: Dict[str, Dict[int, float]] = {}

        all_hits = self._engine.vec_search_multi(
            self.index_name, q_vecs.tolist(), k=k_per_token, nprobe=nprobe
        )
        for q_idx, hits in enumerate(all_hits):
            for hit in hits:
                raw_id = hit["id"]
                doc_id = raw_id.rsplit(":", 1)[0]
                score = float(hit["score"])
                doc_sims = per_doc_max.setdefault(doc_id, {})
                if q_idx not in doc_sims or score > doc_sims[q_idx]:
                    doc_sims[q_idx] = score

        stage_search_ms = (time.perf_counter() - t_search) * 1000

        t_maxsim = time.perf_counter()
        final_scores: Dict[str, float] = {
            doc_id: sum(q_sims.values()) for doc_id, q_sims in per_doc_max.items()
        }
        results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        stage_maxsim_ms = (time.perf_counter() - t_maxsim) * 1000

        meta = {
            "n_candidates": len(per_doc_max),
            "stage_search_ms": round(stage_search_ms, 2),
            "stage_maxsim_ms": round(stage_maxsim_ms, 2),
            "total_ms": round(stage_search_ms + stage_maxsim_ms, 2),
        }
        return results, meta
