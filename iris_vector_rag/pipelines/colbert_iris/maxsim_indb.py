"""
In-database MaxSim scoring for ColBERT late interaction.

Two execution modes:

  Phase 1 — bulk_fetch_maxsim (Python loop, DB storage)
    Fetches all token vectors for candidate docs in one SQL call, then
    computes MaxSim in Python using numpy.  Faster than PyLate baseline
    because doc re-encoding is eliminated.

  Phase 2 — indb_maxsim (per-token HNSW, SQL aggregation)
    For each query token:
      SELECT TOP k doc_id, VECTOR_DOT_PRODUCT(...) AS sim
      FROM RAG.DocumentTokenEmbeddings ORDER BY sim DESC
    Accumulates per-doc max-sim in a Python dict, then sums.
    All vector math runs inside IRIS; uses HNSW when TOP + ORDER BY DESC
    pattern is present (as documented in IRIS vector search guide).

The HNSW path is preferred at scale (>10K token vectors).
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TOKEN_DIM = 128


class MaxSimInDB:
    def __init__(self, conn, token_dim: int = TOKEN_DIM):
        self._conn = conn
        self._token_dim = token_dim

    # ------------------------------------------------------------------
    # Phase 1: bulk fetch — no HNSW required
    # ------------------------------------------------------------------

    def bulk_fetch_maxsim(
        self,
        query_token_vecs: np.ndarray,
        candidate_doc_ids: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Fetch all token vectors for candidate docs in one SQL call;
        compute MaxSim in Python.  Eliminates query-time doc encoding.
        """
        if len(candidate_doc_ids) == 0:
            return []

        q_vecs = np.array(query_token_vecs, dtype=np.float32)
        per_qtok_sims: Dict[str, Dict[int, float]] = {}
        batch_size = 500

        cur = self._conn.cursor()
        try:
            for q_idx, q_vec in enumerate(q_vecs):
                vec_str = "[" + ",".join(f"{v:.6f}" for v in q_vec) + "]"
                for i in range(0, len(candidate_doc_ids), batch_size):
                    batch = candidate_doc_ids[i : i + batch_size]
                    placeholders = ",".join(["?"] * len(batch))
                    cur.execute(
                        f"""
                    SELECT doc_id,
                           VECTOR_DOT_PRODUCT(tok_vec,
                               TO_VECTOR('{vec_str}', FLOAT, {self._token_dim})) AS sim
                    FROM RAG.DocumentTokenEmbeddings
                    WHERE doc_id IN ({placeholders})
                    """,
                        batch,
                    )
                    for row in cur.fetchall():
                        doc_id, sim = row[0], float(row[1])
                        doc_sims = per_qtok_sims.setdefault(doc_id, {})
                        if q_idx not in doc_sims or sim > doc_sims[q_idx]:
                            doc_sims[q_idx] = sim
        finally:
            cur.close()

        if not per_qtok_sims:
            return []

        scores = {
            doc_id: sum(q_sims.values()) for doc_id, q_sims in per_qtok_sims.items()
        }
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # Phase 2: per-token HNSW + SQL aggregation
    # ------------------------------------------------------------------

    def indb_maxsim(
        self,
        query_token_vecs: np.ndarray,
        top_k: int = 10,
        k_per_token: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        For each query token, run a TOP-k ANN query against the token table.
        IRIS uses the HNSW index when TOP + ORDER BY DESC + VECTOR_* is present.
        Accumulate per-doc max-sim, then sum and return top_k docs.
        """
        max_sim: Dict[str, Dict[int, float]] = {}

        cur = self._conn.cursor()
        try:
            for q_idx, q_vec in enumerate(query_token_vecs):
                vec_str = "[" + ",".join(f"{v:.6f}" for v in q_vec) + "]"
                cur.execute(
                    f"""
                    SELECT TOP {k_per_token} doc_id,
                           VECTOR_DOT_PRODUCT(tok_vec,
                               TO_VECTOR('{vec_str}', FLOAT, {self._token_dim})) AS sim
                    FROM RAG.DocumentTokenEmbeddings
                    ORDER BY sim DESC
                    """,
                )
                for row in cur.fetchall():
                    doc_id, sim = row[0], float(row[1])
                    doc_sims = max_sim.setdefault(doc_id, {})
                    if q_idx not in doc_sims or sim > doc_sims[q_idx]:
                        doc_sims[q_idx] = sim
        finally:
            cur.close()

        scores = {doc_id: sum(q_sims.values()) for doc_id, q_sims in max_sim.items()}
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # EXPLAIN helper (tests use this to verify HNSW activation)
    # ------------------------------------------------------------------

    def explain_indb_query(self, q_vec: np.ndarray, k_per_token: int = 50) -> str:
        vec_str = "[" + ",".join(f"{v:.6f}" for v in q_vec) + "]"
        sql = (
            f"SELECT TOP {k_per_token} doc_id, "
            f"VECTOR_DOT_PRODUCT(tok_vec, "
            f"TO_VECTOR('{vec_str}', FLOAT, {self._token_dim})) AS sim "
            f"FROM RAG.DocumentTokenEmbeddings "
            f"ORDER BY sim DESC"
        )
        cur = self._conn.cursor()
        try:
            cur.execute(f"EXPLAIN {sql}")
            rows = cur.fetchall()
            return "\n".join(str(r) for r in rows)
        finally:
            cur.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_vec(raw) -> Optional[np.ndarray]:
        if raw is None:
            return None
        try:
            if isinstance(raw, (list, tuple)):
                return np.array(raw, dtype=np.float32)
            s = str(raw).strip().strip("[]")
            return np.array([float(x) for x in s.split(",")], dtype=np.float32)
        except Exception:
            return None
