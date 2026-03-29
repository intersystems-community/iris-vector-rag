import logging
import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TOKEN_DIM = 128
IN_LIST_BATCH = 500
UPDATE_BATCH = 1000


class PLAIDNotBuiltError(RuntimeError):
    pass


def _format_vec_str(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vec) + "]"


class PLAIDBuilder:
    def __init__(self, conn, token_dim: int = TOKEN_DIM):
        self._conn = conn
        self._token_dim = token_dim

    @staticmethod
    def recommended_k(n_tokens: int) -> int:
        raw = max(16, n_tokens // 400)
        exp = round(math.log2(raw))
        return max(16, min(4096, 2**exp))

    def build(self, n_clusters: Optional[int] = None) -> Dict:
        from sklearn.cluster import MiniBatchKMeans

        n_tokens = self._fetch_token_count()
        if n_tokens == 0:
            raise ValueError("No token vectors found — run ingest before build()")

        if n_clusters is not None and n_clusters <= 0:
            raise ValueError(f"n_clusters must be > 0, got {n_clusters}")
        k = n_clusters if n_clusters is not None else self.recommended_k(n_tokens)
        k = min(k, max(1, n_tokens // 4))
        logger.info(f"PLAID build: n_tokens={n_tokens}, k={k}")

        t0 = time.perf_counter()
        vecs, doc_ids, tok_positions = self._fetch_all_tokens()

        km = MiniBatchKMeans(
            n_clusters=k, random_state=42, n_init=10, batch_size=min(10000, n_tokens)
        )
        labels = km.fit_predict(vecs)
        centroids = km.cluster_centers_.astype(np.float32)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / np.where(norms == 0, 1.0, norms)

        self._drop_centroid_tables()
        self._insert_centroids(centroids)
        self._assign_token_centroids(doc_ids, tok_positions, labels.tolist())
        self._populate_doc_centroids()
        self._conn.commit()

        elapsed = time.perf_counter() - t0
        logger.info(f"PLAID build complete in {elapsed:.1f}s")
        return {"n_clusters": k, "n_tokens": n_tokens, "elapsed_s": round(elapsed, 2)}

    def _fetch_token_count(self) -> int:
        cur = self._conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
            return cur.fetchone()[0]
        finally:
            cur.close()

    def _fetch_all_tokens(self) -> Tuple[np.ndarray, List[str], List[int]]:
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

        doc_ids, tok_positions, vecs = [], [], []
        for row in rows:
            doc_ids.append(row[0])
            tok_positions.append(row[1])
            raw = row[2]
            v = np.array(
                (
                    [float(x) for x in raw.strip("[]").split(",")]
                    if isinstance(raw, str)
                    else raw
                ),
                dtype=np.float32,
            )
            vecs.append(v)

        return np.array(vecs, dtype=np.float32), doc_ids, tok_positions

    def _drop_centroid_tables(self) -> None:
        cur = self._conn.cursor()
        for tbl in ["RAG.ColBERTDocCentroids", "RAG.ColBERTCentroids"]:
            try:
                cur.execute(f"DROP TABLE {tbl}")
            except Exception:
                pass
        for ddl in [
            "CREATE TABLE RAG.ColBERTCentroids (centroid_id INTEGER NOT NULL, centroid_vec VECTOR(FLOAT, 128) NOT NULL, PRIMARY KEY (centroid_id))",
            "CREATE TABLE RAG.ColBERTDocCentroids (centroid_id INTEGER NOT NULL, doc_id VARCHAR(64) NOT NULL, PRIMARY KEY (centroid_id, doc_id))",
        ]:
            try:
                cur.execute(ddl)
            except Exception as e:
                if "-201" not in str(e):
                    raise
        try:
            self._conn.commit()
        except Exception:
            pass
        cur.close()

    def _insert_centroids(self, centroids: np.ndarray) -> None:
        cur = self._conn.cursor()
        try:
            for cid, vec in enumerate(centroids):
                vec_str = _format_vec_str(vec)
                cur.execute(
                    f"INSERT INTO RAG.ColBERTCentroids (centroid_id, centroid_vec) "
                    f"VALUES (?, TO_VECTOR(?, FLOAT, {self._token_dim}))",
                    [cid, vec_str],
                )
            self._conn.commit()
        finally:
            cur.close()

    def _assign_token_centroids(
        self, doc_ids: List[str], tok_positions: List[int], labels: List[int]
    ) -> None:
        cur = self._conn.cursor()
        try:
            for i in range(0, len(doc_ids), UPDATE_BATCH):
                batch_docs = doc_ids[i : i + UPDATE_BATCH]
                batch_pos = tok_positions[i : i + UPDATE_BATCH]
                batch_labels = labels[i : i + UPDATE_BATCH]
                for doc_id, tok_pos, label in zip(batch_docs, batch_pos, batch_labels):
                    cur.execute(
                        "UPDATE RAG.DocumentTokenEmbeddings "
                        "SET centroid_id = ? WHERE doc_id = ? AND tok_pos = ?",
                        [label, doc_id, tok_pos],
                    )
                self._conn.commit()
        finally:
            cur.close()

    def _populate_doc_centroids(self) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "INSERT INTO RAG.ColBERTDocCentroids (centroid_id, doc_id) "
                "SELECT DISTINCT centroid_id, doc_id FROM RAG.DocumentTokenEmbeddings "
                "WHERE centroid_id IS NOT NULL"
            )
            self._conn.commit()
        finally:
            cur.close()


class PLAIDSearcher:
    def __init__(self, conn, token_dim: int = TOKEN_DIM):
        self._conn = conn
        self._token_dim = token_dim

    def _centroids_built(self) -> bool:
        cur = self._conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM RAG.ColBERTCentroids")
            return cur.fetchone()[0] > 0
        except Exception:
            return False
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
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        hit_centroid_ids = self._stage1_centroid_scan(q_vecs, n_probe)
        stage1_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        candidate_ids = self._stage15_candidate_expansion(hit_centroid_ids)
        stage15_ms = (time.perf_counter() - t0) * 1000

        if not candidate_ids:
            return [], {
                "candidate_count": 0,
                "total_docs": 0,
                "pruning_ratio": 0.0,
                "stage1_ms": stage1_ms,
                "stage15_ms": stage15_ms,
                "stage2_ms": 0.0,
            }

        t0 = time.perf_counter()
        results = self._stage2_exact_maxsim(q_vecs, candidate_ids, top_k)
        stage2_ms = (time.perf_counter() - t0) * 1000

        total_docs = self._total_doc_count()
        meta = {
            "candidate_count": len(candidate_ids),
            "total_docs": total_docs,
            "pruning_ratio": len(candidate_ids) / max(total_docs, 1),
            "stage1_ms": round(stage1_ms, 2),
            "stage15_ms": round(stage15_ms, 2),
            "stage2_ms": round(stage2_ms, 2),
            "total_ms": round((time.perf_counter() - t_total) * 1000, 2),
        }
        return results, meta

    def _stage1_centroid_scan(self, q_vecs: np.ndarray, n_probe: int) -> List[int]:
        hit_ids = set()
        cur = self._conn.cursor()
        try:
            for q_vec in q_vecs:
                vec_str = _format_vec_str(q_vec)
                cur.execute(
                    f"SELECT TOP {n_probe} centroid_id "
                    f"FROM RAG.ColBERTCentroids "
                    f"ORDER BY VECTOR_DOT_PRODUCT(centroid_vec, "
                    f"TO_VECTOR('{vec_str}', FLOAT, {self._token_dim})) DESC"
                )
                for row in cur.fetchall():
                    hit_ids.add(int(row[0]))
        finally:
            cur.close()
        return list(hit_ids)

    def _stage15_candidate_expansion(self, centroid_ids: List[int]) -> List[str]:
        if not centroid_ids:
            return []
        candidate_ids: set = set()
        cur = self._conn.cursor()
        try:
            for i in range(0, len(centroid_ids), IN_LIST_BATCH):
                batch = centroid_ids[i : i + IN_LIST_BATCH]
                placeholders = ",".join(["?"] * len(batch))
                cur.execute(
                    f"SELECT DISTINCT doc_id FROM RAG.ColBERTDocCentroids "
                    f"WHERE centroid_id IN ({placeholders})",
                    batch,
                )
                for row in cur.fetchall():
                    candidate_ids.add(row[0])
        finally:
            cur.close()
        return list(candidate_ids)

    def _stage2_exact_maxsim(
        self, q_vecs: np.ndarray, candidate_ids: List[str], top_k: int
    ) -> List[Tuple[str, float]]:
        per_qtok_sims: Dict[str, Dict[int, float]] = {}
        cur = self._conn.cursor()
        try:
            for i in range(0, len(candidate_ids), IN_LIST_BATCH):
                batch = candidate_ids[i : i + IN_LIST_BATCH]
                placeholders = ",".join(["?"] * len(batch))
                for q_idx, q_vec in enumerate(q_vecs):
                    vec_str = _format_vec_str(q_vec)
                    cur.execute(
                        f"SELECT doc_id, "
                        f"VECTOR_DOT_PRODUCT(tok_vec, TO_VECTOR('{vec_str}', FLOAT, {self._token_dim})) AS sim "
                        f"FROM RAG.DocumentTokenEmbeddings "
                        f"WHERE doc_id IN ({placeholders})",
                        batch,
                    )
                    for row in cur.fetchall():
                        doc_id, sim = row[0], float(row[1])
                        doc_sims = per_qtok_sims.setdefault(doc_id, {})
                        if q_idx not in doc_sims or sim > doc_sims[q_idx]:
                            doc_sims[q_idx] = sim
        finally:
            cur.close()

        scores = {
            doc_id: sum(q_sims.values()) for doc_id, q_sims in per_qtok_sims.items()
        }
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def _total_doc_count(self) -> int:
        cur = self._conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM RAG.ColBERTDocuments")
            return cur.fetchone()[0]
        except Exception:
            return 0
        finally:
            cur.close()

    def search_via_sp(
        self,
        conn,
        query_token_vecs: np.ndarray,
        top_k: int = 10,
        n_probe: int = 4,
    ) -> Tuple[List[Tuple[str, float]], Dict]:
        import json as _json

        q_vecs = np.array(query_token_vecs, dtype=np.float32)
        q_json = _json.dumps(q_vecs.tolist())

        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT RAG.ColBERTSearch_Search(?, ?, ?)",
                [q_json, top_k, n_probe],
            )
            row = cur.fetchone()
            raw_json = row[0]
        finally:
            cur.close()

        raw = _json.loads(raw_json)
        results = [(str(d), float(s)) for d, s in raw.get("results", [])]
        meta = {
            "n_centroids": raw.get("n_centroids", 0),
            "n_candidates": raw.get("n_candidates", 0),
            "stage1_ms": raw.get("stage1_ms", 0.0),
            "stage15_ms": raw.get("stage15_ms", 0.0),
            "stage2_ms": raw.get("stage2_ms", 0.0),
            "total_ms": raw.get("total_ms", 0.0),
        }
        return results, meta
