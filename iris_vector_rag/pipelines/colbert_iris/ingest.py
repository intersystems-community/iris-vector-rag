"""
ColBERT ingestion: tokenise + embed documents, store in IRIS.

Design choices:
  • Token dimension fixed at 128 (GTE-ModernColBERT-v1 projects to 128-d)
  • Vectors L2-normalised at ingestion (dot product == cosine at query time)
  • Commit every COMMIT_BATCH rows to bound transaction size
  • No HNSW index during ingest — caller creates it after load completes
"""

import json
import logging
import time
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

COMMIT_BATCH = 500
TOKEN_DIM = 128


class ColBERTIngestor:
    def __init__(self, conn, model=None, token_dim: int = TOKEN_DIM):
        self._conn = conn
        self._model = model
        self._token_dim = token_dim

    def set_model(self, model) -> None:
        self._model = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_documents(
        self,
        docs: List[Dict[str, Any]],
        batch_size: int = 32,
        use_vecindex: bool = False,
        vecindex_searcher=None,
    ) -> Dict[str, Any]:
        if self._model is None:
            raise RuntimeError(
                "ColBERTIngestor: model not set — call set_model() first"
            )

        if use_vecindex and vecindex_searcher is None:
            from .vecindex_phase2 import VecIndexSearcher
            vecindex_searcher = VecIndexSearcher(self._conn, token_dim=self._token_dim)
            vecindex_searcher._ensure_index_created()

        t0 = time.perf_counter()
        total_tokens = 0
        vecindex_count = 0
        failed = 0

        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            texts = [d["text"] for d in batch]
            token_embeddings = self._encode_batch(texts)

            for doc, tok_embs in zip(batch, token_embeddings):
                try:
                    self._insert_doc(doc)
                    self._insert_tokens(doc["doc_id"], tok_embs)
                    total_tokens += len(tok_embs)
                    if use_vecindex and vecindex_searcher is not None:
                        for pos, vec in enumerate(tok_embs):
                            vecindex_searcher._fast_insert(doc["doc_id"], pos, vec)
                            vecindex_count += 1
                except Exception as e:
                    logger.warning(f"Failed to ingest doc {doc.get('doc_id')}: {e}")
                    failed += 1

            self._conn.commit()
            logger.debug(f"Ingested batch {i // batch_size + 1} ({len(batch)} docs)")

        elapsed = time.perf_counter() - t0
        stats: Dict[str, Any] = {
            "docs_ingested": len(docs) - failed,
            "docs_failed": failed,
            "total_tokens": total_tokens,
            "elapsed_s": round(elapsed, 2),
            "docs_per_sec": round((len(docs) - failed) / elapsed, 1) if elapsed > 0 else 0,
        }

        if use_vecindex and vecindex_searcher is not None:
            t_build = time.perf_counter()
            vecindex_searcher.build()
            stats["vecindex_count"] = vecindex_count
            stats["vecindex_build_ms"] = round((time.perf_counter() - t_build) * 1000, 1)
            stats["vecindex_ingest_mode"] = "engine_insert"

        return stats

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Encode a batch of texts, returning list of (n_tokens, dim) arrays."""
        embeddings = self._model.encode(texts, is_query=False)
        result = []
        for emb in embeddings:
            arr = np.array(emb, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            arr = self._normalise(arr)
            result.append(arr)
        return result

    @staticmethod
    def _normalise(vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vecs / norms

    def _insert_doc(self, doc: Dict[str, Any]) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                """
                DELETE FROM RAG.ColBERTDocuments WHERE doc_id = ?
                """,
                [doc["doc_id"]],
            )
            cur.execute(
                """
                INSERT INTO RAG.ColBERTDocuments
                    (doc_id, parent_id, chunk_index, text_content, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    doc["doc_id"],
                    doc.get("parent_id"),
                    doc.get("chunk_index", 0),
                    doc["text"],
                    json.dumps(doc.get("metadata", {})),
                ],
            )
        finally:
            cur.close()

    def _insert_tokens(self, doc_id: str, tok_vecs: np.ndarray) -> None:
        cur = self._conn.cursor()
        try:
            cur.execute(
                "DELETE FROM RAG.DocumentTokenEmbeddings WHERE doc_id = ?",
                [doc_id],
            )
            for pos, vec in enumerate(tok_vecs):
                vec_str = "[" + ",".join(f"{v:.6f}" for v in vec) + "]"
                cur.execute(
                    f"""
                    INSERT INTO RAG.DocumentTokenEmbeddings
                        (doc_id, tok_pos, tok_vec)
                    VALUES (?, ?, TO_VECTOR(?, FLOAT, {self._token_dim}))
                    """,
                    [doc_id, pos, vec_str],
                )
        finally:
            cur.close()
