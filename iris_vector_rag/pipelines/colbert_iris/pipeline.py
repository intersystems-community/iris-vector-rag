import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ...core.models import Document
from ..basic import BasicRAGPipeline
from .ingest import ColBERTIngestor
from .maxsim_indb import MaxSimInDB
from .schema import ColBERTSchema

logger = logging.getLogger(__name__)

TOKEN_DIM = 128


class IRISColBERTPipeline(BasicRAGPipeline):
    """ColBERT pipeline backed by IRIS in-database MaxSim scoring.

    Replaces PyLate's query-time doc encoding with pre-stored token
    embeddings in RAG.DocumentTokenEmbeddings, then executes MaxSim
    either via bulk-fetch (Phase 1) or per-token HNSW (Phase 2).
    """

    def __init__(self, connection_manager=None, config_manager=None, **kwargs):
        super().__init__(connection_manager, config_manager, **kwargs)

        self._raw_conn = self.connection_manager.get_connection()
        self._schema = ColBERTSchema(self._raw_conn)
        self._ingestor = ColBERTIngestor(self._raw_conn, token_dim=TOKEN_DIM)
        self._maxsim = MaxSimInDB(self._raw_conn, token_dim=TOKEN_DIM)

        self._model = None
        self._model_name = "lightonai/GTE-ModernColBERT-v1"
        self._use_hnsw = kwargs.get("use_hnsw", True)
        self._k_per_token = kwargs.get("k_per_token", 50)
        self._rerank_factor = kwargs.get("rerank_factor", 2)

        self._schema.create_tables(token_dim=TOKEN_DIM)

    def _load_model(self):
        if self._model is not None:
            return
        try:
            import importlib
            pylate = importlib.import_module("pylate")
            self._model = pylate.models.ColBERT(model_name_or_path=self._model_name)
            self._ingestor.set_model(self._model)
            logger.info(f"Loaded ColBERT model: {self._model_name}")
        except Exception as e:
            raise RuntimeError(f"ColBERT model load failed: {e}") from e

    def load_documents(self, documents=None, documents_path=None, **kwargs) -> Dict[str, Any]:
        self._load_model()

        result = super().load_documents(documents=documents, documents_path=documents_path, **kwargs)

        docs_to_ingest = documents or []
        raw_docs = [
            {
                "doc_id": d.id if hasattr(d, "id") and d.id else f"doc_{i}",
                "text": d.page_content,
                "metadata": d.metadata if hasattr(d, "metadata") else {},
            }
            for i, d in enumerate(docs_to_ingest)
        ]

        if raw_docs:
            stats = self._ingestor.ingest_documents(raw_docs)
            self._schema.create_hnsw_index()
            self._schema.create_doc_index()
            result["colbert_ingest"] = stats
            logger.info(f"ColBERT ingest: {stats}")

        return result

    def query(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        self._load_model()

        t0 = time.perf_counter()

        initial_k = min(top_k * self._rerank_factor, 100)
        parent_kwargs = dict(kwargs, generate_answer=False)
        parent_result = super().query(query, top_k=initial_k, **parent_kwargs)
        candidate_docs = parent_result.get("retrieved_documents", [])

        q_vecs = self._encode_query(query)

        if self._use_hnsw:
            scored = self._maxsim.indb_maxsim(q_vecs, top_k=top_k, k_per_token=self._k_per_token)
        else:
            candidate_ids = [
                d.id if hasattr(d, "id") and d.id else d.page_content[:64]
                for d in candidate_docs
            ]
            scored = self._maxsim.bulk_fetch_maxsim(q_vecs, candidate_ids, top_k=top_k)

        scored_ids = {doc_id for doc_id, _ in scored}
        final_docs = self._fetch_docs_by_ids([d for d, _ in scored])

        generate_answer = kwargs.get("generate_answer", True)
        answer = None
        if generate_answer and self.llm_func and final_docs:
            try:
                answer = self._generate_answer(query, final_docs, kwargs.get("custom_prompt"))
            except Exception as e:
                logger.warning(f"Answer generation failed: {e}")
                answer = "Error generating answer"
        elif not generate_answer:
            answer = None
        elif not final_docs:
            answer = "No relevant documents found."
        else:
            answer = "No LLM configured. Retrieved documents only."

        elapsed = time.perf_counter() - t0
        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": final_docs,
            "contexts": [d.page_content for d in final_docs],
            "sources": self._extract_sources(final_docs) if kwargs.get("include_sources", True) else [],
            "execution_time": round(elapsed, 3),
            "metadata": {
                "pipeline_type": "iris_colbert",
                "num_retrieved": len(final_docs),
                "initial_candidates": len(candidate_docs),
                "use_hnsw": self._use_hnsw,
                "k_per_token": self._k_per_token,
                "rerank_factor": self._rerank_factor,
                "processing_time": round(elapsed, 3),
                "generated_answer": generate_answer and answer is not None,
            },
        }

    def _encode_query(self, query: str) -> np.ndarray:
        embs = self._model.encode([query], is_query=True)
        arr = np.array(embs[0], dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    def _fetch_docs_by_ids(self, doc_ids: List[str]) -> List[Document]:
        if not doc_ids:
            return []
        cur = self._raw_conn.cursor()
        docs = []
        try:
            for doc_id in doc_ids:
                cur.execute(
                    "SELECT doc_id, text_content, metadata "
                    "FROM RAG.ColBERTDocuments WHERE doc_id = ?",
                    [doc_id],
                )
                row = cur.fetchone()
                if row:
                    import json as _json
                    meta = {}
                    try:
                        meta = _json.loads(row[2] or "{}")
                    except Exception:
                        pass
                    docs.append(Document(id=row[0], page_content=row[1] or "", metadata=meta))
        finally:
            cur.close()
        return docs
