"""
E2E: `RAGPipeline.ingest()` then `clear()` against a real IRIS vector store.

The pipeline is a minimal concrete subclass; the store, table and embeddings are real.
NO MOCKS.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest

from iris_vector_rag.core.base import RAGPipeline
from iris_vector_rag.core.models import Document


class StorePipeline(RAGPipeline):
    """Concrete pipeline that follows the base's declared signatures."""

    def load_documents(
        self, documents_path: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        docs = kwargs["documents"]
        self._store_documents(docs)
        return {"documents_loaded": len(docs)}

    def query(self, query, top_k=5, generate_answer=True, **kwargs):
        docs = self.vector_store.get_all_documents()[:top_k]
        return {
            "query": query,
            "answer": "",
            "retrieved_documents": docs,
            "contexts": [d.page_content for d in docs],
            "execution_time": 0.0,
            "metadata": {},
        }


@pytest.mark.true_e2e
class TestIngestRetrieveClear:
    def test_ingest_retrieve_clear_leaves_the_store_empty(
        self, fresh_iris_vector_store, e2e_database_cleanup
    ):
        store = fresh_iris_vector_store
        pipeline = StorePipeline(connection_manager=object(), vector_store=store)
        docs = [
            Document(id=f"e2e_test_core_{i}", page_content=f"Contract text {i}")
            for i in range(3)
        ]

        pipeline.ingest(docs)

        retrieved = pipeline.retrieve("contract", top_k=10)
        assert isinstance(retrieved, list)
        assert {d.id for d in retrieved} >= {d.id for d in docs}

        pipeline.clear()

        assert pipeline.retrieve("contract", top_k=10) == []
        assert store.get_all_documents() == []
