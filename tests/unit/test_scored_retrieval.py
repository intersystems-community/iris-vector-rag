"""A query's retrieved documents carry their similarity score when the store can
give it, and the MCP serializer passes it on (the "Document missing score" failure
of tests/integration/test_mcp_basic_rag.py). `search_by_text` stays score-free
(FR-014); the scored sibling is `search_by_text_with_score`."""

from __future__ import annotations

from iris_vector_rag.core.models import Document
from iris_vector_rag.mcp.technique_handlers import _serialize_document
from iris_vector_rag.pipelines.basic import BasicRAGPipeline
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore


def test_store_scored_text_search_returns_pairs(monkeypatch):
    store = IRISVectorStore.__new__(IRISVectorStore)
    doc = Document(id="a", page_content="alpha", metadata={})
    monkeypatch.setattr(store, "_embed_query", lambda q: [0.1, 0.2])
    seen = {}

    def by_embedding(embedding, top_k, flt):
        seen.update(embedding=embedding, top_k=top_k, flt=flt)
        return [(doc, 0.9)]

    monkeypatch.setattr(store, "similarity_search_by_embedding", by_embedding)
    assert store.search_by_text_with_score(
        "q", top_k=3, metadata_filter={"k": "v"}
    ) == [(doc, 0.9)]
    assert seen == {"embedding": [0.1, 0.2], "top_k": 3, "flt": {"k": "v"}}


class ScoringStore:
    def search_by_text(self, query, top_k=5, metadata_filter=None):
        raise AssertionError("a store that can score must be asked for scores")

    def search_by_text_with_score(self, query, top_k=5, metadata_filter=None):
        return [
            (Document(id="a", page_content="alpha", metadata={"source": "s"}), 0.8),
            (Document(id="b", page_content="beta", metadata={}), 0.4),
        ]


class PlainStore:
    def search_by_text(self, query, top_k=5, metadata_filter=None):
        return [Document(id="a", page_content="alpha", metadata={})]


def _pipeline(store):
    pipeline = BasicRAGPipeline.__new__(BasicRAGPipeline)
    pipeline.llm_func = lambda p: "answer"
    pipeline.vector_store = store
    pipeline.use_iris_embedding = False
    pipeline.embedding_config = None
    pipeline.default_top_k = 5
    pipeline._ensure_initialized = lambda: None
    pipeline._generate_answer = lambda q, d, c=None: "answer"
    return pipeline


def test_documents_carry_the_score_in_rank_order():
    result = _pipeline(ScoringStore()).query("q", top_k=2)
    docs = result["retrieved_documents"]
    assert [d.id for d in docs] == ["a", "b"]
    assert [d.metadata["score"] for d in docs] == [0.8, 0.4]
    assert docs[0].metadata["similarity"] == 0.8
    assert docs[0].metadata["source"] == "s"


def test_a_store_without_scores_still_works():
    result = _pipeline(PlainStore()).query("q", top_k=1)
    (doc,) = result["retrieved_documents"]
    assert "score" not in doc.metadata


def test_serializer_passes_the_score_on():
    doc = Document(id="a", page_content="alpha", metadata={"score": 0.8})
    assert _serialize_document(doc)["score"] == 0.8


def test_serializer_fills_score_from_similarity():
    doc = Document(id="a", page_content="alpha", metadata={"similarity": 0.7})
    assert _serialize_document(doc)["score"] == 0.7
