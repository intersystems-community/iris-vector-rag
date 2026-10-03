"""
E2E: fusion returns real content and sane scores against a real IRIS.

Ingest a few documents through the real vector store (real embedder), mirror them into
the (id, text) table the iris-vector-graph text engine reads, then query every mode
through `BasicRAGPipeline`-equivalent dispatch (`RetrievalEngine`). NO MOCKS.
"""

from __future__ import annotations

import pytest

from iris_vector_rag.core.models import Document
from iris_vector_rag.core.query_options import QueryOptions
from iris_vector_rag.retrieval.engine import RetrievalEngine
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

CORPUS = [
    ("e2e_test_fuse_1", "Insulin regulates blood glucose levels.", "bio"),
    (
        "e2e_test_fuse_2",
        "Diabetes affects glucose metabolism and insulin response.",
        "bio",
    ),
    ("e2e_test_fuse_3", "Neural networks learn from training data.", "cs"),
    ("e2e_test_fuse_4", "Transformer architecture uses attention mechanisms.", "cs"),
]
TEXT_TABLE = "E2EFusionDocs"


@pytest.fixture
def fused_corpus(fresh_iris_vector_store: IRISVectorStore, e2e_database_cleanup):
    store = fresh_iris_vector_store
    docs = [
        Document(id=i, page_content=t, metadata={"source": s}) for i, t, s in CORPUS
    ]
    store.add_documents(docs, auto_chunk=False)

    conn = store._get_connection()
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {TEXT_TABLE}")
    cursor.execute(f"CREATE TABLE {TEXT_TABLE} (id VARCHAR(100), text VARCHAR(2000))")
    for i, t, _ in CORPUS:
        cursor.execute(f"INSERT INTO {TEXT_TABLE} (id, text) VALUES (?, ?)", [i, t])
    conn.commit()
    cursor.close()
    yield store, conn
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {TEXT_TABLE}")
    conn.commit()
    cursor.close()


@pytest.fixture
def table_bound_text_engine():
    """The text engine reads a table named `docs` unless told otherwise; aim it at ours."""
    import iris_vector_graph.text_search as ts

    real = ts.TextSearchEngine.search_documents

    def search_documents(self, query_text, k=50, table_name=TEXT_TABLE):
        return real(self, query_text, k=k, table_name=table_name)

    ts.TextSearchEngine.search_documents = search_documents
    yield
    ts.TextSearchEngine.search_documents = real


@pytest.mark.true_e2e
class TestFusionE2E:
    def _engine(self, store, conn):
        return RetrievalEngine(vector_store=store, connection=conn)

    @pytest.mark.parametrize("mode", ["text", "rrf", "hybrid"])
    def test_every_hit_has_content_and_the_overlap_ranks_first(
        self, fused_corpus, table_bound_text_engine, mode
    ):
        store, conn = fused_corpus
        opts = QueryOptions(query="insulin", retrieval=mode, top_k=3)

        docs = self._engine(store, conn).retrieve(opts)

        assert docs, f"{mode} returned nothing"
        assert all(d.page_content.strip() for d in docs), [
            d.id for d in docs if not d.page_content.strip()
        ]
        assert len({d.id for d in docs}) == len(docs), "one entry per document"
        if mode != "text":
            top = docs[0]
            # Both documents mentioning insulin are in the text leg and the vector
            # leg's top; the best of them leads with a positive fused score.
            assert top.id in {"e2e_test_fuse_1", "e2e_test_fuse_2"}
            assert top.metadata["fusion_score"] > 0
            assert top.metadata["vector_score"] > 0
            assert top.metadata["text_score"] > 0

    def test_a_filter_scopes_every_mode(self, fused_corpus, table_bound_text_engine):
        store, conn = fused_corpus
        for mode in ("vector", "text", "rrf", "hybrid"):
            opts = QueryOptions(
                query="insulin",
                retrieval=mode,
                top_k=4,
                metadata_filter={"source": "bio"},
            )
            docs = self._engine(store, conn).retrieve(opts)
            assert docs, mode
            assert {d.metadata["source"] for d in docs} == {"bio"}, mode

    def test_a_threshold_drops_weak_vector_hits(
        self, fused_corpus, table_bound_text_engine
    ):
        store, conn = fused_corpus
        opts = QueryOptions(
            query="insulin", retrieval="vector", top_k=4, similarity_threshold=0.99
        )
        docs = self._engine(store, conn).retrieve(opts)
        assert docs == [], "no document is a 0.99 match for one word"

        opts = QueryOptions(
            query="insulin", retrieval="vector", top_k=4, similarity_threshold=0.01
        )
        docs = self._engine(store, conn).retrieve(opts)
        assert docs and all(d.metadata["score"] >= 0.01 for d in docs)
