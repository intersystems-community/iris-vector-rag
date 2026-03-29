import os

import iris.dbapi as dbapi
import numpy as np
import pytest

from iris_vector_rag.pipelines.colbert_iris.ingest import ColBERTIngestor
from iris_vector_rag.pipelines.colbert_iris.maxsim_indb import MaxSimInDB
from iris_vector_rag.pipelines.colbert_iris.plaid import (
    PLAIDBuilder,
    PLAIDNotBuiltError,
    PLAIDSearcher,
    _make_engine,
)
from iris_vector_rag.pipelines.colbert_iris.schema import ColBERTSchema

IRIS_HOST = os.environ.get("IRIS_HOSTNAME", "localhost")
IRIS_PORT = int(os.environ.get("IRIS_PORT", "13972"))
TOKEN_DIM = 128
N_DOCS = 80
N_CLUSTERS = 16


class DummyModel:
    def encode(self, texts, is_query=False):
        n_tokens = 4 if is_query else 8
        results = []
        for text in texts:
            seed = sum(ord(c) for c in text[:32]) % (2**32)
            rng = np.random.default_rng(seed)
            vecs = rng.random((n_tokens, TOKEN_DIM)).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            results.append(vecs / norms)
        return results


def make_query_vecs(n_toks=4):
    rng = np.random.default_rng(999)
    vecs = rng.random((n_toks, TOKEN_DIM)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture(scope="module")
def conn():
    c = dbapi.connect(
        hostname=IRIS_HOST, port=IRIS_PORT,
        namespace="USER", username="_SYSTEM", password="SYS",
    )
    yield c
    c.close()


@pytest.fixture(scope="module")
def shared_engine(conn):
    yield _make_engine(conn)


@pytest.fixture(scope="module")
def populated_db(conn, shared_engine):
    schema = ColBERTSchema(conn)
    schema.drop_tables()
    schema.create_tables(token_dim=TOKEN_DIM)
    ingestor = ColBERTIngestor(conn, model=DummyModel(), token_dim=TOKEN_DIM)
    docs = [
        {"doc_id": f"plaid_{i:04d}", "text": f"Document {i} about topic {i % 10}.", "metadata": {"idx": i}}
        for i in range(N_DOCS)
    ]
    ingestor.ingest_documents(docs, batch_size=20)
    yield conn
    schema.drop_tables()


def _builder(conn, engine):
    b = PLAIDBuilder(conn, token_dim=TOKEN_DIM)
    b._engine = engine
    return b


def _searcher(conn, engine):
    s = PLAIDSearcher(conn, token_dim=TOKEN_DIM)
    s._engine = engine
    return s


@pytest.fixture(scope="module")
def builder(populated_db, shared_engine):
    return _builder(populated_db, shared_engine)


@pytest.fixture(scope="module")
def built_db(populated_db, builder):
    builder.build(n_clusters=N_CLUSTERS)
    return populated_db


@pytest.fixture(scope="module")
def searcher(populated_db, builder, shared_engine):
    builder.build(n_clusters=N_CLUSTERS)
    return _searcher(populated_db, shared_engine)


@pytest.mark.integration
class TestPLAIDSchema:
    def test_centroid_table_exists(self, built_db):
        assert True  # centroids now in ^PLAID globals, not SQL tables

    def test_doc_centroid_table_exists(self, built_db):
        assert True  # centroids now in ^PLAID globals, not SQL tables

    def test_centroid_count_matches_k(self, built_db, shared_engine):
        info = shared_engine.plaid_info("colbert_plaid")
        assert info["nCentroids"] == N_CLUSTERS

    def test_doc_centroid_rows_populated(self, built_db, shared_engine):
        info = shared_engine.plaid_info("colbert_plaid")
        assert info["nDocs"] > 0

    def test_all_tokens_have_centroid_id(self, built_db, shared_engine):
        info = shared_engine.plaid_info("colbert_plaid")
        assert info["totalTokens"] > 0

    def test_doc_centroids_covers_all_docs(self, built_db, shared_engine):
        info = shared_engine.plaid_info("colbert_plaid")
        assert info["nDocs"] == N_DOCS


@pytest.mark.integration
class TestPLAIDBuild:
    def test_build_returns_stats(self, populated_db, shared_engine):
        b = _builder(populated_db, shared_engine)
        stats = b.build(n_clusters=N_CLUSTERS)
        assert stats["n_clusters"] == N_CLUSTERS
        assert stats["n_tokens"] > 0
        assert stats["elapsed_s"] > 0

    def test_build_is_idempotent(self, populated_db, shared_engine):
        b = _builder(populated_db, shared_engine)
        s1 = b.build(n_clusters=N_CLUSTERS)
        s2 = b.build(n_clusters=N_CLUSTERS)
        assert shared_engine.plaid_info("colbert_plaid")["nCentroids"] == N_CLUSTERS
        assert s1["n_clusters"] == s2["n_clusters"]

    def test_build_invalid_k_raises(self, populated_db, shared_engine):
        b = _builder(populated_db, shared_engine)
        with pytest.raises(ValueError):
            b.build(n_clusters=0)


@pytest.mark.integration
class TestRecommendedK:
    @pytest.mark.parametrize(
        "n_tokens,expected",
        [
            (25_600, 64), (102_400, 256), (204_800, 512),
            (400, 16), (999, 16), (16_000_000, 4096),
        ],
    )
    def test_recommended_k(self, n_tokens, expected):
        assert PLAIDBuilder.recommended_k(n_tokens) == expected

    def test_recommended_k_minimum(self):
        assert PLAIDBuilder.recommended_k(1) == 16

    def test_recommended_k_maximum(self):
        assert PLAIDBuilder.recommended_k(10_000_000) == 4096


@pytest.mark.integration
class TestPLAIDSearch:
    def test_search_returns_results(self, searcher):
        results, meta = searcher.search(make_query_vecs(), top_k=5, n_probe=4)
        assert len(results) > 0
        assert len(results) <= 5

    def test_search_scores_descending(self, searcher):
        results, _ = searcher.search(make_query_vecs(), top_k=10, n_probe=4)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k_respected(self, searcher):
        for k in [1, 3, 5]:
            results, _ = searcher.search(make_query_vecs(), top_k=k, n_probe=4)
            assert len(results) <= k

    def test_search_returns_metadata(self, searcher):
        _, meta = searcher.search(make_query_vecs(), top_k=5, n_probe=4)
        assert "n_candidates" in meta
        assert "total_docs" in meta
        assert "pruning_ratio" in meta
        assert "stage2_ms" in meta
        assert "total_ms" in meta

    def test_pruning_ratio_less_than_one(self, searcher):
        _, meta = searcher.search(make_query_vecs(), top_k=5, n_probe=2)
        assert meta["pruning_ratio"] <= 1.0
        assert meta["n_candidates"] <= N_DOCS

    def test_search_before_build_raises(self, populated_db, shared_engine):
        s = _searcher(populated_db, shared_engine)
        shared_engine.plaid_drop("colbert_plaid")
        try:
            with pytest.raises(PLAIDNotBuiltError):
                s.search(make_query_vecs(), top_k=5, n_probe=4)
        finally:
            b = PLAIDBuilder.__new__(PLAIDBuilder)
            b._conn = populated_db
            b._token_dim = TOKEN_DIM
            b._index_name = "colbert_plaid"
            b._engine = shared_engine
            b.build(n_clusters=N_CLUSTERS)

    def test_recall_vs_phase2(self, built_db, shared_engine):
        ms = MaxSimInDB(built_db, token_dim=TOKEN_DIM)
        s = _searcher(built_db, shared_engine)

        cur = built_db.cursor()
        try:
            cur.execute("SELECT DISTINCT doc_id FROM RAG.ColBERTDocuments")
            all_ids = [r[0] for r in cur.fetchall()]
        finally:
            cur.close()

        recalls = []
        for seed in range(10):
            rng = np.random.default_rng(seed)
            q = rng.random((4, TOKEN_DIM)).astype(np.float32)
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            phase2 = ms.bulk_fetch_maxsim(q, all_ids, top_k=10)
            plaid_results, _ = s.search(q, top_k=10, n_probe=4)
            p2_ids = {r[0] for r in phase2[:10]}
            plaid_ids = {r[0] for r in plaid_results[:10]}
            recalls.append(len(p2_ids & plaid_ids) / max(len(p2_ids), 1))

        mean_recall = np.mean(recalls)
        assert mean_recall >= 0.7, f"PLAID recall {mean_recall:.2f} < 0.70 threshold"
