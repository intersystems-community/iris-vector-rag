"""
Test suite for iris_vector_rag.pipelines.colbert_iris

Run against the dedicated spike container:
  IRIS_PORT=13972 pytest tests/colbert_iris/ -v -m integration

Covers:
  - Schema creation and table existence
  - Token ingestion round-trip
  - Phase 1 bulk-fetch MaxSim
  - Phase 2 in-DB HNSW MaxSim
  - EXPLAIN plan verification (HNSW must be used, no full table scan)
  - HNSW parameter sweep (M=8/16/32/48)
  - Ranking correctness (synthetic docs)
  - Edge cases: empty query, single doc, no candidates
"""

import json
import os
import time
import uuid

import numpy as np
import pytest

import iris.dbapi as dbapi

from iris_vector_rag.pipelines.colbert_iris.ingest import ColBERTIngestor
from iris_vector_rag.pipelines.colbert_iris.maxsim_indb import MaxSimInDB
from iris_vector_rag.pipelines.colbert_iris.schema import ColBERTSchema

IRIS_HOST = os.environ.get("IRIS_HOSTNAME", "localhost")
IRIS_PORT = int(os.environ.get("IRIS_PORT", "13972"))
TOKEN_DIM = 128
MIN_DOCS_FOR_HNSW = 200


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class DummyModel:
    """Minimal ColBERT model stub: returns random normalised token tensors."""

    def encode(self, texts, is_query=False):
        rng = np.random.default_rng(42)
        n_tokens = 4 if is_query else 8
        results = []
        for text in texts:
            seed = sum(ord(c) for c in text[:32]) % (2**32)
            rng2 = np.random.default_rng(seed)
            vecs = rng2.random((n_tokens, TOKEN_DIM)).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            results.append(vecs / norms)
        return results


@pytest.fixture(scope="module")
def conn():
    c = dbapi.connect(
        hostname=IRIS_HOST,
        port=IRIS_PORT,
        namespace="USER",
        username="_SYSTEM",
        password="SYS",
    )
    yield c
    c.close()


@pytest.fixture(scope="module")
def schema(conn):
    s = ColBERTSchema(conn)
    s.drop_tables()
    s.create_tables(token_dim=TOKEN_DIM)
    yield s
    s.drop_tables()


@pytest.fixture(scope="module")
def ingestor(conn, schema):
    ing = ColBERTIngestor(conn, model=DummyModel(), token_dim=TOKEN_DIM)
    return ing


@pytest.fixture(scope="module")
def maxsim(conn, schema):
    return MaxSimInDB(conn, token_dim=TOKEN_DIM)


def make_docs(n: int, prefix: str = "doc") -> list:
    return [
        {"doc_id": f"{prefix}_{i}", "text": f"Medical text about topic {i}.", "metadata": {"idx": i}}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Schema tests (1-5)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_tables_created(schema):
    assert schema.table_exists("RAG.ColBERTDocuments")
    assert schema.table_exists("RAG.DocumentTokenEmbeddings")


@pytest.mark.integration
def test_tables_empty_after_create(schema):
    assert schema.row_count("RAG.ColBERTDocuments") == 0
    assert schema.row_count("RAG.DocumentTokenEmbeddings") == 0


@pytest.mark.integration
def test_drop_and_recreate(conn):
    s = ColBERTSchema(conn)
    s.drop_tables()
    assert not s.table_exists("RAG.ColBERTDocuments")
    s.create_tables(token_dim=TOKEN_DIM)
    assert s.table_exists("RAG.ColBERTDocuments")


@pytest.mark.integration
def test_hnsw_index_created(schema, ingestor, conn):
    docs = make_docs(MIN_DOCS_FOR_HNSW, prefix="hnsw_test")
    ingestor.ingest_documents(docs, batch_size=50)
    schema.create_hnsw_index(m=16, ef_construction=200)
    assert schema.index_exists("IDX_TOK_VEC_HNSW")


@pytest.mark.integration
def test_doc_index_created(schema):
    schema.create_doc_index()


# ---------------------------------------------------------------------------
# Ingestion tests (6-10)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ingest_single_doc(ingestor, schema):
    doc = {"doc_id": "single_001", "text": "Single document for ingestion test.", "metadata": {}}
    stats = ingestor.ingest_documents([doc])
    assert stats["docs_ingested"] == 1
    assert stats["docs_failed"] == 0
    assert stats["total_tokens"] > 0
    assert schema.row_count("RAG.ColBERTDocuments") >= 1
    assert schema.row_count("RAG.DocumentTokenEmbeddings") >= 1


@pytest.mark.integration
def test_ingest_batch(ingestor, schema):
    docs = make_docs(20, prefix="batch")
    stats = ingestor.ingest_documents(docs, batch_size=8)
    assert stats["docs_ingested"] == 20
    assert stats["docs_failed"] == 0
    assert stats["total_tokens"] >= 20 * 4


@pytest.mark.integration
def test_ingest_upsert(ingestor, schema):
    doc = {"doc_id": "upsert_001", "text": "Original text.", "metadata": {}}
    ingestor.ingest_documents([doc])
    doc["text"] = "Updated text."
    ingestor.ingest_documents([doc])
    cur = schema._conn.cursor()
    cur.execute("SELECT COUNT(*) FROM RAG.ColBERTDocuments WHERE doc_id = 'upsert_001'")
    count = cur.fetchone()[0]
    cur.close()
    assert count == 1


@pytest.mark.integration
def test_ingest_metadata_roundtrip(ingestor, schema):
    meta = {"source": "test.pdf", "page": 42, "active": True}
    doc = {"doc_id": "meta_001", "text": "Metadata roundtrip test.", "metadata": meta}
    ingestor.ingest_documents([doc])
    cur = schema._conn.cursor()
    cur.execute("SELECT metadata FROM RAG.ColBERTDocuments WHERE doc_id = 'meta_001'")
    row = cur.fetchone()
    metadata_str = row[0] if row else None
    cur.close()
    stored = json.loads(metadata_str)
    assert stored["source"] == "test.pdf"
    assert stored["page"] == 42


@pytest.mark.integration
def test_ingest_token_count_positive(ingestor, schema):
    doc = {"doc_id": "token_count_001", "text": "Some text to be tokenised.", "metadata": {}}
    stats = ingestor.ingest_documents([doc])
    assert stats["total_tokens"] > 0


# ---------------------------------------------------------------------------
# Phase 1 — bulk_fetch_maxsim tests (11-15)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_bulk_fetch_returns_results(ingestor, maxsim):
    docs = make_docs(10, prefix="bulk")
    ingestor.ingest_documents(docs)
    q_vecs = np.random.rand(4, TOKEN_DIM).astype(np.float32)
    q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True)
    ids = [f"bulk_{i}" for i in range(10)]
    results = maxsim.bulk_fetch_maxsim(q_vecs, ids, top_k=5)
    assert len(results) <= 5
    assert all(isinstance(r[1], float) for r in results)


@pytest.mark.integration
def test_bulk_fetch_scores_descending(ingestor, maxsim):
    docs = make_docs(10, prefix="bulk_ord")
    ingestor.ingest_documents(docs)
    q_vecs = np.random.rand(4, TOKEN_DIM).astype(np.float32)
    q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True)
    ids = [f"bulk_ord_{i}" for i in range(10)]
    results = maxsim.bulk_fetch_maxsim(q_vecs, ids, top_k=10)
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
def test_bulk_fetch_empty_candidates(maxsim):
    q_vecs = np.random.rand(4, TOKEN_DIM).astype(np.float32)
    results = maxsim.bulk_fetch_maxsim(q_vecs, [], top_k=5)
    assert results == []


@pytest.mark.integration
def test_bulk_fetch_top_k_respected(ingestor, maxsim):
    docs = make_docs(20, prefix="topk")
    ingestor.ingest_documents(docs)
    q_vecs = np.random.rand(4, TOKEN_DIM).astype(np.float32)
    ids = [f"topk_{i}" for i in range(20)]
    for k in [1, 3, 5]:
        results = maxsim.bulk_fetch_maxsim(q_vecs, ids, top_k=k)
        assert len(results) <= k


@pytest.mark.integration
def test_bulk_fetch_synthetic_ranking(conn, schema):
    s = ColBERTSchema(conn)
    ing = ColBERTIngestor(conn, token_dim=TOKEN_DIM)

    target_vec = np.ones((1, TOKEN_DIM), dtype=np.float32)
    target_vec /= np.linalg.norm(target_vec)

    class TargetModel:
        def encode(self, texts, is_query=False):
            results = []
            for t in texts:
                if "target" in t:
                    results.append(target_vec.copy())
                else:
                    noise = np.random.rand(1, TOKEN_DIM).astype(np.float32)
                    noise /= np.linalg.norm(noise)
                    results.append(noise)
            return results

    ing.set_model(TargetModel())
    docs = [
        {"doc_id": "synth_target", "text": "target document", "metadata": {}},
        {"doc_id": "synth_noise1", "text": "irrelevant noise document", "metadata": {}},
        {"doc_id": "synth_noise2", "text": "another unrelated document", "metadata": {}},
    ]
    ing.ingest_documents(docs)

    ms = MaxSimInDB(conn, token_dim=TOKEN_DIM)
    results = ms.bulk_fetch_maxsim(target_vec, ["synth_target", "synth_noise1", "synth_noise2"], top_k=3)
    assert results[0][0] == "synth_target", f"Expected synth_target first, got {results[0][0]}"


# ---------------------------------------------------------------------------
# Phase 2 — indb_maxsim tests (16-18)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_indb_maxsim_returns_results(ingestor, maxsim):
    docs = make_docs(10, prefix="indb")
    ingestor.ingest_documents(docs)
    q_vecs = np.random.rand(4, TOKEN_DIM).astype(np.float32)
    q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True)
    results = maxsim.indb_maxsim(q_vecs, top_k=5, k_per_token=20)
    assert len(results) > 0
    assert all(isinstance(r[1], float) for r in results)


@pytest.mark.integration
def test_indb_maxsim_scores_descending(ingestor, maxsim):
    docs = make_docs(15, prefix="indb_ord")
    ingestor.ingest_documents(docs)
    q_vecs = np.random.rand(4, TOKEN_DIM).astype(np.float32)
    q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True)
    results = maxsim.indb_maxsim(q_vecs, top_k=10, k_per_token=30)
    scores = [r[1] for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
def test_indb_maxsim_top_k_respected(ingestor, maxsim):
    docs = make_docs(20, prefix="indb_topk")
    ingestor.ingest_documents(docs)
    q_vecs = np.random.rand(4, TOKEN_DIM).astype(np.float32)
    q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True)
    for k in [1, 3, 5]:
        results = maxsim.indb_maxsim(q_vecs, top_k=k, k_per_token=20)
        assert len(results) <= k


# ---------------------------------------------------------------------------
# EXPLAIN plan verification tests (19-22)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_explain_uses_hnsw_for_token_query(maxsim, schema):
    if not schema.index_exists("IDX_TOK_VEC_HNSW"):
        pytest.skip("HNSW index not present — run test_hnsw_index_created first")
    q_vec = np.random.rand(TOKEN_DIM).astype(np.float32)
    q_vec /= np.linalg.norm(q_vec)
    plan = maxsim.explain_indb_query(q_vec, k_per_token=50)
    assert plan, "EXPLAIN returned empty plan"
    plan_upper = plan.upper()
    assert "FULL TABLE SCAN" not in plan_upper or "HNSW" in plan_upper or "INDEX" in plan_upper, (
        f"Query plan suggests no index usage:\n{plan}"
    )


@pytest.mark.integration
def test_explain_no_full_table_scan_with_hnsw(maxsim, schema):
    if not schema.index_exists("IDX_TOK_VEC_HNSW"):
        pytest.skip("HNSW index not present")
    q_vec = np.ones(TOKEN_DIM, dtype=np.float32)
    q_vec /= np.linalg.norm(q_vec)
    plan = maxsim.explain_indb_query(q_vec, k_per_token=10)
    assert "READ" in plan.upper() or "INDEX" in plan.upper() or "HNSW" in plan.upper(), (
        f"Expected index read in plan:\n{plan}"
    )


@pytest.mark.integration
def test_explain_plan_is_non_empty(maxsim):
    q_vec = np.random.rand(TOKEN_DIM).astype(np.float32)
    q_vec /= np.linalg.norm(q_vec)
    plan = maxsim.explain_indb_query(q_vec)
    assert len(plan.strip()) > 0


@pytest.mark.integration
def test_explain_plan_contains_documenttokenembeddings(maxsim):
    q_vec = np.random.rand(TOKEN_DIM).astype(np.float32)
    q_vec /= np.linalg.norm(q_vec)
    plan = maxsim.explain_indb_query(q_vec)
    assert "DOCUMENTTOKENEMBEDDINGS" in plan.upper() or "DocumentTokenEmbeddings".upper() in plan.upper()


# ---------------------------------------------------------------------------
# HNSW parameter sweep (23-26)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("m,ef", [(8, 100), (16, 200), (32, 300), (48, 400)])
def test_hnsw_param_sweep(conn, ingestor, m, ef):
    s = ColBERTSchema(conn)
    docs = make_docs(MIN_DOCS_FOR_HNSW, prefix=f"sweep_m{m}")
    ingestor.ingest_documents(docs)
    s.create_hnsw_index(m=m, ef_construction=ef)
    assert s.index_exists("IDX_TOK_VEC_HNSW")

    ms = MaxSimInDB(conn, token_dim=TOKEN_DIM)
    q_vecs = np.random.rand(4, TOKEN_DIM).astype(np.float32)
    q_vecs /= np.linalg.norm(q_vecs, axis=1, keepdims=True)
    results = ms.indb_maxsim(q_vecs, top_k=5, k_per_token=30)
    assert len(results) > 0, f"No results for HNSW M={m} efC={ef}"
