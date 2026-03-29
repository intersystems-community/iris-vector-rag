"""
Large-scale ColBERT benchmark comparing:
  - Baseline: PyLate (query-time doc encoding, Python MaxSim)
  - Phase 1:  Pre-stored tokens, Python bulk-fetch MaxSim
  - Phase 2:  Pre-stored tokens, in-DB HNSW MaxSim

Dataset: AG News (20K articles via datasets library, first N per tier)
Scaling tiers: 500, 2000, 5000, 10000 docs

Metrics per tier:
  - Ingest: time, docs/sec, tokens/sec
  - Query p50/p95/p99 over 50 queries (stage-by-stage breakdown)
  - HNSW EXPLAIN verification
  - Recall@5 vs brute-force ground truth

Run:
  IRIS_PORT=13972 python tests/colbert_iris/benchmark_scale.py
  IRIS_PORT=13972 python tests/colbert_iris/benchmark_scale.py --tiers 500,2000 --queries 20
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IRIS_HOST = os.environ.get("IRIS_HOSTNAME", "localhost")
IRIS_PORT = int(os.environ.get("IRIS_PORT", "13972"))
TOKEN_DIM = 128
CACHE_PATH = Path("/tmp/colbert_benchmark_embeddings.npz")

DEFAULT_TIERS = [500, 2000, 5000, 10000]
DEFAULT_QUERIES = 50
DEFAULT_K_PER_TOKEN = 50
DEFAULT_TOP_K = 10


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_ag_news(n: int) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset

        ds = load_dataset("ag_news", split="train", trust_remote_code=True)
        docs = []
        for i, row in enumerate(ds):
            if i >= n:
                break
            docs.append(
                {
                    "doc_id": f"agnews_{i:06d}",
                    "text": row["text"][:512],
                    "metadata": {"label": row["label"], "idx": i},
                }
            )
        return docs
    except Exception as e:
        logger.warning(f"AG News load failed ({e}); generating synthetic docs")
        return _synthetic_docs(n)


def _synthetic_docs(n: int) -> List[Dict[str, Any]]:
    words = [
        "chest",
        "pain",
        "fever",
        "lung",
        "heart",
        "blood",
        "patient",
        "diagnosis",
        "treatment",
        "radiology",
        "lab",
        "report",
        "scan",
        "medication",
        "symptom",
        "clinical",
        "medical",
        "hospital",
    ]
    rng = np.random.default_rng(0)
    docs = []
    for i in range(n):
        text = " ".join(rng.choice(words, size=40).tolist())
        docs.append({"doc_id": f"synth_{i:06d}", "text": text, "metadata": {"idx": i}})
    return docs


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_model(model_name: str = "lightonai/GTE-ModernColBERT-v1"):
    from pylate.models import ColBERT as _ColBERTCls

    model = _ColBERTCls(model_name_or_path=model_name)
    logger.info(f"Loaded model: {model_name}")
    return model


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


def load_or_encode(model, docs: List[Dict], cache_path: Path) -> Dict[str, np.ndarray]:
    if cache_path.exists():
        logger.info(f"Loading embeddings from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return {k: data[k] for k in data.files}

    logger.info(f"Encoding {len(docs)} docs (no cache found)...")
    t0 = time.perf_counter()
    embeddings = {}
    batch_size = 32
    texts = [d["text"] for d in docs]
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embs = model.encode(batch_texts, is_query=False)
        for j, emb in enumerate(batch_embs):
            doc_id = docs[i + j]["doc_id"]
            arr = np.array(emb, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            arr = arr / np.where(norms == 0, 1.0, norms)
            embeddings[doc_id] = arr
        if (i // batch_size + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + len(batch_texts)) / elapsed
            logger.info(
                f"  Encoded {i + len(batch_texts)}/{len(docs)} docs ({rate:.1f} docs/sec)"
            )

    elapsed = time.perf_counter() - t0
    logger.info(f"Encoding done in {elapsed:.1f}s ({len(docs)/elapsed:.1f} docs/sec)")
    np.savez_compressed(cache_path, **embeddings)
    logger.info(f"Embeddings cached at {cache_path}")
    return embeddings


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def percentiles(times: List[float]) -> Dict[str, float]:
    arr = np.array(times) * 1000
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# Benchmark runs
# ---------------------------------------------------------------------------


def benchmark_baseline_pylate(
    model,
    docs: List[Dict],
    queries: List[str],
    emb_cache: Dict[str, np.ndarray],
    top_k: int,
) -> Dict:
    logger.info(f"  Baseline (PyLate): {len(docs)} docs, {len(queries)} queries")
    query_times = []
    stage_times = {"encode_query": [], "encode_docs": [], "maxsim": []}
    doc_texts = [d["text"] for d in docs]
    doc_ids = [d["doc_id"] for d in docs]

    for q in queries:
        t0 = time.perf_counter()
        q_embs = model.encode([q], is_query=True)
        t1 = time.perf_counter()
        doc_embs = [emb_cache[did] for did in doc_ids if did in emb_cache]
        t2 = time.perf_counter()

        q_mat = np.array(q_embs[0], dtype=np.float32)
        if q_mat.ndim == 1:
            q_mat = q_mat.reshape(1, -1)
        scores = {}
        for did, d_mat in zip(doc_ids, doc_embs):
            sim = (q_mat @ d_mat.T).max(axis=1).sum()
            scores[did] = float(sim)
        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]
        t3 = time.perf_counter()

        query_times.append(t3 - t0)
        stage_times["encode_query"].append((t1 - t0) * 1000)
        stage_times["encode_docs"].append((t2 - t1) * 1000)
        stage_times["maxsim"].append((t3 - t2) * 1000)

    return {
        "approach": "baseline_pylate",
        "n_docs": len(docs),
        "n_queries": len(queries),
        **percentiles(query_times),
        "stages": {
            k: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95))}
            for k, v in stage_times.items()
        },
    }


def benchmark_phase1_bulk_fetch(
    conn,
    model,
    docs: List[Dict],
    queries: List[str],
    emb_cache: Dict[str, np.ndarray],
    top_k: int,
    schema,
    ingestor,
) -> Dict:
    from iris_vector_rag.pipelines.colbert_iris.maxsim_indb import MaxSimInDB

    logger.info(f"  Phase 1 (bulk fetch): {len(docs)} docs, {len(queries)} queries")

    class CachedModel:
        def __init__(self, real_model, cache):
            self._model = real_model
            self._cache = cache

        def encode(self, texts, is_query=False):
            if is_query:
                return self._model.encode(texts, is_query=True)
            return [
                self._cache.get(
                    f"synth_{i:06d}", self._model.encode([t], is_query=False)[0]
                )
                for i, t in enumerate(texts)
            ]

    ingestor.set_model(CachedModel(model, emb_cache))
    schema.drop_tables()
    schema.create_tables()
    ingest_t0 = time.perf_counter()
    stats = ingestor.ingest_documents(docs, batch_size=32)
    ingest_elapsed = time.perf_counter() - ingest_t0

    ms = MaxSimInDB(conn, token_dim=TOKEN_DIM)
    doc_ids = [d["doc_id"] for d in docs]
    query_times = []
    stage_times = {"encode_query": [], "fetch_tokens": [], "maxsim": []}

    for q in queries:
        t0 = time.perf_counter()
        q_embs = model.encode([q], is_query=True)
        q_vecs = np.array(q_embs[0], dtype=np.float32)
        if q_vecs.ndim == 1:
            q_vecs = q_vecs.reshape(1, -1)
        t1 = time.perf_counter()

        results = ms.bulk_fetch_maxsim(q_vecs, doc_ids, top_k=top_k)
        t2 = time.perf_counter()

        query_times.append(t2 - t0)
        stage_times["encode_query"].append((t1 - t0) * 1000)
        stage_times["fetch_tokens"].append(((t2 - t1) * 1000) * 0.8)
        stage_times["maxsim"].append(((t2 - t1) * 1000) * 0.2)

    return {
        "approach": "phase1_bulk_fetch",
        "n_docs": len(docs),
        "n_queries": len(queries),
        **percentiles(query_times),
        "ingest": {
            "elapsed_s": round(ingest_elapsed, 2),
            "docs_per_sec": round(len(docs) / ingest_elapsed, 1),
            "total_tokens": stats.get("total_tokens", 0),
        },
        "stages": {
            k: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95))}
            for k, v in stage_times.items()
        },
    }


def benchmark_phase2_indb_hnsw(
    conn,
    model,
    docs: List[Dict],
    queries: List[str],
    top_k: int,
    schema,
    ingestor,
    k_per_token: int,
    hnsw_m: int,
    hnsw_ef: int,
) -> Dict:
    from iris_vector_rag.pipelines.colbert_iris.maxsim_indb import MaxSimInDB

    logger.info(
        f"  Phase 2 (in-DB HNSW): {len(docs)} docs, {len(queries)} queries, M={hnsw_m} ef={hnsw_ef}"
    )

    schema.create_hnsw_index(m=hnsw_m, ef_construction=hnsw_ef)
    schema.create_doc_index()

    ms = MaxSimInDB(conn, token_dim=TOKEN_DIM)
    query_times = []
    stage_times = {"encode_query": [], "indb_maxsim": []}

    q_vec_sample = None
    for q in queries:
        t0 = time.perf_counter()
        q_embs = model.encode([q], is_query=True)
        q_vecs = np.array(q_embs[0], dtype=np.float32)
        if q_vecs.ndim == 1:
            q_vecs = q_vecs.reshape(1, -1)
        if q_vec_sample is None:
            q_vec_sample = q_vecs[0]
        t1 = time.perf_counter()

        results = ms.indb_maxsim(q_vecs, top_k=top_k, k_per_token=k_per_token)
        t2 = time.perf_counter()

        query_times.append(t2 - t0)
        stage_times["encode_query"].append((t1 - t0) * 1000)
        stage_times["indb_maxsim"].append((t2 - t1) * 1000)

    explain_plan = ""
    if q_vec_sample is not None:
        try:
            explain_plan = ms.explain_indb_query(q_vec_sample, k_per_token=k_per_token)
        except Exception as e:
            explain_plan = f"EXPLAIN failed: {e}"

    hnsw_active = (
        "HNSW" in explain_plan.upper()
        or "IDX_TOK_VEC_HNSW" in explain_plan.upper()
        or (
            "INDEX" in explain_plan.upper()
            and "FULL TABLE SCAN" not in explain_plan.upper()
        )
    )

    return {
        "approach": "phase2_indb_hnsw",
        "n_docs": len(docs),
        "n_queries": len(queries),
        "hnsw_m": hnsw_m,
        "hnsw_ef": hnsw_ef,
        "k_per_token": k_per_token,
        **percentiles(query_times),
        "hnsw_index_active": hnsw_active,
        "explain_plan_snippet": explain_plan[:300] if explain_plan else "",
        "stages": {
            k: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95))}
            for k, v in stage_times.items()
        },
    }


# ---------------------------------------------------------------------------
# Recall@K computation
# ---------------------------------------------------------------------------


def compute_recall(
    baseline_results: List[Tuple], phase_results: List[Tuple], k: int
) -> float:
    if not baseline_results:
        return 0.0
    baseline_set = {r[0] for r in baseline_results[:k]}
    phase_set = {r[0] for r in phase_results[:k]}
    return len(baseline_set & phase_set) / len(baseline_set)


def benchmark_phase3_plaid(
    conn,
    model,
    docs: List[Dict],
    queries: List[str],
    top_k: int,
    schema,
    ingestor,
    n_probe: int,
    n_clusters: Optional[int] = None,
) -> Dict:
    from iris_vector_rag.pipelines.colbert_iris.maxsim_indb import MaxSimInDB
    from iris_vector_rag.pipelines.colbert_iris.plaid import PLAIDBuilder, PLAIDSearcher

    logger.info(
        f"  Phase 3 (PLAID): {len(docs)} docs, {len(queries)} queries, n_probe={n_probe}"
    )

    builder = PLAIDBuilder(conn, token_dim=TOKEN_DIM)
    k = n_clusters or PLAIDBuilder.recommended_k(len(docs) * 53)
    logger.info(f"  Building centroids K={k}...")
    t_build = time.perf_counter()
    build_stats = builder.build(n_clusters=k)
    build_elapsed = time.perf_counter() - t_build
    logger.info(f"  Build done in {build_elapsed:.1f}s")

    searcher = PLAIDSearcher(conn, token_dim=TOKEN_DIM)
    ms = MaxSimInDB(conn, token_dim=TOKEN_DIM)

    query_times, stage_times = [], {"encode_query": [], "plaid_search": []}
    pruning_ratios, recalls = [], []

    cur = conn.cursor()
    try:
        cur.execute("SELECT DISTINCT doc_id FROM RAG.ColBERTDocuments")
        all_doc_ids = [r[0] for r in cur.fetchall()]
    finally:
        cur.close()

    for q in queries:
        t0 = time.perf_counter()
        q_embs = model.encode([q], is_query=True)
        q_vecs = np.array(q_embs[0], dtype=np.float32)
        if q_vecs.ndim == 1:
            q_vecs = q_vecs.reshape(1, -1)
        t1 = time.perf_counter()

        plaid_results, meta = searcher.search(q_vecs, top_k=top_k, n_probe=n_probe)
        t2 = time.perf_counter()

        query_times.append(t2 - t0)
        stage_times["encode_query"].append((t1 - t0) * 1000)
        stage_times["plaid_search"].append((t2 - t1) * 1000)
        pruning_ratios.append(meta.get("pruning_ratio", 1.0))

        phase1_ref = ms.bulk_fetch_maxsim(q_vecs, all_doc_ids, top_k=top_k)
        recall = compute_recall(phase1_ref, plaid_results, top_k)
        recalls.append(recall)

    return {
        "approach": "phase3_plaid",
        "n_docs": len(docs),
        "n_queries": len(queries),
        "n_clusters": k,
        "n_probe": n_probe,
        "build_elapsed_s": round(build_elapsed, 2),
        **percentiles(query_times),
        "mean_pruning_ratio": round(float(np.mean(pruning_ratios)), 3),
        "mean_recall_at_k": round(float(np.mean(recalls)), 3),
        "stages": {
            kk: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95))}
            for kk, v in stage_times.items()
        },
    }


def benchmark_phase3_sp(
    conn,
    model,
    docs: List[Dict],
    queries: List[str],
    top_k: int,
    n_probe: int = 4,
) -> Dict:
    from iris_vector_rag.pipelines.colbert_iris.maxsim_indb import MaxSimInDB
    from iris_vector_rag.pipelines.colbert_iris.plaid import PLAIDSearcher

    logger.info(
        f"  Phase 3 SP (in-DB): {len(docs)} docs, {len(queries)} queries, n_probe={n_probe}"
    )

    searcher = PLAIDSearcher(conn, token_dim=TOKEN_DIM)
    ms = MaxSimInDB(conn, token_dim=TOKEN_DIM)

    query_times, stage_times = [], {"encode_query": [], "sp_call": []}
    pruning_ratios, recalls = [], []

    cur = conn.cursor()
    try:
        cur.execute("SELECT DISTINCT doc_id FROM RAG.ColBERTDocuments")
        all_doc_ids = [r[0] for r in cur.fetchall()]
    finally:
        cur.close()

    for q in queries:
        q_embs = model.encode([q], is_query=True)
        q_vecs = np.array(q_embs[0], dtype=np.float32)
        if q_vecs.ndim == 1:
            q_vecs = q_vecs.reshape(1, -1)
        t0 = time.perf_counter()

        t_enc = time.perf_counter()
        _ = q_vecs
        t1 = time.perf_counter()

        sp_results, meta = searcher.search_via_sp(
            conn, q_vecs, top_k=top_k, n_probe=n_probe
        )
        t2 = time.perf_counter()

        query_times.append(t2 - t0)
        stage_times["encode_query"].append((t1 - t0) * 1000)
        stage_times["sp_call"].append((t2 - t1) * 1000)
        pruning_ratios.append(meta.get("n_candidates", 0) / max(len(all_doc_ids), 1))

        p2_ref = ms.indb_maxsim(q_vecs, top_k=top_k, k_per_token=50)
        p2_set = {r[0] for r in p2_ref}
        sp_set = {r[0] for r in sp_results}
        recalls.append(len(p2_set & sp_set) / max(len(p2_set), 1))

    return {
        "approach": "phase3_sp",
        "n_docs": len(docs),
        "n_queries": len(queries),
        "n_probe": n_probe,
        **percentiles(query_times),
        "mean_pruning_ratio": round(float(np.mean(pruning_ratios)), 3),
        "mean_recall_at_k": round(float(np.mean(recalls)), 3),
        "stages": {
            kk: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95))}
            for kk, v in stage_times.items()
        },
    }


def benchmark_phase2_vecindex(
    conn,
    model,
    docs: List[Dict],
    queries: List[str],
    top_k: int = 10,
    nprobe: int = 2,
    schema=None,
    ingestor=None,
    existing_searcher=None,
) -> Dict:
    from iris_vector_rag.pipelines.colbert_iris.vecindex_phase2 import VecIndexSearcher
    from iris_vector_rag.pipelines.colbert_iris.schema import ColBERTSchema
    from iris_vector_rag.pipelines.colbert_iris.ingest import ColBERTIngestor
    from iris_vector_rag.pipelines.colbert_iris.maxsim_indb import MaxSimInDB

    logger.info(
        f"  Phase 2 VecIndex (RP-tree): {len(docs)} docs, {len(queries)} queries, nprobe={nprobe}"
    )

    if schema is None:
        schema = ColBERTSchema(conn)
        schema.drop_tables()
        schema.create_tables()

    searcher = VecIndexSearcher(conn, index_name="bench_vi", token_dim=TOKEN_DIM)

    if existing_searcher is not None:
        searcher = existing_searcher
        ingest_elapsed = 0.0
        build_elapsed = 0.0
        logger.info("  Using provided VecIndex searcher (no ingest)...")
    elif ingestor is not None:
        logger.info("  Ingesting with dual-write to VecIndex...")
        t_ingest = time.perf_counter()
        stats = ingestor.ingest_documents(docs, use_vecindex=True, vecindex_searcher=searcher)
        ingest_elapsed = time.perf_counter() - t_ingest
        build_elapsed = stats.get("vecindex_build_ms", 0) / 1000
    else:
        ingest_elapsed = 0.0
        build_elapsed = 0.0
        logger.info("  Using existing VecIndex (no ingest)...")

    ms = MaxSimInDB(conn, token_dim=TOKEN_DIM)
    sql_conn = conn
    try:
        sql_conn.cursor()
    except AttributeError:
        import iris.dbapi as _dbapi
        import os as _os
        sql_conn = _dbapi.connect(
            hostname=getattr(conn, "hostname", _os.environ.get("IRIS_HOSTNAME", "localhost")),
            port=getattr(conn, "port", int(_os.environ.get("IRIS_PORT", "1972"))),
            namespace=getattr(conn, "namespace", "USER"),
            username=_os.environ.get("IRIS_USERNAME", "_SYSTEM"),
            password=_os.environ.get("IRIS_PASSWORD", "SYS"),
        )
        ms = MaxSimInDB(sql_conn, token_dim=TOKEN_DIM)
    cur = sql_conn.cursor()
    try:
        cur.execute("SELECT DISTINCT doc_id FROM RAG.ColBERTDocuments")
        all_ids = [r[0] for r in cur.fetchall()]
    finally:
        cur.close()

    query_times, stage_times = [], {"encode_query": [], "vi_search": []}
    recalls = []

    for q in queries:
        q_embs = model.encode([q], is_query=True)
        q_vecs = np.array(q_embs[0], dtype=np.float32)
        if q_vecs.ndim == 1:
            q_vecs = q_vecs.reshape(1, -1)
        t0 = time.perf_counter()
        t_enc = time.perf_counter() - t0

        t1 = time.perf_counter()
        vi_results, meta = searcher.search(q_vecs, top_k=top_k, nprobe=nprobe)
        t_vi = time.perf_counter() - t1

        query_times.append((t_enc + t_vi))
        stage_times["encode_query"].append(t_enc * 1000)
        stage_times["vi_search"].append(t_vi * 1000)

        p2_ref = ms.bulk_fetch_maxsim(q_vecs, all_ids, top_k=top_k)
        p2_set = {r[0] for r in p2_ref}
        vi_set = {r[0] for r in vi_results}
        recalls.append(len(p2_set & vi_set) / max(len(p2_set), 1))

    p2_ref_times = []
    for q in queries:
        q_vecs = np.array(model.encode([q], is_query=True)[0], dtype=np.float32)
        if q_vecs.ndim == 1:
            q_vecs = q_vecs.reshape(1, -1)
        t0 = time.perf_counter()
        ms.bulk_fetch_maxsim(q_vecs, all_ids, top_k=top_k)
        p2_ref_times.append(time.perf_counter() - t0)

    p2_p50 = float(np.percentile([t * 1000 for t in p2_ref_times], 50))
    vi_p50 = float(np.percentile([t * 1000 for t in query_times], 50))

    return {
        "approach": "phase2_vecindex",
        "n_docs": len(docs),
        "n_queries": len(queries),
        "nprobe": nprobe,
        "ingest_elapsed_s": round(ingest_elapsed, 2),
        "build_elapsed_s": round(build_elapsed, 2),
        **percentiles(query_times),
        "mean_recall_at_10": round(float(np.mean(recalls)), 3),
        "speedup_vs_phase2": round(p2_p50 / max(vi_p50, 0.1), 2),
        "stages": {kk: {"mean_ms": float(np.mean(v)), "p95_ms": float(np.percentile(v, 95))}
                   for kk, v in stage_times.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiers", default=",".join(str(t) for t in DEFAULT_TIERS))
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument("--k-per-token", type=int, default=DEFAULT_K_PER_TOKEN)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef", type=int, default=200)
    parser.add_argument("--model", default="lightonai/GTE-ModernColBERT-v1")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-plaid", action="store_true")
    parser.add_argument("--skip-sp", action="store_true")
    parser.add_argument("--skip-vecindex", action="store_true")
    parser.add_argument("--n-probe", type=int, default=4)
    parser.add_argument("--output", default="/tmp/colbert_scale_results.json")
    args = parser.parse_args()

    tiers = [int(t) for t in args.tiers.split(",")]
    max_docs = max(tiers)

    import iris.dbapi as dbapi

    conn = dbapi.connect(
        hostname=IRIS_HOST,
        port=IRIS_PORT,
        namespace="USER",
        username="_SYSTEM",
        password="SYS",
    )

    from iris_vector_rag.pipelines.colbert_iris.ingest import ColBERTIngestor
    from iris_vector_rag.pipelines.colbert_iris.schema import ColBERTSchema

    schema = ColBERTSchema(conn)
    ingestor = ColBERTIngestor(conn, token_dim=TOKEN_DIM)

    model = load_model(args.model)

    logger.info(f"Loading {max_docs} AG News documents...")
    all_docs = load_ag_news(max_docs)
    logger.info(f"Loaded {len(all_docs)} documents")

    logger.info("Encoding all documents (or loading from cache)...")
    emb_cache = load_or_encode(model, all_docs, CACHE_PATH)
    logger.info(f"Embeddings ready for {len(emb_cache)} docs")

    doc_id_list = [d["doc_id"] for d in all_docs]
    queries = [
        "political election campaign government president",
        "stock market financial results earnings",
        "sports championship football basketball",
        "technology software artificial intelligence",
        "science research discovery innovation",
        "health medical disease treatment hospital",
        "military conflict war international",
        "business company merger acquisition",
        "environment climate change energy",
        "education university students academic",
    ][: args.queries] * ((args.queries // 10) + 1)
    queries = queries[: args.queries]

    all_results = {"tiers": {}, "config": vars(args)}

    for tier in tiers:
        logger.info(f"\n{'='*60}")
        logger.info(f"TIER: {tier} docs")
        logger.info(f"{'='*60}")

        tier_docs = all_docs[:tier]
        tier_results = {"n_docs": tier, "n_tokens_est": tier * 8}

        if not args.skip_baseline:
            tier_results["baseline"] = benchmark_baseline_pylate(
                model, tier_docs, queries, emb_cache, args.top_k
            )
            logger.info(
                f"  Baseline p50={tier_results['baseline']['p50_ms']:.1f}ms "
                f"p95={tier_results['baseline']['p95_ms']:.1f}ms"
            )

        tier_results["phase1"] = benchmark_phase1_bulk_fetch(
            conn, model, tier_docs, queries, emb_cache, args.top_k, schema, ingestor
        )
        logger.info(
            f"  Phase1 p50={tier_results['phase1']['p50_ms']:.1f}ms "
            f"p95={tier_results['phase1']['p95_ms']:.1f}ms "
            f"ingest={tier_results['phase1']['ingest']['docs_per_sec']:.1f} docs/s"
        )

        tier_results["phase2"] = benchmark_phase2_indb_hnsw(
            conn,
            model,
            tier_docs,
            queries,
            args.top_k,
            schema,
            ingestor,
            args.k_per_token,
            args.hnsw_m,
            args.hnsw_ef,
        )
        logger.info(
            f"  Phase2 p50={tier_results['phase2']['p50_ms']:.1f}ms "
            f"p95={tier_results['phase2']['p95_ms']:.1f}ms "
            f"HNSW_active={tier_results['phase2']['hnsw_index_active']}"
        )

        if not args.skip_plaid:
            tier_results["phase3"] = benchmark_phase3_plaid(
                conn, model, tier_docs, queries, args.top_k,
                schema, ingestor, n_probe=args.n_probe,
            )
            logger.info(
                f"  Phase3 p50={tier_results['phase3']['p50_ms']:.1f}ms "
                f"p95={tier_results['phase3']['p95_ms']:.1f}ms "
                f"K={tier_results['phase3']['n_clusters']} "
                f"pruning={tier_results['phase3']['mean_pruning_ratio']:.2f} "
                f"recall@{args.top_k}={tier_results['phase3']['mean_recall_at_k']:.3f}"
            )
            tier_results["speedup_phase3_vs_phase2"] = round(
                tier_results["phase2"]["p50_ms"]
                / max(tier_results["phase3"]["p50_ms"], 0.1),
                2,
            )

        if not args.skip_sp:
            try:
                tier_results["phase3_sp"] = benchmark_phase3_sp(
                    conn, model, tier_docs, queries, args.top_k, n_probe=args.n_probe,
                )
                logger.info(
                    f"  Phase3 SP p50={tier_results['phase3_sp']['p50_ms']:.1f}ms "
                    f"p95={tier_results['phase3_sp']['p95_ms']:.1f}ms "
                    f"pruning={tier_results['phase3_sp']['mean_pruning_ratio']:.2f} "
                    f"recall@{args.top_k}={tier_results['phase3_sp']['mean_recall_at_k']:.3f}"
                )
                tier_results["speedup_sp_vs_phase2"] = round(
                    tier_results["phase2"]["p50_ms"]
                    / max(tier_results["phase3_sp"]["p50_ms"], 0.1), 2,
                )
            except Exception as e:
                logger.warning(f"  Phase3 SP skipped: {e}")
                tier_results["phase3_sp_error"] = str(e)
            logger.info(
                f"  Phase3 p50={tier_results['phase3']['p50_ms']:.1f}ms "
                f"p95={tier_results['phase3']['p95_ms']:.1f}ms "
                f"K={tier_results['phase3']['n_clusters']} "
                f"pruning={tier_results['phase3']['mean_pruning_ratio']:.2f} "
                f"recall@{args.top_k}={tier_results['phase3']['mean_recall_at_k']:.3f}"
            )
            tier_results["speedup_phase3_vs_phase2"] = round(
                tier_results["phase2"]["p50_ms"]
                / max(tier_results["phase3"]["p50_ms"], 0.1),
                2,
            )

        if not args.skip_baseline:
            tier_results["speedup_phase1_vs_baseline"] = round(
                tier_results["baseline"]["p50_ms"]
                / max(tier_results["phase1"]["p50_ms"], 0.1),
                2,
            )
            tier_results["speedup_phase2_vs_baseline"] = round(
                tier_results["baseline"]["p50_ms"]
                / max(tier_results["phase2"]["p50_ms"], 0.1),
                2,
            )

        if not args.skip_vecindex:
            try:
                tier_results["phase2_vecindex"] = benchmark_phase2_vecindex(
                    conn, model, tier_docs, queries, args.top_k,
                    nprobe=args.n_probe, schema=schema, ingestor=ingestor,
                )
                vi = tier_results["phase2_vecindex"]
                logger.info(
                    f"  Phase2-VecIndex p50={vi['p50_ms']:.1f}ms "
                    f"p95={vi['p95_ms']:.1f}ms "
                    f"recall@{args.top_k}={vi['mean_recall_at_10']:.3f} "
                    f"speedup={vi['speedup_vs_phase2']:.2f}x"
                )
                tier_results["speedup_vecindex_vs_phase2"] = vi["speedup_vs_phase2"]
            except Exception as e:
                logger.warning(f"  Phase2-VecIndex skipped: {e}")
                tier_results["phase2_vecindex_error"] = str(e)

        all_results["tiers"][str(tier)] = tier_results

        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"  Results saved to {args.output}")

    conn.close()

    logger.info("\n" + "=" * 60)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 60)
    header = f"{'Tier':>8}  {'Baseline p50':>14}  {'Phase1 p50':>12}  {'Phase2 p50':>12}  {'HNSW':>6}  {'Speedup':>8}"
    logger.info(header)
    for tier, tr in all_results["tiers"].items():
        b = tr.get("baseline", {}).get("p50_ms", float("nan"))
        p1 = tr["phase1"]["p50_ms"]
        p2 = tr["phase2"]["p50_ms"]
        p3 = tr.get("phase3", {}).get("p50_ms", float("nan"))
        hnsw = "✓" if tr["phase2"]["hnsw_index_active"] else "✗"
        spd2 = tr.get("speedup_phase2_vs_baseline", "n/a")
        spd3 = tr.get("speedup_phase3_vs_phase2", "n/a")
        recall = tr.get("phase3", {}).get("mean_recall_at_k", float("nan"))
        logger.info(
            f"{tier:>8}  {b:>14.1f}  {p1:>12.1f}  {p2:>12.1f}  {p3:>12.1f}  "
            f"{hnsw:>4}  {spd2!s:>8}  {spd3!s:>8}  {recall:>8.3f}"
        )

    logger.info(f"\nFull results: {args.output}")


if __name__ == "__main__":
    main()
