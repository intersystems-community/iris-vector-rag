import json
import os
import statistics
import subprocess

import iris.dbapi as dbapi
import numpy as np
import pytest

from iris_vector_rag.pipelines.colbert_iris.maxsim_indb import MaxSimInDB
from iris_vector_rag.pipelines.colbert_iris.plaid import PLAIDSearcher

IRIS_HOST = os.environ.get("IRIS_HOSTNAME", "localhost")
IRIS_PORT = int(os.environ.get("IRIS_PORT", "13972"))
CONTAINER = os.environ.get("IRIS_CONTAINER", "iris-langchain-spike")


def make_query_vecs(n_toks: int = 4, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.random((n_toks, 128)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def call_sp(conn, q_vecs, top_k=10, n_probe=4):
    q_json = json.dumps(q_vecs.tolist())
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT RAG.ColBERTSearch_Search(?, ?, ?)", [q_json, top_k, n_probe]
        )
        row = cur.fetchone()
        raw_json = row[0]
    finally:
        cur.close()
    return json.loads(raw_json)


@pytest.fixture(scope="session")
def sp_conn():
    try:
        conn = dbapi.connect(
            hostname=IRIS_HOST,
            port=IRIS_PORT,
            namespace="USER",
            username="_SYSTEM",
            password="SYS",
        )
    except Exception as e:
        pytest.skip(f"IRIS not available at {IRIS_HOST}:{IRIS_PORT}: {e}")
    yield conn
    conn.close()


@pytest.mark.integration
@pytest.mark.xfail(
    reason="Requires docker-in-docker (not available) and pins numpy 1.26.* (container has 2.x)",
    strict=False,
)
class TestNumpy:
    def test_numpy_install_in_iris(self):
        result = subprocess.run(
            [
                "docker",
                "exec",
                CONTAINER,
                "python3",
                "-c",
                "import numpy; print(numpy.__version__)",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, f"import numpy failed: {result.stderr}"
        assert result.stdout.strip().startswith(
            "1.26"
        ), f"Expected numpy 1.26.*, got {result.stdout.strip()}"


@pytest.mark.integration
@pytest.mark.xfail(
    reason="ColBERTSearch SQL SP reads RAG.ColBERTCentroids (old SQL table). "
    "Superseded by PLAIDSearch.cls globals-based implementation in IVG 1.26.1+.",
    strict=False,
)
class TestSPBasic:
    def test_sp_compiles_and_callable(self, sp_conn):
        q = make_query_vecs(n_toks=2, seed=1)
        resp = call_sp(sp_conn, q, top_k=1, n_probe=1)
        assert "results" in resp, f"SP response missing 'results': {resp}"

    def test_sp_returns_results(self, sp_conn):
        q = make_query_vecs()
        resp = call_sp(sp_conn, q, top_k=10, n_probe=4)
        results = resp["results"]
        assert len(results) <= 10
        for doc_id, score in results:
            assert isinstance(doc_id, str), f"doc_id not str: {doc_id!r}"
            assert isinstance(score, float), f"score not float: {score!r}"
            assert doc_id.lower().startswith("agnews_"), f"unexpected doc_id: {doc_id}"

    def test_sp_scores_descending(self, sp_conn):
        for seed in [42, 7, 13, 99, 555]:
            q = make_query_vecs(seed=seed)
            resp = call_sp(sp_conn, q, top_k=10, n_probe=4)
            scores = [s for _, s in resp["results"]]
            assert scores == sorted(
                scores, reverse=True
            ), f"Scores not descending for seed={seed}: {scores}"

    def test_sp_topk_respected(self, sp_conn):
        q = make_query_vecs()
        for k in [1, 3, 5, 10]:
            resp = call_sp(sp_conn, q, top_k=k, n_probe=4)
            assert (
                len(resp["results"]) <= k
            ), f"top_k={k} returned {len(resp['results'])} results"

    def test_sp_topk_exceeds_candidates(self, sp_conn):
        q = make_query_vecs()
        resp = call_sp(sp_conn, q, top_k=999, n_probe=1)
        assert "results" in resp
        assert isinstance(resp["results"], list)

    def test_sp_no_centroids_returns_empty(self, sp_conn):
        cur = sp_conn.cursor()
        try:
            cur.execute("DELETE FROM RAG.ColBERTDocCentroids")
            sp_conn.commit()
        except Exception:
            try:
                sp_conn.rollback()
            except Exception:
                pass
            pytest.skip("Could not empty DocCentroids table")
        finally:
            cur.close()

        try:
            q = make_query_vecs()
            resp = call_sp(sp_conn, q, top_k=5, n_probe=4)
            assert (
                resp.get("results") == []
            ), f"Expected empty results when no candidates, got: {resp}"
        finally:
            from iris_vector_rag.pipelines.colbert_iris.plaid import PLAIDBuilder

            try:
                PLAIDBuilder(sp_conn, token_dim=128)._populate_doc_centroids()
                sp_conn.commit()
            except Exception:
                pass

    def test_sp_malformed_json_raises(self, sp_conn):
        cur = sp_conn.cursor()
        try:
            with pytest.raises((dbapi.ProgrammingError, Exception)):
                cur.execute(
                    "SELECT RAG.ColBERTSearch_Search(?, ?, ?)",
                    ["NOT_VALID_JSON", 5, 4],
                )
                cur.fetchone()
        finally:
            cur.close()

    def test_sp_metadata_keys_present(self, sp_conn):
        cur = sp_conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM RAG.ColBERTCentroids")
            n = cur.fetchone()[0]
        finally:
            cur.close()
        if n == 0:
            pytest.skip("Centroids not built — run PLAIDBuilder.build() first")
        q = make_query_vecs()
        resp = call_sp(sp_conn, q, top_k=5, n_probe=4)
        for key in [
            "results",
            "n_centroids",
            "n_candidates",
            "stage1_ms",
            "stage15_ms",
            "stage2_ms",
            "total_ms",
        ]:
            assert key in resp, f"Missing metadata key: {key!r} in {resp}"


@pytest.mark.integration
class TestCallSyntax:
    @pytest.mark.xfail(
        reason="RAG.ColBERTSearch_Search SQL function reads RAG.ColBERTCentroids "
        "(old SQL table). Superseded by PLAIDSearch.cls in IVG 1.26.1+.",
        strict=False,
    )
    def test_call_syntax_works_via_dbapi(self, sp_conn):
        q = make_query_vecs()
        q_json = json.dumps(q.tolist())
        cur = sp_conn.cursor()
        try:
            cur.execute("SELECT RAG.ColBERTSearch_Search(?, ?, ?)", [q_json, 5, 4])
            row = cur.fetchone()
            raw_json = row[0]
        finally:
            cur.close()
        assert row is not None
        parsed = json.loads(raw_json)
        assert "results" in parsed

    def test_search_via_sp_return_type(self, sp_conn):
        searcher = PLAIDSearcher(sp_conn, token_dim=128)
        q = make_query_vecs()
        results, meta = searcher.search_via_sp(sp_conn, q, top_k=5, n_probe=4)
        assert isinstance(results, list)
        assert isinstance(meta, dict)
        for item in results:
            assert isinstance(item, tuple) and len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    @pytest.mark.xfail(
        reason="Overlap depends on K/centroid config; PLAID not production-ready at this scale"
    )
    def test_search_via_sp_matches_search(self, sp_conn):
        cur = sp_conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM RAG.ColBERTCentroids")
            n_centroids = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM RAG.ColBERTDocCentroids")
            n_doc_centroids = cur.fetchone()[0]
        finally:
            cur.close()
        if n_centroids == 0 or n_doc_centroids == 0:
            pytest.skip("Centroids not built — run PLAIDBuilder.build() first")

        from iris_vector_rag.pipelines.colbert_iris.maxsim_indb import MaxSimInDB

        ms = MaxSimInDB(sp_conn, token_dim=128)
        searcher = PLAIDSearcher(sp_conn, token_dim=128)
        cur = sp_conn.cursor()
        try:
            cur.execute("SELECT DISTINCT doc_id FROM RAG.ColBERTDocuments")
            all_ids = [r[0] for r in cur.fetchall()]
        finally:
            cur.close()

        overlaps = []
        for seed in [42, 7, 13]:
            q = make_query_vecs(seed=seed)
            phase1_results = ms.bulk_fetch_maxsim(q, all_ids[:200], top_k=10)
            sp_results, _ = searcher.search_via_sp(sp_conn, q, top_k=10, n_probe=4)
            if not sp_results:
                overlaps.append(0.0)
                continue
            p1_set = {r[0] for r in phase1_results[:10]}
            sp_set = {r[0] for r in sp_results[:10]}
            overlap = len(p1_set & sp_set) / max(len(p1_set), 1)
            overlaps.append(overlap)

        assert (
            statistics.mean(overlaps) >= 0.3
        ), f"Mean overlap {statistics.mean(overlaps):.2f} < 0.3 — SP returning no results for these queries"

    @pytest.mark.xfail(
        reason="Stage 2 latency depends on K/candidate count; PLAID SP not production-ready"
    )
    def test_stage_timing_assertions(self, sp_conn):
        searcher = PLAIDSearcher(sp_conn, token_dim=128)
        q = make_query_vecs()
        timings = []
        for i in range(6):
            _, meta = searcher.search_via_sp(sp_conn, q, top_k=10, n_probe=4)
            if i > 0:
                timings.append(meta)

        s1 = statistics.median([t["stage1_ms"] for t in timings])
        s15 = statistics.median([t["stage15_ms"] for t in timings])
        s2 = statistics.median([t["stage2_ms"] for t in timings])
        assert s1 <= 15, f"Stage 1 median {s1:.1f}ms > 15ms"
        assert s15 <= 30, f"Stage 1.5 median {s15:.1f}ms > 30ms"
        assert s2 <= 600, f"Stage 2 median {s2:.1f}ms > 600ms (warm, ARM64)"

    def test_nprobe_tradeoff(self, sp_conn):
        searcher = PLAIDSearcher(sp_conn, token_dim=128)
        q = make_query_vecs(seed=77)

        r_n1, m1 = searcher.search_via_sp(sp_conn, q, top_k=10, n_probe=1)
        r_n4, m4 = searcher.search_via_sp(sp_conn, q, top_k=10, n_probe=4)
        r_n8, m8 = searcher.search_via_sp(sp_conn, q, top_k=10, n_probe=8)

        assert (
            m1["total_ms"] <= m8["total_ms"] * 1.5
        ), f"n_probe=1 ({m1['total_ms']:.0f}ms) not faster than n_probe=8 ({m8['total_ms']:.0f}ms)"
        assert (
            m1["n_candidates"] <= m8["n_candidates"]
        ), f"n_probe=1 candidates ({m1['n_candidates']}) ≤ n_probe=8 ({m8['n_candidates']})"

        n4_set = {r[0] for r in r_n4[:10]}
        n1_set = {r[0] for r in r_n1[:10]}
        n8_set = {r[0] for r in r_n8[:10]}
        recall_n8 = len(n8_set & n4_set) / max(len(n4_set), 1)
        recall_n1 = len(n1_set & n4_set) / max(len(n4_set), 1)
        assert (
            m8["n_candidates"] >= m1["n_candidates"]
        ), f"n_probe=8 candidates ({m8['n_candidates']}) < n_probe=1 ({m1['n_candidates']})"
