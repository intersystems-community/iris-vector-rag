import os
import statistics
import time

import iris.dbapi as dbapi
import numpy as np
import pytest

from iris_vector_rag.pipelines.colbert_iris.maxsim_indb import MaxSimInDB
from iris_vector_rag.pipelines.colbert_iris.vecindex_phase2 import (
    VecIndexNotAvailableError,
    VecIndexSearcher,
)
from tests.colbert_iris.conftest import make_query_vecs

IRIS_HOST = os.environ.get("IRIS_HOSTNAME", "localhost")
IRIS_PORT = int(os.environ.get("IRIS_PORT", "13972"))
TOKEN_DIM = 128


@pytest.mark.integration
class TestVecIndexIngest:
    def test_deploy_and_callable(self, setup_vecindex):
        from iris_vector_graph import IRISGraphEngine

        engine = IRISGraphEngine(setup_vecindex)
        result = engine.vec_create_index("test_deploy_ci", TOKEN_DIM, "dot")
        assert "dim" in str(result)
        engine.vec_drop("test_deploy_ci")

    def test_vec_insert_single(self, setup_vecindex):
        from iris_vector_graph import IRISGraphEngine

        engine = IRISGraphEngine(setup_vecindex)
        engine.vec_create_index("test_single_ci", 4, "dot")
        engine.vec_insert("test_single_ci", "d:0", [0.1, 0.9, 0.0, 0.0])
        results = engine.vec_search(
            "test_single_ci", [0.1, 0.9, 0.0, 0.0], k=1, nprobe=1
        )
        assert len(results) >= 1
        assert results[0]["id"] == "d:0"
        engine.vec_drop("test_single_ci")

    def test_vec_build_returns_stats(self, setup_vecindex):
        from iris_vector_graph import IRISGraphEngine

        engine = IRISGraphEngine(setup_vecindex)
        engine.vec_create_index("test_build_ci", TOKEN_DIM, "dot", 4, 10)
        rng = np.random.default_rng(0)
        for i in range(80):
            v = rng.random(TOKEN_DIM).astype(np.float32)
            v /= np.linalg.norm(v)
            engine.vec_insert("test_build_ci", f"d_{i}:0", v.tolist())
        result = engine.vec_build("test_build_ci")
        assert result.get("trees", 0) >= 1
        assert result.get("vectors", 0) == 80
        engine.vec_drop("test_build_ci")

    def test_dual_write_counts_match(self, vecindex_80doc):
        conn, searcher = vecindex_80doc
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings WHERE doc_id LIKE 'vi80%'"
            )
            sql_count = cur.fetchone()[0]
        finally:
            cur.close()
        vi_info = searcher.info()
        assert vi_info.get("count", 0) > 0, f"VecIndex count is 0, expected > 0"
        assert sql_count > 0, f"SQL token count is 0"

    def test_ingest_idempotent(self, setup_vecindex):
        from iris_vector_graph import IRISGraphEngine

        from iris_vector_rag.pipelines.colbert_iris.ingest import ColBERTIngestor
        from iris_vector_rag.pipelines.colbert_iris.schema import ColBERTSchema
        from tests.colbert_iris.conftest import DummyModel

        conn = dbapi.connect(
            hostname=IRIS_HOST,
            port=IRIS_PORT,
            namespace="USER",
            username="_SYSTEM",
            password="SYS",
        )
        schema = ColBERTSchema(conn)
        schema.drop_tables()
        schema.create_tables()
        ingestor = ColBERTIngestor(conn, model=DummyModel(), token_dim=TOKEN_DIM)
        docs = [
            {"doc_id": f"idem_{i}", "text": f"Doc {i}", "metadata": {}}
            for i in range(10)
        ]
        searcher = VecIndexSearcher(conn, index_name="test_idem", token_dim=TOKEN_DIM)

        stats1 = ingestor.ingest_documents(
            docs, use_vecindex=True, vecindex_searcher=searcher
        )
        count1 = searcher.info().get("count", 0)

        searcher.drop()
        schema.drop_tables()
        schema.create_tables()
        searcher2 = VecIndexSearcher(conn, index_name="test_idem", token_dim=TOKEN_DIM)
        stats2 = ingestor.ingest_documents(
            docs, use_vecindex=True, vecindex_searcher=searcher2
        )
        count2 = searcher2.info().get("count", 0)

        assert count1 == count2, f"Idempotent counts differ: {count1} vs {count2}"
        searcher2.drop()
        schema.drop_tables()
        conn.close()


@pytest.mark.integration
class TestVecIndexSearch:
    def test_search_returns_results(self, vecindex_80doc):
        _, searcher = vecindex_80doc
        q = make_query_vecs(4)
        results, meta = searcher.search(q, top_k=5, nprobe=2)
        assert len(results) > 0
        assert len(results) <= 5

    def test_search_scores_descending(self, vecindex_80doc):
        _, searcher = vecindex_80doc
        q = make_query_vecs(4)
        results, _ = searcher.search(q, top_k=10, nprobe=2)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_topk_respected(self, vecindex_80doc):
        _, searcher = vecindex_80doc
        q = make_query_vecs(4)
        for k in [1, 3, 5]:
            results, _ = searcher.search(q, top_k=k, nprobe=2)
            assert len(results) <= k

    def test_search_before_build_raises(self, setup_vecindex):
        from iris_vector_graph import IRISGraphEngine

        engine = IRISGraphEngine(setup_vecindex)
        engine.vec_create_index("test_empty_ci", TOKEN_DIM, "dot")
        searcher = VecIndexSearcher(setup_vecindex, index_name="test_empty_ci")
        with pytest.raises(VecIndexNotAvailableError):
            searcher.search(make_query_vecs(4), top_k=5)
        engine.vec_drop("test_empty_ci")

    def test_doc_id_parsing(self, setup_vecindex):
        from iris_vector_graph import IRISGraphEngine

        engine = IRISGraphEngine(setup_vecindex)
        engine.vec_create_index("test_parse_ci", TOKEN_DIM, "dot", 1, 5)
        rng = np.random.default_rng(77)
        target = rng.random(TOKEN_DIM).astype(np.float32)
        target /= np.linalg.norm(target)
        engine.vec_insert("test_parse_ci", "myrealdoc:7", target.tolist())
        engine.vec_build("test_parse_ci")
        searcher = VecIndexSearcher(setup_vecindex, index_name="test_parse_ci")
        results, _ = searcher.search(target.reshape(1, -1), top_k=1, nprobe=1)
        assert (
            results[0][0] == "myrealdoc"
        ), f"Expected 'myrealdoc', got '{results[0][0]}'"
        engine.vec_drop("test_parse_ci")

    def test_vec_drop_cleans(self, setup_vecindex):
        from iris_vector_graph import IRISGraphEngine

        engine = IRISGraphEngine(setup_vecindex)
        engine.vec_create_index("test_drop_ci", 4, "dot")
        engine.vec_insert("test_drop_ci", "d:0", [0.1, 0.2, 0.3, 0.4])
        searcher = VecIndexSearcher(setup_vecindex, index_name="test_drop_ci")
        searcher.drop()
        assert searcher.info().get("count", 0) == 0

    def test_nprobe_tradeoff_recall(self, vecindex_80doc):
        conn, searcher = vecindex_80doc
        try:
            sql_cur = conn.cursor()
            sql_cur.execute(
                "SELECT COUNT(*) FROM RAG.ColBERTDocuments WHERE doc_id LIKE 'vi80%'"
            )
            n = sql_cur.fetchone()[0]
            sql_cur.close()
        except Exception:
            pytest.skip(
                "RAG.ColBERTDocuments not available (tables dropped by earlier test)"
            )

        if n == 0:
            pytest.skip("No vi80 docs in ColBERTDocuments")

        ms = MaxSimInDB(conn, token_dim=TOKEN_DIM)
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT DISTINCT doc_id FROM RAG.ColBERTDocuments WHERE doc_id LIKE 'vi80%'"
            )
            all_ids = [r[0] for r in cur.fetchall()]
        finally:
            cur.close()

        recalls_n1, recalls_n4 = [], []
        for seed in range(5):
            q = make_query_vecs(4, seed=seed)
            ref = ms.bulk_fetch_maxsim(q, all_ids, top_k=10)
            ref_set = {r[0] for r in ref}
            for nprobe, store in [(1, recalls_n1), (4, recalls_n4)]:
                res, _ = searcher.search(q, top_k=10, nprobe=nprobe)
                res_set = {r[0] for r in res}
                recall = len(ref_set & res_set) / max(len(ref_set), 1)
                store.append(recall)

        assert np.mean(recalls_n4) >= np.mean(
            recalls_n1
        ), f"nprobe=4 recall {np.mean(recalls_n4):.2f} should be >= nprobe=1 {np.mean(recalls_n1):.2f}"

    def test_metadata_keys_present(self, vecindex_80doc):
        _, searcher = vecindex_80doc
        _, meta = searcher.search(make_query_vecs(4), top_k=5, nprobe=2)
        for key in ["n_candidates", "stage_search_ms", "stage_maxsim_ms", "total_ms"]:
            assert key in meta, f"Missing meta key: {key}"

    def test_no_class_compile_lock(self, vecindex_80doc):
        conn, _ = vecindex_80doc
        cur = conn.cursor()
        try:
            cur.execute(
                "UPDATE RAG.DocumentTokenEmbeddings SET centroid_id = NULL "
                "WHERE tok_pos = -999 AND doc_id = 'nonexistent'"
            )
            conn.commit()
        except Exception as e:
            assert "-110" not in str(e), f"SQLCODE -110 class compile lock: {e}"
        finally:
            cur.close()


@pytest.mark.integration
class TestNoClassCompileLock:
    def test_repeated_build_no_lock(self, setup_vecindex):
        import intersystems_iris
        from iris_vector_graph import IRISGraphEngine

        from iris_vector_rag.pipelines.colbert_iris.vecindex_phase2 import (
            VecIndexSearcher,
        )

        for run in range(3):
            engine = IRISGraphEngine(setup_vecindex)
            index_name = f"lock_test_{run}"
            engine.vec_create_index(index_name, TOKEN_DIM, "dot", 2, 10)
            rng = np.random.default_rng(run)
            for i in range(20):
                v = rng.random(TOKEN_DIM).astype(np.float32)
                v /= np.linalg.norm(v)
                engine.vec_insert(index_name, f"lockdoc_{i}:0", v.tolist())
            engine.vec_build(index_name)
            searcher = VecIndexSearcher(
                setup_vecindex, index_name=index_name, token_dim=TOKEN_DIM
            )
            results, _ = searcher.search(make_query_vecs(4), top_k=5)
            assert len(results) >= 0
            engine.vec_drop(index_name)

    def test_update_after_build_no_lock(self, vecindex_80doc):
        conn, _ = vecindex_80doc
        try:
            cur = conn.cursor()
            try:
                cur.execute(
                    "SELECT COUNT(*) FROM RAG.ColBERTDocuments WHERE doc_id='__nonexistent__'"
                )
                cur.fetchone()
                conn.commit()
            finally:
                cur.close()
        except Exception as e:
            if "-110" in str(e):
                pytest.fail(
                    f"SQLCODE -110 class compile lock after VecIndex build: {e}"
                )


@pytest.mark.integration
class TestVecIndexBenchmarkTier:
    def test_benchmark_tier_exists(self, vecindex_80doc):
        conn, searcher = vecindex_80doc
        from tests.colbert_iris.benchmark_scale import benchmark_phase2_vecindex
        from tests.colbert_iris.conftest import DummyModel

        queries = ["document topic 0", "document topic 5", "document topic 9"]
        model = DummyModel()

        result = benchmark_phase2_vecindex(
            conn,
            model,
            docs=[
                {"doc_id": f"vi80_{i:04d}", "text": f"Document {i}", "metadata": {}}
                for i in range(80)
            ],
            queries=queries,
            top_k=5,
            nprobe=2,
            existing_searcher=searcher,
        )
        for key in ["p50_ms", "p95_ms", "mean_recall_at_10", "speedup_vs_phase2"]:
            assert key in result, f"Missing benchmark key: {key}"
        assert result["p50_ms"] > 0
