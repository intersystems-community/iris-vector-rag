"""E2E tests for Feature 081: global and mix retrieval modes.

Requires a .DAT fixture with ≥10 documents, a populated knowledge graph
(RAG.Entities + RAG.EntityRelationships), and relation embeddings.

These tests are marked xfail when the required fixture or KG data is absent.
"""

import pytest

pytestmark = pytest.mark.e2e

# ─── Shared fixture ───────────────────────────────────────────────────────────


def _has_kg_with_relation_embeddings(conn) -> bool:
    """True when EntityRelationships table has ≥1 row with non-NULL relation_embedding."""
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM RAG.EntityRelationships WHERE relation_embedding IS NOT NULL"
        )
        row = cur.fetchone()
        cur.close()
        return bool(row and int(row[0]) > 0)
    except Exception:
        return False


@pytest.fixture(scope="module")
def e2e_conn():
    try:
        import iris

        conn = iris.connect("localhost", 51972, "USER", "_SYSTEM", "SYS")
        yield conn
        conn.close()
    except Exception as exc:
        pytest.skip(f"IRIS unavailable for e2e tests: {exc}")


@pytest.fixture(scope="module")
def e2e_pipeline(e2e_conn):
    from iris_vector_rag.core.connection import ConnectionManager
    from iris_vector_rag.config.manager import ConfigurationManager
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline

    conn_mgr = ConnectionManager()
    cfg_mgr = ConfigurationManager()
    pipeline = BasicRAGPipeline(conn_mgr, cfg_mgr)
    return pipeline


# ─── TestGlobalMode ───────────────────────────────────────────────────────────


class TestGlobalMode:
    """E2E tests for retrieval='global' — theme-level relation embedding search."""

    def test_global_mode_runs_without_error(self, e2e_conn, e2e_pipeline):
        """pipeline.query with retrieval='global' completes without exception."""
        if not _has_kg_with_relation_embeddings(e2e_conn):
            pytest.xfail(
                "No relation embeddings in IRIS — run embed_and_store() to populate"
            )

        from iris_vector_rag.retrieval.engine import RetrievalEngine
        from iris_vector_rag.core.query_options import QueryOptions
        from unittest.mock import MagicMock

        engine = RetrievalEngine(
            vector_store=e2e_pipeline.vector_store,
            connection=e2e_conn,
        )
        engine.keyword_extractor = MagicMock()
        engine.keyword_extractor.extract.return_value = (["systemic risk", "financial stability"], [])

        opts = QueryOptions(query="What are systemic risks to financial stability?", retrieval="global", top_k=5)
        result = engine._retrieve_global(opts)

        assert isinstance(result, dict)
        assert "retrieved_documents" in result
        assert "metadata" in result
        assert result["metadata"].get("error") is None

    def test_global_result_has_required_metadata_keys(self, e2e_conn, e2e_pipeline):
        """Global result metadata contains high_level_keywords and degraded."""
        if not _has_kg_with_relation_embeddings(e2e_conn):
            pytest.xfail("No relation embeddings in IRIS")

        from iris_vector_rag.retrieval.engine import RetrievalEngine
        from iris_vector_rag.core.query_options import QueryOptions
        from unittest.mock import MagicMock

        engine = RetrievalEngine(
            vector_store=e2e_pipeline.vector_store,
            connection=e2e_conn,
        )
        engine.keyword_extractor = MagicMock()
        engine.keyword_extractor.extract.return_value = (["regulatory capital"], ["Basel III"])

        opts = QueryOptions(query="regulatory capital requirements", retrieval="global", top_k=5)
        result = engine._retrieve_global(opts)

        meta = result["metadata"]
        assert "high_level_keywords" in meta
        assert "degraded" in meta
        assert isinstance(meta["high_level_keywords"], list)
        assert isinstance(meta["degraded"], bool)

    @pytest.mark.xfail(reason="Requires labeled expected doc — validate manually with KG data")
    def test_global_recall_thematic_query(self, e2e_conn, e2e_pipeline):
        """Recall@K: thematic query surfaces a doc that vector-only misses.

        This test is xfail because it requires a specific labeled document
        in the KG that is retrievable by relation embedding but not by dense vector.
        """
        assert False, "Mark as xfail; validate manually with populated KG"


# ─── TestMixMode ─────────────────────────────────────────────────────────────


class TestMixMode:
    """E2E tests for retrieval='mix' — comprehensive RRF-fused retrieval."""

    def test_mix_mode_runs_without_error(self, e2e_conn, e2e_pipeline):
        """pipeline.query with retrieval='mix' completes without exception."""
        if not _has_kg_with_relation_embeddings(e2e_conn):
            pytest.xfail("No relation embeddings in IRIS")

        from iris_vector_rag.retrieval.engine import RetrievalEngine
        from iris_vector_rag.core.query_options import QueryOptions
        from unittest.mock import MagicMock

        engine = RetrievalEngine(
            vector_store=e2e_pipeline.vector_store,
            connection=e2e_conn,
        )
        engine.keyword_extractor = MagicMock()
        engine.keyword_extractor.extract.return_value = (["systemic risk"], ["Basel III"])

        opts = QueryOptions(query="systemic risk and capital requirements", retrieval="mix", top_k=5)
        result = engine._retrieve_mix(opts)

        assert isinstance(result, dict)
        assert result["metadata"]["fusion_method"] == "rrf"

    def test_mix_metadata_counts_are_ints(self, e2e_conn, e2e_pipeline):
        """Mix result metadata has integer count fields."""
        if not _has_kg_with_relation_embeddings(e2e_conn):
            pytest.xfail("No relation embeddings in IRIS")

        from iris_vector_rag.retrieval.engine import RetrievalEngine
        from iris_vector_rag.core.query_options import QueryOptions
        from unittest.mock import MagicMock

        engine = RetrievalEngine(
            vector_store=e2e_pipeline.vector_store,
            connection=e2e_conn,
        )
        engine.keyword_extractor = MagicMock()
        engine.keyword_extractor.extract.return_value = (["risk"], ["entity"])

        opts = QueryOptions(query="risk analysis", retrieval="mix", top_k=5)
        result = engine._retrieve_mix(opts)

        meta = result["metadata"]
        assert isinstance(meta["low_level_count"], int)
        assert isinstance(meta["high_level_count"], int)
        assert isinstance(meta["naive_count"], int)

    def test_mix_with_weights_uses_weighted_fusion(self, e2e_conn, e2e_pipeline):
        """Mix with explicit weights uses weighted_score fusion method."""
        if not _has_kg_with_relation_embeddings(e2e_conn):
            pytest.xfail("No relation embeddings in IRIS")

        from iris_vector_rag.retrieval.engine import RetrievalEngine
        from iris_vector_rag.core.query_options import QueryOptions
        from unittest.mock import MagicMock

        engine = RetrievalEngine(
            vector_store=e2e_pipeline.vector_store,
            connection=e2e_conn,
        )
        engine.keyword_extractor = MagicMock()
        engine.keyword_extractor.extract.return_value = (["risk"], ["entity"])

        opts = QueryOptions(
            query="risk analysis",
            retrieval="mix",
            top_k=5,
            weights={"relation": 0.7, "vector": 0.3},
        )
        result = engine._retrieve_mix(opts)
        assert result["metadata"]["fusion_method"] == "weighted_score"

    def test_default_query_uses_vector_not_mix(self, e2e_conn, e2e_pipeline):
        """pipeline.query without retrieval= uses default, not mix (backward compat)."""
        from iris_vector_rag.core.query_options import normalize_query_params

        opts = normalize_query_params(query="test query")
        assert opts.retrieval is None or opts.retrieval == "vector"
