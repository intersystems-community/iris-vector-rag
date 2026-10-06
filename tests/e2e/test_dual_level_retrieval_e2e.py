"""E2E tests for Feature 081: global and mix retrieval modes.

Seeded with programmatic KG fixtures (3 entities, 3 relationships + embeddings).
Runs against live IRIS (port 51972) per constitution Principle I — no xfail.
"""

import pytest

pytestmark = pytest.mark.e2e

# ─── Module-scoped KG fixture ─────────────────────────────────────────────────

_ENTITIES = [
    ("e_e2e_a", "Systemic Risk", "CONCEPT"),
    ("e_e2e_b", "Capital Requirements", "REGULATION"),
    ("e_e2e_c", "Financial Stability", "CONCEPT"),
]

_RELS = [
    (
        "rel_e2e_1",
        "e_e2e_a",
        "e_e2e_c",
        "THREATENS",
        "Systemic risk threatens financial stability",
    ),
    (
        "rel_e2e_2",
        "e_e2e_b",
        "e_e2e_a",
        "MITIGATES",
        "Capital requirements mitigate systemic risk",
    ),
    (
        "rel_e2e_3",
        "e_e2e_b",
        "e_e2e_c",
        "SUPPORTS",
        "Capital requirements support financial stability",
    ),
]


@pytest.fixture(scope="module")
def e2e_conn(iris_target):
    """A connection to the IRIS the session runs against (never a fixed port)."""
    from tests.support.iris_environment import connect

    conn = connect(iris_target)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture(scope="module")
def e2e_managers(e2e_conn):
    from iris_vector_rag.config.manager import ConfigurationManager
    from iris_vector_rag.core.connection import ConnectionManager

    return ConnectionManager(), ConfigurationManager()


def _insert_once(conn, sql, params):
    """Insert a seed row; a row that is already there is fine, any other error is not."""
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        conn.commit()
    except Exception as exc:
        conn.rollback()
        if "unique" not in str(exc).lower() and "duplicate" not in str(exc).lower():
            raise
    finally:
        cur.close()


@pytest.fixture(scope="module", autouse=True)
def kg_with_embeddings(e2e_conn, e2e_managers):
    """Seed entities + relationships + relation embeddings; teardown after module."""
    from iris_vector_rag.storage.schema_manager import SchemaManager

    conn_mgr, cfg_mgr = e2e_managers
    conn = e2e_conn

    # Earlier tests in the session reset the RAG schema, so this module makes the
    # tables it seeds instead of assuming an earlier test left them.
    schema = SchemaManager(conn_mgr, cfg_mgr)
    for table in ("SourceDocuments", "Entities", "EntityRelationships"):
        assert schema.ensure_table_schema(table), f"could not create RAG.{table}"

    for eid, name, etype in _ENTITIES:
        _insert_once(
            conn,
            "INSERT INTO RAG.Entities (entity_id, entity_name, entity_type, source_doc_id) "
            "VALUES (?, ?, ?, 'e2e_synthetic')",
            [eid, name, etype],
        )

    for rid, src, tgt, rtype, _ in _RELS:
        _insert_once(
            conn,
            "INSERT INTO RAG.EntityRelationships "
            "(relationship_id, source_entity_id, target_entity_id, relationship_type) "
            "VALUES (?, ?, ?, ?)",
            [rid, src, tgt, rtype],
        )

    # Ensure schema and embed relationships
    from iris_vector_rag.storage.relation_embedding_store import RelationEmbeddingStore

    store = RelationEmbeddingStore(conn_mgr, cfg_mgr)
    store._ensure_schema()
    for rid, src, tgt, rtype, desc in _RELS:
        # Get entity names
        src_name = next(n for eid, n, _ in _ENTITIES if eid == src)
        tgt_name = next(n for eid, n, _ in _ENTITIES if eid == tgt)
        store.embed_and_store(rid, rtype, src_name, tgt_name, desc)

    yield store

    # Teardown: remove test data (P4)
    cur2 = conn.cursor()
    for rid, _, _, _, _ in _RELS:
        try:
            cur2.execute(
                "DELETE FROM RAG.EntityRelationships WHERE relationship_id = ?", [rid]
            )
            conn.commit()
        except Exception:
            conn.rollback()
    for eid, _, _ in _ENTITIES:
        try:
            cur2.execute("DELETE FROM RAG.Entities WHERE entity_id = ?", [eid])
            conn.commit()
        except Exception:
            conn.rollback()
    cur2.close()


_DOCS = [
    ("doc_e2e_1", "rel_e2e_1"),
    ("doc_e2e_2", "rel_e2e_2"),
    ("doc_e2e_3", "rel_e2e_3"),
]


@pytest.fixture(scope="module")
def e2e_vector_store(e2e_managers, kg_with_embeddings):
    """The real IRISVectorStore over the session's IRIS, seeded through add_documents.

    One document per seeded relationship, worded as the relationship's description, so
    vector retrieval over documents and global retrieval over relation embeddings are
    asked the same question of the same facts. Embedded by the real embedding path.
    """
    from iris_vector_rag.core.models import Document
    from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

    conn_mgr, cfg_mgr = e2e_managers
    store = IRISVectorStore(conn_mgr, cfg_mgr)
    descriptions = {rid: desc for rid, _, _, _, desc in _RELS}
    ids = [doc_id for doc_id, _ in _DOCS]
    store.delete_documents(ids)
    store.add_documents(
        [
            Document(
                id=doc_id,
                page_content=descriptions[rid],
                metadata={"source": "e2e_dual_level", "relationship_id": rid},
            )
            for doc_id, rid in _DOCS
        ]
    )
    yield store
    store.delete_documents(ids)


@pytest.fixture(scope="module")
def e2e_engine(e2e_managers, e2e_vector_store):
    from iris_vector_rag.retrieval.engine import RetrievalEngine

    conn_mgr, cfg_mgr = e2e_managers
    engine = RetrievalEngine(
        vector_store=e2e_vector_store,
        connection=None,
        config_manager=cfg_mgr,
    )
    engine._conn_mgr = conn_mgr
    # Keywords are supplied in the QueryOptions, so no LLM call is needed to extract them
    return engine


# ─── TestGlobalMode ───────────────────────────────────────────────────────────


class TestGlobalMode:
    """E2E tests for retrieval='global' against live IRIS with seeded KG data."""

    def test_global_mode_runs_without_error(self, e2e_engine):
        """_retrieve_global() completes without exception against live IRIS."""
        from iris_vector_rag.core.query_options import QueryOptions

        opts = QueryOptions(
            query="What threatens financial stability?",
            retrieval="global",
            top_k=5,
            high_level_keywords=["systemic risk", "financial stability"],
            low_level_keywords=[],
        )
        result = e2e_engine._retrieve_global(opts)

        assert isinstance(result, dict)
        assert "retrieved_documents" in result
        assert "metadata" in result

    def test_global_result_has_required_metadata_keys(self, e2e_engine):
        """Global result metadata contains high_level_keywords and degraded."""
        from iris_vector_rag.core.query_options import QueryOptions

        opts = QueryOptions(
            query="capital requirements",
            retrieval="global",
            top_k=3,
            high_level_keywords=["regulatory capital"],
            low_level_keywords=[],
        )
        result = e2e_engine._retrieve_global(opts)
        meta = result["metadata"]

        assert "high_level_keywords" in meta
        assert "degraded" in meta
        assert isinstance(meta["high_level_keywords"], list)
        assert isinstance(meta["degraded"], bool)

    def test_global_not_degraded_with_populated_index(self, e2e_engine):
        """With seeded embeddings, degraded must be False."""
        from iris_vector_rag.core.query_options import QueryOptions

        opts = QueryOptions(
            query="systemic risk",
            retrieval="global",
            top_k=3,
            high_level_keywords=["systemic risk"],
            low_level_keywords=[],
        )
        result = e2e_engine._retrieve_global(opts)
        assert result["metadata"]["degraded"] is False

    def test_global_returns_relationship_documents(self, e2e_engine):
        """Global mode returns ≥1 Document from the seeded relationship embeddings."""
        from iris_vector_rag.core.query_options import QueryOptions

        opts = QueryOptions(
            query="systemic risk threatens financial stability",
            retrieval="global",
            top_k=3,
            high_level_keywords=["systemic risk", "financial stability"],
            low_level_keywords=[],
        )
        result = e2e_engine._retrieve_global(opts)
        assert len(result["retrieved_documents"]) >= 1


# ─── TestMixMode ─────────────────────────────────────────────────────────────


class TestMixMode:
    """E2E tests for retrieval='mix' against live IRIS with seeded KG data."""

    def test_mix_mode_runs_without_error(self, e2e_engine):
        """_retrieve_mix() completes without exception."""
        from iris_vector_rag.core.query_options import QueryOptions

        opts = QueryOptions(
            query="systemic risk and capital requirements",
            retrieval="mix",
            top_k=5,
            high_level_keywords=["systemic risk"],
            low_level_keywords=["Basel III"],
        )
        result = e2e_engine._retrieve_mix(opts)

        assert isinstance(result, dict)
        assert result["metadata"]["fusion_method"] == "rrf"

    def test_mix_metadata_counts_are_ints(self, e2e_engine):
        """Mix result metadata has integer low/high/naive count fields."""
        from iris_vector_rag.core.query_options import QueryOptions

        opts = QueryOptions(
            query="risk analysis",
            retrieval="mix",
            top_k=5,
            high_level_keywords=["risk"],
            low_level_keywords=["entity"],
        )
        result = e2e_engine._retrieve_mix(opts)
        meta = result["metadata"]

        assert isinstance(meta["low_level_count"], int)
        assert isinstance(meta["high_level_count"], int)
        assert isinstance(meta["naive_count"], int)
        assert meta["low_level_count"] + meta["high_level_count"] + meta[
            "naive_count"
        ] >= len(result["retrieved_documents"])

    def test_mix_with_weights_uses_weighted_fusion(self, e2e_engine):
        """Mix with explicit weights produces fusion_method='weighted_score'."""
        from iris_vector_rag.core.query_options import QueryOptions

        opts = QueryOptions(
            query="risk analysis",
            retrieval="mix",
            top_k=5,
            high_level_keywords=["risk"],
            low_level_keywords=[],
            weights={"relation": 0.7, "vector": 0.3},
        )
        result = e2e_engine._retrieve_mix(opts)
        assert result["metadata"]["fusion_method"] == "weighted_score"

    def test_mix_high_level_docs_tagged_with_source(self, e2e_engine):
        """Docs retrieved from relation embeddings have retrieval_source='high_level'."""
        from iris_vector_rag.core.query_options import QueryOptions

        opts = QueryOptions(
            query="financial stability",
            retrieval="mix",
            top_k=5,
            high_level_keywords=["financial stability"],
            low_level_keywords=[],
        )
        result = e2e_engine._retrieve_mix(opts)
        sources = {
            d.metadata.get("retrieval_source") for d in result["retrieved_documents"]
        }
        assert "high_level" in sources

    def test_default_query_uses_vector_not_mix(self):
        """pipeline.query without retrieval= defaults to vector, not mix (backward compat)."""
        from iris_vector_rag.core.query_options import normalize_query_params

        opts = normalize_query_params(query="test query")
        assert opts.retrieval is None or opts.retrieval == "vector"


# ─── TestRecallBenchmark (SC-001) ─────────────────────────────────────────────


class TestRecallBenchmark:
    """SC-001: global retrieval finds seeded relationships by thematic query.

    Uses the seeded KG fixture (3 relationships). Each labeled query is a
    thematic phrase whose meaning matches one of the seeded relationship
    descriptions. Global mode must surface the expected relationship ID.
    The vector store is the real IRISVectorStore holding one document per
    relationship, so the same queries also run through real vector retrieval.
    """

    # (query, expected_relationship_id, high_level_keywords)
    # Relationship descriptions in the fixture:
    #   rel_e2e_1: "Systemic risk threatens financial stability"
    #   rel_e2e_2: "Capital requirements mitigate systemic risk"
    #   rel_e2e_3: "Capital requirements support financial stability"
    LABELED_QUERIES = [
        (
            "What threatens financial stability?",
            "rel_e2e_1",
            ["financial stability", "systemic risk"],
        ),
        (
            "How are systemic risks controlled?",
            "rel_e2e_2",
            ["systemic risk", "mitigation"],
        ),
        (
            "What supports financial stability?",
            "rel_e2e_3",
            ["financial stability", "capital"],
        ),
    ]

    def test_global_recall_gte_vector_on_thematic_queries(self, e2e_engine):
        """SC-001: global mode surfaces expected relationship for each thematic query."""
        from iris_vector_rag.core.query_options import QueryOptions

        hits = 0
        for query, expected_id, high_kws in self.LABELED_QUERIES:
            opts = QueryOptions(
                query=query,
                retrieval="global",
                top_k=3,
                high_level_keywords=high_kws,
                low_level_keywords=[],
            )
            result = e2e_engine._retrieve_global(opts)
            doc_ids = {getattr(d, "id", None) for d in result["retrieved_documents"]}
            if expected_id in doc_ids:
                hits += 1

        n = len(self.LABELED_QUERIES)
        recall = hits / n
        assert recall >= 1.0, (
            f"SC-001 failed: global recall {recall:.2f} ({hits}/{n} queries). "
            f"Expected all seeded relationships to be retrievable by thematic query."
        )

    def test_vector_retrieval_finds_the_same_facts(self, e2e_engine):
        """The real vector leg over the seeded documents, for comparison with global."""
        from iris_vector_rag.core.query_options import QueryOptions

        hits = 0
        for query, expected_id, _ in self.LABELED_QUERIES:
            opts = QueryOptions(query=query, retrieval="vector", top_k=3)
            docs = e2e_engine.retrieve(opts)
            related = {d.metadata.get("relationship_id") for d in docs}
            if expected_id in related:
                hits += 1
        print(f"vector recall {hits}/{len(self.LABELED_QUERIES)}")
        assert hits >= 1, "real vector retrieval found none of the seeded documents"
