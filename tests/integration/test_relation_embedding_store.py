"""Integration tests for RelationEmbeddingStore against live IRIS — Feature 081.

Uses programmatic fixtures (3 relationships, <10 entities) — too small for .DAT.
Teardown removes the added column and index to restore pre-test schema state
(constitution P4: test isolation).
"""

import pytest

pytestmark = pytest.mark.integration


# Override the autouse iris_connection fixture from integration/conftest.py.
# That conftest probes only SQLAlchemy-compatible ports; this project uses
# iris.dbapi on port 51972.  Provide a no-op override so cleanup_test_data
# does not skip our tests.
@pytest.fixture(autouse=True)
def iris_connection():  # noqa: F811
    """Override integration-conftest iris_connection to prevent skip cascade."""
    yield None


@pytest.fixture(scope="module")
def conn_mgr():
    from iris_vector_rag.core.connection import ConnectionManager

    return ConnectionManager()


@pytest.fixture(scope="module")
def cfg_mgr():
    from iris_vector_rag.config.manager import ConfigurationManager

    return ConfigurationManager()


@pytest.fixture(scope="module")
def store(conn_mgr, cfg_mgr):
    from iris_vector_rag.storage.relation_embedding_store import RelationEmbeddingStore

    s = RelationEmbeddingStore(conn_mgr, cfg_mgr)
    s._ensure_schema()
    yield s
    # Teardown: restore pre-test schema (P4)
    conn = conn_mgr.get_connection("iris")
    cur = conn.cursor()
    for sql in [
        "DROP INDEX idx_hnsw_rel_embedding ON RAG.EntityRelationships",
        "ALTER TABLE RAG.EntityRelationships DROP COLUMN relation_embedding",
    ]:
        try:
            cur.execute(sql)
            conn.commit()
        except Exception:
            conn.rollback()
    cur.close()


@pytest.fixture(scope="module", autouse=True)
def base_entities(conn_mgr):
    """Insert minimal entity rows for FK constraints; clean up after."""
    conn = conn_mgr.get_connection("iris")
    cur = conn.cursor()
    rows = [
        ("e_int_a", "Basel III", "REGULATION"),
        ("e_int_b", "Capital Requirements", "CONCEPT"),
        ("e_int_c", "Systemic Risk", "CONCEPT"),
    ]
    for eid, name, etype in rows:
        try:
            cur.execute(
                "INSERT INTO RAG.Entities (entity_id, entity_name, entity_type) VALUES (?, ?, ?)",
                [eid, name, etype],
            )
            conn.commit()
        except Exception:
            conn.rollback()
    yield
    cur2 = conn.cursor()
    for eid, _, _ in rows:
        try:
            cur2.execute("DELETE FROM RAG.Entities WHERE entity_id = ?", [eid])
            conn.commit()
        except Exception:
            conn.rollback()
    cur2.close()
    cur.close()


@pytest.fixture(scope="module", autouse=True)
def base_relationships(conn_mgr, base_entities):
    """Insert 3 relationship rows (no embedding yet) for store tests."""
    conn = conn_mgr.get_connection("iris")
    cur = conn.cursor()
    rels = [
        ("rel_i001", "e_int_a", "e_int_b", "CAUSES"),
        ("rel_i002", "e_int_b", "e_int_c", "RELATED_TO"),
        ("rel_i003", "e_int_a", "e_int_c", "INFLUENCES"),
    ]
    for rid, src, tgt, rtype in rels:
        try:
            cur.execute(
                "INSERT INTO RAG.EntityRelationships "
                "(relationship_id, source_entity_id, target_entity_id, relationship_type) "
                "VALUES (?, ?, ?, ?)",
                [rid, src, tgt, rtype],
            )
            conn.commit()
        except Exception:
            conn.rollback()
    yield
    cur2 = conn.cursor()
    for rid, _, _, _ in rels:
        try:
            cur2.execute(
                "DELETE FROM RAG.EntityRelationships WHERE relationship_id = ?", [rid]
            )
            conn.commit()
        except Exception:
            conn.rollback()
    cur2.close()
    cur.close()


def test_ensure_schema_idempotent(store):
    """_ensure_schema() twice must not raise."""
    store._ensure_schema()  # already called in fixture; call again
    conn = store._conn_mgr.get_connection("iris")
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT COUNT(*) FROM RAG.EntityRelationships WHERE relation_embedding IS NOT NULL"
        )
        assert cur.fetchone() is not None
    finally:
        cur.close()


def test_embed_and_store_via_to_vector(store):
    """embed_and_store() must write a real embedding using TO_VECTOR path."""
    store.embed_and_store(
        relationship_id="rel_i001",
        relationship_type="CAUSES",
        source_entity="Basel III",
        target_entity="Capital Requirements",
        description="Basel III regulation caused stricter capital requirements.",
    )
    assert store.count_embedded() >= 1


def test_search_returns_score_float(store):
    """search() must return ≤top_k results, each with a float 'score'."""
    query_vec = [0.01 * (i % 100) for i in range(384)]
    results = store.search(query_vec, top_k=2)
    assert isinstance(results, list)
    assert len(results) <= 2
    for row in results:
        assert "score" in row, f"missing 'score' in {row}"
        assert isinstance(row["score"], float)


def test_incremental_add_does_not_touch_existing(store):
    """Adding a second embedding must not alter the count of existing rows."""
    count_before = store.count_embedded()
    store.embed_and_store(
        "rel_i002", "RELATED_TO", "Capital Requirements", "Systemic Risk"
    )
    assert store.count_embedded() == count_before + 1


def test_count_embedded_correct_after_three_inserts(store):
    """count_embedded() must equal 3 after all three relationships are embedded."""
    store.embed_and_store("rel_i003", "INFLUENCES", "Basel III", "Systemic Risk")
    assert store.count_embedded() == 3
