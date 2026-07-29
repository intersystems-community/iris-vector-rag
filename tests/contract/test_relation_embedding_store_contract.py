"""Contract tests for RelationEmbeddingStore — Feature 081."""

from unittest.mock import MagicMock, patch, call


def _make_store():
    from iris_vector_rag.storage.relation_embedding_store import RelationEmbeddingStore

    conn_mgr = MagicMock()
    cfg_mgr = MagicMock()
    return RelationEmbeddingStore(conn_mgr, cfg_mgr)


def test_embed_and_store_uses_update_sql_with_to_vector():
    """embed_and_store() must execute UPDATE ... TO_VECTOR(?, FLOAT, 384) SQL."""
    store = _make_store()

    fake_vec = [0.1] * 384
    mock_emb_mgr = MagicMock()
    mock_emb_mgr.embed_text.return_value = fake_vec

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    store._conn_mgr.get_connection.return_value = mock_conn

    with patch.object(store, "_get_embedding_manager", return_value=mock_emb_mgr):
        store.embed_and_store(
            relationship_id="rel_001",
            relationship_type="CAUSES",
            source_entity="Basel III",
            target_entity="Capital Requirements",
            description="Basel III caused stricter capital requirements.",
        )

    mock_cursor.execute.assert_called_once()
    sql, params = mock_cursor.execute.call_args[0]
    assert "UPDATE" in sql.upper()
    assert "TO_VECTOR" in sql.upper()
    assert "relation_embedding" in sql
    # params: [embedding_str, relationship_id]
    assert params[1] == "rel_001"


def test_search_calls_vector_similarity_search_with_cosine():
    """search() must use metric='COSINE' via vector_similarity_search."""
    store = _make_store()
    query_vec = [0.2] * 384

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    store._conn_mgr.get_connection.return_value = mock_conn

    with patch("iris_vector_rag.storage.relation_embedding_store.RelationEmbeddingStore.search") as _:
        pass  # just verifying the patch path exists

    with patch("iris_vector_graph.dbapi_utils.vector_similarity_search", return_value=[]) as mock_vss:
        results = store.search(query_vec, top_k=5)

    mock_vss.assert_called_once()
    call_kwargs = mock_vss.call_args.kwargs
    assert call_kwargs["metric"] == "COSINE"
    assert call_kwargs["dtype"] == "FLOAT"
    assert call_kwargs["table_name"] == "RAG.EntityRelationships"
    assert call_kwargs["top_k"] == 5
    assert results == []


def test_count_embedded_queries_non_null_rows():
    """count_embedded() must run a SELECT COUNT(*) WHERE relation_embedding IS NOT NULL."""
    store = _make_store()

    cursor = MagicMock()
    cursor.fetchone.return_value = (42,)
    with patch.object(store, "_get_cursor", return_value=cursor):
        result = store.count_embedded()

    assert result == 42
    sql_called = cursor.execute.call_args[0][0]
    assert "COUNT(*)" in sql_called.upper()
    assert "IS NOT NULL" in sql_called.upper()


def test_embed_and_store_no_raise_on_second_call():
    """embed_and_store() called twice for same id must not raise (UPDATE is idempotent)."""
    store = _make_store()

    fake_vec = [0.0] * 384
    mock_emb_mgr = MagicMock()
    mock_emb_mgr.embed_text.return_value = fake_vec

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    store._conn_mgr.get_connection.return_value = mock_conn

    with patch.object(store, "_get_embedding_manager", return_value=mock_emb_mgr):
        # First update
        store.embed_and_store("rel_dup", "REL", "A", "B")
        # Second update (same id) — UPDATE is idempotent, must not raise
        store.embed_and_store("rel_dup", "REL", "A", "B")
