"""Contract tests for RelationEmbeddingStore schema migration — Feature 081."""

from unittest.mock import MagicMock, patch, call


def _make_store():
    from iris_vector_rag.storage.relation_embedding_store import RelationEmbeddingStore

    conn_mgr = MagicMock()
    cfg_mgr = MagicMock()
    return RelationEmbeddingStore(conn_mgr, cfg_mgr)


def test_ensure_schema_idempotent():
    """Calling _ensure_schema() twice must not raise."""
    store = _make_store()
    store._ensure_schema()
    store._ensure_schema()  # second call — must be a no-op / idempotent


def test_count_embedded_returns_int():
    """count_embedded() returns an int (mocked cursor)."""
    store = _make_store()
    cursor = MagicMock()
    cursor.fetchone.return_value = (7,)
    with patch.object(store, "_get_cursor", return_value=cursor):
        result = store.count_embedded()
    assert isinstance(result, int)
    assert result == 7
