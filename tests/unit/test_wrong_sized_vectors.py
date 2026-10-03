"""A vector of the wrong width is refused, never padded, truncated or skipped.

ST-13  a failed insert aborts the batch instead of returning a partial id list
ST-23  normalize_vector no longer pads or truncates
G6 point 6  DimensionMismatchError on insert and on query, in ivg's wording
CR-09  an embedding failure is not a zero vector (store and engine)
"""

from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from iris_vector_rag.common import db_vector_utils
from iris_vector_rag.exceptions import (
    DimensionMismatchError,
    EmbeddingError,
    VectorStoreDataError,
)
from tests.unit.test_vector_store_shape import DIM, FakeCursor, document, make_store

# ---------------------------------------------------------------------------
# normalize_vector and insert_vector
# ---------------------------------------------------------------------------


def test_normalize_vector_returns_exact_width_vectors_unchanged():
    assert db_vector_utils.normalize_vector([1, 2, 3], 3) == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("vector", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0]])
def test_normalize_vector_refuses_to_pad_or_truncate(vector):
    with pytest.raises(DimensionMismatchError, match="dimension mismatch"):
        db_vector_utils.normalize_vector(vector, 3)


def test_normalize_vector_accepts_numpy():
    np = pytest.importorskip("numpy")
    assert db_vector_utils.normalize_vector(np.ones(4, dtype="float32"), 4) == [1.0] * 4
    with pytest.raises(DimensionMismatchError):
        db_vector_utils.normalize_vector(np.ones(5), 4)


def test_the_message_says_what_to_do():
    with pytest.raises(DimensionMismatchError) as info:
        db_vector_utils.normalize_vector([0.0] * 5, 3)
    message = str(info.value)
    assert "5" in message and "3" in message
    assert "migrate" in message.lower()


def test_insert_vector_refuses_before_any_sql_runs():
    cursor = MagicMock()
    with pytest.raises(DimensionMismatchError):
        db_vector_utils.insert_vector(
            cursor=cursor,
            table_name="RAG.T",
            vector_column="embedding",
            vector_data=[0.0] * 5,
            dimension=3,
            key_columns={"id": "x"},
        )
    cursor.execute.assert_not_called()


def test_the_error_is_a_value_error_and_a_vector_store_error():
    assert issubclass(DimensionMismatchError, ValueError)
    assert issubclass(DimensionMismatchError, VectorStoreDataError)


# ---------------------------------------------------------------------------
# the store
# ---------------------------------------------------------------------------


def test_storing_a_wrong_sized_embedding_raises_and_commits_nothing(recorded_catalog):
    cursor = FakeCursor()
    store = make_store("canonical", cursor, recorded_catalog)
    connection = store.connection_manager.get_connection.return_value

    with patch("iris_vector_rag.storage.vector_store_iris.insert_vector") as insert:
        with pytest.raises(DimensionMismatchError, match="dimension mismatch"):
            store._store_documents(
                [document("a"), document("b")],
                embeddings=[[0.1] * DIM, [0.1] * (DIM + 1)],
            )

    # The session connection autocommits, so a rollback cannot un-write the first
    # document: every width is checked before any row is.
    insert.assert_not_called()
    connection.commit.assert_not_called()
    connection.rollback.assert_called()


def test_a_failed_insert_aborts_the_batch(recorded_catalog):
    """ST-13: a `False` from insert_vector used to be logged, skipped and committed."""
    cursor = FakeCursor()
    store = make_store("canonical", cursor, recorded_catalog)
    connection = store.connection_manager.get_connection.return_value

    with patch(
        "iris_vector_rag.storage.vector_store_iris.insert_vector",
        side_effect=[True, False],
    ):
        with pytest.raises(VectorStoreDataError, match="b") as info:
            store._store_documents(
                [document("a"), document("b")], embeddings=[[0.1] * DIM, [0.1] * DIM]
            )
    assert info.value.details["stored_ids"] == ["a"]

    connection.commit.assert_not_called()
    connection.rollback.assert_called()


def test_a_query_of_the_wrong_width_raises_the_same_error(recorded_catalog):
    store = make_store("canonical", FakeCursor(), recorded_catalog)
    with pytest.raises(DimensionMismatchError, match="dimension mismatch"):
        store.similarity_search_by_embedding([0.1] * (DIM + 2), 5)


def test_store_ctor_does_not_resolve_the_dimension(recorded_catalog):
    store = make_store("canonical", FakeCursor(), recorded_catalog)
    store.schema_manager.get_vector_dimension.assert_not_called()
    assert store.vector_dimension == DIM
    store.schema_manager.get_vector_dimension.assert_called_once()


# ---------------------------------------------------------------------------
# CR-09: no zero vector, no mock sniffing
# ---------------------------------------------------------------------------


def test_text_search_with_a_mocked_config_does_not_fabricate_a_zero_vector(
    recorded_catalog,
):
    """The store used to notice a spec'd Mock config and search with [0.0] * n."""
    store = make_store("canonical", FakeCursor(), recorded_catalog)
    with patch(
        "iris_vector_rag.embeddings.manager.EmbeddingManager.embed_text",
        side_effect=EmbeddingError("no model"),
    ):
        with pytest.raises(EmbeddingError, match="no model"):
            store.similarity_search("a question")


def test_relation_search_embedding_failure_raises():
    from iris_vector_rag.retrieval import engine as engine_module

    holder = engine_module.RetrievalEngine.__new__(engine_module.RetrievalEngine)
    holder._config_manager = MagicMock()
    with patch(
        "iris_vector_rag.embeddings.manager.EmbeddingManager.embed_text",
        side_effect=EmbeddingError("model gone"),
    ):
        with pytest.raises(EmbeddingError, match="model gone"):
            holder._embed_text_for_search("theme")


def test_relation_search_without_a_config_manager_raises():
    from iris_vector_rag.retrieval import engine as engine_module

    holder = engine_module.RetrievalEngine.__new__(engine_module.RetrievalEngine)
    holder._config_manager = None
    with pytest.raises(EmbeddingError, match="configuration"):
        holder._embed_text_for_search("theme")


def test_hybrid_fusion_does_not_fall_back_to_text_only_when_the_query_embedding_fails():
    from iris_vector_rag.pipelines.hybrid_graphrag_retrieval import (
        HybridRetrievalMethods,
    )

    fusion = MagicMock()
    manager = MagicMock()
    manager.get_embeddings.side_effect = EmbeddingError("no model")
    methods = HybridRetrievalMethods(
        MagicMock(), fusion, MagicMock(), MagicMock(), manager
    )
    with pytest.raises(EmbeddingError):
        methods.retrieve_via_hybrid_fusion("q", 3, lambda *_: "")
    fusion.multi_modal_search.assert_not_called()


def test_rrf_keeps_the_embedding_error_instead_of_a_generic_value_error():
    from iris_vector_rag.pipelines.hybrid_graphrag_retrieval import (
        HybridRetrievalMethods,
    )

    manager = MagicMock()
    manager.get_embeddings.side_effect = EmbeddingError("no model")
    methods = HybridRetrievalMethods(
        MagicMock(), MagicMock(), MagicMock(), MagicMock(), manager
    )
    with pytest.raises(EmbeddingError):
        methods.retrieve_via_rrf("q", 3, lambda *_: "")


# ---------------------------------------------------------------------------
# relation embeddings: the width is the authority's
# ---------------------------------------------------------------------------


class _Cfg:
    def get(self, key, default=None):
        if key == "embeddings":
            return {"provider": "stub", "dimension": 7}
        return default


def test_relation_embedding_sql_uses_the_authoritys_width():
    from iris_vector_rag.storage.relation_embedding_store import RelationEmbeddingStore

    conn = MagicMock()
    cursor = conn.cursor.return_value
    store = RelationEmbeddingStore(MagicMock(get_connection=lambda *_: conn), _Cfg())
    store._ensure_schema()
    store.embed_and_store("r1", "KNOWS", "a", "b")

    executed = [call.args[0] for call in cursor.execute.call_args_list]
    assert any("VECTOR(FLOAT, 7)" in sql for sql in executed)
    assert any("TO_VECTOR(?, FLOAT, 7)" in sql for sql in executed)
    assert not any("384" in sql for sql in executed)


# ---------------------------------------------------------------------------
# engine: the dimension property reads the schema manager's authority
# ---------------------------------------------------------------------------


def test_engine_embedding_dimension_is_the_schema_managers():
    from iris_vector_rag.core.engine import IRISVectorEngine

    store = MagicMock()
    store.schema_manager.base_embedding_dimension = 1536
    engine = IRISVectorEngine.__new__(IRISVectorEngine)
    with patch.object(
        IRISVectorEngine, "vector_store", PropertyMock(return_value=store)
    ):
        assert engine.embedding_dimension == 1536


# ---------------------------------------------------------------------------
# ST-13 follow-on: re-adding an existing document must not depend on an error message
# ---------------------------------------------------------------------------


class _ExistingRowCursor(FakeCursor):
    """A table that already holds the row; the driver's duplicate-key error is garbled.

    ivg's upsert falls back to UPDATE only when the error text contains UNIQUE or
    DUPLICATE. This driver can report the duplicate as `<LIST ERROR> ...`, so the
    store must not rely on it.
    """

    def execute(self, sql, params=None):
        upper = " ".join(str(sql).upper().split())
        if upper.startswith("INSERT"):
            self.executed.append((sql, params))
            raise RuntimeError("<LIST ERROR> Incorrect list format, unsupported type")
        if upper.startswith("SELECT COUNT(*)"):
            self.executed.append((sql, params))
            self._result = [(1,)]
            return
        super().execute(sql, params)


def test_re_adding_an_existing_document_updates_it_whatever_the_duplicate_error_says(
    recorded_catalog,
):
    cursor = _ExistingRowCursor()
    store = make_store("canonical", cursor, recorded_catalog)

    ids = store._store_documents([document("a", "new text")], embeddings=[[0.1] * DIM])

    assert ids == ["a"]
    update = [sql for sql in cursor.sql if sql.upper().startswith("UPDATE")]
    assert len(update) == 1 and "TO_VECTOR" in update[0].upper()
    params = [p for sql, p in cursor.executed if sql.upper().startswith("UPDATE")][0]
    assert "new text" in params and params[-1] == "a"


def test_a_failed_insert_of_a_new_document_still_raises(recorded_catalog):
    cursor = _ExistingRowCursor()
    cursor.execute = lambda sql, params=None: (
        (_ for _ in ()).throw(  # type: ignore[assignment]
            RuntimeError("the table is gone")
        )
        if str(sql).upper().startswith("INSERT")
        else FakeCursor.execute(cursor, sql, params)
    )
    cursor.fetchone = lambda: (0,)  # type: ignore[assignment]
    store = make_store("canonical", cursor, recorded_catalog)
    with pytest.raises(VectorStoreDataError, match="a"):
        store._store_documents([document("a")], embeddings=[[0.1] * DIM])
