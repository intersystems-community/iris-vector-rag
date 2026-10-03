"""The catalog helpers in IRISSQLUtils must not close a connection they did not open.

`get_iris_dbapi_connection()` hands back the process-wide cached connection, not a fresh
one. Three helpers called it "temporary" and closed it in a `finally` block, which closed
the connection their own caller was in the middle of using: `initialize_schema`'s vector
index step died with `<COMMUNICATION LINK ERROR> Connection closed` every time it ran
against a real instance, and so did the `store_documents` call that triggered it.

The cache owns that connection's lifetime. These tests pin the ownership rule.
"""

from unittest.mock import MagicMock, patch

import pytest

from iris_vector_rag.common.iris_sql_utils import IRISSQLUtils


@pytest.fixture
def shared_connection():
    """Stand in for the cached connection that get_iris_dbapi_connection returns."""
    connection = MagicMock(name="cached_connection")
    connection.cursor.return_value.fetchone.return_value = [0]
    connection.cursor.return_value.fetchall.return_value = []
    with patch(
        "iris_vector_rag.common.iris_sql_utils.get_iris_dbapi_connection",
        return_value=connection,
    ):
        yield connection


def test_check_table_exists_leaves_the_shared_connection_open(shared_connection):
    IRISSQLUtils.check_table_exists(shared_connection, "RAG.SourceDocuments")
    shared_connection.close.assert_not_called()


def test_check_index_exists_leaves_the_shared_connection_open(shared_connection):
    IRISSQLUtils.check_index_exists(
        shared_connection,
        "RAG_SourceDocuments_embedding_vector_idx",
        "RAG.SourceDocuments",
    )
    shared_connection.close.assert_not_called()


def test_get_table_columns_leaves_the_shared_connection_open(shared_connection):
    IRISSQLUtils.get_table_columns(shared_connection, "RAG.SourceDocuments")
    shared_connection.close.assert_not_called()


def test_helpers_still_close_the_cursors_they_open(shared_connection):
    """Cursors are this code's own; only the connection belongs to the cache."""
    IRISSQLUtils.check_table_exists(shared_connection, "RAG.SourceDocuments")
    shared_connection.cursor.return_value.close.assert_called()
