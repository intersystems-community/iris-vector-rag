"""ColBERT ingest never hands back a connection with an open transaction.

`_transactional` turns autocommit off for the ingest and restores it after. A failure
that is not an Exception (a pytest-timeout, KeyboardInterrupt) escapes the per-document
rollback; the connection then goes back to the shared session with uncommitted writes
holding row locks that block the next DDL on the same tables.
"""

import pytest

from iris_vector_rag.pipelines.colbert_iris.ingest import ColBERTIngestor


class Conn:
    def __init__(self):
        self.autocommit = True
        self.rollbacks = 0
        self.commits = 0

    def rollback(self):
        self.rollbacks += 1

    def commit(self):
        self.commits += 1


def test_autocommit_is_restored_after_a_clean_run():
    conn = Conn()
    with ColBERTIngestor(conn)._transactional():
        assert conn.autocommit is False
    assert conn.autocommit is True
    assert conn.rollbacks == 0


@pytest.mark.parametrize("error", [RuntimeError("x"), KeyboardInterrupt()])
def test_an_escaping_failure_rolls_back_before_autocommit_returns(error):
    conn = Conn()
    with pytest.raises(type(error)):
        with ColBERTIngestor(conn)._transactional():
            raise error
    assert conn.rollbacks == 1
    assert conn.autocommit is True
