"""ColBERT in-database MaxSim binds each query-token vector (TO_VECTOR(?, FLOAT, dim)).

Writing the 128-number vector into the SQL compiled a new statement for every query
token of every query. The bound form works for FLOAT and DOUBLE columns on IRIS 2026.1
and later (common/vector_sql_utils.build_bound_vector_dot_sql). No database here: the
connection records what was sent.
"""

from __future__ import annotations

import numpy as np

from iris_vector_rag.pipelines.colbert_iris.maxsim_indb import MaxSimInDB

DIM = 4


class RecordingCursor:
    def __init__(self, log, rows):
        self.log, self.rows = log, rows

    def execute(self, sql, params=None):
        self.log.append((" ".join(sql.split()), params))

    def fetchall(self):
        return self.rows

    def close(self):
        pass


class RecordingConnection:
    def __init__(self, rows=()):
        self.log = []
        self.rows = list(rows)

    def cursor(self):
        return RecordingCursor(self.log, self.rows)


QUERY = np.array([[0.5, 0.25, 0.125, 1.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)


def vec(row):
    return "[" + ",".join(f"{v:.6f}" for v in row) + "]"


def test_bulk_fetch_binds_the_token_vector_before_the_document_ids():
    conn = RecordingConnection(rows=[("d1", 0.5)])
    MaxSimInDB(conn, token_dim=DIM).bulk_fetch_maxsim(QUERY, ["d1", "d2"], top_k=2)

    assert len(conn.log) == 2  # one statement per query token
    for (sql, params), row in zip(conn.log, QUERY):
        assert f"TO_VECTOR(?, FLOAT, {DIM})" in sql
        assert "'[" not in sql
        assert params == [vec(row), "d1", "d2"]
    assert conn.log[0][0] == conn.log[1][0], "the same statement for every token"


def test_indb_maxsim_binds_the_token_vector():
    conn = RecordingConnection(rows=[("d1", 0.5)])
    MaxSimInDB(conn, token_dim=DIM).indb_maxsim(QUERY, top_k=2, k_per_token=3)

    for (sql, params), row in zip(conn.log, QUERY):
        assert sql.startswith("SELECT TOP 3 doc_id,")
        assert f"TO_VECTOR(?, FLOAT, {DIM})" in sql and "'[" not in sql
        assert params == [vec(row)]


def test_explain_reads_the_plan_of_the_statement_the_search_runs():
    conn = RecordingConnection(rows=[("d1", 0.5)])
    ms = MaxSimInDB(conn, token_dim=DIM)
    ms.explain_indb_query(QUERY[0], k_per_token=3)
    ms.indb_maxsim(QUERY[:1], top_k=1, k_per_token=3)

    (explain_sql, explain_params), (search_sql, search_params) = conn.log
    assert explain_sql == f"EXPLAIN {search_sql}"
    # EXPLAIN gets the statement with its marker unbound: the DB-API refuses
    # parameter values for EXPLAIN, and the plan does not depend on the vector.
    assert explain_params is None
    assert search_params == [vec(QUERY[0])]
