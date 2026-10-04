"""E2E gate for colbert-vendored-objectscript: the kept class compiles and searches.

Loads `RAG.ColBERTSearch` into a container private to this module (never a shared
database), builds the three tables it reads, and calls it as the SQL procedure the
ColBERT pipeline uses. OS-11: a doc_id with an apostrophe must neither break nor
inject into the candidate query.
"""

import json
import subprocess
from pathlib import Path

import pytest

from tests.support.iris_environment import connect, start_ephemeral

CLS = (
    Path(__file__).resolve().parents[2]
    / "iris_vector_rag"
    / "pipelines"
    / "colbert_iris"
    / "sp"
    / "ColBERTSearch.cls"
)


@pytest.fixture(scope="module")
def conn():
    container, target = start_ephemeral()
    try:
        subprocess.run(
            ["docker", "cp", str(CLS), f"{container.name}:/tmp/ColBERTSearch.cls"],
            check=True,
            capture_output=True,
        )
        loaded = subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                container.name,
                "iris",
                "session",
                "IRIS",
                "-U",
                "USER",
            ],
            input='Set sc=$SYSTEM.OBJ.Load("/tmp/ColBERTSearch.cls","ck")\nHalt\n',
            capture_output=True,
            text=True,
        )
        assert "Load finished successfully" in loaded.stdout, loaded.stdout[-2000:]
        c = connect(target)
        cur = c.cursor()
        for ddl in (
            "CREATE TABLE RAG.ColBERTCentroids (centroid_id INTEGER PRIMARY KEY, centroid_vec VECTOR(FLOAT,2))",
            "CREATE TABLE RAG.ColBERTDocCentroids (centroid_id INTEGER, doc_id VARCHAR(64))",
            "CREATE TABLE RAG.DocumentTokenEmbeddings (doc_id VARCHAR(64), tok_vec VECTOR(FLOAT,2))",
        ):
            cur.execute(ddl)
        cur.execute(
            "INSERT INTO RAG.ColBERTCentroids VALUES (1, TO_VECTOR('[1.0,0.0]', FLOAT, 2))"
        )
        for doc in ("plain", "o'brien"):
            cur.execute("INSERT INTO RAG.ColBERTDocCentroids VALUES (1, ?)", [doc])
            cur.execute(
                "INSERT INTO RAG.DocumentTokenEmbeddings VALUES (?, TO_VECTOR('[1.0,0.0]', FLOAT, 2))",
                [doc],
            )
        c.commit()
        yield c
        c.close()
    finally:
        container.stop()


def _search(conn, top_k=5, n_probe=1):
    cur = conn.cursor()
    cur.execute(
        "SELECT RAG.ColBERTSearch_Search(?, ?, ?)",
        [json.dumps([[1.0, 0.0]]), top_k, n_probe],
    )
    return json.loads(cur.fetchone()[0])


def test_search_compiles_and_ranks_documents(conn):
    result = _search(conn)
    assert {d.lower() for d, _ in result["results"]} == {"plain", "o'brien"}


def test_doc_id_with_an_apostrophe_is_searched_not_injected(conn):
    result = _search(conn)
    assert "error" not in result
    assert result["n_candidates"] == 2
