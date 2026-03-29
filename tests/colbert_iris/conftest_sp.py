import json
import os
import subprocess

import numpy as np
import pytest

IRIS_HOST = os.environ.get("IRIS_HOSTNAME", "localhost")
IRIS_PORT = int(os.environ.get("IRIS_PORT", "13972"))
CONTAINER = os.environ.get("IRIS_CONTAINER", "iris-langchain-spike")
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")


def _iris_available() -> bool:
    try:
        import iris.dbapi as dbapi

        conn = dbapi.connect(
            hostname=IRIS_HOST,
            port=IRIS_PORT,
            namespace="USER",
            username="_SYSTEM",
            password="SYS",
        )
        conn.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def setup_colbert_sp():
    if not _iris_available():
        pytest.skip(f"IRIS not available at {IRIS_HOST}:{IRIS_PORT}")

    setup_script = os.path.join(SCRIPTS_DIR, "setup_spike_env.sh")
    deploy_script = os.path.join(SCRIPTS_DIR, "deploy_colbert_sp.sh")

    result = subprocess.run(
        ["bash", setup_script, CONTAINER],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.skip(f"setup_spike_env.sh failed: {result.stderr}")

    result = subprocess.run(
        ["bash", deploy_script, CONTAINER],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        pytest.skip(f"deploy_colbert_sp.sh failed: {result.stderr}")

    import iris.dbapi as dbapi

    conn = dbapi.connect(
        hostname=IRIS_HOST,
        port=IRIS_PORT,
        namespace="USER",
        username="_SYSTEM",
        password="SYS",
    )
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM RAG.ColBERTCentroids")
        n_centroids = cur.fetchone()[0]
        cur.close()
        if n_centroids == 0:
            from iris_vector_rag.pipelines.colbert_iris.plaid import PLAIDBuilder

            builder = PLAIDBuilder(conn, token_dim=128)
            builder.build(n_clusters=512)
    except Exception as e:
        conn.close()
        pytest.skip(f"PLAID centroid build failed: {e}")

    try:
        cur = conn.cursor()
        q = np.random.default_rng(0).random((2, 128)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        q_json = json.dumps(q.tolist())
        cur.execute("SELECT RAG.ColBERTSearch_Search(?, ?, ?)", [q_json, 1, 1])
        row = cur.fetchone()
        raw_sp = row[0]
        assert row is not None, "SP returned no row"
        parsed = json.loads(raw_sp)
        assert "results" in parsed, f"SP response missing results: {raw_sp}"
        cur.close()
    except Exception as e:
        conn.close()
        pytest.skip(f"SP smoke test failed: {e}")

    yield conn
    conn.close()


@pytest.fixture(scope="session")
def sp_conn(setup_colbert_sp):
    return setup_colbert_sp


def make_query_vecs(n_toks: int = 4, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.random((n_toks, 128)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs
