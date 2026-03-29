import json
import os
import subprocess

import numpy as np
import pytest

IRIS_HOST = os.environ.get("IRIS_HOSTNAME", "localhost")
IRIS_PORT = int(os.environ.get("IRIS_PORT", "13972"))
CONTAINER = os.environ.get("IRIS_CONTAINER", "iris-langchain-spike")
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.environ.get(
    "COLBERT_SCRIPTS_DIR",
    os.path.join(_THIS_DIR, "..", "..", "scripts"),
)
TOKEN_DIM = 128


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


class DummyModel:
    def encode(self, texts, is_query=False):
        n_tokens = 4 if is_query else 8
        results = []
        for text in texts:
            seed = sum(ord(c) for c in text[:32]) % (2**32)
            rng = np.random.default_rng(seed)
            vecs = rng.random((n_tokens, TOKEN_DIM)).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            results.append(vecs / norms)
        return results


def _vecindex_already_deployed() -> bool:
    try:
        import intersystems_iris

        conn = intersystems_iris.createConnection(
            hostname=IRIS_HOST,
            port=IRIS_PORT,
            namespace="USER",
            username="_SYSTEM",
            password="SYS",
        )
        iris_obj = conn.createIRIS()
        result = iris_obj.classMethodValue("Graph.KG.VecIndex", "Info", "__probe__")
        conn.close()
        return '"name"' in str(result)
    except Exception:
        return False


@pytest.fixture(scope="session")
def setup_vecindex():
    if not _iris_available():
        pytest.skip(f"IRIS not available at {IRIS_HOST}:{IRIS_PORT}")

    if not _vecindex_already_deployed():
        deploy_script = os.path.join(SCRIPTS_DIR, "deploy_vecindex.sh")
        result = subprocess.run(
            ["bash", deploy_script, CONTAINER],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            pytest.skip(f"deploy_vecindex.sh failed: {result.stderr}")

    import intersystems_iris
    from iris_vector_graph import IRISGraphEngine

    conn = intersystems_iris.createConnection(
        hostname=IRIS_HOST,
        port=IRIS_PORT,
        namespace="USER",
        username="_SYSTEM",
        password="SYS",
    )

    try:
        engine = IRISGraphEngine(conn)
        engine.vec_create_index("__smoke_test__", 4, "dot")
        engine.vec_insert("__smoke_test__", "s:0", [0.1, 0.2, 0.3, 0.4])
        results = engine.vec_search(
            "__smoke_test__", [0.1, 0.2, 0.3, 0.4], k=1, nprobe=1
        )
        assert len(results) >= 1, f"VecIndex smoke test returned no results: {results}"
        engine.vec_drop("__smoke_test__")
    except Exception as e:
        conn.close()
        pytest.skip(f"VecIndex smoke test failed: {e}")

    yield conn
    conn.close()


@pytest.fixture(scope="session")
def vecindex_conn(setup_vecindex):
    return setup_vecindex


@pytest.fixture(scope="module")
def vecindex_80doc(setup_vecindex):
    import intersystems_iris
    from iris_vector_graph import IRISGraphEngine

    from iris_vector_rag.pipelines.colbert_iris.ingest import ColBERTIngestor
    from iris_vector_rag.pipelines.colbert_iris.schema import ColBERTSchema
    from iris_vector_rag.pipelines.colbert_iris.vecindex_phase2 import VecIndexSearcher

    conn = intersystems_iris.createConnection(
        hostname=IRIS_HOST,
        port=IRIS_PORT,
        namespace="USER",
        username="_SYSTEM",
        password="SYS",
    )
    schema = ColBERTSchema(conn)
    schema.drop_tables()
    schema.create_tables()

    ingestor = ColBERTIngestor(conn, model=DummyModel(), token_dim=TOKEN_DIM)
    docs = [
        {
            "doc_id": f"vi80_{i:04d}",
            "text": f"Document {i} about topic {i % 10}.",
            "metadata": {},
        }
        for i in range(80)
    ]
    searcher = VecIndexSearcher(conn, index_name="test_vi80", token_dim=TOKEN_DIM)
    ingestor.ingest_documents(docs, use_vecindex=True, vecindex_searcher=searcher)

    yield conn, searcher

    searcher.drop()
    schema.drop_tables()
    conn.close()


def make_query_vecs(n_toks: int = 4, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.random((n_toks, TOKEN_DIM)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs
