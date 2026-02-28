from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text()


def test_no_iris_connect_usage_in_library():
    targets = [
        "iris_vector_rag/common/iris_dbapi_connector.py",
        "iris_vector_rag/common/utils.py",
        "iris_vector_rag/common/environment_manager.py",
        "iris_vector_rag/common/iris_connection.py",
        "iris_vector_rag/pipelines/hybrid_graphrag.py",
    ]
    for rel_path in targets:
        content = _read(rel_path)
        assert "iris.connect" not in content, f"Found iris.connect in {rel_path}"


def test_create_connection_used_in_library():
    targets = [
        "iris_vector_rag/common/utils.py",
        "iris_vector_rag/common/environment_manager.py",
        "iris_vector_rag/common/iris_connection.py",
        "iris_vector_rag/pipelines/hybrid_graphrag.py",
    ]
    for rel_path in targets:
        content = _read(rel_path)
        assert "createConnection" in content, f"Missing createConnection in {rel_path}"
