"""Verify IRIS connection patterns in the library.

Ensures the codebase uses the standard intersystems-irispython connection
pattern (import iris → iris.connect) consistent with iris-vector-graph.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text()


def test_uses_import_iris_pattern():
    """iris_connection.py must use 'import iris' (official intersystems-irispython pattern)."""
    content = _read("iris_vector_rag/common/iris_connection.py")
    assert "import iris" in content, "Must use 'import iris' from intersystems-irispython"


def test_no_import_iris_dbapi_directly():
    """Should not import iris.dbapi directly — use 'import iris' then iris.connect()."""
    content = _read("iris_vector_rag/common/iris_connection.py")
    assert "import iris.dbapi" not in content, (
        "Should use 'import iris' not 'import iris.dbapi' — "
        "iris.connect() is the standard pattern per intersystems-irispython docs"
    )
