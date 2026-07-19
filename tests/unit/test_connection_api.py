"""Verify IRIS connection patterns in the library.

Ensures the codebase uses the standard intersystems-irispython connection
pattern (import iris → iris.connect) consistent with iris-vector-graph.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text()


def test_uses_import_iris_pattern():
    """iris_connection.py must use 'import iris' (official intersystems-irispython pattern)."""
    content = _read("iris_vector_rag/common/iris_connection.py")
    assert (
        "import iris" in content
    ), "Must use 'import iris' from intersystems-irispython"


def test_no_import_iris_dbapi_directly():
    """Should not import iris.dbapi directly — use 'import iris' then iris.connect()."""
    content = _read("iris_vector_rag/common/iris_connection.py")
    assert "import iris.dbapi" not in content, (
        "Should use 'import iris' not 'import iris.dbapi' — "
        "iris.connect() is the standard pattern per intersystems-irispython docs"
    )


# ---------------------------------------------------------------------------
# T001 — embedded-mode branch: when IRISINSTALLDIR set, use iris.dbapi.connect(path=...)
# ---------------------------------------------------------------------------


class TestEmbeddedModeSupport:
    """T001: get_iris_connection() uses embedded path when IRISINSTALLDIR is set."""

    def test_embedded_kernel_mode_skips_tcp(self):
        """When iris.runtime reports embedded-kernel, use iris.dbapi.connect(namespace=...)."""
        import iris_vector_rag.common.iris_connection as conn_module

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor

        mock_runtime_ctx = MagicMock()
        mock_runtime_ctx.state = "embedded-kernel"

        mock_iris = MagicMock()
        mock_iris.runtime.get.return_value = mock_runtime_ctx
        mock_iris.dbapi.connect.return_value = mock_conn

        with patch.dict(sys.modules, {"iris": mock_iris}):
            with patch.object(conn_module, "_connection_cache", {}):
                result = conn_module.get_iris_connection(
                    host="localhost",
                    port=1972,
                    namespace="USER",
                    username="_SYSTEM",
                    password="SYS",
                )

        mock_iris.dbapi.connect.assert_called_once()
        call_kwargs = mock_iris.dbapi.connect.call_args
        assert call_kwargs.kwargs.get("namespace") == "USER" or (
            call_kwargs.args and "USER" in str(call_kwargs.args)
        ), "embedded-kernel must call iris.dbapi.connect with namespace"
        assert result is mock_conn

    def test_irisinstalldir_triggers_embedded_local(self, tmp_path, monkeypatch):
        """When IRISINSTALLDIR is set and runtime is unavailable, configure embedded-local."""
        import iris_vector_rag.common.iris_connection as conn_module

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor

        mock_runtime_ctx_unavailable = MagicMock()
        mock_runtime_ctx_unavailable.state = "unavailable"

        mock_runtime_ctx_embedded = MagicMock()
        mock_runtime_ctx_embedded.state = "embedded-local"

        mock_iris = MagicMock()
        mock_iris.runtime.get.side_effect = [
            mock_runtime_ctx_unavailable,  # first check
            mock_runtime_ctx_embedded,  # after configure()
        ]
        mock_iris.runtime.configure.return_value = mock_runtime_ctx_embedded
        mock_iris.dbapi.connect.return_value = mock_conn

        monkeypatch.setenv("IRISINSTALLDIR", str(tmp_path))

        with patch.dict(sys.modules, {"iris": mock_iris}):
            with patch.object(conn_module, "_connection_cache", {}):
                result = conn_module.get_iris_connection(
                    host="localhost",
                    port=1972,
                    namespace="USER",
                    username="_SYSTEM",
                    password="SYS",
                )

        mock_iris.runtime.configure.assert_called_once_with(
            mode="embedded", install_dir=str(tmp_path)
        )
        mock_iris.dbapi.connect.assert_called_once()
        assert result is mock_conn


# ---------------------------------------------------------------------------
# T002 — connection_pool.py: importing the module must not raise even without iris
# ---------------------------------------------------------------------------


class TestConnectionPoolLazyImport:
    """T002: connection_pool module import must not fail when iris is unavailable."""

    def test_module_importable_without_iris(self):
        """Importing connection_pool must not raise even when iris is not installed."""
        saved = sys.modules.pop("iris_vector_rag.common.connection_pool", None)
        saved_iris = sys.modules.pop("iris", None)
        saved_iris_dbapi = sys.modules.pop("iris.dbapi", None)

        try:
            with patch.dict(sys.modules, {"iris": None, "iris.dbapi": None}):
                import iris_vector_rag.common.connection_pool  # must not raise
        finally:
            if saved:
                sys.modules["iris_vector_rag.common.connection_pool"] = saved
            if saved_iris is not None:
                sys.modules["iris"] = saved_iris
            if saved_iris_dbapi is not None:
                sys.modules["iris.dbapi"] = saved_iris_dbapi


# ---------------------------------------------------------------------------
# T004 — colbert _ensure_native_conn must route through get_iris_connection
# ---------------------------------------------------------------------------


class TestColbertEnsureNativeConn:
    """T004: colbert _ensure_native_conn() must call get_iris_connection(), not intersystems_iris."""

    def test_plaid_ensure_native_conn_calls_get_iris_connection(self):
        """plaid._ensure_native_conn calls get_iris_connection when conn is not native."""
        from iris_vector_rag.pipelines.colbert_iris import plaid

        mock_result_conn = MagicMock()
        fake_conn = MagicMock(spec=[])  # no IRISConnection attrs

        # The function does `from iris_vector_rag.common.iris_connection import get_iris_connection`
        # at call time, so we patch the source module's name.
        with patch(
            "iris_vector_rag.common.iris_connection.get_iris_connection",
            return_value=mock_result_conn,
        ) as mock_get:
            result = plaid._ensure_native_conn(fake_conn)

        mock_get.assert_called_once()
        assert result is mock_result_conn

    def test_vecindex_ensure_native_conn_calls_get_iris_connection(self):
        """vecindex_phase2._ensure_native_conn calls get_iris_connection when conn is not native."""
        from iris_vector_rag.pipelines.colbert_iris import vecindex_phase2

        mock_result_conn = MagicMock()
        fake_conn = MagicMock(spec=[])

        with patch(
            "iris_vector_rag.common.iris_connection.get_iris_connection",
            return_value=mock_result_conn,
        ) as mock_get:
            result = vecindex_phase2._ensure_native_conn(fake_conn)

        mock_get.assert_called_once()
        assert result is mock_result_conn


# ---------------------------------------------------------------------------
# T005 — hybrid_graphrag connection path must call get_iris_connection
# ---------------------------------------------------------------------------


class TestHybridGraphRAGConnection:
    """T005: hybrid_graphrag must use get_iris_connection(), not inline hasattr fan-out."""

    def test_no_hasattr_iris_fanout_in_hybrid_graphrag(self):
        """hybrid_graphrag.py must not contain the 3-way hasattr fan-out."""
        content = _read("iris_vector_rag/pipelines/hybrid_graphrag.py")
        assert (
            'hasattr(_iris_mod, "createConnection")' not in content
        ), "hybrid_graphrag must not contain inline hasattr createConnection fan-out"
        assert (
            'hasattr(_iris_mod, "dbapi")' not in content
        ), "hybrid_graphrag must not contain inline hasattr dbapi fan-out"

    def test_no_intersystems_iris_createconnection_in_colbert(self):
        """colbert files must not call intersystems_iris.createConnection directly."""
        for rel in [
            "iris_vector_rag/pipelines/colbert_iris/plaid.py",
            "iris_vector_rag/pipelines/colbert_iris/vecindex_phase2.py",
        ]:
            content = _read(rel)
            assert (
                "intersystems_iris.createConnection" not in content
            ), f"{rel} must not call intersystems_iris.createConnection directly"
