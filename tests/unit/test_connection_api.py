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


@pytest.fixture
def fresh_sessions():
    """No session outlives a test that substitutes the `iris` module.

    Since 083 the embedded decision belongs to the session's `EmbeddedDriver`, and
    sessions are cached per identity, so a leftover one would answer with the driver
    the previous test chose.
    """
    from iris_vector_rag.session import registry

    registry.reset_sessions_for_tests()
    yield
    registry.reset_sessions_for_tests()


class TestEmbeddedModeSupport:
    """T001: the embedded path is taken when the runtime says so, or IRISINSTALLDIR is set.

    The decision moved into `session/drivers.py` (083-iris-session, FR-006); these
    tests still enter through the connection function callers use.
    """

    def test_embedded_kernel_mode_skips_tcp(self, fresh_sessions):
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
            result = conn_module._session_connection(
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

    def test_irisinstalldir_triggers_embedded_local(
        self, tmp_path, monkeypatch, fresh_sessions
    ):
        """When IRISINSTALLDIR is set and runtime is unavailable, configure embedded-local."""
        import iris_vector_rag.common.iris_connection as conn_module

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = (1,)
        mock_conn.cursor.return_value = mock_cursor

        # The runtime reports "unavailable" until it is configured, and the session
        # asks more than once (once to pick the driver, once to connect), so the
        # state is a fact about the runtime rather than a fixed sequence of answers.
        runtime_state = {"value": "unavailable"}

        def runtime_get():
            context = MagicMock()
            context.state = runtime_state["value"]
            return context

        def runtime_configure(mode, install_dir):
            runtime_state["value"] = "embedded-local"
            return runtime_get()

        mock_iris = MagicMock()
        mock_iris.runtime.get.side_effect = runtime_get
        mock_iris.runtime.configure.side_effect = runtime_configure
        mock_iris.dbapi.connect.return_value = mock_conn

        monkeypatch.setenv("IRISINSTALLDIR", str(tmp_path))

        with patch.dict(sys.modules, {"iris": mock_iris}):
            result = conn_module._session_connection(
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
# T004 — colbert _ensure_native_conn must route through the session
# ---------------------------------------------------------------------------


class TestColbertEnsureNativeConn:
    """T004: colbert _ensure_native_conn() must reach the session, not intersystems_iris."""

    def test_plaid_ensure_native_conn_reaches_the_session(self):
        """plaid._ensure_native_conn asks the session when conn is not native."""
        from iris_vector_rag.pipelines.colbert_iris import plaid

        mock_result_conn = MagicMock()
        fake_conn = MagicMock(spec=[])  # no IRISConnection attrs

        # The function imports the name at call time, so patch the source module's.
        with patch(
            "iris_vector_rag.common.iris_connection._session_connection",
            return_value=mock_result_conn,
        ) as mock_get:
            result = plaid._ensure_native_conn(fake_conn)

        mock_get.assert_called_once()
        assert result is mock_result_conn

    def test_vecindex_ensure_native_conn_reaches_the_session(self):
        """vecindex_phase2._ensure_native_conn asks the session when conn is not native."""
        from iris_vector_rag.pipelines.colbert_iris import vecindex_phase2

        mock_result_conn = MagicMock()
        fake_conn = MagicMock(spec=[])

        with patch(
            "iris_vector_rag.common.iris_connection._session_connection",
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
