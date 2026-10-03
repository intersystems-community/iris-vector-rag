"""The superseded entry points still work, and say so exactly once.

0.15 keeps every old name callable with its old signature, because breaking an
import is a worse migration than a warning. What each one must not do any more is
open its own connection: they all hand out the session's, so the protections apply
even to a caller that never heard of the session.

Feature: 083-iris-session (FR-015, US5)
"""

import warnings

import pytest

from iris_vector_rag.session import registry
from iris_vector_rag.testing.fake_driver import FakeDriver, fake_identity


@pytest.fixture
def substituted(monkeypatch):
    """Every `session_for` in the library answers with one fake session.

    The deprecated names must reach the session and nothing else, so the fake
    driver is the proof: a connection came back and no socket was opened.
    """
    from iris_vector_rag.session import IRISSession

    registry.reset_sessions_for_tests()
    session = IRISSession(identity=fake_identity(), driver=FakeDriver(rows=[(1,)]))
    monkeypatch.setattr(registry, "session_for", lambda *a, **k: session)
    import iris_vector_rag.session as session_package

    monkeypatch.setattr(session_package, "session_for", lambda *a, **k: session)
    yield session
    registry.reset_sessions_for_tests()


def _warnings_from(call):
    """Every warning one call raises, with the filters out of the way."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    return result, [w for w in caught if issubclass(w.category, DeprecationWarning)]


DEPRECATED_CONNECTION_CALLS = [
    (
        "common.iris_connection.get_iris_connection",
        lambda: __import__(
            "iris_vector_rag.common.iris_connection", fromlist=["x"]
        ).get_iris_connection(),
    ),
    (
        "common.iris_dbapi_connector.get_iris_dbapi_connection",
        lambda: __import__(
            "iris_vector_rag.common.iris_dbapi_connector", fromlist=["x"]
        ).get_iris_dbapi_connection(),
    ),
    (
        "common.iris_connection_manager.get_iris_dbapi_connection",
        lambda: __import__(
            "iris_vector_rag.common.iris_connection_manager", fromlist=["x"]
        ).get_iris_dbapi_connection(),
    ),
    (
        "common.iris_connection_manager.IRISConnectionManager",
        lambda: (
            __import__("iris_vector_rag.common.iris_connection_manager", fromlist=["x"])
            .IRISConnectionManager()
            .get_connection()
        ),
    ),
]


class TestOneWarningEach:
    @pytest.mark.parametrize(
        "name,call",
        DEPRECATED_CONNECTION_CALLS,
        ids=[n for n, _ in DEPRECATED_CONNECTION_CALLS],
    )
    def test_a_superseded_name_warns_once_and_still_answers(
        self, name, call, substituted
    ):
        connection, deprecations = _warnings_from(call)

        assert (
            connection is substituted.connection_for_adapter()
        ), f"{name} handed out a connection the session does not own"
        assert (
            len(deprecations) == 1
        ), f"{name} raised {len(deprecations)} deprecation warnings, want exactly 1"
        assert (
            "session" in str(deprecations[0].message).lower()
        ), f"{name} does not name the session as its replacement"

    def test_the_pool_the_api_uses_warns_once_and_serves_the_session(
        self, substituted, monkeypatch
    ):
        import iris_vector_rag.common.connection_pool as pool_module

        monkeypatch.setattr(pool_module, "session_for", lambda *a, **k: substituted)

        pool, deprecations = _warnings_from(
            lambda: pool_module.IRISConnectionPool(
                host="localhost",
                port=1972,
                namespace="USER",
                username="_SYSTEM",
                password="SYS",
            )
        )

        assert len(deprecations) == 1
        assert "session" in str(deprecations[0].message).lower()
        with pool.get_connection() as connection:
            assert connection is substituted.connection_for_adapter()

    def test_the_odbc_era_manager_warns_under_its_new_name(self, substituted):
        from iris_vector_rag.common.connection_manager import LegacyConnectionManager

        _, deprecations = _warnings_from(lambda: LegacyConnectionManager())

        assert len(deprecations) == 1
        assert "session" in str(deprecations[0].message).lower()

    @pytest.mark.parametrize(
        "module_name",
        [
            "iris_vector_rag.common.iris_connection",
            "iris_vector_rag.common.iris_connection_manager",
            "iris_vector_rag.common.iris_dbapi_connector",
            "iris_vector_rag.common.connection_pool",
            "iris_vector_rag.common.connection_manager",
        ],
    )
    def test_importing_a_deprecated_module_is_silent(self, module_name):
        """A warning at import time cannot be acted on: only the call warns.

        Run in a fresh interpreter with deprecations promoted to errors, because in
        this one the module is imported already and would never warn again.
        """
        import subprocess
        import sys

        finished = subprocess.run(
            [
                sys.executable,
                "-W",
                "error::DeprecationWarning",
                "-c",
                f"import {module_name}",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert (
            finished.returncode == 0
        ), f"importing {module_name} warned or failed:\n{finished.stderr}"


class TestTheSeamThatSurvives:
    """`core.connection.ConnectionManager` is the library's own seam, not a rival."""

    def test_the_storage_facing_manager_does_not_warn(self, substituted):
        from iris_vector_rag.core.connection import ConnectionManager

        manager, deprecations = _warnings_from(lambda: ConnectionManager())

        assert deprecations == [], (
            "core.connection.ConnectionManager is what storage and the pipelines "
            "are handed; warning on it would fire from library internals a caller "
            "cannot change"
        )
        assert manager.get_connection() is substituted.connection_for_adapter()


class TestNoInternalCallerWarns:
    """A warning a caller cannot act on is noise, so the library never raises one.

    Every deprecated name has a twin that does the same work silently. The library's
    own modules call the twin; only code outside the package reaches the warning.
    """

    # The names whose call emits the warning, and the modules allowed to call them.
    WARNING_CALLS = {
        "get_iris_connection",
        "get_iris_dbapi_connection",
        "IRISConnectionManager",
        "IRISConnectionPool",
        "LegacyConnectionManager",
    }
    DEFINING_MODULES = {
        "iris_vector_rag/common/iris_connection.py",
        "iris_vector_rag/common/iris_connection_manager.py",
        "iris_vector_rag/common/iris_dbapi_connector.py",
        "iris_vector_rag/common/connection_pool.py",
        "iris_vector_rag/common/connection_manager.py",
    }

    def test_no_module_in_the_library_calls_a_deprecated_name(self):
        import ast
        import pathlib

        package = pathlib.Path(__import__("iris_vector_rag").__file__).parent
        root = package.parent
        offenders = []
        for path in sorted(package.rglob("*.py")):
            relative = path.relative_to(root).as_posix()
            if relative in self.DEFINING_MODULES:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                called = getattr(function, "id", None) or getattr(
                    function, "attr", None
                )
                if called in self.WARNING_CALLS:
                    offenders.append(f"{relative}:{node.lineno} calls {called}()")

        assert offenders == [], (
            "these library modules would raise a DeprecationWarning their callers "
            "cannot act on; call the non-warning twin instead:\n  "
            + "\n  ".join(offenders)
        )

    def test_the_twins_do_the_same_work_without_warning(self, substituted):
        from iris_vector_rag.common import iris_connection, iris_dbapi_connector

        silent_calls = [
            lambda: iris_connection._session_connection(None, None, None, None, None),
            lambda: iris_dbapi_connector._session_connection(),
        ]
        for call in silent_calls:
            connection, deprecations = _warnings_from(call)
            assert connection is substituted.connection_for_adapter()
            assert deprecations == []

    def test_the_pool_the_api_builds_is_silent(self, substituted, monkeypatch):
        import iris_vector_rag.common.connection_pool as pool_module

        monkeypatch.setattr(pool_module, "session_for", lambda *a, **k: substituted)

        pool, deprecations = _warnings_from(
            lambda: pool_module._session_pool(
                host="localhost",
                port=1972,
                namespace="USER",
                username="_SYSTEM",
                password="SYS",
                pool_size=4,
            )
        )

        assert deprecations == []
        assert isinstance(pool, pool_module.IRISConnectionPool)
        assert pool.get_stats()["pool_size"] == 4
        with pool.get_connection() as connection:
            assert connection is substituted.connection_for_adapter()


class TestBehaviourIsUnchanged:
    def test_explicit_parameters_still_reach_the_identity(self, monkeypatch):
        """The old signature still decides which instance is meant (FR-015)."""
        from iris_vector_rag.common import iris_connection

        seen = {}

        def capture(*args, **kwargs):
            seen.update(kwargs)
            return fake_session_stub()

        def fake_session_stub():
            from iris_vector_rag.session import IRISSession

            return IRISSession(identity=fake_identity(), driver=FakeDriver())

        import iris_vector_rag.session as session_package

        monkeypatch.setattr(session_package, "session_for", capture)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iris_connection.get_iris_connection(
                host="db.example.com",
                port=1234,
                namespace="RAGTEST",
                username="someone",
                password="secret",
            )

        identity = seen["identity"]
        assert (identity.host, identity.port) == ("db.example.com", 1234)
        assert (identity.namespace, identity.username) == ("RAGTEST", "someone")
        assert identity.password == "secret"

    def test_invalid_parameters_still_fail_before_any_driver_work(self):
        """Validation errors keep their attributes, and no warning replaces them."""
        from iris_vector_rag.common.exceptions import ValidationError
        from iris_vector_rag.common.iris_connection import get_iris_connection

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValidationError) as caught:
                get_iris_connection(
                    host="localhost",
                    port=99999,
                    namespace="USER",
                    username="_SYSTEM",
                    password="SYS",
                )

        assert caught.value.parameter_name == "port"
        assert "65535" in caught.value.valid_range
