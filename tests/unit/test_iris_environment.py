"""The `iris_connection` environment: pinned project container or an ephemeral one.

Decision record: docs/remediation/2026-10-review/tickets/G9-verify-fixes-needing-iris.md.
No silent fallback, no mock, no skip: an unreachable IRIS fails loudly with a hint.
"""

from pathlib import Path

import pytest

from tests.support import iris_environment as env_mod
from tests.support.iris_environment import (
    ForeignContainerError,
    IrisTarget,
    IrisUnreachable,
    connect,
    declared_container,
    resolve_pinned,
)

TESTS = Path(__file__).resolve().parents[1]
DECLARED = "iris-vector-rag-iris"


def _docker_port(mapping):
    return lambda name: mapping.get(name)


class TestDeclaration:
    def test_reads_the_container_from_claude_md(self, tmp_path):
        (tmp_path / "CLAUDE.md").write_text(
            "text\nIRIS_CONTAINER: my-iris\nIRIS_PORT: 4242\n"
        )
        assert declared_container(tmp_path) == "my-iris"

    def test_this_repository_declares_its_container(self):
        assert declared_container(TESTS.parent) == DECLARED

    def test_missing_declaration_is_an_error_not_a_default(self, tmp_path):
        with pytest.raises(IrisUnreachable, match="IRIS_CONTAINER"):
            declared_container(tmp_path)


class TestResolvePinned:
    def test_unset_means_start_an_ephemeral_container(self):
        assert (
            resolve_pinned({}, declared=DECLARED, docker_port=_docker_port({})) is None
        )
        assert (
            resolve_pinned(
                {"IRIS_CONTAINER_NAME": ""},
                declared=DECLARED,
                docker_port=_docker_port({}),
            )
            is None
        )

    def test_pinned_project_container_uses_the_port_docker_reports(self):
        target = resolve_pinned(
            {"IRIS_CONTAINER_NAME": DECLARED},
            declared=DECLARED,
            docker_port=_docker_port({DECLARED: 51972}),
        )
        assert isinstance(target, IrisTarget)
        assert (target.container, target.host, target.port) == (
            DECLARED,
            "localhost",
            51972,
        )
        assert target.ephemeral is False
        assert (target.username, target.password, target.namespace) == (
            "_SYSTEM",
            "SYS",
            "USER",
        )

    def test_credentials_follow_the_devtester_variable_names(self):
        target = resolve_pinned(
            {
                "IRIS_CONTAINER_NAME": DECLARED,
                "IRIS_USERNAME": "SuperUser",
                "IRIS_PASSWORD": "pw",
                "IRIS_NAMESPACE": "RAG",
            },
            declared=DECLARED,
            docker_port=_docker_port({DECLARED: 51972}),
        )
        assert (target.username, target.password, target.namespace) == (
            "SuperUser",
            "pw",
            "RAG",
        )

    @pytest.mark.parametrize("other", ["los-iris", "opsreview-iris", "aihub-iris-116"])
    def test_another_projects_container_is_refused(self, other):
        with pytest.raises(ForeignContainerError) as err:
            resolve_pinned(
                {"IRIS_CONTAINER_NAME": other},
                declared=DECLARED,
                docker_port=_docker_port({other: 11972}),
            )
        assert other in str(err.value) and DECLARED in str(err.value)

    def test_ci_may_pin_its_own_throwaway_container(self):
        target = resolve_pinned(
            {"IRIS_CONTAINER_NAME": "iris", "CI": "true"},
            declared=DECLARED,
            docker_port=_docker_port({"iris": 1972}),
        )
        assert (target.container, target.port) == ("iris", 1972)

    def test_port_falls_back_to_iris_port_when_docker_cannot_say(self):
        target = resolve_pinned(
            {"IRIS_CONTAINER_NAME": "iris", "CI": "true", "IRIS_PORT": "1972"},
            declared=DECLARED,
            docker_port=_docker_port({}),
        )
        assert target.port == 1972

    def test_no_port_anywhere_fails_with_the_reset_hint(self):
        with pytest.raises(IrisUnreachable, match="make iris-reset"):
            resolve_pinned(
                {"IRIS_CONTAINER_NAME": DECLARED},
                declared=DECLARED,
                docker_port=_docker_port({}),
            )


class TestConnect:
    TARGET = IrisTarget(DECLARED, "localhost", 51972, "USER", "_SYSTEM", "SYS", False)

    class Conn:
        def __init__(self):
            self.statements = []

        def cursor(self):
            outer = self

            class Cur:
                def execute(self, sql, *a):
                    outer.statements.append(sql)

                def fetchone(self):
                    return (1,)

                def close(self):
                    pass

            return Cur()

        def close(self):
            pass

    def test_warm_up_runs_a_query_and_returns_the_real_connection(self):
        conn = self.Conn()
        got = connect(self.TARGET, connector=lambda t: conn, attempts=1, delay=0)
        assert got is conn
        assert conn.statements == ["SELECT 1"]

    def test_warm_up_retries_a_wedged_server(self):
        calls = []

        def flaky(target):
            calls.append(1)
            if len(calls) < 3:
                raise OSError("connection reset")
            return self.Conn()

        connect(self.TARGET, connector=flaky, attempts=5, delay=0)
        assert len(calls) == 3

    def test_unreachable_pinned_container_fails_loudly_with_the_hint(self):
        def dead(target):
            raise OSError("refused")

        with pytest.raises(IrisUnreachable) as err:
            connect(self.TARGET, connector=dead, attempts=2, delay=0)
        message = str(err.value)
        assert "make iris-reset" in message
        assert DECLARED in message and "51972" in message
        assert "refused" in message

    def test_unreachable_ephemeral_container_does_not_suggest_reset(self):
        ephemeral = IrisTarget(
            "tmp", "localhost", 40000, "USER", "_SYSTEM", "SYS", True
        )
        with pytest.raises(IrisUnreachable) as err:
            connect(
                ephemeral,
                connector=lambda t: (_ for _ in ()).throw(OSError("x")),
                attempts=1,
                delay=0,
            )
        assert "make iris-reset" not in str(err.value)


class TestNoMocks:
    def test_the_fixture_does_not_route_through_the_mock_pool(self):
        source = (TESTS / "conftest.py").read_text()
        body = source.split("def iris_connection", 1)[1].split("\ndef ", 1)[0]
        assert "connection_pool" not in body
        assert "Mock" not in body

    def test_the_pool_no_longer_fabricates_a_database(self):
        source = (TESTS / "support" / "connection_pool.py").read_text()
        assert "MockConnection" not in source
        assert "MockCursor" not in source

    def test_module_exposes_no_skip_or_mock_fallback(self):
        source = Path(env_mod.__file__).read_text()
        assert "pytest.skip" not in source
        assert "Mock(" not in source


class TestExportedEnvironment:
    """Code under test resolves its own connection from IRIS_* variables; they must agree."""

    def test_target_exports_the_devtester_variable_names(self):
        target = IrisTarget("c", "localhost", 51972, "RAG", "SuperUser", "pw", False)
        assert target.environment() == {
            "IRIS_HOST": "localhost",
            "IRIS_PORT": "51972",
            "IRIS_USERNAME": "SuperUser",
            "IRIS_PASSWORD": "pw",
            "IRIS_NAMESPACE": "RAG",
        }

    def test_the_session_fixture_points_the_code_under_test_at_the_target(self):
        source = (TESTS / "conftest.py").read_text()
        body = source.split("def iris_target", 1)[1].split("\ndef iris_connection", 1)[
            0
        ]
        assert "environment()" in body


class TestWarmUpCommand:
    """`python -m tests.support.iris_environment warm-up`, the last step of make iris-reset."""

    def test_warm_up_succeeds_against_a_live_pinned_container(self, capsys):
        conn = TestConnect.Conn()
        code = env_mod.main(
            ["warm-up"],
            env={"IRIS_CONTAINER_NAME": DECLARED},
            declared=DECLARED,
            docker_port=_docker_port({DECLARED: 51972}),
            connector=lambda t: conn,
        )
        assert code == 0
        assert "51972" in capsys.readouterr().out

    def test_warm_up_without_a_pinned_container_is_an_error(self, capsys):
        code = env_mod.main(
            ["warm-up"], env={}, declared=DECLARED, docker_port=_docker_port({})
        )
        assert code == 2
        assert "IRIS_CONTAINER_NAME" in capsys.readouterr().err

    def test_warm_up_failure_exits_nonzero_with_the_hint(self, capsys):
        def dead(target):
            raise OSError("refused")

        code = env_mod.main(
            ["warm-up"],
            env={"IRIS_CONTAINER_NAME": DECLARED},
            declared=DECLARED,
            docker_port=_docker_port({DECLARED: 51972}),
            connector=dead,
            attempts=1,
            delay=0,
        )
        assert code == 1
        assert "make iris-reset" in capsys.readouterr().err


class TestEnvironmentIsAutomaticWhereIrisIsNeeded:
    """A contract/integration/e2e test file run on its own must see the same IRIS.

    Without this, code under test fell back to localhost:1972 unless an earlier test
    had happened to request iris_target.
    """

    def _conftest(self):
        return (TESTS / "conftest.py").read_text()

    def test_an_autouse_fixture_activates_iris_target_for_iris_directories(self):
        source = self._conftest()
        assert "def _iris_environment" in source
        body = source.split("def _iris_environment", 1)[1].split("\n\n\n", 1)[0]
        assert 'getfixturevalue("iris_target")' in body
        for directory in ("integration", "e2e", "contract"):
            assert f'"{directory}"' in body

    def test_unit_tests_do_not_need_docker_or_iris(self):
        body = (
            self._conftest().split("def _iris_environment", 1)[1].split("\n\n\n", 1)[0]
        )
        assert '"unit"' not in body
