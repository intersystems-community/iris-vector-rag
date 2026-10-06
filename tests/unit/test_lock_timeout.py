"""A statement that waits on a lock gives up; a test session cannot hang on one.

Test connections (tests.support.iris_environment.connect) and the library's own network
connections (when IRIS_LOCK_TIMEOUT is set, as the test session sets it) run
`SET OPTION LOCK_TIMEOUT=n` right after they open, so the server abandons a lock wait
with SQLCODE -110 instead of leaving the caller in a driver call that no signal can break.
"""

import pytest

from iris_vector_rag.session.drivers import NetworkDriver, lock_timeout_statement
from iris_vector_rag.session.identity import SessionIdentity
from tests.support.iris_environment import IrisTarget, connect


class Conn:
    def __init__(self, fail_on=None):
        self.statements = []
        self.fail_on = fail_on

    def cursor(self):
        outer = self

        class Cur:
            def execute(self, sql, *a):
                outer.statements.append(sql)
                if outer.fail_on and outer.fail_on in sql:
                    raise RuntimeError("refused")

            def fetchone(self):
                return (1,)

            def close(self):
                pass

        return Cur()

    def close(self):
        pass


TARGET = IrisTarget("c", "localhost", 1, "USER", "_SYSTEM", "SYS", True)


class TestTestConnections:
    def test_connect_sets_a_30_second_lock_timeout_by_default(self):
        conn = Conn()
        connect(TARGET, connector=lambda t: conn, attempts=1, delay=0)
        assert conn.statements == ["SELECT 1", "SET OPTION LOCK_TIMEOUT=30"]

    def test_the_timeout_can_be_chosen(self):
        conn = Conn()
        connect(TARGET, connector=lambda t: conn, attempts=1, delay=0, lock_timeout=2)
        assert conn.statements[-1] == "SET OPTION LOCK_TIMEOUT=2"

    def test_a_server_that_refuses_the_option_is_an_unreachable_iris_not_a_silent_skip(
        self,
    ):
        from tests.support.iris_environment import IrisUnreachable

        with pytest.raises(IrisUnreachable):
            connect(
                TARGET,
                connector=lambda t: Conn(fail_on="LOCK_TIMEOUT"),
                attempts=1,
                delay=0,
            )


class TestLibraryConnections:
    def test_statement_is_none_unless_asked_for(self, monkeypatch):
        monkeypatch.delenv("IRIS_LOCK_TIMEOUT", raising=False)
        assert lock_timeout_statement() is None

    def test_statement_follows_the_environment(self, monkeypatch):
        monkeypatch.setenv("IRIS_LOCK_TIMEOUT", "30")
        assert lock_timeout_statement() == "SET OPTION LOCK_TIMEOUT=30"

    @pytest.mark.parametrize("bad", ["", "abc", "0", "-5"])
    def test_a_nonsense_value_is_ignored(self, monkeypatch, bad):
        monkeypatch.setenv("IRIS_LOCK_TIMEOUT", bad)
        assert lock_timeout_statement() is None

    def test_network_driver_applies_it_to_the_connection_it_opens(self, monkeypatch):
        monkeypatch.setenv("IRIS_LOCK_TIMEOUT", "30")
        conn = Conn()
        driver = NetworkDriver()
        monkeypatch.setattr(driver, "_precheck", lambda identity: None)
        monkeypatch.setattr(driver, "_raw_connect", lambda identity: conn)
        driver.connect(SessionIdentity.from_env())
        assert conn.statements == ["SET OPTION LOCK_TIMEOUT=30"]

    def test_network_driver_leaves_the_server_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("IRIS_LOCK_TIMEOUT", raising=False)
        conn = Conn()
        driver = NetworkDriver()
        monkeypatch.setattr(driver, "_precheck", lambda identity: None)
        monkeypatch.setattr(driver, "_raw_connect", lambda identity: conn)
        driver.connect(SessionIdentity.from_env())
        assert conn.statements == []
