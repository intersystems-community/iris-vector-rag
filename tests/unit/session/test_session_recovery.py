"""An expired default password is recovered once, and reported.

Feature: 083-iris-session (FR-005, US1)
"""

import logging

import pytest

from iris_vector_rag.session import IRISSession
from iris_vector_rag.session.errors import PasswordExpired
from iris_vector_rag.testing.fake_driver import FakeDriver, fake_identity, fake_session


def test_an_expired_password_is_recovered_and_the_connect_retried(caplog):
    session = fake_session(expired_password=True, rows=[(1,)])

    with caplog.at_level(logging.INFO):
        session.execute("SELECT 1")

    assert session.driver.recoveries == 1
    assert session.driver.connects == 2
    assert "password" in caplog.text.lower()


def test_recovery_is_attempted_once_per_session():
    class ClaimsSuccessButChangesNothing(FakeDriver):
        def recover_password(self, identity):
            self.recoveries += 1
            return True  # the password is still expired afterwards

    driver = ClaimsSuccessButChangesNothing(expired_password=True)
    session = IRISSession(identity=fake_identity(), driver=driver)

    with pytest.raises(PasswordExpired):
        session.execute("SELECT 1")
    with pytest.raises(PasswordExpired):
        session.execute("SELECT 1")

    assert driver.recoveries == 1


def test_a_driver_that_cannot_recover_reports_the_expiry():
    driver = FakeDriver(expired_password=True)
    driver.recover_password = lambda identity: False
    session = IRISSession(identity=fake_identity(), driver=driver)

    with pytest.raises(PasswordExpired) as caught:
        session.execute("SELECT 1")

    assert "fake-host:51972/USER" in str(caught.value)


def test_the_identity_answers_without_connecting():
    session = fake_session()

    assert session.identity.label == "fake-host:51972/USER"
    assert session.driver_name == "fake"
    assert session.driver.connects == 0
