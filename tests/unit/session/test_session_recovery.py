"""An expired default password is recognised and reported, never "fixed".

The library does not run ``docker exec`` or ``UnExpireUserPasswords``: a container that
requires a password change is configured (``ChangePassword=0``) or created by
iris-devtester, whose container factory owns that story.

Feature: 083-iris-session (FR-005, US1); G11 of the 2026-10 remediation review
"""

import pytest

from iris_vector_rag.session.drivers import Driver, NetworkDriver
from iris_vector_rag.session.errors import PasswordExpired
from iris_vector_rag.testing.fake_driver import FakeDriver, fake_identity, fake_session


def test_an_expired_password_raises_the_drivers_error_with_a_hint():
    session = fake_session(expired_password=True, rows=[(1,)])

    with pytest.raises(PasswordExpired) as caught:
        session.execute("SELECT 1")

    message = str(caught.value)
    assert "fake-host:51972/USER" in message
    assert "Password change required for user" in message, "the driver's own error"
    assert "ChangePassword=0" in message, "how to configure the container"
    assert "iris-devtester" in message, "who owns creating a container"
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_an_expired_password_is_not_retried_and_opens_no_backoff():
    session = fake_session(expired_password=True)

    with pytest.raises(PasswordExpired):
        session.execute("SELECT 1")
    with pytest.raises(PasswordExpired):
        session.execute("SELECT 1")

    assert session.driver.connects == 2, "one connect per call, no recovery retry"
    assert PasswordExpired.opens_backoff is False


def test_the_detection_is_kept_on_the_network_driver():
    driver = NetworkDriver()

    assert driver.password_expired(RuntimeError("Password change required")) is True
    assert driver.password_expired(RuntimeError("Access Denied")) is False


def test_no_password_recovery_remains_in_the_library():
    assert not hasattr(Driver, "recover_password")
    assert not hasattr(NetworkDriver, "recover_password")
    assert not hasattr(NetworkDriver, "_container_publishing")
    assert not hasattr(FakeDriver, "recover_password")


def test_the_identity_answers_without_connecting():
    session = fake_session()

    assert session.identity.label == "fake-host:51972/USER"
    assert session.driver_name == "fake"
    assert session.driver.connects == 0
