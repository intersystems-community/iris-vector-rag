"""Which driver a process uses is a decision that gets logged, not an import accident.

Feature: 083-iris-session (FR-006, FR-007, US4)
"""

import logging

import pytest

from iris_vector_rag.session import SessionIdentity
from iris_vector_rag.session.drivers import (
    Driver,
    EmbeddedDriver,
    NetworkDriver,
    select_driver,
)
from iris_vector_rag.session.errors import NoUsableDriver

IDENTITY = SessionIdentity(
    host="localhost",
    port=51972,
    namespace="USER",
    username="_SYSTEM",
    password="SYS",
)


@pytest.fixture
def availability(monkeypatch):
    def set_it(embedded: bool, network: bool):
        monkeypatch.setattr(EmbeddedDriver, "available", lambda self: embedded)
        monkeypatch.setattr(NetworkDriver, "available", lambda self: network)

    return set_it


def test_the_embedded_driver_wins_when_it_is_usable(availability, caplog):
    availability(embedded=True, network=True)

    with caplog.at_level(logging.INFO):
        driver = select_driver(IDENTITY)

    assert driver.name == "embedded"
    assert "embedded" in caplog.text


def test_the_network_driver_is_used_when_the_embedded_one_is_not_available(
    availability, caplog
):
    availability(embedded=False, network=True)

    with caplog.at_level(logging.INFO):
        driver = select_driver(IDENTITY)

    assert driver.name == "network"
    assert "network" in caplog.text


def test_no_usable_driver_names_everything_that_was_tried(availability):
    availability(embedded=False, network=False)

    with pytest.raises(NoUsableDriver) as caught:
        select_driver(IDENTITY)

    message = str(caught.value)
    assert "embedded" in message and "network" in message
    assert "localhost:51972/USER" in message


def test_preferring_the_network_driver_is_possible(availability):
    availability(embedded=True, network=True)

    assert select_driver(IDENTITY, prefer_embedded=False).name == "network"


def test_availability_never_opens_a_connection(monkeypatch):
    # available() answering by connecting would defeat the point of asking.
    def explode(*args, **kwargs):
        raise AssertionError("available() must not connect")

    monkeypatch.setattr(EmbeddedDriver, "connect", explode)
    monkeypatch.setattr(NetworkDriver, "connect", explode)

    EmbeddedDriver().available()
    NetworkDriver().available()


def test_the_real_drivers_satisfy_the_protocol():
    assert isinstance(EmbeddedDriver(), Driver)
    assert isinstance(NetworkDriver(), Driver)


def test_the_fake_driver_satisfies_the_protocol_without_importing_a_real_one():
    from tests.support.fake_driver import FakeDriver

    assert isinstance(FakeDriver(), Driver)


class TestADriverKnowsItsOwnClosedConnections:
    """Whether a handle is closed is a local fact, so asking costs no round trip."""

    def test_the_network_driver_reports_a_closed_connection(self):
        class Handle:
            def __init__(self, closed):
                self._closed = closed

            def isClosed(self):  # the name the IRIS DBAPI driver uses
                return self._closed

        driver = NetworkDriver()

        assert driver.is_closed(Handle(True)) is True
        assert driver.is_closed(Handle(False)) is False

    def test_a_handle_that_cannot_say_is_treated_as_open(self):
        """A driver that offers no answer must not be assumed dead."""

        class Silent:
            pass

        assert NetworkDriver().is_closed(Silent()) is False
        assert EmbeddedDriver().is_closed(Silent()) is False
