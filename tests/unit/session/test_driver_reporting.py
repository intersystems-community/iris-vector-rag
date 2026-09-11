"""Which driver was chosen, and why, is on the record.

Driver selection used to be import-time `try/except` in several modules: which one
a process ended up with depended on import order and nothing said so.

Feature: 083-iris-session (FR-006, FR-007, US4)
"""

import logging

import pytest

from iris_vector_rag.session import IRISSession
from iris_vector_rag.session.drivers import EmbeddedDriver, NetworkDriver, select_driver
from iris_vector_rag.session.errors import NoUsableDriver
from iris_vector_rag.testing.fake_driver import FakeDriver, fake_identity


@pytest.fixture
def availability(monkeypatch):
    """Decide what this process can use, without touching a driver."""

    def decide(embedded: bool, network: bool):
        monkeypatch.setattr(EmbeddedDriver, "available", lambda self: embedded)
        monkeypatch.setattr(NetworkDriver, "available", lambda self: network)

    return decide


def test_the_choice_names_the_driver_and_the_instance(availability, caplog):
    availability(embedded=False, network=True)

    with caplog.at_level(logging.INFO, logger="iris_vector_rag.session.drivers"):
        driver = select_driver(fake_identity())

    assert driver.name == "network"
    assert "network" in caplog.text
    assert "fake-host:51972/USER" in caplog.text


def test_nothing_usable_says_what_was_tried_and_why(availability):
    availability(embedded=False, network=False)

    with pytest.raises(NoUsableDriver) as caught:
        select_driver(fake_identity())

    message = str(caught.value)
    assert "embedded" in message
    assert "network" in message
    assert "fake-host:51972/USER" in message


def test_the_reason_a_driver_was_rejected_is_the_drivers_own(availability):
    availability(embedded=False, network=False)

    with pytest.raises(NoUsableDriver) as caught:
        select_driver(fake_identity())

    # Not a generic "unavailable": the reason comes from unavailable_because().
    assert "(" in str(caught.value) and ")" in str(caught.value)


def test_a_substituted_driver_means_no_selection_at_all(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a session given a driver must not choose one")

    monkeypatch.setattr("iris_vector_rag.session.drivers.select_driver", refuse)

    session = IRISSession(identity=fake_identity(), driver=FakeDriver(rows=[(1,)]))

    assert session.driver_name == "fake"
    assert session.one("SELECT 1") == (1,)


def test_the_dbapi_entry_point_hands_out_the_sessions_connection(monkeypatch):
    from iris_vector_rag.common import iris_dbapi_connector
    from iris_vector_rag.testing.fake_driver import fake_session

    session = fake_session()
    monkeypatch.setattr("iris_vector_rag.session.session_for", lambda *a, **k: session)

    connection = iris_dbapi_connector.get_iris_dbapi_connection()

    assert connection is session.connection_for_adapter()
    assert session.driver.connects == 1


def test_the_dbapi_module_is_not_imported_at_import_time():
    from iris_vector_rag.common import iris_dbapi_connector

    # Importing a connector module must not decide anything about drivers.
    assert "irisdbapi" not in vars(iris_dbapi_connector)
    assert callable(iris_dbapi_connector.get_iris_dbapi_module)
