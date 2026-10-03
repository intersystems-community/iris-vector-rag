"""The engine and the pipelines share one session, and report real numbers.

Feature: 083-iris-session (FR-004, FR-014, US1)
"""

from unittest.mock import MagicMock

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.engine import IRISVectorEngine
from iris_vector_rag.testing.fake_driver import FakeConnection, FakeDriver, fake_session


@pytest.fixture
def registered(monkeypatch):
    """Every `session_for` in the process answers with one fake session."""
    session = fake_session(rows=[(1,)], health_interval=3600)
    monkeypatch.setattr(
        "iris_vector_rag.session.session_for", lambda *args, **kwargs: session
    )
    return session


def test_the_engine_shares_the_managers_session(registered):
    engine = IRISVectorEngine(ConnectionManager(config_manager=ConfigurationManager()))

    assert engine.session is registered
    assert engine.connection is registered.connection_for_adapter()
    assert registered.driver.connects == 1  # one connection, not one per caller


def test_repeated_asks_do_not_open_a_second_connection(registered):
    engine = IRISVectorEngine(ConnectionManager(config_manager=ConfigurationManager()))

    for _ in range(10):
        engine.connection
        engine.session

    assert registered.driver.connects == 1


def test_a_caller_supplied_connection_is_borrowed_never_closed():
    connection = FakeConnection(FakeDriver())
    engine = IRISVectorEngine(connection, config_manager=ConfigurationManager())

    session = engine.session

    assert session.owns_connection is False
    assert engine.connection is connection
    session.close()
    assert connection.closed is False


def test_the_embedding_dimension_is_the_configured_one_not_a_guess():
    # A project configured for 384 that reported 1536 built a table it could not
    # query (083 research, FR-014).
    engine = IRISVectorEngine(
        FakeConnection(FakeDriver()), config_manager=ConfigurationManager()
    )
    engine._vector_store = object()  # no schema manager to ask

    assert engine.embedding_dimension == 384


def test_the_schema_is_asked_before_the_configuration():
    engine = IRISVectorEngine(
        FakeConnection(FakeDriver()), config_manager=ConfigurationManager()
    )
    store = MagicMock()
    store.schema_manager.get_embedding_dimension.return_value = 768
    engine._vector_store = store

    assert engine.embedding_dimension == 768


def test_the_retrieval_engine_is_handed_a_connection_that_exists(registered):
    # Until 0.15 this read `connection_manager.connection` -- an attribute the
    # manager never had -- so every non-vector retrieval mode ran with None.
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline

    manager = ConnectionManager(config_manager=ConfigurationManager())
    assert getattr(manager, "connection", None) is None

    pipeline = BasicRAGPipeline.__new__(BasicRAGPipeline)
    pipeline.connection_manager = manager

    assert pipeline._retrieval_connection() is registered.connection_for_adapter()


def test_no_connection_manager_means_no_connection():
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline

    pipeline = BasicRAGPipeline.__new__(BasicRAGPipeline)
    pipeline.connection_manager = None

    assert pipeline._retrieval_connection() is None
