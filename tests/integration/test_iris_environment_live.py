"""E2E gate for test-environment: `iris_connection` is a real IRIS, both ways."""

import os
from pathlib import Path

from tests.support.iris_environment import (
    connect,
    declared_container,
    resolve_pinned,
    start_ephemeral,
)

REPO = Path(__file__).resolve().parents[2]


def _scalar(conn, sql):
    cursor = conn.cursor()
    cursor.execute(sql)
    value = cursor.fetchone()[0]
    cursor.close()
    return value


def test_iris_connection_is_a_live_database(iris_connection, iris_target):
    assert _scalar(iris_connection, "SELECT 1") == 1
    assert _scalar(iris_connection, "SELECT $ZVERSION").startswith("IRIS")
    assert not hasattr(iris_connection, "_source_documents")  # the old mock's state


def test_pinned_runs_use_only_the_declared_container(iris_target):
    if not iris_target.ephemeral:
        assert iris_target.container in (declared_container(REPO), "iris")


def test_an_ephemeral_container_starts_and_answers():
    container, target = start_ephemeral()
    try:
        assert target.ephemeral is True
        conn = connect(target)
        try:
            assert _scalar(conn, "SELECT 1") == 1
        finally:
            conn.close()
    finally:
        container.stop()


def test_resolve_pinned_matches_the_environment_this_run_uses():
    pinned = resolve_pinned(os.environ, declared=declared_container(REPO))
    assert (pinned is None) == (not os.environ.get("IRIS_CONTAINER_NAME"))
