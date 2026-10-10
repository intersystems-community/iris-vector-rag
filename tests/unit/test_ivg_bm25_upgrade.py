"""Upgrading BM25 indexes to ivg 4.2.0's schema 2.

ivg 4.2.0's `Insert` and `Delete` refuse an index built earlier until
`engine.bm25_upgrade(name)` has run once; `Search` keeps working. IVR reports what
it would do (dry run), upgrades only what is still schema 1, and can be run again.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from iris_vector_rag.integrations.ivg_bm25 import (
    Bm25IndexResult,
    Bm25UpgradeReport,
    upgrade_bm25_indexes,
)


class FakeEngine:
    """The slice of IRISGraphEngine the upgrade uses, with BM25 state in a dict."""

    def __init__(self, indexes, other_indexes=("vec_tokens",), fail_on=()):
        self.indexes = {name: dict(info) for name, info in indexes.items()}
        self.other = tuple(other_indexes)
        self.fail_on = set(fail_on)
        self.upgrade_calls = []

    def list_indexes(self):
        listed = [SimpleNamespace(name=n, type="fulltext") for n in self.indexes]
        listed += [SimpleNamespace(name=n, type="vector") for n in self.other]
        return sorted(listed, key=lambda index: index.name)

    def bm25_info(self, name):
        info = self.indexes.get(name)
        if info is None:
            # ivg's wrapper adds "type" to the class's "{}" for an index that is not there.
            return {"type": "bm25"}
        return {"N": info["N"], "schema": info.get("schema", 1), "type": "bm25"}

    def bm25_upgrade(self, name):
        self.upgrade_calls.append(name)
        if name not in self.indexes:
            return {"upgraded": 0, "missing": 1}
        if name in self.fail_on:
            raise RuntimeError(f"BM25 index '{name}' is locked by a writer")
        info = self.indexes[name]
        if info.get("schema", 1) >= 2:
            return {"upgraded": 0, "already": 1}
        info["schema"] = 2
        return {"upgraded": 1, "docs": info["N"], "vocab": 9, "postings": 40}


def _old(n=5):
    return {"N": n}


def _current(n=5):
    return {"N": n, "schema": 2}


def test_it_upgrades_every_fulltext_index_and_only_those():
    engine = FakeEngine({"logs": _old(), "notes": _old(3)})

    report = upgrade_bm25_indexes(engine)

    assert engine.upgrade_calls == ["logs", "notes"]  # not the vector index
    assert report.names == ("logs", "notes")
    assert report.upgraded == ("logs", "notes")
    assert report["logs"].docs == 5
    assert report["logs"].postings == 40
    assert report.ok


def test_an_index_already_in_schema_2_is_left_alone():
    engine = FakeEngine({"fresh": _current(), "logs": _old()})

    report = upgrade_bm25_indexes(engine)

    assert engine.upgrade_calls == ["logs"]
    assert report.current == ("fresh",)
    assert report["fresh"].action == "already_current"
    assert report["fresh"].schema == 2


def test_running_it_twice_changes_nothing_the_second_time():
    engine = FakeEngine({"logs": _old()})

    first = upgrade_bm25_indexes(engine)
    second = upgrade_bm25_indexes(engine)

    assert first.upgraded == ("logs",)
    assert second.upgraded == ()
    assert second.current == ("logs",)
    assert engine.upgrade_calls == ["logs"]


def test_a_dry_run_reports_and_writes_nothing():
    engine = FakeEngine({"fresh": _current(), "logs": _old(7)})

    report = upgrade_bm25_indexes(engine, dry_run=True)

    assert engine.upgrade_calls == []
    assert engine.indexes["logs"].get("schema", 1) == 1
    assert report.dry_run is True
    assert report.pending == ("logs",)
    assert report["logs"].action == "would_upgrade"
    assert report["logs"].docs == 7
    assert report.current == ("fresh",)
    assert report.upgraded == ()
    assert report.ok


def test_named_indexes_only():
    engine = FakeEngine({"a": _old(), "b": _old()})

    report = upgrade_bm25_indexes(engine, names=["b"])

    assert engine.upgrade_calls == ["b"]
    assert report.names == ("b",)


def test_a_name_with_no_index_is_reported_missing_not_created():
    engine = FakeEngine({"a": _old()})

    report = upgrade_bm25_indexes(engine, names=["nope", "a"])

    assert report.missing == ("nope",)
    assert report["nope"].schema is None
    assert report["a"].action == "upgraded"
    assert engine.upgrade_calls == ["a"]
    assert report.ok  # nothing to upgrade is not a failure


def test_an_index_that_vanishes_before_the_upgrade_is_missing_not_upgraded():
    class Racing(FakeEngine):
        def bm25_info(self, name):
            return {"N": 4, "schema": 1, "type": "bm25"}  # there when read ...

    engine = Racing({})  # ... gone when upgraded

    report = upgrade_bm25_indexes(engine, names=["gone"])

    assert report.missing == ("gone",)
    assert report.upgraded == ()


def test_one_failure_is_recorded_and_the_rest_still_run():
    engine = FakeEngine({"a": _old(), "b": _old(), "c": _old()}, fail_on={"b"})

    report = upgrade_bm25_indexes(engine)

    assert report.failed == ("b",)
    assert "locked by a writer" in report["b"].error
    assert report.upgraded == ("a", "c")
    assert not report.ok


def test_no_fulltext_index_is_an_empty_report():
    report = upgrade_bm25_indexes(FakeEngine({}))

    assert report.results == ()
    assert report.ok


def test_the_report_names_the_call_that_fixes_a_schema_1_index():
    report = upgrade_bm25_indexes(FakeEngine({"logs": _old()}), dry_run=True)

    assert "bm25_upgrade('logs')" in report["logs"].hint


def test_an_engine_with_no_bm25_upgrade_is_an_error_that_says_so():
    class Old:
        def list_indexes(self):
            return []

    with pytest.raises(RuntimeError, match=r"iris-vector-graph>=4\.2\.0"):
        upgrade_bm25_indexes(Old(), names=["x"])


def test_result_types_are_immutable():
    report = upgrade_bm25_indexes(FakeEngine({"a": _old()}))

    with pytest.raises(Exception):
        report.results = ()  # type: ignore[misc]
    assert isinstance(report, Bm25UpgradeReport)
    assert isinstance(report["a"], Bm25IndexResult)


# ---------------------------------------------------------------------------
# SchemaManager: the upgrade path
# ---------------------------------------------------------------------------


def _manager():
    from unittest.mock import MagicMock

    from iris_vector_rag.storage.schema_manager import SchemaManager

    manager = SchemaManager.__new__(SchemaManager)
    manager.base_embedding_dimension = 384
    manager.connection_manager = MagicMock()
    manager.connection_manager.session.identity = None
    manager.connection_manager.get_connection.return_value = object()
    return manager


def test_schema_manager_upgrades_bm25_indexes_through_its_engine():
    from unittest.mock import patch

    manager = _manager()
    engine = FakeEngine({"logs": _old()})

    with patch.object(manager, "_graph_engine", return_value=engine):
        report = manager.upgrade_bm25_indexes()

    assert report.upgraded == ("logs",)


def test_schema_manager_bm25_dry_run_writes_nothing():
    from unittest.mock import patch

    manager = _manager()
    engine = FakeEngine({"logs": _old()})

    with patch.object(manager, "_graph_engine", return_value=engine):
        report = manager.upgrade_bm25_indexes(names=["logs"], dry_run=True)

    assert report.pending == ("logs",)
    assert engine.upgrade_calls == []


def test_the_full_upgrade_runs_storage_first_then_bm25():
    """The storage step deploys the 4.2.0 classes; BM25 `Upgrade` lives in one of them."""
    from unittest.mock import patch

    manager = _manager()
    order = []
    storage = object()
    engine = FakeEngine({"logs": _old()})

    def fake_storage(dry_run=False, **options):
        order.append(("storage", dry_run, options))
        return storage

    def fake_engine(connection):
        order.append("engine")
        return engine

    with (
        patch.object(manager, "upgrade_iris_vector_graph_storage", fake_storage),
        patch.object(manager, "_graph_engine", fake_engine),
    ):
        result = manager.upgrade_iris_vector_graph(steps=("embeddings",))

    assert order == [("storage", False, {"steps": ("embeddings",)}), "engine"]
    assert result.storage is storage
    assert result.bm25.upgraded == ("logs",)
    assert result.ok


def test_the_full_upgrade_dry_run_previews_both_and_writes_neither():
    from unittest.mock import patch

    manager = _manager()
    engine = FakeEngine({"logs": _old()})
    seen = {}

    def fake_storage(dry_run=False, **options):
        seen["dry_run"] = dry_run
        return object()

    with (
        patch.object(manager, "upgrade_iris_vector_graph_storage", fake_storage),
        patch.object(manager, "_graph_engine", return_value=engine),
    ):
        result = manager.upgrade_iris_vector_graph(dry_run=True)

    assert seen["dry_run"] is True
    assert result.bm25.pending == ("logs",)
    assert engine.upgrade_calls == []


def test_the_full_upgrade_is_not_ok_when_an_index_fails():
    from unittest.mock import patch

    manager = _manager()
    engine = FakeEngine({"logs": _old()}, fail_on={"logs"})

    with (
        patch.object(
            manager, "upgrade_iris_vector_graph_storage", return_value=object()
        ),
        patch.object(manager, "_graph_engine", return_value=engine),
    ):
        result = manager.upgrade_iris_vector_graph()

    assert result.bm25.failed == ("logs",)
    assert not result.ok


def test_named_bm25_indexes_reach_the_upgrade():
    from unittest.mock import patch

    manager = _manager()
    engine = FakeEngine({"a": _old(), "b": _old()})

    with (
        patch.object(
            manager, "upgrade_iris_vector_graph_storage", return_value=object()
        ),
        patch.object(manager, "_graph_engine", return_value=engine),
    ):
        result = manager.upgrade_iris_vector_graph(bm25_indexes=["b"])

    assert result.bm25.names == ("b",)
