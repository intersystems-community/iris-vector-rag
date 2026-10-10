"""IVGTextSearchBackend and ivg 4.2.0's BM25 common-term cutoff.

`engine.bm25_search(name, query, k, max_df_fraction=0.0, report=False)` skips query
terms that occur in more than that fraction of the documents. It is off by default and
IVR leaves it off unless `retrieval.bm25.max_df_fraction` (or the constructor argument)
asks for it.
"""

from __future__ import annotations

import logging

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.pipelines.ivg_bm25_backend import IVGTextSearchBackend


class Engine:
    def __init__(self, schema=2, info_error=None):
        self.calls = []
        self.info_calls = 0
        self._schema = schema
        self._info_error = info_error

    def bm25_search(self, name, query, k=10, max_df_fraction=0.0, report=False):
        self.calls.append(
            {"name": name, "query": query, "k": k, "max_df_fraction": max_df_fraction}
        )
        return [("n1", 1.5), ("n2", 0.5)]

    def bm25_info(self, name):
        self.info_calls += 1
        if self._info_error:
            raise self._info_error
        return {"N": 3, "schema": self._schema}


class LegacySignatureEngine:
    """ivg before 4.2.0: `bm25_search(name, query, k)` and nothing else."""

    def __init__(self):
        self.calls = []

    def bm25_search(self, name, query, k=10):
        self.calls.append((name, query, k))
        return [("n1", 1.0)]


def test_by_default_the_cutoff_is_not_passed_and_results_are_unchanged():
    engine = Engine()
    backend = IVGTextSearchBackend(engine, "logs")

    assert backend.search("timeout error", 5) == [("n1", 1.5), ("n2", 0.5)]
    assert engine.calls == [
        {"name": "logs", "query": "timeout error", "k": 5, "max_df_fraction": 0.0}
    ]


def test_an_engine_without_the_option_still_works_while_it_is_off():
    engine = LegacySignatureEngine()

    assert IVGTextSearchBackend(engine, "logs").search("q", 3) == [("n1", 1.0)]
    assert engine.calls == [("logs", "q", 3)]


def test_the_cutoff_is_passed_when_set():
    engine = Engine()

    IVGTextSearchBackend(engine, "logs", max_df_fraction=0.5).search("error", 4)

    assert engine.calls[0]["max_df_fraction"] == 0.5


@pytest.mark.parametrize("bad", [-0.1, 1.1, True, "half", float("nan")])
def test_a_bad_cutoff_is_refused_at_construction(bad):
    with pytest.raises(ValueError, match="max_df_fraction"):
        IVGTextSearchBackend(Engine(), "logs", max_df_fraction=bad)


@pytest.mark.parametrize("edge", [0, 0.0, 1, 1.0])
def test_the_ends_of_the_range_are_valid(edge):
    assert IVGTextSearchBackend(Engine(), "logs", max_df_fraction=edge)


# -- configuration ------------------------------------------------------------


def _config(**retrieval):
    manager = ConfigurationManager()
    manager._config.setdefault("retrieval", {})["bm25"] = dict(retrieval)
    return manager


def test_the_shipped_default_is_off():
    manager = ConfigurationManager()

    assert manager.get("retrieval:bm25:max_df_fraction") == 0.0


def test_from_config_reads_retrieval_bm25_max_df_fraction():
    engine = Engine()

    backend = IVGTextSearchBackend.from_config(
        engine, "logs", _config(max_df_fraction=0.9)
    )
    backend.search("q", 2)

    assert engine.calls[0]["max_df_fraction"] == 0.9


def test_from_config_without_the_key_is_off():
    manager = ConfigurationManager()
    manager._config.get("retrieval", {}).pop("bm25", None)
    engine = Engine()

    IVGTextSearchBackend.from_config(engine, "logs", manager).search("q", 2)

    assert engine.calls[0]["max_df_fraction"] == 0.0


def test_from_config_names_the_key_when_the_value_is_bad():
    with pytest.raises(ValueError, match=r"retrieval\.bm25\.max_df_fraction"):
        IVGTextSearchBackend.from_config(Engine(), "logs", _config(max_df_fraction=7))


def test_an_explicit_argument_beats_the_configuration():
    engine = Engine()

    IVGTextSearchBackend.from_config(
        engine, "logs", _config(max_df_fraction=0.9), max_df_fraction=0.2
    ).search("q", 2)

    assert engine.calls[0]["max_df_fraction"] == 0.2


# -- a schema-1 index ---------------------------------------------------------


def test_searching_a_schema_1_index_warns_once_and_still_searches(caplog):
    engine = Engine(schema=1)
    backend = IVGTextSearchBackend(engine, "logs")

    with caplog.at_level(logging.WARNING, logger="iris_vector_rag.pipelines"):
        first = backend.search("q", 2)
        backend.search("q", 2)

    assert first == [("n1", 1.5), ("n2", 0.5)]
    warnings = [r for r in caplog.records if "schema 1" in r.getMessage()]
    assert len(warnings) == 1
    assert "bm25_upgrade('logs')" in warnings[0].getMessage()
    assert engine.info_calls == 1


def test_a_current_index_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="iris_vector_rag.pipelines"):
        IVGTextSearchBackend(Engine(schema=2), "logs").search("q", 2)

    assert not [r for r in caplog.records if "schema" in r.getMessage()]


def test_a_failing_info_call_never_breaks_search():
    engine = Engine(info_error=RuntimeError("no such class"))

    assert IVGTextSearchBackend(engine, "logs").search("q", 2)


def test_an_engine_that_cannot_report_a_schema_is_left_alone():
    assert IVGTextSearchBackend(LegacySignatureEngine(), "logs").search("q", 1)
