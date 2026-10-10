"""E2E: an index built before ivg 4.2.0 is upgraded to schema 2 and then written to.

The schema-1 index is written the way ivg 3.x wrote it (term frequencies, lengths, a
stored IDF, `N` and `avgdl`; no document frequency, token record or `schema`), straight
into `^BM25Idx`, so what is exercised is the real `Graph.KG.BM25Index` on the session's
IRIS, not a stand-in.
"""

from __future__ import annotations

import math
import uuid

import pytest

from iris_vector_rag.integrations.ivg import create_graph_engine
from iris_vector_rag.pipelines.ivg_bm25_backend import IVGTextSearchBackend

pytestmark = pytest.mark.integration

DOCS = {
    "d1": "timeout while connecting to the primary replica",
    "d2": "replica lag is rising on the standby",
    "d3": "disk usage is normal on the standby",
}


@pytest.fixture
def connection(iris_target, ivg_deployed):
    import iris.dbapi

    conn = iris.dbapi.connect(
        hostname=iris_target.host,
        port=iris_target.port,
        namespace=iris_target.namespace,
        username=iris_target.username,
        password=iris_target.password,
    )
    yield conn
    conn.close()


def _credentials(target):
    return {
        "host": target.host,
        "port": target.port,
        "namespace": target.namespace,
        "username": target.username,
        "password": target.password,
    }


@pytest.fixture
def engines(connection, iris_target):
    """Make engines; close the class-method connection each one opens.

    ivg lists indexes when an engine is built, so build it after the index exists.
    The community licence allows five connections: an engine left open holds one.
    """
    made = []

    def make():
        engine = create_graph_engine(connection, credentials=_credentials(iris_target))
        made.append(engine)
        return engine

    yield make
    for engine in made:
        native_connection = getattr(engine, "_native_conn", None)
        if native_connection is not None:
            native_connection.close()


@pytest.fixture
def native(connection):
    import iris

    return iris.createIRIS(connection)


@pytest.fixture
def index_name(engines, native):
    name = f"ivr_bm25_{uuid.uuid4().hex[:8]}"
    yield name
    engines().bm25_drop(name)


def write_schema_1(native, name, docs, k1=1.5, b=0.75):
    """An index as ivg 3.x wrote it: tf, len, stored idf, N, avgdl; no df, doc, schema."""
    df = {}
    total = 0
    for doc_id, text in docs.items():
        tokens = text.split()
        total += len(tokens)
        native.set(len(tokens), "BM25Idx", name, "len", doc_id)
        for term in set(tokens):
            native.set(tokens.count(term), "BM25Idx", name, "tf", term, doc_id)
            df[term] = df.get(term, 0) + 1
    n = len(docs)
    for term, count in df.items():
        native.set(
            math.log((n - count + 0.5) / (count + 0.5) + 1),
            "BM25Idx",
            name,
            "idf",
            term,
        )
    native.set(n, "BM25Idx", name, "cfg", "N")
    native.set(total / n, "BM25Idx", name, "cfg", "avgdl")
    native.set(k1, "BM25Idx", name, "cfg", "k1")
    native.set(b, "BM25Idx", name, "cfg", "b")
    native.set(len(df), "BM25Idx", name, "cfg", "vocab_size")


def test_a_schema_1_index_searches_but_refuses_writes(engines, native, index_name):
    write_schema_1(native, index_name, DOCS)
    engine = engines()

    assert engine.bm25_info(index_name)["schema"] == 1
    assert {i for i, _ in engine.bm25_search(index_name, "replica", 10)} == {"d1", "d2"}
    with pytest.raises(Exception, match="Upgrade"):
        engine.bm25_insert(index_name, "d4", "a new document")


def test_the_dry_run_finds_the_index_and_changes_nothing(
    engines, native, index_name, ivg_deployed
):
    write_schema_1(native, index_name, DOCS)

    report = ivg_deployed.upgrade_bm25_indexes(dry_run=True)

    assert index_name in report.pending
    assert report[index_name].docs == len(DOCS)
    assert engines().bm25_info(index_name)["schema"] == 1


def test_the_upgrade_converts_the_index_and_writes_then_work(
    engines, native, index_name, ivg_deployed
):
    write_schema_1(native, index_name, DOCS)

    report = ivg_deployed.upgrade_bm25_indexes(names=[index_name])

    assert report.upgraded == (index_name,)
    assert report[index_name].docs == len(DOCS)
    assert report[index_name].postings and report[index_name].vocab
    engine = engines()
    assert engine.bm25_info(index_name)["schema"] == 2
    assert engine.bm25_insert(index_name, "d4", "standby promotion finished")
    assert {i for i, _ in engine.bm25_search(index_name, "standby", 10)} == {
        "d2",
        "d3",
        "d4",
    }
    assert engine.bm25_delete(index_name, "d2")
    assert "d2" not in {i for i, _ in engine.bm25_search(index_name, "standby", 10)}


def test_upgrading_again_changes_nothing(engines, native, index_name, ivg_deployed):
    write_schema_1(native, index_name, DOCS)
    ivg_deployed.upgrade_bm25_indexes(names=[index_name])

    again = ivg_deployed.upgrade_bm25_indexes(names=[index_name])

    assert again.upgraded == ()
    assert again.current == (index_name,)
    assert again.ok


def test_an_index_that_is_not_there_is_reported_not_created(ivg_deployed):
    report = ivg_deployed.upgrade_bm25_indexes(names=["ivr_no_such_index"])

    assert report.missing == ("ivr_no_such_index",)
    assert report.ok


def test_the_full_upgrade_previews_then_converts(
    engines, native, index_name, ivg_deployed
):
    write_schema_1(native, index_name, DOCS)

    preview = ivg_deployed.upgrade_iris_vector_graph(dry_run=True)
    assert index_name in preview.bm25.pending
    assert engines().bm25_info(index_name)["schema"] == 1

    done = ivg_deployed.upgrade_iris_vector_graph(bm25_indexes=[index_name])
    assert done.bm25.upgraded == (index_name,)
    assert done.ok
    assert engines().bm25_info(index_name)["schema"] == 2


def test_the_backend_warns_about_schema_1_and_searches(
    engines, native, index_name, caplog
):
    import logging

    write_schema_1(native, index_name, DOCS)
    backend = IVGTextSearchBackend(engines(), index_name)

    with caplog.at_level(logging.WARNING, logger="iris_vector_rag.pipelines"):
        hits = backend.search("replica", 10)

    assert {i for i, _ in hits} == {"d1", "d2"}
    assert any("schema 1" in r.getMessage() for r in caplog.records)


def test_max_df_fraction_skips_a_term_in_most_documents(engines, index_name):
    engine = engines()
    for doc_id, text in {
        "a": "common zebra",
        "b": "common apple",
        "c": "common pear",
    }.items():
        engine.bm25_insert(index_name, doc_id, text)

    plain = IVGTextSearchBackend(engine, index_name).search("common zebra", 10)
    cut = IVGTextSearchBackend(engine, index_name, max_df_fraction=0.5).search(
        "common zebra", 10
    )

    assert {i for i, _ in plain} == {"a", "b", "c"}  # "common" scores every document
    assert [i for i, _ in cut] == ["a"]  # "common" is in 100% of them: skipped
