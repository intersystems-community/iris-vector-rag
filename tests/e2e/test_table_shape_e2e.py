"""E2E: the pipeline answers from a legacy-shaped table (feature 082, US1 gate).

The table here is the one a 0.13 release created: its text lives in `content`, not
`text_content`. Nothing in this test names a column -- the shape resolved from the
catalog decides what is written and what is read, which is the whole claim of US1.

Real IRIS (`iris-vector-rag-iris` on 51972), real embeddings, real LLM. If the
reader and the writer disagreed about which column holds the text, the retrieved
documents would come back empty and the answer would say so.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from iris_vector_rag.common.iris_connection import get_iris_connection
from iris_vector_rag.common.utils import get_llm_func
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.basic import BasicRAGPipeline
from iris_vector_rag.storage.catalog import RecordedCatalog
from iris_vector_rag.storage.shape_resolver import TableShapeResolver
from iris_vector_rag.storage.table_shape import CANONICAL_DOCUMENT_SHAPE, ShapeKind
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from tests.fixtures.shapes.ddl import (
    SHAPE_SCHEMA,
    SHAPE_TABLES,
    create_statement,
    drop_statement,
)

pytestmark = [pytest.mark.e2e, pytest.mark.true_e2e]

#: `doc_id` / `content` / `embedding` -- the spelling shipped before 0.15, and the
#: shape the store used to write one column of and read the other.
LEGACY_SHAPE = "fixture"

#: The recordings the unit suite describes shapes from.
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "shapes"

DOCUMENTS = [
    Document(
        id="shape-e2e-warfarin",
        page_content=(
            "Patients taking warfarin have their INR checked every four weeks "
            "once the dose has been stable for three consecutive readings."
        ),
        metadata={"source": "shape-e2e"},
    ),
    Document(
        id="shape-e2e-metformin",
        page_content=(
            "Metformin is taken with meals to reduce gastrointestinal upset and "
            "requires no routine coagulation monitoring."
        ),
        metadata={"source": "shape-e2e"},
    ),
    Document(
        id="shape-e2e-weather",
        page_content="The forecast for the weekend is overcast with light winds.",
        metadata={"source": "shape-e2e"},
    ),
]

QUESTION = "How often is the INR checked for a patient on a stable warfarin dose?"


@pytest.fixture
def legacy_table(monkeypatch):
    """A legacy-shaped table, with the library configured to use it.

    The table name reaches the store through configuration, the same way a caller
    with an existing table would supply it.
    """
    connection = get_iris_connection()
    table = f"{SHAPE_SCHEMA}.{SHAPE_TABLES[LEGACY_SHAPE]}"
    cursor = connection.cursor()
    try:
        cursor.execute(drop_statement(LEGACY_SHAPE, SHAPE_SCHEMA))
        cursor.execute(create_statement(LEGACY_SHAPE, SHAPE_SCHEMA))
        connection.commit()
    finally:
        cursor.close()

    monkeypatch.setenv("RAG_STORAGE__IRIS__TABLE_NAME", table)
    try:
        yield table
    finally:
        cursor = connection.cursor()
        try:
            cursor.execute(drop_statement(LEGACY_SHAPE, SHAPE_SCHEMA))
            connection.commit()
        except Exception:  # noqa: BLE001 - teardown is advisory
            pass
        cursor.close()
        connection.close()


@pytest.fixture
def legacy_pipeline(legacy_table):
    """A real BasicRAG pipeline over the legacy-shaped table."""
    # Calling this first also loads `.env`, which is where the key lives locally.
    llm_func = get_llm_func(provider="openai", model_name="gpt-4o-mini")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("answer generation needs OPENAI_API_KEY")

    config = ConfigurationManager()
    connections = ConnectionManager(config)
    return BasicRAGPipeline(
        connection_manager=connections,
        config_manager=config,
        llm_func=llm_func,
        vector_store=IRISVectorStore(connections, config),
    )


def test_us1(legacy_pipeline):
    """A legacy-shaped table serves a full pipeline query, answer included."""
    legacy_pipeline.load_documents(documents=DOCUMENTS)

    result = legacy_pipeline.query(QUESTION, top_k=3)

    assert result["error"] is None, result["error"]

    retrieved = result["retrieved_documents"]
    assert (
        retrieved
    ), "the legacy-shaped table must be readable by the store that wrote it"
    assert retrieved[0].id == "shape-e2e-warfarin", [doc.id for doc in retrieved]
    # The text presented is the text written, out of whichever column the shape says
    # holds it: the failure this replaces returned rows with empty content.
    assert retrieved[0].page_content == DOCUMENTS[0].page_content

    answer = result["answer"]
    assert answer, "an answer must be generated from the retrieved documents"
    assert "four weeks" in answer.lower(), answer


def test_us2(legacy_pipeline):
    """A session's worth of queries reads the table's shape once (US2 gate).

    Retrieval only: what is under test is how often the catalog is consulted, and a
    generated answer per query would buy nothing but latency.
    """
    legacy_pipeline.load_documents(documents=DOCUMENTS)
    store = legacy_pipeline.vector_store

    for _ in range(20):
        result = legacy_pipeline.query(QUESTION, top_k=3, generate_answer=False)
        assert result["error"] is None, result["error"]
        assert result["retrieved_documents"][0].id == "shape-e2e-warfarin"

    assert store.shapes.reads == 1


def canonical_diff(shape) -> str:
    """What this shape has that the canonical one does not, and the reverse.

    `is_canonical` is one bit, and one bit cannot say which creation path ran. The
    columns, the role bindings and the vector spec are the three things it looks at,
    so all three are reported here.
    """
    have = set(shape.columns)
    want = set(CANONICAL_DOCUMENT_SHAPE.columns)
    embedding = shape.columns.get("embedding")
    return (
        f"{shape.qualified_name}: extra={sorted(have - want)} "
        f"missing={sorted(want - have)} "
        f"roles={ {role.name: name for role, name in sorted(shape.roles.items(), key=lambda item: item[0].name)} } "
        f"vector={embedding.vector if embedding is not None else None}"
    )


@pytest.fixture
def fresh_instance(monkeypatch):
    """A schema with no document table, as a fresh install has.

    The scratch schema stands in for the fresh instance: the library creates
    `SourceDocuments` in whatever schema it is configured for, so this is the real
    bootstrap path without dropping the shared `RAG` table.
    """
    monkeypatch.setenv("IRIS_SCHEMA_PREFIX", SHAPE_SCHEMA)
    table = f"{SHAPE_SCHEMA}.SourceDocuments"
    monkeypatch.setenv("RAG_STORAGE__IRIS__TABLE_NAME", table)

    connection = get_iris_connection()

    def drop():
        cursor = connection.cursor()
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {table}")
            connection.commit()
        except Exception:  # noqa: BLE001 - teardown is advisory
            pass
        cursor.close()

    drop()
    try:
        yield table
    finally:
        drop()
        connection.close()


def test_us3(fresh_instance):
    """A fresh instance bootstrapped through the pipeline has the canonical shape.

    US3's claim end to end: the table the library creates for itself is the one it
    describes as canonical, and it answers a real question on the first run -- no
    migration step, no second shape.
    """
    llm_func = get_llm_func(provider="openai", model_name="gpt-4o-mini")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("answer generation needs OPENAI_API_KEY")

    config = ConfigurationManager()
    connections = ConnectionManager(config)
    pipeline = BasicRAGPipeline(
        connection_manager=connections,
        config_manager=config,
        llm_func=llm_func,
        vector_store=IRISVectorStore(connections, config),
    )

    pipeline.load_documents(documents=DOCUMENTS)

    shape = pipeline.vector_store.shape()
    assert shape.qualified_name == fresh_instance
    assert shape.is_canonical, canonical_diff(shape)
    assert shape.kind is ShapeKind.CANONICAL

    result = pipeline.query(QUESTION, top_k=3)
    assert result["error"] is None, result["error"]
    assert result["retrieved_documents"][0].id == "shape-e2e-warfarin"
    assert "four weeks" in result["answer"].lower(), result["answer"]


def test_us4(legacy_table):
    """A recorded shape and the live catalog reach the same verdict (US4 gate).

    Two stores over one real table: one resolves the shape from IRIS, the other only
    ever reads `tests/fixtures/shapes/` -- the recordings every shape-dependent unit
    test describes its table with. Both are asked the same question against real
    data, so a recording that has drifted from the DDL it claims to describe fails
    here instead of quietly making the unit suite agree with nothing.

    No LLM: what is under test is which columns each store decides to read.
    """
    config = ConfigurationManager()
    connections = ConnectionManager(config)

    live = IRISVectorStore(connections, config)
    live.add_documents(DOCUMENTS)

    recorded = IRISVectorStore(
        connections,
        config,
        shapes=TableShapeResolver(RecordedCatalog.from_fixture_dir(FIXTURE_DIR)),
    )

    assert recorded.shape() == live.shape()
    assert recorded.shape().qualified_name == legacy_table

    live_docs = live.search_by_text(QUESTION, top_k=3)
    recorded_docs = recorded.search_by_text(QUESTION, top_k=3)

    assert [doc.id for doc in live_docs][0] == "shape-e2e-warfarin"
    assert [doc.id for doc in recorded_docs] == [doc.id for doc in live_docs]
    # Same rows and the same text out of the same column: the recording decides the
    # SQL the second store sends, and it reads the table the first one wrote.
    assert [doc.page_content for doc in recorded_docs] == [
        doc.page_content for doc in live_docs
    ]
