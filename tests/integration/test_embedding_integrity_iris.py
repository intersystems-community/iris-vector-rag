"""Embedding integrity against a real IRIS (G6 points 5 and 6).

The dimension the table is created at is the embedding manager's; a vector of any other
width is refused at insert and at query, and nothing is stored. Runs on the ephemeral
container `iris_target` starts, never on a mock. The embedder is the real sentence-transformer; the one fault injected
is a failing `embed_text` in the ingest-failure test, because that failure is the subject.
"""

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Document
from iris_vector_rag.exceptions import DimensionMismatchError, EmbeddingError
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from tests.fixtures.shapes.ddl import VECTOR_DIMENSION, create_statement

pytestmark = pytest.mark.integration

WIDTH = 384  # the real embedder's width
PREFIX = "EITEST"


@pytest.fixture
def stack(iris_connection):
    """A real config, schema manager and store on a schema of their own."""
    config = ConfigurationManager()
    connections = ConnectionManager(config)
    schema = SchemaManager(connections, config, schema_prefix=PREFIX)

    cursor = iris_connection.cursor()
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {PREFIX}.SourceDocuments")
        iris_connection.commit()
    finally:
        cursor.close()

    assert schema.ensure_table_schema("SourceDocuments")
    store = IRISVectorStore(
        connections,
        config,
        schema_manager=schema,
    )
    store.table_name = f"{PREFIX}.SourceDocuments"
    yield store, schema

    cursor = iris_connection.cursor()
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {PREFIX}.SourceDocuments")
        iris_connection.commit()
    finally:
        cursor.close()


def _count(connection, table):
    cursor = connection.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        return cursor.fetchone()[0]
    finally:
        cursor.close()


def test_the_table_is_created_at_the_embedding_managers_width(stack, iris_connection):
    store, schema = stack
    assert schema.base_embedding_dimension == WIDTH
    store.add_documents([Document(id="d1", page_content="hello", metadata={})])
    cursor = iris_connection.cursor()
    try:
        # Succeeds only if the column holds WIDTH-dimensional vectors.
        cursor.execute(
            f"SELECT VECTOR_COSINE(embedding, TO_VECTOR(?, FLOAT, {WIDTH})) "
            f"FROM {PREFIX}.SourceDocuments",
            ["[" + ",".join(["0.5"] * WIDTH) + "]"],
        )
        assert cursor.fetchone() is not None
    finally:
        cursor.close()


def test_a_wrong_sized_embedding_is_refused_and_nothing_is_stored(
    stack, iris_connection
):
    store, _ = stack
    documents = [
        Document(id="ok", page_content="fits", metadata={}),
        Document(id="too-wide", page_content="does not", metadata={}),
    ]
    with pytest.raises(DimensionMismatchError, match="dimension mismatch"):
        store._store_documents(
            documents, embeddings=[[0.1] * WIDTH, [0.1] * (WIDTH + 1)]
        )
    assert _count(iris_connection, f"{PREFIX}.SourceDocuments") == 0


def test_a_wrong_sized_query_is_refused(stack):
    store, _ = stack
    store.add_documents([Document(id="d1", page_content="hello", metadata={})])
    with pytest.raises(DimensionMismatchError):
        store.similarity_search_by_embedding([0.1] * (WIDTH * 2), 3)


def test_a_right_sized_query_finds_the_document(stack):
    store, _ = stack
    store.add_documents([Document(id="d1", page_content="hello", metadata={})])
    results = store.similarity_search("hello", 1)
    assert [doc.id for doc in results] == ["d1"]


def test_embedding_failure_during_ingest_raises_and_stores_nothing(
    stack, iris_connection, monkeypatch
):
    store, _ = stack

    # FAULT INJECTION: the embedder fails; store, schema and IRIS stay real.
    def fail(self, text):
        raise EmbeddingError("the model is gone")

    monkeypatch.setattr(
        "iris_vector_rag.embeddings.manager.EmbeddingManager.embed_text", fail
    )
    with pytest.raises(EmbeddingError):
        store.add_documents([Document(id="d1", page_content="hello", metadata={})])
    assert _count(iris_connection, f"{PREFIX}.SourceDocuments") == 0


NO_METADATA_WIDTH = VECTOR_DIMENSION  # the recorded shape's own width


@pytest.fixture
def no_metadata_store(stack, iris_connection):
    store, _ = stack
    table = f"{PREFIX}.NoMeta"
    cursor = iris_connection.cursor()
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
        cursor.execute(create_statement("no_metadata", PREFIX, table="NoMeta"))
        iris_connection.commit()
    finally:
        cursor.close()
    store.table_name = table
    store._shape_cache = {}
    yield store, table
    cursor = iris_connection.cursor()
    try:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
        iris_connection.commit()
    finally:
        cursor.close()


def test_re_adding_a_document_updates_it_in_every_table_shape(
    no_metadata_store, iris_connection
):
    """ivg's upsert waits for an error message containing UNIQUE; this driver's duplicate
    key error does not, so the second add used to be logged as a failure and skipped.
    The store now checks for the row itself."""
    store, table = no_metadata_store
    first = Document(id="d1", page_content="first text", metadata={"source": "a"})
    second = Document(id="d1", page_content="second text", metadata={"source": "b"})

    def vector(seed):  # varied values: a flat vector does not trigger the driver quirk
        return [((seed * 7 + i) % 13) / 13.0 for i in range(NO_METADATA_WIDTH)]

    assert store._store_documents([first], embeddings=[vector(1)]) == ["d1"]
    # Identical again: the duplicate-key error for this is the one the driver garbles.
    assert store._store_documents([first], embeddings=[vector(1)]) == ["d1"]
    assert store._store_documents([second], embeddings=[vector(2)]) == ["d1"]

    cursor = iris_connection.cursor()
    try:
        cursor.execute(f"SELECT content, title FROM {table} WHERE doc_id = 'd1'")
        row = cursor.fetchone()
        text, title = row[0], row[1]
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        assert cursor.fetchone()[0] == 1
    finally:
        cursor.close()
    assert (text, title) == ("second text", "b")
