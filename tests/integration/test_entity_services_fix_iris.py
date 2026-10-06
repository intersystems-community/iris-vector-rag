"""Entity services against a real IRIS (feature `entity-services-fix`).

Works in a schema of its own (`ESFIX`): DDL autocommits, so nothing here can be rolled
back and nothing here is left in `RAG`. Runs on the ephemeral container `iris_target`
starts. The LLM is the real OpenAI model (OPENAI_API_KEY from the environment), the
embedder is the real sentence-transformer and the vector store is the real
`IRISVectorStore`. The one fault injected is an LLM timeout on one document (Story 2),
because that fault is what the story is about.

Story 1  a batch of entities is stored with confidence, upserted, chunked, mixed
         embeddings included, and read back by document and by type.
Story 2  a document whose extraction fails is reported in `failed`, the rest are
         stored; every document failing raises.
"""

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Document, Entity, Relationship
from iris_vector_rag.pipelines.graphrag import GraphRAGPipeline
from iris_vector_rag.services.entity_extraction import (
    EntityExtractionError,
    EntityExtractionService,
)
from iris_vector_rag.services.storage import EntityStorageAdapter
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

pytestmark = pytest.mark.integration

WIDTH = 384  # the real embedder's width
SCHEMA = "ESFIX"
TABLES = ("EntityRelationships", "Entities", "DocumentChunks", "SourceDocuments")


def _drop_all(connection):
    cursor = connection.cursor()
    try:
        for table in TABLES:
            cursor.execute(f"DROP TABLE IF EXISTS {SCHEMA}.{table}")
        connection.commit()
    finally:
        cursor.close()


def _scalar(connection, sql, params=None):
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params or [])
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        cursor.close()


@pytest.fixture
def stack(iris_connection):
    config = ConfigurationManager()
    config._config.setdefault("storage", {}).setdefault("iris", {})[
        "table_name"
    ] = f"{SCHEMA}.SourceDocuments"
    config._config.setdefault("entity_extraction", {})["storage"] = {
        "entities_table": f"{SCHEMA}.Entities",
        "relationships_table": f"{SCHEMA}.EntityRelationships",
        "batch_size": 4,
    }
    connections = ConnectionManager(config)
    _drop_all(iris_connection)
    manager = SchemaManager(connections, config, schema_prefix=SCHEMA)
    assert manager.ensure_table_schema("SourceDocuments")
    cursor = iris_connection.cursor()
    for doc in ("d1", "d2", "d3"):
        cursor.execute(
            f"INSERT INTO {SCHEMA}.SourceDocuments (doc_id, text_content) VALUES (?, ?)",
            [doc, f"text of {doc}"],
        )
    iris_connection.commit()
    cursor.close()
    adapter = EntityStorageAdapter(connections, config._config)
    yield config, connections, adapter
    _drop_all(iris_connection)


def _entity(n, doc="d1", embedding=None, confidence=0.8, kind="PERSON"):
    return Entity(
        id=f"e{n}",
        text=f"name{n}",
        entity_type=kind,
        confidence=confidence,
        start_offset=0,
        end_offset=1,
        source_document_id=doc,
        metadata={} if embedding is None else {"embedding": embedding},
    )


# --- Story 1: the batch processor ------------------------------------------------------


def _add_vector_column(connection, adapter):
    """The standard Entities table has no vector column; a legacy one may."""
    assert adapter.create_tables_if_not_exist() is True
    cursor = connection.cursor()
    try:
        cursor.execute(
            f"ALTER TABLE {SCHEMA}.Entities ADD embedding VECTOR(FLOAT, {WIDTH})"
        )
        connection.commit()
    finally:
        cursor.close()


def test_a_mixed_batch_is_stored_with_confidence_and_read_back(stack, iris_connection):
    _config, _connections, adapter = stack
    _add_vector_column(iris_connection, adapter)
    entities = [
        _entity(1, embedding=[0.1] * WIDTH, confidence=0.91),
        _entity(2, confidence=0.42, kind="ORGANIZATION"),
        _entity(3, doc="d2", embedding=[0.2] * WIDTH),
    ]

    assert adapter.store_entities_batch(entities) == 3

    assert _scalar(iris_connection, f"SELECT COUNT(*) FROM {SCHEMA}.Entities") == 3
    assert _scalar(
        iris_connection,
        f"SELECT confidence FROM {SCHEMA}.Entities WHERE entity_id = ?",
        ["e2"],
    ) == pytest.approx(0.42)
    assert _scalar(
        iris_connection,
        f"SELECT VECTOR_COSINE(embedding, embedding) FROM {SCHEMA}.Entities"
        " WHERE entity_id = ?",
        ["e1"],
    ) == pytest.approx(1.0, abs=1e-3)
    assert {e.id for e in adapter.get_entities_by_document("d1")} == {"e1", "e2"}
    assert [e.id for e in adapter.get_entities_by_type("ORGANIZATION")] == ["e2"]
    assert len(adapter.get_entities_by_type("PERSON", limit=1)) == 1
    # min_confidence search sees batch-stored rows (they used to get no confidence)
    found = adapter.search_entities("name2", fuzzy=False, min_confidence=0.4)
    assert [f["entity_id"] for f in found] == ["e2"]


def test_an_embedding_the_table_cannot_hold_is_reported_not_dropped(
    stack, iris_connection
):
    from iris_vector_rag.services.storage import EntityStorageError

    _config, _connections, adapter = stack  # canonical Entities: no vector column

    with pytest.raises(EntityStorageError) as caught:
        adapter.store_entities_batch([_entity(1), _entity(2, embedding=[0.1] * WIDTH)])

    assert [f["entity_id"] for f in caught.value.failed] == ["e2"]
    assert "embedding" in caught.value.failed[0]["reason"].lower()
    assert caught.value.stored == 1


def test_storing_the_same_ids_again_updates_them(stack, iris_connection):
    _config, _connections, adapter = stack
    adapter.store_entities_batch([_entity(1, confidence=0.3), _entity(2)])

    stored = adapter.store_entities_batch(
        [_entity(1, confidence=0.99), _entity(4, doc="d2")]
    )

    assert stored == 2
    assert _scalar(iris_connection, f"SELECT COUNT(*) FROM {SCHEMA}.Entities") == 3
    assert _scalar(
        iris_connection,
        f"SELECT confidence FROM {SCHEMA}.Entities WHERE entity_id = ?",
        ["e1"],
    ) == pytest.approx(0.99)


def test_batch_size_chunks_a_large_call(stack, iris_connection):
    _config, _connections, adapter = stack
    assert adapter.batch_processor.batch_size == 4

    assert adapter.store_entities_batch([_entity(n) for n in range(11)]) == 11

    assert _scalar(iris_connection, f"SELECT COUNT(*) FROM {SCHEMA}.Entities") == 11
    stats = adapter.get_storage_stats()
    assert stats["entities_count"] == 11
    assert stats["entity_types"] == {"PERSON": 11}


def test_relationships_read_back_by_document(stack):
    _config, _connections, adapter = stack
    adapter.store_entities_batch([_entity(1), _entity(2)])
    rel = Relationship(
        source_entity_id="e1",
        target_entity_id="e2",
        relationship_type="knows",
        confidence=0.7,
        source_document_id="d1",
    )

    assert adapter.store_relationships_batch([rel]) == 1

    got = adapter.get_relationships_by_document("d1")
    assert [(r.source_entity_id, r.target_entity_id) for r in got] == [("e1", "e2")]
    assert adapter.get_storage_stats()["relationships_count"] == 1


def test_a_row_iris_rejects_is_reported_not_swallowed(stack, iris_connection):
    from iris_vector_rag.services.storage import EntityStorageError

    _config, _connections, adapter = stack
    _add_vector_column(iris_connection, adapter)
    too_wide = _entity(2, embedding=[0.1] * (WIDTH + 3))  # wrong vector width

    with pytest.raises(EntityStorageError) as caught:
        adapter.store_entities_batch([_entity(1), too_wide, _entity(3)])

    assert [f["entity_id"] for f in caught.value.failed] == ["e2"]
    assert caught.value.stored == 2
    assert _scalar(iris_connection, f"SELECT COUNT(*) FROM {SCHEMA}.Entities") == 2


# --- Story 2: load_documents -----------------------------------------------------------


OPENAI_MODEL = "gpt-4o-mini"

TEXTS = {
    "d1": "Dr. John Smith, a cardiologist at Mayo Clinic, prescribed aspirin to treat the patient's hypertension.",
    "d2": "DOCUMENT-TWO-MARKER: Dr. Lee at Cleveland Clinic prescribed insulin for diabetes.",
    "d3": "Dr. Maria Garcia at Johns Hopkins Hospital treated the patient's asthma with albuterol.",
}


def _real_llm():
    from iris_vector_rag.common.utils import get_llm_func

    return get_llm_func(provider="openai", model_name=OPENAI_MODEL)


def _llm_timing_out_on_document_two(real):
    """The real LLM, except that the call for document two times out (fault injection)."""

    def llm(prompt):
        if "DOCUMENT-TWO-MARKER" in prompt:
            raise TimeoutError("llm timed out")
        return real(prompt)

    return llm


def _pipeline(stack, llm):
    config, connections, _adapter = stack
    service = EntityExtractionService(
        config_manager=config, connection_manager=connections, llm_func=llm
    )
    pipeline = object.__new__(GraphRAGPipeline)
    pipeline.entity_extraction_service = service
    pipeline.entity_extraction_enabled = True
    pipeline.vector_store = IRISVectorStore(
        connection_manager=connections, config_manager=config
    )
    pipeline.connection_manager = connections
    pipeline.config_manager = config
    return pipeline


def _docs():
    return [Document(page_content=TEXTS[d], id=d) for d in ("d1", "d2", "d3")]


def test_load_documents_stores_the_rest_and_lists_the_failed_one(
    stack, iris_connection
):
    _config, _connections, adapter = stack
    llm = _llm_timing_out_on_document_two(_real_llm())

    result = _pipeline(stack, llm).load_documents("", documents=_docs())

    assert [f["document_id"] for f in result["failed"]] == ["d2"]
    assert "timed out" in result["failed"][0]["reason"]
    assert result["documents_loaded"] == 3
    # the real LLM found entities in the two documents whose call did not time out
    found = {d: adapter.get_entities_by_document(d) for d in ("d1", "d3")}
    assert all(found[d] for d in found), f"the real LLM extracted nothing: {found}"
    assert adapter.get_entities_by_document("d2") == []
    for entities in found.values():
        assert all(0.0 < e.confidence <= 1.0 for e in entities)
    stored = _scalar(iris_connection, f"SELECT COUNT(*) FROM {SCHEMA}.Entities")
    assert stored == sum(len(v) for v in found.values())
    # every pair of entities in a document is related (the pipeline's co-occurrence rule)
    expected_pairs = sum(len(v) * (len(v) - 1) // 2 for v in found.values())
    assert (
        _scalar(iris_connection, f"SELECT COUNT(*) FROM {SCHEMA}.EntityRelationships")
        == expected_pairs
    )
    # the real vector store holds all three documents, including the one that failed
    assert (
        _scalar(iris_connection, f"SELECT COUNT(*) FROM {SCHEMA}.SourceDocuments") == 3
    )


def test_load_documents_raises_when_every_document_fails(stack):
    # FAULT INJECTION: the LLM is down for every document; nothing else is replaced.
    def llm(prompt):
        raise ConnectionError("llm down")

    with pytest.raises(EntityExtractionError, match="every one of 3"):
        _pipeline(stack, llm).load_documents("", documents=_docs())
