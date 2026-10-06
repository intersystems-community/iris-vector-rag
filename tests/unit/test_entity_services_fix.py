"""The two live entity services tell the truth (feature `entity-services-fix`).

CR-11  extraction swallowed every error and returned `[]`; a failed document is now
       reported in `failed` and the batch continues; it raises only when every
       document fails.
CR-06  `BatchEntityProcessor` builds one tuple width per statement, writes
       `confidence`, upserts, and resolves the source document.
CR-13  `batch_size` chunks the work, a failed chunk retries smaller, and
       `validate_count` compares against the database.
CR-10  the storage stubs return real rows or raise.
CR-17  a failed write never shows in search; the KG tables follow the entities
       table's schema.

No database here: a recording cursor stands in for IRIS. The live checks are in
tests/integration/test_entity_services_fix_iris.py.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, Mock

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.models import Document, Entity, Relationship
from iris_vector_rag.services import entity_extraction as ee
from iris_vector_rag.services.batch_entity_processor import BatchEntityProcessor
from iris_vector_rag.services.entity_extraction import (
    EntityExtractionError,
    EntityExtractionService,
)
from iris_vector_rag.services.storage import EntityStorageAdapter

# --------------------------------------------------------------------------- service


def _config(**extraction):
    cfg = Mock(spec=ConfigurationManager)
    cfg.get.return_value = extraction
    cfg._config = {"entity_extraction": extraction}
    return cfg


def _docs(*ids):
    return [Document(page_content=f"text of {i}", id=i) for i in ids]


def _llm_failing_on(bad_id, reply='[{"text": "Ada", "type": "PERSON"}]'):
    def llm(prompt):
        if f"text of {bad_id}" in prompt:
            raise TimeoutError("llm timed out")
        return reply

    return llm


def test_a_failed_document_is_listed_and_the_batch_continues():
    service = EntityExtractionService(_config(), llm_func=_llm_failing_on("d2"))

    result = service.extract_batch(_docs("d1", "d2", "d3"))

    assert [f["document_id"] for f in result.failed] == ["d2"]
    assert "timed out" in result.failed[0]["reason"]
    assert set(result.per_document_entities) == {"d1", "d3"}
    assert all(result.per_document_entities[d] for d in ("d1", "d3"))
    assert result.success_status is False
    assert "d2" in result.error_message


def test_a_clean_batch_has_an_empty_failed_list():
    service = EntityExtractionService(_config(), llm_func=_llm_failing_on("nobody"))

    result = service.extract_batch(_docs("d1", "d2"))

    assert result.failed == []
    assert result.success_status is True


def test_every_document_failing_raises(monkeypatch):
    # the batch is retried with a backoff before it gives up (FR-005): don't wait it out
    monkeypatch.setattr("iris_vector_rag.common.batch_utils.time.sleep", lambda s: None)

    def llm(prompt):
        raise ConnectionError("llm down")

    service = EntityExtractionService(_config(), llm_func=llm)

    with pytest.raises(EntityExtractionError, match="llm down"):
        service.extract_batch(_docs("d1", "d2"))


def test_no_llm_configured_raises_instead_of_returning_nothing():
    service = EntityExtractionService(_config())

    with pytest.raises(EntityExtractionError, match="llm"):
        service.extract_entities(_docs("d1")[0])


def test_an_unparseable_llm_reply_is_a_failure_not_an_empty_list():
    service = EntityExtractionService(_config(), llm_func=lambda p: '[{"text": ')

    with pytest.raises(EntityExtractionError, match="parse"):
        service.extract_entities(_docs("d1")[0])


def test_confidence_threshold_types_and_max_entities_are_applied():
    reply = json.dumps(
        [
            {"text": "Ada", "type": "PERSON", "confidence": 0.9},
            {"text": "Bob", "type": "PERSON", "confidence": 0.1},
            {"text": "Acme", "type": "ORGANIZATION", "confidence": 0.9},
            {"text": "Cy", "type": "PERSON", "confidence": 0.8},
            {"text": "Dee", "type": "PERSON", "confidence": 0.95},
        ]
    )
    service = EntityExtractionService(
        _config(confidence_threshold=0.5, entity_types=["PERSON"], max_entities=2),
        llm_func=lambda p: reply,
    )

    entities = service.extract_entities(_docs("d1")[0])

    assert [e.text for e in entities] == ["Dee", "Ada"]  # best two, PERSON only, >= 0.5


def test_process_document_does_not_claim_storage_it_did_not_do():
    service = EntityExtractionService(
        _config(), llm_func=lambda p: '[{"text": "Ada", "type": "PERSON"}]'
    )

    outcome = service.process_document(_docs("d1")[0])

    assert outcome["stored"] is False
    assert outcome["entities_extracted"] == 1


def test_process_document_raises_when_the_adapter_cannot_store():
    service = EntityExtractionService(
        _config(), llm_func=lambda p: '[{"text": "Ada", "type": "PERSON"}]'
    )
    service.storage_adapter = Mock()
    service.storage_adapter.store_entity.return_value = False

    with pytest.raises(EntityExtractionError, match="store"):
        service.process_document(_docs("d1")[0])


def test_extract_batch_with_dspy_is_gone():
    assert not hasattr(EntityExtractionService, "extract_batch_with_dspy")


def test_extract_with_ontology_works_on_the_service_and_warns():
    service = EntityExtractionService(
        _config(), llm_func=lambda p: '[{"text": "Ada", "type": "PERSON"}]'
    )

    with pytest.warns(DeprecationWarning, match="extract_entities"):
        entities = service.extract_with_ontology("Ada wrote code", _docs("d1")[0])

    assert [e.text for e in entities] == ["Ada"]


def test_the_base_class_no_longer_offers_a_method_that_cannot_work():
    assert not hasattr(ee.OntologyAwareEntityExtractor, "extract_with_ontology")


# --------------------------------------------------------------------------- batch processor


@pytest.fixture(autouse=True)
def _no_catalog(monkeypatch):
    """The recording cursors here cannot answer catalog queries.

    The legacy-table repair (`ensure_confidence_column`) has its own unit tests and a
    live one in tests/integration/test_legacy_entities_table.py; these tests are about
    the SQL the writers send.
    """
    for module in (
        "iris_vector_rag.services.batch_entity_processor",
        "iris_vector_rag.services.storage",
    ):
        monkeypatch.setattr(f"{module}.ensure_confidence_column", lambda *a, **k: False)


class FakeCursor:
    """Records statements and answers the processor's reads from `db_ids`."""

    def __init__(self, db_ids=(), fail=None):
        self.db_ids = set(db_ids)
        self.fail = fail  # callable(sql, rows) -> exception or None
        self.many: list[tuple[str, list]] = []
        self.executed: list[tuple[str, list]] = []
        self._result: list = []

    def execute(self, sql, params=None):
        sql = " ".join(str(sql).split())
        params = list(params or [])
        self.executed.append((sql, params))
        if "COUNT(*)" in sql:
            self._result = [(len([p for p in params if p in self.db_ids]),)]
        elif sql.startswith("SELECT entity_id"):
            self._result = [(p,) for p in params if p in self.db_ids]
        else:
            self._result = []

    def executemany(self, sql, rows):
        sql = " ".join(str(sql).split())
        rows = [tuple(r) for r in rows]
        if self.fail:
            exc = self.fail(sql, rows)
            if exc:
                raise exc
        self.many.append((sql, rows))
        if sql.startswith("INSERT"):
            self.db_ids.update(r[0] for r in rows)

    def fetchall(self):
        return list(self._result)

    def fetchone(self):
        return self._result[0] if self._result else None

    def close(self):
        pass


def _processor(cursor, batch_size=32, resolver=None):
    conn = MagicMock()
    conn.cursor.return_value = cursor
    manager = Mock()
    manager.get_connection.return_value = conn
    processor = BatchEntityProcessor(
        manager, {}, batch_size=batch_size, source_document_resolver=resolver
    )
    processor._conn = conn
    return processor


def _entity(n, *, embedding=None, confidence=0.8, doc="d1"):
    metadata = {} if embedding is None else {"embedding": embedding}
    return Entity(
        id=f"e{n}",
        text=f"name{n}",
        entity_type="PERSON",
        confidence=confidence,
        start_offset=0,
        end_offset=1,
        source_document_id=doc,
        metadata=metadata,
    )


def test_a_mixed_batch_sends_one_tuple_width_per_statement():
    cursor = FakeCursor()
    processor = _processor(cursor)
    entities = [
        _entity(1, embedding=[0.1, 0.2, 0.3]),
        _entity(2),
        _entity(3, embedding=[0.3, 0.2, 0.1]),
    ]

    result = processor.store_entities_batch(entities)

    assert result["entities_stored"] == 3
    assert result["validation_passed"] is True
    for sql, rows in cursor.many:
        assert len({len(r) for r in rows}) == 1, sql
        assert sql.count("?") == len(rows[0]), sql


def test_an_empty_embedding_is_no_embedding():
    cursor = FakeCursor()
    processor = _processor(cursor)

    result = processor.store_entities_batch([_entity(1, embedding=[]), _entity(2)])

    assert result["entities_stored"] == 2
    assert all("TO_VECTOR" not in sql for sql, _ in cursor.many)


def test_the_vector_width_comes_from_the_embedding_not_a_constant():
    cursor = FakeCursor()
    processor = _processor(cursor)

    processor.store_entities_batch([_entity(1, embedding=[0.5] * 7)])

    sql, _rows = cursor.many[0]
    assert "TO_VECTOR(?, FLOAT, 7)" in sql
    assert "384" not in sql


def test_confidence_is_written():
    cursor = FakeCursor()
    processor = _processor(cursor)

    processor.store_entities_batch([_entity(1, confidence=0.42)])

    sql, rows = cursor.many[0]
    assert "confidence" in sql
    assert 0.42 in rows[0]


def test_an_existing_entity_id_is_updated_not_inserted():
    cursor = FakeCursor(db_ids={"e1"})
    processor = _processor(cursor)

    result = processor.store_entities_batch([_entity(1), _entity(2)])

    assert result["entities_stored"] == 2
    kinds = {sql.split()[0]: rows for sql, rows in cursor.many}
    assert [r[-1] for r in kinds["UPDATE"]] == ["e1"]
    assert [r[0] for r in kinds["INSERT"]] == ["e2"]


def test_a_repeated_id_inside_one_batch_keeps_the_last():
    cursor = FakeCursor()
    processor = _processor(cursor)

    result = processor.store_entities_batch(
        [_entity(1, confidence=0.1), _entity(1, confidence=0.9)]
    )

    assert result["entities_stored"] == 1
    assert result["validation_passed"] is True
    inserted = [
        r for sql, rows in cursor.many if sql.startswith("INSERT") for r in rows
    ]
    assert len(inserted) == 1 and 0.9 in inserted[0]


def test_the_source_document_goes_through_the_resolver():
    cursor = FakeCursor()
    processor = _processor(cursor, resolver=lambda cur, raw: f"resolved-{raw}")

    processor.store_entities_batch([_entity(1, doc="chunk-7")])

    _sql, rows = cursor.many[0]
    assert "resolved-chunk-7" in rows[0]


def test_batch_size_chunks_the_work():
    cursor = FakeCursor()
    processor = _processor(cursor, batch_size=10)

    result = processor.store_entities_batch([_entity(n) for n in range(25)])

    inserts = [rows for sql, rows in cursor.many if sql.startswith("INSERT")]
    assert [len(r) for r in inserts] == [10, 10, 5]
    assert result["entities_stored"] == 25
    assert processor._conn.commit.call_count >= 3


def test_a_failing_chunk_is_retried_smaller_and_only_the_bad_row_is_reported():
    def fail(sql, rows):
        if any(r[0] == "e3" for r in rows):
            return RuntimeError("row e3 rejected")

    cursor = FakeCursor(fail=fail)
    processor = _processor(cursor, batch_size=8)

    result = processor.store_entities_batch([_entity(n) for n in range(6)])

    assert result["entities_stored"] == 5
    assert result["validation_passed"] is False
    assert [f["entity_id"] for f in result["failed"]] == ["e3"]
    assert "rejected" in result["failed"][0]["reason"]
    assert "e3" in result["error"]
    assert processor._conn.rollback.called


def test_validate_count_compares_with_the_database():
    class ForgetfulCursor(FakeCursor):
        def executemany(self, sql, rows):  # accepts the rows, keeps none
            self.many.append((sql, [tuple(r) for r in rows]))

    cursor = ForgetfulCursor()
    processor = _processor(cursor)

    result = processor.store_entities_batch(
        [_entity(1), _entity(2)], validate_count=True
    )

    assert result["validation_passed"] is False
    assert "count" in result["error"].lower()


def test_entities_and_relationships_reports_the_entity_error():
    def fail(sql, rows):
        return RuntimeError("boom")

    processor = _processor(FakeCursor(fail=fail))

    result = processor.store_entities_and_relationships([_entity(1)], [])

    assert result["validation_passed"] is False
    assert "boom" in result["error"]


def test_the_claim_of_atomicity_is_gone():
    doc = BatchEntityProcessor.store_entities_and_relationships.__doc__
    assert "NONE are stored" not in doc
    assert "not atomic" in doc.lower()


# --------------------------------------------------------------------------- storage adapter


class ScriptedCursor:
    def __init__(self, rows=(), execute_error=None):
        self.rows = list(rows)
        self.execute_error = execute_error
        self.statements: list[tuple[str, list]] = []

    def execute(self, sql, params=None):
        self.statements.append((" ".join(str(sql).split()), list(params or [])))
        if self.execute_error and (
            sql.lstrip().upper().startswith(("INSERT", "UPDATE"))
        ):
            raise self.execute_error

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def close(self):
        pass


def _adapter(cursor, entities_table="ESFIX.Entities"):
    conn = MagicMock()
    conn.cursor.return_value = cursor
    manager = Mock()
    manager.get_connection.return_value = conn
    adapter = EntityStorageAdapter(
        manager,
        {
            "entity_extraction": {
                "storage": {
                    "entities_table": entities_table,
                    "relationships_table": "ESFIX.EntityRelationships",
                }
            }
        },
    )
    adapter._tables_ensured = True
    adapter._ensure_entity_columns = lambda: None
    adapter._resolve_source_document = lambda cur, src: src
    adapter._conn = conn
    return adapter


def test_a_failed_write_does_not_appear_in_search():
    cursor = ScriptedCursor(execute_error=RuntimeError("insert refused"))
    adapter = _adapter(cursor)

    stored = adapter.store_entity(_entity(1))

    assert stored is False
    assert "e1" not in adapter._entity_cache
    assert adapter.search_entities("name1", fuzzy=True) == []


def test_a_committed_write_is_cached():
    adapter = _adapter(ScriptedCursor(rows=[(0,)]))

    assert adapter.store_entity(_entity(1)) is True
    assert "e1" in adapter._entity_cache


def test_store_entity_vector_width_follows_the_embedding():
    cursor = ScriptedCursor(rows=[(0,)])
    adapter = _adapter(cursor)

    adapter.store_entity(_entity(1, embedding=[0.1] * 5))

    sql = next(s for s, _ in cursor.statements if s.startswith("INSERT"))
    assert "TO_VECTOR(?, FLOAT, 5)" in sql


def test_get_entities_by_document_queries_by_document():
    cursor = ScriptedCursor(rows=[("e1", "Ada", "PERSON", "d1", "desc", 0.9)])
    adapter = _adapter(cursor)

    entities = adapter.get_entities_by_document("d1")

    sql, params = cursor.statements[-1]
    assert "FROM ESFIX.Entities" in sql and "source_doc_id = ?" in sql
    assert params == ["d1"]
    assert [(e.id, e.text, e.entity_type, e.confidence) for e in entities] == [
        ("e1", "Ada", "PERSON", 0.9)
    ]
    assert entities[0].source_document_id == "d1"


def test_get_entities_by_type_is_capped_with_top():
    cursor = ScriptedCursor(rows=[])
    adapter = _adapter(cursor)

    assert adapter.get_entities_by_type("PERSON", limit=5) == []

    sql, params = cursor.statements[-1]
    assert sql.startswith("SELECT TOP 5 ") and "FETCH FIRST" not in sql
    assert params == ["PERSON"]


def test_get_relationships_by_document_returns_relationships():
    cursor = ScriptedCursor(rows=[("r1", "e1", "e2", "knows", 0.7, "d1")])
    adapter = _adapter(cursor)

    rels = adapter.get_relationships_by_document("d1")

    sql, params = cursor.statements[-1]
    assert "FROM ESFIX.EntityRelationships" in sql and "source_document = ?" in sql
    assert params == ["d1"]
    assert isinstance(rels[0], Relationship)
    assert (rels[0].id, rels[0].source_entity_id, rels[0].target_entity_id) == (
        "r1",
        "e1",
        "e2",
    )


def test_a_read_that_cannot_run_raises():
    cursor = ScriptedCursor()
    cursor.execute = Mock(side_effect=RuntimeError("table gone"))
    adapter = _adapter(cursor)

    for call in (
        lambda: adapter.get_entities_by_document("d1"),
        lambda: adapter.get_entities_by_type("PERSON"),
        lambda: adapter.get_relationships_by_document("d1"),
        lambda: adapter.get_storage_stats(),
    ):
        with pytest.raises(RuntimeError, match="table gone"):
            call()


def test_storage_stats_are_counted_not_invented():
    class StatsCursor(ScriptedCursor):
        def execute(self, sql, params=None):
            super().execute(sql, params)
            s = " ".join(sql.split())
            if "GROUP BY entity_type" in s:
                self.rows = [("PERSON", 3), ("ORGANIZATION", 1)]
            elif "GROUP BY relationship_type" in s:
                self.rows = [("knows", 2)]
            elif "FROM ESFIX.Entities" in s:
                self.rows = [(4,)]
            else:
                self.rows = [(2,)]

    adapter = _adapter(StatsCursor())

    stats = adapter.get_storage_stats()

    assert stats["entities_count"] == 4
    assert stats["relationships_count"] == 2
    assert stats["entity_types"] == {"PERSON": 3, "ORGANIZATION": 1}
    assert stats["relationship_types"] == {"knows": 2}


def test_create_tables_if_not_exist_really_ensures_the_tables():
    adapter = _adapter(ScriptedCursor())
    adapter._ensure_kg_tables = Mock()

    assert adapter.create_tables_if_not_exist() is True

    adapter._ensure_kg_tables.assert_called_once()


def test_create_tables_if_not_exist_raises_when_it_cannot():
    adapter = _adapter(ScriptedCursor())
    adapter._ensure_kg_tables = Mock(side_effect=RuntimeError("no ddl rights"))

    with pytest.raises(RuntimeError, match="no ddl rights"):
        adapter.create_tables_if_not_exist()


def test_the_kg_tables_are_ensured_in_the_entities_table_schema(monkeypatch):
    seen = {}

    class FakeSchemaManager:
        def __init__(self, connection_manager, config_manager, schema_prefix=None):
            seen["prefix"] = schema_prefix

        def ensure_table_schema(self, name):
            seen.setdefault("tables", []).append(name)
            return True

    monkeypatch.setattr(
        "iris_vector_rag.services.storage.SchemaManager", FakeSchemaManager
    )
    adapter = _adapter(ScriptedCursor())
    adapter._tables_ensured = False
    adapter._ensure_entity_columns = lambda: None

    adapter._ensure_kg_tables()

    assert seen["prefix"] == "ESFIX"
    assert seen["tables"] == ["Entities", "EntityRelationships"]


def test_search_docstring_does_not_promise_sql_like():
    doc = EntityStorageAdapter.search_entities.__doc__
    assert "SQL LIKE" not in doc
    assert "Levenshtein" in doc


# --------------------------------------------------------------------------- GraphRAG load


def _pipeline(service, storage):
    from iris_vector_rag.pipelines.graphrag import GraphRAGPipeline

    pipeline = object.__new__(GraphRAGPipeline)
    pipeline.entity_extraction_service = service
    pipeline.entity_extraction_enabled = True
    pipeline.vector_store = Mock()
    pipeline.connection_manager = Mock()
    pipeline.config_manager = Mock()
    pipeline.config_manager._config = {}
    return pipeline


@pytest.fixture
def no_schema_or_adapter(monkeypatch):
    stored = []
    adapter = Mock()
    adapter.store_entities_batch.side_effect = lambda ents: (
        stored.extend(e.source_document_id for e in ents) or len(ents)
    )
    adapter.store_relationships_batch.side_effect = lambda rels: len(rels)
    monkeypatch.setattr(
        "iris_vector_rag.services.storage.EntityStorageAdapter",
        lambda **kwargs: adapter,
    )
    monkeypatch.setattr(
        "iris_vector_rag.pipelines.graphrag.SchemaManager", lambda *a, **k: Mock()
    )
    return stored, adapter


def test_load_documents_reports_a_failed_document_and_stores_the_rest(
    no_schema_or_adapter,
):
    stored, _ = no_schema_or_adapter
    service = EntityExtractionService(_config(), llm_func=_llm_failing_on("d2"))
    pipeline = _pipeline(service, None)

    result = pipeline.load_documents("", documents=_docs("d1", "d2", "d3"))

    assert [f["document_id"] for f in result["failed"]] == ["d2"]
    assert "timed out" in result["failed"][0]["reason"]
    assert sorted(stored) == ["d1", "d3"]
    assert result["documents_loaded"] == 3


def test_load_documents_lists_a_document_whose_entities_could_not_be_stored(
    no_schema_or_adapter,
):
    from iris_vector_rag.services.storage import EntityStorageError

    _, adapter = no_schema_or_adapter
    calls = []

    def store(ents):
        calls.append(ents[0].source_document_id)
        if ents[0].source_document_id == "d1":
            raise EntityStorageError("1 of 1 entities not stored")
        return len(ents)

    adapter.store_entities_batch.side_effect = store
    service = EntityExtractionService(_config(), llm_func=_llm_failing_on("nobody"))

    result = _pipeline(service, None).load_documents("", documents=_docs("d1", "d2"))

    assert calls == ["d1", "d2"]
    assert [f["document_id"] for f in result["failed"]] == ["d1"]


def test_load_documents_raises_when_every_document_fails(
    no_schema_or_adapter, monkeypatch
):
    # a batch that fails as a whole is retried with a backoff (FR-005): don't wait it out
    monkeypatch.setattr("iris_vector_rag.common.batch_utils.time.sleep", lambda s: None)

    def llm(prompt):
        raise ConnectionError("llm down")

    service = EntityExtractionService(_config(), llm_func=llm)

    with pytest.raises(EntityExtractionError, match="every one of 6"):
        _pipeline(service, None).load_documents(
            "", documents=_docs("a", "b", "c", "d", "e", "f")  # two batches of 5 and 1
        )


def test_a_query_without_an_llm_finds_no_seed_entities_instead_of_crashing(caplog):
    service = EntityExtractionService(_config())  # no llm_func
    pipeline = _pipeline(service, None)

    with caplog.at_level("WARNING"):
        seeds = pipeline._find_seed_entities("what treats diabetes?")

    assert seeds == []
    assert "Seed entity extraction failed" in caplog.text
