"""The test infrastructure's idea of the document table is the library's (T058).

`tests/utils/schema_models.py` carried its own column list for `RAG.SourceDocuments`
-- `id`, `source`, `content`, `metadata`, `created_at`, `test_run_id`, and no
embedding column at all. Two accounts of one table, and the consequences were the
ones that account predicts:

* `SchemaValidator` rejected a canonical document table, because it has `doc_id` and
  not `id`, so `database_with_clean_schema` called `SchemaResetter`.
* `SchemaResetter` then replaced it with the legacy shape, which cannot answer a
  vector search at all -- and never put it back. Every later test in the session
  vector-searching that table failed, which is the order dependence that kept
  `test_graphrag_handler_execute_hybrid_search` in `ci_known_failures.txt`.
* The validator could not even accept the resetter's own output: it expects `JSON`
  where the resetter writes `VARCHAR(MAX)`, which IRIS reports back as
  `LONGVARCHAR`. Validation therefore failed on every run, so the reset was not a
  rare repair -- it happened for every test class that asked for a clean schema.

The document and chunk tables are now described by `CANONICAL_DOCUMENT_SHAPE` and
`CANONICAL_CHUNK_SHAPE`, and created by the same `create_table_statement` the library
uses, so the test infrastructure cannot disagree with the code under test (FR-006).
"""

from __future__ import annotations

from iris_vector_rag.storage.table_shape import (
    CANONICAL_CHUNK_SHAPE,
    CANONICAL_DOCUMENT_SHAPE,
    ColumnRole,
    create_table_statement,
)
from tests.fixtures.schema_reset import SchemaResetter
from tests.utils.schema_models import ColumnType, get_expected_rag_schema
from tests.utils.schema_validator import SchemaValidator


def definition_for(table: str):
    for schema_def in get_expected_rag_schema():
        if schema_def.table_name == table:
            return schema_def
    raise AssertionError(f"no expected schema for {table}")


def test_expected_document_columns_are_the_canonical_ones():
    definition = definition_for("SourceDocuments")

    assert [column.name for column in definition.columns] == [
        column.name
        for column in sorted(
            CANONICAL_DOCUMENT_SHAPE.columns.values(), key=lambda c: c.position
        )
    ]
    assert definition.primary_key == "doc_id"


def test_expected_chunk_columns_are_the_canonical_ones():
    definition = definition_for("DocumentChunks")

    assert [column.name for column in definition.columns] == [
        column.name
        for column in sorted(
            CANONICAL_CHUNK_SHAPE.columns.values(), key=lambda c: c.position
        )
    ]


def test_the_expected_document_table_is_vector_searchable():
    """The point of the whole exercise: a reset leaves a table search can use."""
    definition = definition_for("SourceDocuments")
    embedding = CANONICAL_DOCUMENT_SHAPE.column_for(ColumnRole.EMBEDDING).name

    column = definition.get_column(embedding)
    assert column is not None, f"no {embedding} column in the expected schema"
    assert column.column_type is ColumnType.VECTOR


def test_the_resetter_creates_the_document_table_the_library_creates():
    """One `CREATE TABLE` for one table, rendered from the shape."""
    statement = SchemaResetter()._build_create_table_sql(
        definition_for("SourceDocuments")
    )

    assert statement.strip() == create_table_statement(CANONICAL_DOCUMENT_SHAPE)


def test_a_vector_column_satisfies_a_vector_expectation():
    """`INFORMATION_SCHEMA` reports `VECTOR(FLOAT,384)` as `varchar`.

    A driver limitation, not a schema defect, so the validator cannot demand the
    word VECTOR back -- doing so is what made validation always fail.
    """
    validator = SchemaValidator()

    assert validator.satisfies(ColumnType.VECTOR, "varchar")
    assert validator.satisfies(ColumnType.VECTOR, "varbinary")


def test_a_long_text_column_satisfies_a_json_expectation():
    """The resetter writes `VARCHAR(MAX)` for JSON; IRIS reports `longvarchar`."""
    validator = SchemaValidator()

    assert validator.satisfies(ColumnType.JSON, "longvarchar")
    assert validator.satisfies(ColumnType.JSON, "varchar")
    assert not validator.satisfies(ColumnType.JSON, "integer")


def test_the_graph_tables_are_not_described_a_second_time():
    """`Entities` and `Relationships` were hand-written here, and wrong.

    The library writes `entity_id` / `source_doc_id` / `confidence` and names the
    second table `EntityRelationships`. The hand-written `Entities` had a required
    `id` and no `confidence`, so `SchemaResetter` (run for every test class that asked
    for a clean schema, because `Relationships` never exists and validation always
    failed) left `RAG.Entities` in a shape every later entity write failed on: the
    GraphRAG and ColBERT integration tests then errored at setup in a session that
    had run the contract suite first. The library creates its own graph tables on
    demand, so the test infrastructure does not describe them at all.
    """
    described = {schema_def.table_name for schema_def in get_expected_rag_schema()}

    assert described == {"SourceDocuments", "DocumentChunks"}


def test_the_resetter_drops_stale_graph_tables_before_the_document_table():
    """A graph table left in a non-canonical shape is removed, children first."""
    statements = []

    class Cursor:
        def execute(self, sql, *args):
            statements.append(sql)

    class Connection:
        def cursor(self):
            return Cursor()

        def commit(self):
            pass

        def rollback(self):
            pass

    import iris_vector_rag.session as session_module

    class Session:
        def connection_for_adapter(self):
            return Connection()

    original = session_module.session_for
    session_module.session_for = lambda *a, **k: Session()
    try:
        SchemaResetter().reset_schema()
    finally:
        session_module.session_for = original

    drops = [s for s in statements if s.startswith("DROP TABLE")]
    names = [s.rsplit(" ", 1)[1] for s in drops]
    assert names.index("RAG.EntityRelationships") < names.index("RAG.Entities")
    assert names.index("RAG.Entities") < names.index("RAG.SourceDocuments")
    assert not any(s.startswith("CREATE TABLE RAG.Entit") for s in statements)
