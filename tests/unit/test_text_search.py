"""The text leg's SQL, the schema manager's text index, and the store's text search.

Everything here is a fake cursor or a recorded catalog: the statements are the
behaviour under test. `tests/integration/test_text_search_iris.py` runs the same
statements against a real IRIS, which is where their meaning is verified.
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock, patch

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.exceptions import SchemaIndexError, TextIndexMissingError
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.table_shape import RoleNotPresent
from iris_vector_rag.storage.text_search import (
    TEXT_INDEX_NAME,
    build_text_search_sql,
    create_text_index_statement,
    rank_procedure,
    search_terms,
)
from tests.fixtures.shapes.ddl import SHAPE_SCHEMA, SHAPE_TABLES
from tests.unit.test_vector_store_shape import FakeCursor, make_store

# --- search_terms -----------------------------------------------------------


def test_a_question_becomes_an_or_of_its_content_words():
    assert search_terms("What does insulin do?") == "insulin"
    assert search_terms("insulin and glucose regulation") == (
        "insulin OR glucose OR regulation"
    )


def test_operators_wildcards_and_punctuation_cannot_reach_the_index():
    assert (
        search_terms("insulin* OR (glucose NOT sugar)") == "insulin OR glucose OR sugar"
    )
    assert search_terms("'; DROP TABLE x; --") == "drop OR table OR x"


def test_terms_are_lower_cased_deduplicated_and_capped():
    assert search_terms("Insulin INSULIN insulin") == "insulin"
    many = " ".join(f"w{i}" for i in range(100))
    assert len(search_terms(many).split(" OR ")) == 32


def test_a_query_with_nothing_to_search_for_is_empty():
    assert search_terms("") == ""
    assert search_terms("???") == ""
    assert search_terms("and or not") == ""


def test_a_query_of_only_stop_words_keeps_them():
    assert search_terms("what is the") == "what OR is OR the"


# --- the statements ---------------------------------------------------------


def test_the_index_ddl_names_schema_table_column_and_class():
    assert create_text_index_statement("RAG", "SourceDocuments", "text_content") == (
        f"CREATE INDEX {TEXT_INDEX_NAME} ON RAG.SourceDocuments (text_content) "
        "AS %iFind.Index.Basic(LANGUAGE='en', LOWER=1)"
    )


@pytest.mark.parametrize("bad", ["x; DROP TABLE y", "a b", "", "t)--"])
def test_the_ddl_refuses_unsafe_identifiers(bad):
    with pytest.raises(ValueError):
        create_text_index_statement("RAG", bad, "text_content")
    with pytest.raises(ValueError):
        create_text_index_statement("RAG", "T", bad)


def test_the_rank_procedure_is_named_from_the_class_not_the_table():
    assert rank_procedure("RAG", "RAG.SourceDocuments") == (
        f"RAG.SourceDocuments_{TEXT_INDEX_NAME}Rank"
    )
    # RAG.llm_cache is class RAG.llmcache
    assert rank_procedure("RAG", "RAG.llmcache", "Ix") == "RAG.llmcache_IxRank"


def test_the_search_sql_uses_find_syntax_a_top_and_a_rank_order():
    sql = build_text_search_sql(
        table="RAG.SourceDocuments",
        id_column="doc_id",
        index_name=TEXT_INDEX_NAME,
        rank_proc="RAG.SourceDocuments_SourceDocumentsTextIdxRank",
        top_k=7,
    )
    assert sql.startswith("SELECT TOP 7 doc_id, ")
    assert f"%ID %FIND search_index({TEXT_INDEX_NAME}, ?)" in sql
    assert (
        "ORDER BY CAST(RAG.SourceDocuments_SourceDocumentsTextIdxRank(%ID, ?) AS DOUBLE) DESC"
        in sql
    )
    assert sql.count("?") == 3
    for forbidden in ("FETCH FIRST", "LIKE", ";", "%FIND(text"):
        assert forbidden not in sql


def test_the_filter_predicate_joins_the_search_with_and():
    sql = build_text_search_sql(
        table="RAG.T",
        id_column="id",
        index_name="Ix",
        rank_proc="RAG.T_IxRank",
        top_k=3,
        extra_where="(meta LIKE '%x%' OR meta LIKE '%y%')",
    )
    assert "search_index(Ix, ?) AND (meta LIKE '%x%' OR meta LIKE '%y%') ORDER" in sql
    assert "((" not in sql, "the IRIS DBAPI miscounts parameters around a doubled paren"


# --- SchemaManager.ensure_text_index ----------------------------------------


class Recorder:
    def __init__(self, fail_on=None):
        self.statements = []
        self.closed = False
        self.fail_on = fail_on

    def execute(self, sql, params=None):
        self.statements.append(sql)
        if self.fail_on and self.fail_on in sql:
            raise RuntimeError("iFind is not available here")

    def close(self):
        self.closed = True


class RecordedIndexes:
    def __init__(self, *names):
        self.names = {n.upper() for n in names}

    def __call__(self, connection):
        return self

    def index_names(self, schema, table, cursor=None):
        return set(self.names)


def manager_over(recorded_catalog, cursor, shape="canonical"):
    connection = MagicMock()
    connection.cursor.return_value = cursor
    connections = MagicMock()
    connections.get_connection.return_value = connection
    with patch.object(SchemaManager, "ensure_schema_metadata_table"):
        manager = SchemaManager(
            connections,
            ConfigurationManager(),
            schema_prefix=SHAPE_SCHEMA,
            base_embedding_dimension=384,
        )
    manager._snapshot = lambda table, schema=None: recorded_catalog.snapshot(
        schema or SHAPE_SCHEMA, table
    )
    return manager, connection, SHAPE_TABLES[shape]


def test_the_index_goes_on_the_column_that_fills_the_text_role(recorded_catalog):
    for shape, column in [("canonical", "text_content"), ("fixture", "content")]:
        cursor = Recorder()
        manager, connection, table = manager_over(recorded_catalog, cursor, shape)
        with patch(
            "iris_vector_rag.storage.schema_manager.IRISCatalogReader",
            RecordedIndexes(),
        ):
            assert manager.ensure_text_index(table) == TEXT_INDEX_NAME
        assert cursor.statements == [
            create_text_index_statement(SHAPE_SCHEMA, table, column)
        ], shape
        connection.commit.assert_called_once()
        assert cursor.closed


def test_an_existing_index_is_left_alone(recorded_catalog):
    cursor = Recorder()
    manager, connection, table = manager_over(recorded_catalog, cursor)
    with patch(
        "iris_vector_rag.storage.schema_manager.IRISCatalogReader",
        RecordedIndexes(TEXT_INDEX_NAME),
    ):
        assert manager.ensure_text_index(table) == TEXT_INDEX_NAME
    assert cursor.statements == []


def test_a_refused_index_raises_and_rolls_back(recorded_catalog):
    cursor = Recorder(fail_on="CREATE INDEX")
    manager, connection, table = manager_over(recorded_catalog, cursor)
    with patch(
        "iris_vector_rag.storage.schema_manager.IRISCatalogReader", RecordedIndexes()
    ):
        with pytest.raises(SchemaIndexError, match="iFind is not available"):
            manager.ensure_text_index(table)
    connection.rollback.assert_called_once()
    connection.commit.assert_not_called()
    assert cursor.closed


def test_a_missing_table_raises(recorded_catalog):
    manager, _, _ = manager_over(recorded_catalog, Recorder())
    with pytest.raises(SchemaIndexError, match="does not exist"):
        manager.ensure_text_index("NoSuchTable")


def test_a_qualified_name_names_its_own_schema(recorded_catalog):
    cursor = Recorder()
    manager, _, table = manager_over(recorded_catalog, cursor)
    manager.schema_prefix = "Elsewhere"
    with patch(
        "iris_vector_rag.storage.schema_manager.IRISCatalogReader", RecordedIndexes()
    ):
        manager.ensure_text_index(f"{SHAPE_SCHEMA}.{table}")
    assert f"ON {SHAPE_SCHEMA}.{table} " in cursor.statements[0]


def test_ensure_table_schema_for_source_documents_ensures_the_index_and_reports_failure():
    with patch.object(SchemaManager, "ensure_schema_metadata_table"):
        manager = SchemaManager(
            Mock(), ConfigurationManager(), base_embedding_dimension=384
        )
    with patch.object(
        manager, "_ensure_standard_table", return_value=True
    ), patch.object(
        manager, "ensure_text_index", return_value=TEXT_INDEX_NAME
    ) as ensure:
        assert manager.ensure_table_schema("SourceDocuments") is True
        assert manager.ensure_table_schema("SourceDocuments") is True
    ensure.assert_called_once_with("SourceDocuments")  # once per manager, not per call

    with patch.object(
        manager, "_ensure_standard_table", return_value=True
    ), patch.object(manager, "_text_indexed", set()), patch.object(
        manager, "ensure_text_index", side_effect=SchemaIndexError("no iFind")
    ):
        with pytest.raises(SchemaIndexError, match="no iFind"):
            manager.ensure_table_schema("SourceDocuments")


def test_other_tables_get_no_text_index():
    with patch.object(SchemaManager, "ensure_schema_metadata_table"):
        manager = SchemaManager(
            Mock(), ConfigurationManager(), base_embedding_dimension=384
        )
    with patch.object(
        manager, "_ensure_standard_table", return_value=True
    ), patch.object(manager, "ensure_text_index") as ensure:
        assert manager.ensure_table_schema("DocumentChunks") is True
    ensure.assert_not_called()


# --- IRISVectorStore.text_search --------------------------------------------


class TextCursor(FakeCursor):
    """Answers the text search, then the by-id fetch the store does to hydrate."""

    def __init__(self, hits, docs):
        super().__init__()
        self.hits = hits
        self.docs = docs

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        if "SEARCH_INDEX" in sql.upper():
            self._result = list(self.hits)
        else:
            self._result = list(self.docs)


def store_with_index(recorded_catalog, cursor, shape="canonical", present=True):
    store = make_store(shape, cursor, recorded_catalog)
    store.schema_manager.has_text_index.return_value = present
    return store


def test_text_search_runs_the_find_query_and_returns_ranked_documents(recorded_catalog):
    cursor = TextCursor(
        hits=[("b", 0.9), ("a", 0.2)],
        docs=[
            ("a", "alpha", None),
            ("b", "beta", None),
        ],
    )
    store = store_with_index(recorded_catalog, cursor)

    docs = store.text_search("insulin glucose", top_k=2)

    assert [d.id for d in docs] == ["b", "a"]
    assert [d.metadata["text_score"] for d in docs] == [0.9, 0.2]
    sql, params = cursor.executed[0]
    assert "%ID %FIND search_index(" in sql and "TOP 2" in sql
    assert "ShapeTest.ShapeCanonical" in sql
    assert params == ["insulin OR glucose"] * 3


def test_text_search_without_the_index_raises_naming_the_fix(recorded_catalog):
    cursor = TextCursor(hits=[], docs=[])
    store = store_with_index(recorded_catalog, cursor, present=False)

    with pytest.raises(TextIndexMissingError) as caught:
        store.text_search("insulin")

    message = str(caught.value)
    assert TEXT_INDEX_NAME in message and "ensure_text_index" in message
    assert "ShapeTest.ShapeCanonical" in message
    assert cursor.executed == []


def test_text_search_with_no_searchable_word_asks_the_database_nothing(
    recorded_catalog,
):
    cursor = TextCursor(hits=[], docs=[])
    store = store_with_index(recorded_catalog, cursor)
    assert store.text_search("???") == []
    assert cursor.executed == []


def test_the_metadata_filter_is_pushed_into_the_sql(recorded_catalog):
    cursor = TextCursor(hits=[("a", 1.0)], docs=[("a", "alpha", '{"source": "bio"}')])
    store = store_with_index(recorded_catalog, cursor)

    docs = store.text_search("alpha", top_k=3, metadata_filter={"source": "bio"})

    sql = cursor.executed[0][0]
    assert "%FIND" in sql and 'LIKE \'%"source":"bio"%\'' in sql
    assert [d.metadata["source"] for d in docs] == ["bio"]
    assert [d.id for d in docs] == ["a"]


def test_a_filter_on_a_table_without_metadata_is_an_error(recorded_catalog):
    cursor = TextCursor(hits=[], docs=[])
    store = store_with_index(recorded_catalog, cursor, shape="no_metadata")
    with pytest.raises(RoleNotPresent):
        store.text_search("alpha", metadata_filter={"source": "bio"})
