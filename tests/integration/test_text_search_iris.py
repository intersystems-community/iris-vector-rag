"""Full-text search over the document table, against a real IRIS.

The text index is created by `SchemaManager`, queried by `IRISVectorStore.text_search`,
and these tests run both for real: iFind builds the index, IRIS ranks the rows. Nothing
is mocked and no embedder is involved: text search reads the text column alone, so the
tables here are populated with SQL (the way a deployment that predates the index has
them). Hybrid and rrf retrieval, which do embed, are in `test_hybrid_rrf_modes.py`.

Each test works in schemas of its own (`TXTSRCH`, `TXTSRCHB`); nothing here touches `RAG`.
"""

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.exceptions import SchemaIndexError, TextIndexMissingError
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.table_shape import RoleNotPresent
from iris_vector_rag.storage.text_search import TEXT_INDEX_NAME
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from tests.fixtures.shapes.ddl import create_statement, drop_statement

pytestmark = pytest.mark.integration

A, B = "TXTSRCH", "TXTSRCHB"

CORPUS = [
    ("d1", "Insulin regulates blood glucose levels.", '{"source": "bio"}'),
    (
        "d2",
        "Diabetes affects glucose metabolism and insulin response.",
        '{"source": "bio"}',
    ),
    ("d3", "Neural networks learn from training data.", '{"source": "cs"}'),
    ("d4", "Transformer architecture uses attention mechanisms.", '{"source": "cs"}'),
    ("d5", "Pancreatic beta cells secrete insulin.", '{"source": "bio"}'),
]


def _run(connection, sql, params=None):
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params or [])
        rows = cursor.fetchall() if cursor.description else None
        connection.commit()
        return rows
    finally:
        cursor.close()


def _indexes(connection, schema, table):
    rows = _run(
        connection,
        "SELECT INDEX_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.INDEXES "
        "WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?",
        [schema, table],
    )
    return {str(name).upper(): str(column) for name, column in rows}


@pytest.fixture
def clean(iris_connection):
    def drop():
        for schema in (A, B):
            for table in ("SourceDocuments", "Legacy", "Mixed"):
                _run(iris_connection, f"DROP TABLE IF EXISTS {schema}.{table}")

    drop()
    yield iris_connection
    drop()


@pytest.fixture
def stack():
    config = ConfigurationManager()
    return config, ConnectionManager(config)


def _manager(stack, schema):
    config, connections = stack
    return SchemaManager(connections, config, schema_prefix=schema)


def _store(stack, monkeypatch, schema, table="SourceDocuments"):
    monkeypatch.setenv("IRIS_SCHEMA_PREFIX", schema)
    monkeypatch.setenv("RAG_STORAGE__IRIS__TABLE_NAME", f"{schema}.{table}")
    config = ConfigurationManager()
    return IRISVectorStore(ConnectionManager(config), config)


def _legacy_table(connection, schema, rows=CORPUS):
    """A populated table from before the text index: `fixture` shape, `content` column."""
    _run(connection, create_statement("fixture", schema, "Legacy"))
    for doc_id, text, metadata in rows:
        _run(
            connection,
            f"INSERT INTO {schema}.Legacy (doc_id, title, content, metadata) "
            "VALUES (?, ?, ?, ?)",
            [doc_id, "t", text, metadata],
        )


# --- the index: fresh table, legacy table ------------------------------------


def test_a_fresh_source_documents_table_gets_its_text_index(stack, clean):
    manager = _manager(stack, A)
    assert manager.ensure_table_schema("SourceDocuments") is True

    assert _indexes(clean, A, "SourceDocuments")[TEXT_INDEX_NAME.upper()] == (
        "text_content"
    )
    assert manager.has_text_index("SourceDocuments") is True


def test_ensuring_twice_builds_one_index_and_a_new_manager_agrees(stack, clean):
    assert _manager(stack, A).ensure_table_schema("SourceDocuments") is True
    before = _indexes(clean, A, "SourceDocuments")

    second = _manager(stack, A)
    assert second.ensure_text_index("SourceDocuments") == TEXT_INDEX_NAME
    assert second.ensure_text_index("SourceDocuments") == TEXT_INDEX_NAME
    assert second.ensure_table_schema("SourceDocuments") is True

    assert _indexes(clean, A, "SourceDocuments") == before


def test_a_legacy_populated_table_gains_the_index_without_losing_data(
    stack, clean, monkeypatch
):
    _legacy_table(clean, A)
    before = _run(clean, f"SELECT doc_id, title, content, metadata FROM {A}.Legacy")
    assert len(before) == len(CORPUS)
    manager = _manager(stack, A)
    assert manager.has_text_index("Legacy") is False

    assert manager.ensure_text_index("Legacy") == TEXT_INDEX_NAME

    after = _run(clean, f"SELECT doc_id, title, content, metadata FROM {A}.Legacy")
    assert sorted(after) == sorted(before), "no row was lost or changed"
    # on the column that fills the text role in that shape, not a canonical one
    assert _indexes(clean, A, "Legacy")[TEXT_INDEX_NAME.upper()] == "content"
    # the build backfilled: rows that were there before the index are found
    hits = _store(stack, monkeypatch, A, "Legacy").text_search("insulin", top_k=10)
    assert {h.id for h in hits} == {"d1", "d2", "d5"}


def test_a_populated_source_documents_table_gains_the_index_through_ensure_table_schema(
    stack, clean, monkeypatch
):
    manager = _manager(stack, A)
    # created by an older release: the table, its rows and no text index
    _run(
        clean,
        create_statement("canonical", A, "SourceDocuments"),
    )
    for doc_id, text, metadata in CORPUS:
        _run(
            clean,
            f"INSERT INTO {A}.SourceDocuments (doc_id, text_content, metadata) "
            "VALUES (?, ?, ?)",
            [doc_id, text, metadata],
        )
    assert manager.has_text_index("SourceDocuments") is False

    assert manager.ensure_table_schema("SourceDocuments") is True

    assert manager.has_text_index("SourceDocuments") is True
    assert _run(clean, f"SELECT COUNT(*) FROM {A}.SourceDocuments")[0][0] == len(CORPUS)
    hits = _store(stack, monkeypatch, A).text_search("transformer", top_k=5)
    assert [h.id for h in hits] == ["d4"]


def test_the_index_is_scoped_to_its_schema(stack, clean, monkeypatch):
    for schema in (A, B):
        assert _manager(stack, schema).ensure_table_schema("SourceDocuments") is True
    _run(
        clean,
        f"INSERT INTO {A}.SourceDocuments (doc_id, text_content) VALUES ('a1', 'kiwi')",
    )
    _run(
        clean,
        f"INSERT INTO {B}.SourceDocuments (doc_id, text_content) VALUES ('b1', 'kiwi')",
    )

    assert [h.id for h in _store(stack, monkeypatch, A).text_search("kiwi")] == ["a1"]
    assert [h.id for h in _store(stack, monkeypatch, B).text_search("kiwi")] == ["b1"]


def test_a_missing_table_is_an_error_not_a_quiet_false(stack, clean):
    with pytest.raises(SchemaIndexError, match="does not exist"):
        _manager(stack, A).ensure_text_index("SourceDocuments")


def test_migrate_table_rebuilds_the_table_with_its_text_index(stack, clean):
    manager = _manager(stack, A)
    assert manager.ensure_table_schema("SourceDocuments") is True
    _run(clean, f"DROP INDEX {TEXT_INDEX_NAME} ON {A}.SourceDocuments")
    assert manager.has_text_index("SourceDocuments") is False

    assert manager.migrate_table("SourceDocuments") is True

    assert manager.has_text_index("SourceDocuments") is True


# --- searching ---------------------------------------------------------------


@pytest.fixture
def legacy_store(stack, clean, monkeypatch):
    _legacy_table(clean, A)
    _manager(stack, A).ensure_text_index("Legacy")
    return _store(stack, monkeypatch, A, "Legacy")


def test_a_keyword_finds_exactly_the_documents_containing_it(legacy_store):
    assert {h.id for h in legacy_store.text_search("glucose")} == {"d1", "d2"}
    assert {h.id for h in legacy_store.text_search("attention")} == {"d4"}
    assert legacy_store.text_search("xylophone") == []


def test_hits_are_complete_documents_scored_best_first(legacy_store):
    hits = legacy_store.text_search("insulin glucose", top_k=10)

    assert {h.id for h in hits} == {"d1", "d2", "d5"}
    assert all(h.page_content for h in hits)
    assert hits[0].page_content == dict((i, t) for i, t, _ in CORPUS)[hits[0].id]
    scores = [h.metadata["text_score"] for h in hits]
    assert all(isinstance(s, float) and s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_matching_ignores_case_and_a_question_finds_its_content_words(legacy_store):
    assert {h.id for h in legacy_store.text_search("INSULIN")} == {"d1", "d2", "d5"}
    assert [
        h.id for h in legacy_store.text_search("What do attention mechanisms do?")
    ] == ["d4"]


def test_operators_and_punctuation_in_a_query_are_harmless(legacy_store):
    assert {h.id for h in legacy_store.text_search("insulin* ) AND (NOT ; --")} >= {
        "d1",
        "d2",
        "d5",
    }
    assert legacy_store.text_search("???") == []
    assert legacy_store.text_search("") == []


def test_top_k_takes_the_best_of_the_ranking(stack, clean, monkeypatch):
    _run(clean, create_statement("fixture", A, "Legacy"))
    for n in range(1, 31):
        _run(
            clean,
            f"INSERT INTO {A}.Legacy (doc_id, title, content) VALUES (?, ?, ?)",
            [f"r{n:02d}", "t", ("marmot " * n) + "and filler words " * (31 - n)],
        )
    _manager(stack, A).ensure_text_index("Legacy")
    store = _store(stack, monkeypatch, A, "Legacy")

    everything = store.text_search("marmot", top_k=30)
    best = store.text_search("marmot", top_k=5)

    assert len(everything) == 30 and len(best) == 5
    assert [h.id for h in best] == [h.id for h in everything[:5]]
    scores = [h.metadata["text_score"] for h in everything]
    assert scores == sorted(scores, reverse=True)


def test_a_very_long_document_is_searched_to_its_end(stack, clean, monkeypatch):
    _run(clean, create_statement("fixture", A, "Legacy"))
    body = "ordinary words about nothing in particular " * 4000  # ~170 KB
    _run(
        clean,
        f"INSERT INTO {A}.Legacy (doc_id, title, content) VALUES (?, ?, ?)",
        ["long", "t", body + "zephyrquill"],
    )
    _manager(stack, A).ensure_text_index("Legacy")

    hits = _store(stack, monkeypatch, A, "Legacy").text_search("zephyrquill")

    assert [h.id for h in hits] == ["long"]
    assert hits[0].page_content.endswith("zephyrquill")


def test_a_metadata_filter_scopes_the_search(legacy_store):
    bio = legacy_store.text_search(
        "insulin", top_k=10, metadata_filter={"source": "bio"}
    )
    cs = legacy_store.text_search("insulin", top_k=10, metadata_filter={"source": "cs"})
    assert {h.id for h in bio} == {"d1", "d2", "d5"}
    assert cs == []
    both = legacy_store.text_search(
        "insulin OR networks", top_k=10, metadata_filter={"source": "cs"}
    )
    assert [h.id for h in both] == ["d3"]
    assert both[0].metadata["source"] == "cs"


def test_a_filter_is_applied_before_top_k(legacy_store):
    """A page of 1 must be the best *filtered* hit, not a filtered best hit."""
    hits = legacy_store.text_search(
        "transformer OR networks OR insulin", top_k=1, metadata_filter={"source": "cs"}
    )
    assert len(hits) == 1 and hits[0].metadata["source"] == "cs"


def test_a_filter_on_a_table_without_metadata_is_an_error(stack, clean, monkeypatch):
    _run(clean, create_statement("no_metadata", A, "Legacy"))
    _manager(stack, A).ensure_text_index("Legacy")
    store = _store(stack, monkeypatch, A, "Legacy")
    with pytest.raises(RoleNotPresent):
        store.text_search("insulin", metadata_filter={"source": "bio"})


# --- failing loudly ------------------------------------------------------------


def test_searching_a_table_with_no_text_index_raises_and_says_how_to_fix_it(
    stack, clean, monkeypatch
):
    _legacy_table(clean, A)  # populated, not indexed
    store = _store(stack, monkeypatch, A, "Legacy")

    with pytest.raises(TextIndexMissingError) as caught:
        store.text_search("insulin")

    message = str(caught.value)
    assert TEXT_INDEX_NAME in message
    assert f"{A}.Legacy" in message
    assert "ensure_text_index" in message


def test_the_error_is_gone_once_the_index_exists(stack, clean, monkeypatch):
    _legacy_table(clean, A)
    store = _store(stack, monkeypatch, A, "Legacy")
    with pytest.raises(TextIndexMissingError):
        store.text_search("insulin")

    _manager(stack, A).ensure_text_index("Legacy")

    assert {h.id for h in store.text_search("insulin")} == {"d1", "d2", "d5"}


def test_the_old_tables_the_engine_used_to_read_are_not_needed(
    stack, clean, monkeypatch
):
    """Regression: the text leg used to read a `docs(id, text)` table nothing created."""
    _run(clean, "DROP TABLE IF EXISTS docs")
    _legacy_table(clean, A)
    _manager(stack, A).ensure_text_index("Legacy")

    assert _store(stack, monkeypatch, A, "Legacy").text_search("insulin")
