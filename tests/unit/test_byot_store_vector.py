"""Vector search over an attached table (feature 092, US1: FR-005, FR-011, FR-012).

The store reads MedTeam's and Štafeta's tables as IRIS recorded them
(`tests/fixtures/byot/catalog/`), through declared columns. The cursor answers the
data queries and records them; no database is involved.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.core.models import Document
from iris_vector_rag.exceptions import AttachedTableReadOnly, DimensionMismatchError
from iris_vector_rag.storage.catalog import RecordedCatalog
from iris_vector_rag.storage.schema_manager import SchemaManager
from iris_vector_rag.storage.shape_resolver import TableShapeResolver
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

CATALOG = Path(__file__).resolve().parents[1] / "fixtures" / "byot" / "catalog"
DIM = 384
QUERY = [0.05] * DIM

MEDTEAM = {
    "table_name": "BYOTMed.EvidenceChunk",
    "id_col": "EvidenceId",
    "text_col": "Text",
    "embedding_col": "Embedding",
    "title_col": "Title",
    "parent_col": "ResourceId",
}
STAFETA = {
    "table_name": "BYOTStf.DocChunk",
    "id_col": ["DocRef", "Seq"],
    "text_col": "Text",
    "embedding_col": "Embedding",
    "parent_col": "DocRef",
}


class TeamCursor:
    """Answers ranked vector queries and by-id fetches; records every statement."""

    rowcount = 1

    def __init__(self, ranked=(), by_id=()):
        self.executed: list = []
        self._ranked = list(ranked)
        self._by_id = list(by_id)
        self._result: list = []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        upper = " ".join(str(sql).upper().split())
        if "INFORMATION_SCHEMA" in upper or "%DICTIONARY" in upper:
            raise AssertionError(f"the store read the catalog itself: {sql}")
        if "VECTOR_COSINE" in upper or "VECTOR_DOT_PRODUCT" in upper:
            self._result = list(self._ranked)
        elif " IN (" in upper:
            self._result = list(self._by_id)
        else:
            self._result = []

    def fetchall(self):
        return self._result

    def fetchone(self):
        return self._result[0] if self._result else None

    def close(self):
        pass

    @property
    def sql(self):
        return [sql for sql, _ in self.executed]


def team_store(settings, cursor, **kwargs) -> IRISVectorStore:
    connection = MagicMock()
    connection.cursor.return_value = cursor
    connections = MagicMock(spec=ConnectionManager)
    connections.get_connection.return_value = connection
    config = MagicMock(spec=ConfigurationManager)

    def config_get(key, default=None):
        if key == "storage:iris":
            return dict(settings)
        return default

    config.get.side_effect = config_get
    config.to_dict.return_value = {}
    schema_manager = MagicMock(spec=SchemaManager)
    return IRISVectorStore(
        connections,
        config,
        schema_manager=schema_manager,
        shapes=TableShapeResolver(RecordedCatalog.from_fixture_dir(CATALOG)),
        **kwargs,
    )


# --- the ranked query -----------------------------------------------------------


def test_the_vector_query_reads_the_declared_columns_and_ranks_by_cosine():
    cursor = TeamCursor(
        ranked=[("P02-doc03#1", "apixaban", "Zpráva P02-doc03", "P02-doc03", 0.98)]
    )
    store = team_store(MEDTEAM, cursor)

    results = store.similarity_search_by_embedding(QUERY, 5)

    sql, params = cursor.executed[0]
    assert sql.startswith('SELECT TOP 5 "EvidenceId", "Text", "Title", "ResourceId"')
    assert 'VECTOR_COSINE("Embedding", TO_VECTOR(?, DOUBLE, 384)) AS score' in sql
    assert "FROM BYOTMed.EvidenceChunk" in sql
    assert '"Embedding" IS NOT NULL' in sql
    assert params == ["[" + ",".join(map(str, QUERY)) + "]"]
    doc, score = results[0]
    assert doc.id == "P02-doc03#1"
    assert doc.page_content == "apixaban"
    assert doc.metadata["title"] == "Zpráva P02-doc03"
    assert doc.metadata["parent_id"] == "P02-doc03"
    assert score == pytest.approx(0.98)


def test_metric_dot_ranks_an_attached_table_by_dot_product():
    cursor = TeamCursor()
    store = team_store(MEDTEAM, cursor, metric="dot")
    store.similarity_search_by_embedding(QUERY, 3)
    assert 'VECTOR_DOT_PRODUCT("Embedding", TO_VECTOR(?, DOUBLE, 384))' in cursor.sql[0]


def test_metric_from_configuration():
    cursor = TeamCursor()
    store = team_store({**MEDTEAM, "metric": "dot"}, cursor)
    store.similarity_search_by_embedding(QUERY, 3)
    assert "VECTOR_DOT_PRODUCT" in cursor.sql[0]


def test_an_unknown_metric_is_refused_when_the_store_is_built():
    with pytest.raises(ValueError, match="metric"):
        team_store(MEDTEAM, TeamCursor(), metric="l2")


def test_a_composite_identifier_comes_back_joined_with_a_hash():
    cursor = TeamCursor(
        ranked=[("DocumentReference/11#3", "text", "DocumentReference/11", 0.82)]
    )
    store = team_store(STAFETA, cursor)

    ((doc, _),) = store.similarity_search_by_embedding(QUERY, 1)

    # `Title` matches a name list but is not declared, so it is not read.
    assert cursor.sql[0].startswith(
        'SELECT TOP 1 "DocRef" || \'#\' || "Seq", "Text", "DocRef", VECTOR_COSINE'
    )
    assert doc.id == "DocumentReference/11#3"
    assert "title" not in doc.metadata
    assert doc.metadata["parent_id"] == "DocumentReference/11"


def test_a_query_of_the_wrong_width_fails_before_any_sql():
    cursor = TeamCursor()
    store = team_store(MEDTEAM, cursor)
    with pytest.raises(DimensionMismatchError) as caught:
        store.similarity_search_by_embedding([0.1] * 1536, 5)
    assert caught.value.got == 1536 and caught.value.expected == 384
    assert cursor.executed == []


# --- read only --------------------------------------------------------------------


@pytest.mark.parametrize(
    "write",
    [
        lambda s: s.add_documents([Document(id="x", page_content="t")]),
        lambda s: s.delete_documents(["x"]),
        lambda s: s.clear_documents(),
    ],
    ids=["add_documents", "delete_documents", "clear_documents"],
)
def test_writes_to_an_attached_table_are_refused_without_sql(write):
    cursor = TeamCursor()
    store = team_store(MEDTEAM, cursor)
    with pytest.raises(AttachedTableReadOnly, match="BYOTMed.EvidenceChunk"):
        write(store)
    assert cursor.executed == []


def test_fetch_by_id_filters_on_the_identifier_expression():
    cursor = TeamCursor(
        by_id=[("DocumentReference/11#3", "text", "DocumentReference/11")]
    )
    store = team_store(STAFETA, cursor)

    docs = store.fetch_documents_by_ids(["DocumentReference/11#3"])

    sql, params = cursor.executed[0]
    assert 'WHERE "DocRef" || \'#\' || "Seq" IN (?)' in sql
    assert params == ["DocumentReference/11#3"]
    assert docs[0].id == "DocumentReference/11#3"
    assert docs[0].page_content == "text"
    assert docs[0].metadata["parent_id"] == "DocumentReference/11"


# --- the caller's embedder ----------------------------------------------------------


def test_the_callers_embedder_embeds_every_text_query():
    calls = []

    def embed(text):
        calls.append(text)
        return QUERY

    cursor = TeamCursor()
    store = team_store(MEDTEAM, cursor, embedder=embed)

    store.search_by_text("HER2", top_k=2)
    store.search_by_text_with_score("KRAS", top_k=2)
    store.similarity_search_with_score("ECOG", k=2)
    store.similarity_search("biopsie", k=2)

    assert calls == ["HER2", "KRAS", "ECOG", "biopsie"]
    ranked = [sql for sql in cursor.sql if "VECTOR_COSINE" in sql]
    assert len(ranked) == 4
