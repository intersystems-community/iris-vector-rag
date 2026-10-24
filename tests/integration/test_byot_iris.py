"""Bring your own table, against a real IRIS (feature 092).

MedTeam's and Štafeta's chunk tables are rebuilt with synthetic rows
(`tests/fixtures/byot/teams.py`), and the library's answer is compared with the answer
each team's own SQL gives on the same rows. The tables live in schemas of their own
(`BYOTMed`, `BYOTStf`), created and dropped around each test. Every test also checks
that the library left both tables exactly as it found them (SC-004).
"""

from __future__ import annotations

import pytest

from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager
from iris_vector_rag.exceptions import AttachedTableReadOnly, DimensionMismatchError
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from tests.fixtures.byot import teams as T

pytestmark = pytest.mark.integration

MEDTEAM = {
    "table_name": T.MED,
    "id_col": "EvidenceId",
    "text_col": "Text",
    "embedding_col": "Embedding",
    "title_col": "Title",
    "parent_col": "ResourceId",
}
STAFETA = {
    "table_name": T.STF,
    "id_col": ["DocRef", "Seq"],
    "text_col": "Text",
    "embedding_col": "Embedding",
    "parent_col": "DocRef",
}


@pytest.fixture
def byot_tables(iris_connection):
    """Both team tables, built before the test and dropped after; unchanged by it."""
    T.create(iris_connection)
    before = T.snapshot(iris_connection)
    yield iris_connection
    after = T.snapshot(iris_connection)
    T.drop(iris_connection)
    assert after == before, "the library changed an attached table"


def store_for(settings, **kwargs) -> IRISVectorStore:
    config = ConfigurationManager()
    return IRISVectorStore(ConnectionManager(config), config, **settings, **kwargs)


def ranked(results):
    return [(doc.id, round(score, 6)) for doc, score in results]


# --- US1: vector search over an attached table --------------------------------------


def test_us1_medteam_matches_the_teams_cosine_ranking(byot_tables):
    # Unscoped here; the patient and model scope is US2's test.
    cursor = byot_tables.cursor()
    try:
        cursor.execute(
            f"SELECT TOP 5 EvidenceId, VECTOR_COSINE(Embedding, TO_VECTOR(?, DOUBLE)) "
            f"AS Score FROM {T.MED} ORDER BY Score DESC",
            [T.vlit(T.MED_QUERY)],
        )
        expected = [(r[0], round(float(r[1]), 6)) for r in cursor.fetchall()]
    finally:
        cursor.close()

    store = store_for(MEDTEAM)
    results = store.similarity_search_by_embedding(T.MED_QUERY, 5)

    assert ranked(results) == expected
    doc = results[0][0]
    assert doc.metadata["parent_id"] == doc.id.rpartition("#")[0]
    assert doc.metadata["title"].startswith("Zpráva ")


def test_us1_stafeta_composite_ids(byot_tables):
    store = store_for(STAFETA)
    results = store.similarity_search_by_embedding(T.STF_QUERY, 3)
    ids = [doc.id for doc, _ in results]
    assert ids[0] == "DocumentReference/11#3"
    assert all(i.startswith("DocumentReference/") and "#" in i for i in ids)
    fetched = store.fetch_documents_by_ids(ids[:2])
    assert {d.id for d in fetched} == set(ids[:2])


def test_us1_a_query_of_the_wrong_width_is_refused(byot_tables):
    store = store_for(MEDTEAM)
    with pytest.raises(DimensionMismatchError):
        store.similarity_search_by_embedding([0.1] * 1536, 5)


def test_us1_writes_are_refused(byot_tables):
    from iris_vector_rag.core.models import Document

    store = store_for(MEDTEAM)
    with pytest.raises(AttachedTableReadOnly):
        store.add_documents([Document(id="x", page_content="t")])
    with pytest.raises(AttachedTableReadOnly):
        store.delete_documents(["P02-doc03#1"])
