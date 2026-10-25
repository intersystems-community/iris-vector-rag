"""`where=` on the store's search methods (feature 092, US2: FR-009, FR-010)."""

from __future__ import annotations

import pytest

from iris_vector_rag.exceptions import RequiredFilterMissing
from tests.unit.test_byot_store_vector import MEDTEAM, QUERY, TeamCursor, team_store

SCOPED = {
    **MEDTEAM,
    "filter_columns": ["PatientId", "EmbeddingModel", "DocDate"],
    "required_filters": ["PatientId", "EmbeddingModel"],
}
WHERE = {"PatientId": "P02", "EmbeddingModel": "mock:hashed@384"}
VECTOR = "[" + ",".join(map(str, QUERY)) + "]"


def test_the_predicate_follows_the_vector_and_its_values_follow_the_vector():
    cursor = TeamCursor()
    store = team_store(SCOPED, cursor)

    store.similarity_search_by_embedding(QUERY, 5, where=WHERE)

    sql, params = cursor.executed[0]
    assert (
        'WHERE "Embedding" IS NOT NULL AND ("PatientId" = ? AND "EmbeddingModel" = ?)'
        in sql
    )
    assert params == [VECTOR, "P02", "mock:hashed@384"]


def test_a_missing_required_filter_fails_before_any_sql():
    cursor = TeamCursor()
    store = team_store(SCOPED, cursor)
    with pytest.raises(RequiredFilterMissing, match="EmbeddingModel"):
        store.similarity_search_by_embedding(QUERY, 5, where={"PatientId": "P02"})
    with pytest.raises(RequiredFilterMissing):
        store.similarity_search_by_embedding(QUERY, 5)
    assert cursor.executed == []


def test_a_required_filter_is_checked_even_before_embedding():
    calls = []
    cursor = TeamCursor()
    store = team_store(SCOPED, cursor, embedder=lambda t: calls.append(t) or QUERY)
    with pytest.raises(RequiredFilterMissing):
        store.search_by_text("HER2", top_k=3)
    assert calls == [] and cursor.executed == []


@pytest.mark.parametrize(
    "search",
    [
        lambda s: s.search_by_text("q", top_k=2, where=WHERE),
        lambda s: s.search_by_text_with_score("q", top_k=2, where=WHERE),
        lambda s: s.search_by_vector(QUERY, top_k=2, where=WHERE),
        lambda s: s.similarity_search_with_score("q", k=2, where=WHERE),
        lambda s: s.similarity_search("q", k=2, where=WHERE),
    ],
    ids=[
        "search_by_text",
        "search_by_text_with_score",
        "search_by_vector",
        "similarity_search_with_score",
        "similarity_search",
    ],
)
def test_every_public_vector_search_forwards_where(search):
    cursor = TeamCursor()
    store = team_store(SCOPED, cursor, embedder=lambda t: QUERY)
    search(store)
    sql, params = cursor.executed[0]
    assert '"PatientId" = ?' in sql
    assert params[1:] == ["P02", "mock:hashed@384"]


def test_where_and_a_metadata_filter_are_both_applied(recorded_catalog):
    # The library's own fixture table: a metadata column and a real `title` column.
    from tests.unit.test_vector_store_shape import FakeCursor, make_store

    cursor = FakeCursor()
    store = make_store("fixture", cursor, recorded_catalog)
    store.filter_spec = type(store.filter_spec)(filter_columns=("title",))

    store.similarity_search_by_embedding(
        [0.1] * 384, 3, filter={"source": "bio"}, where={"title": "Insulin"}
    )

    sql, params = cursor.executed[0]
    assert '"title" = ?' in sql
    assert '"source"' in sql  # the metadata predicate, as before
    assert params[1:] == ["Insulin"]
