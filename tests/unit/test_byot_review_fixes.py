"""Fixes from the two-axis review of feature 092 (spec findings 1 to 4).

1. FR-010: the text leg hydrates its hits under the same ``where`` it ranked them
   with, so a row outside the scope cannot come back under an id that repeats
   (MedTeam's ``EvidenceId`` repeats once per embedding model).
2. FR-009: no search path skips a required filter. Phrase search cannot take
   ``where``, so on a table with required filters it refuses; so does the IRIS
   EMBEDDING search.
3. FR-007: a declared filter column the table lacks is an error on the first search,
   not only when some ``where`` names it.
4. A ``bool`` filter value is refused with a message that says what to use instead.
"""

from __future__ import annotations

import pytest

from iris_vector_rag.exceptions import FilterError, RequiredFilterMissing
from tests.unit.test_byot_store_vector import MEDTEAM, QUERY, TeamCursor, team_store
from tests.unit.test_byot_text import TextCursor

MED_TEXT = {
    **MEDTEAM,
    "filter_columns": ["PatientId", "EmbeddingModel"],
    "required_filters": ["PatientId", "EmbeddingModel"],
    "text_index": "EvidenceText",
}
WHERE = {"PatientId": "P02", "EmbeddingModel": "m1"}


# --- 1. the text leg hydrates under its scope -----------------------------------------


def test_text_hits_are_fetched_under_the_same_where_they_were_ranked_with():
    cursor = TextCursor(
        hits=[("P02-doc03#1", 1.5)],
        docs=[("P02-doc03#1", "in scope", "Zpráva", "P02-doc03")],
    )
    store = team_store(MED_TEXT, cursor)
    store.schema_manager.has_text_index.return_value = True

    store.text_search("apixaban", top_k=3, where=WHERE)

    fetch_sql, fetch_params = cursor.executed[1]
    assert 'WHERE "EvidenceId" IN (?) AND "PatientId" = ? AND "EmbeddingModel" = ?' in (
        fetch_sql
    )
    assert fetch_params == ["P02-doc03#1", "P02", "m1"]


def test_fetch_by_id_without_where_is_unchanged():
    cursor = TeamCursor(by_id=[("P02-doc03#1", "t", "Zpráva", "P02-doc03")])
    store = team_store(MEDTEAM, cursor)
    store.fetch_documents_by_ids(["P02-doc03#1"])
    sql, params = cursor.executed[0]
    assert sql.endswith('WHERE "EvidenceId" IN (?)')
    assert params == ["P02-doc03#1"]


# --- 2. no path skips a required filter ---------------------------------------------


def test_phrase_search_refuses_a_table_with_required_filters():
    cursor = TextCursor()
    store = team_store(MED_TEXT, cursor)
    with pytest.raises(RequiredFilterMissing, match="phrase"):
        store.phrase_search("chest pain")
    assert cursor.executed == []


def test_iris_embedding_search_refuses_a_table_with_required_filters():
    cursor = TeamCursor()
    store = team_store(MED_TEXT, cursor)
    store.use_iris_embedding = True
    store.embedding_config_name = "cfg"
    with pytest.raises(RequiredFilterMissing):
        store.search_with_embedding("HER2", top_k=3)
    assert cursor.executed == []


def test_the_basic_pipeline_does_not_take_the_iris_embedding_branch_with_where():
    from iris_vector_rag.pipelines.basic import BasicRAGPipeline

    class Store:
        use_iris_embedding = True

        def __init__(self):
            self.calls = []

        def search_with_embedding(self, *a, **k):
            self.calls.append("iris_embedding")
            return []

        def search_by_text(self, query, top_k=5, metadata_filter=None, where=None):
            self.calls.append(("search_by_text", where))
            return []

    pipeline = BasicRAGPipeline.__new__(BasicRAGPipeline)
    pipeline.vector_store = Store()
    pipeline.use_iris_embedding = True
    assert pipeline._iris_embedding_applies(where=WHERE, embedding_func=None) is False
    assert pipeline._iris_embedding_applies(where=None, embedding_func=None) is True


# --- 3. a declared filter column must exist, checked on first use ---------------------


def test_a_filter_column_the_table_lacks_fails_the_first_search_even_without_where():
    cursor = TeamCursor()
    store = team_store({**MEDTEAM, "filter_columns": ["PatientId", "Ward"]}, cursor)
    with pytest.raises(FilterError, match="Ward"):
        store.similarity_search_by_embedding(QUERY, 3)
    assert cursor.executed == []


# --- 4. bool values -----------------------------------------------------------------


def test_a_bool_value_is_refused_with_what_to_use_instead():
    cursor = TeamCursor()
    store = team_store({**MEDTEAM, "filter_columns": ["ChunkIndex"]}, cursor)
    with pytest.raises(FilterError, match="0 or 1"):
        store.similarity_search_by_embedding(QUERY, 3, where={"ChunkIndex": True})
