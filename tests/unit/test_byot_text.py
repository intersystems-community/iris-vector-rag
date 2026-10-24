"""The text leg on an attached table (feature 092, US4: FR-014 to FR-016).

The table's own iFind index, prefix terms for an inflected language, the same
real-column filters as the vector leg, and no index creation on a table the library
does not own.
"""

from __future__ import annotations

import pytest

from iris_vector_rag.exceptions import TextIndexMissingError
from iris_vector_rag.storage.text_search import build_text_search_sql, search_terms
from tests.unit.test_byot_store_vector import STAFETA, team_store

# --- prefix terms ----------------------------------------------------------------


@pytest.mark.parametrize(
    "word,term",
    [
        ("vysazena", "vysaz*"),  # Štafeta's own example
        ("antikoagulace", "antikoa*"),  # cut at 7 letters at most
        ("ECOG", "ecog*"),  # 4 letters: the whole word, as a prefix
        ("biopsií", "biop*"),
    ],
)
def test_prefix_terms_follow_stafetas_rule(word, term):
    assert search_terms(word, language="cs", terms="prefix") == term


def test_words_under_four_letters_stay_whole():
    assert search_terms("CT bez progrese", language="cs", terms="prefix") == (
        "ct OR bez OR progr*"
    )


def test_prefix_terms_keep_the_languages_stop_words():
    assert search_terms("the fractures", terms="prefix") == "fractu*"


def test_words_is_todays_behaviour():
    assert search_terms("vysazena antikoagulace", language="cs") == (
        "vysazena OR antikoagulace"
    )
    assert search_terms("vysazena", language="cs", terms="words") == "vysazena"


def test_an_unknown_terms_mode_is_refused():
    with pytest.raises(ValueError, match="terms"):
        search_terms("x", terms="stems")


# --- the SQL ---------------------------------------------------------------------


def test_the_text_sql_selects_an_identifier_expression_and_takes_where_params():
    sql = build_text_search_sql(
        table="BYOTStf.DocChunk",
        id_column="DocRef",
        id_expression='"DocRef" || \'#\' || "Seq"',
        index_name="DocChunkText",
        rank_proc="BYOTStf.DocChunk_DocChunkTextRank",
        top_k=5,
        extra_where='"PatientRef" = ? AND "AuthoredOn" < CAST(? AS DATE)',
    )
    assert sql.startswith('SELECT TOP 5 "DocRef" || \'#\' || "Seq", ')
    assert "%ID %FIND search_index(DocChunkText, ?)" in sql
    assert 'AND "PatientRef" = ? AND "AuthoredOn" < CAST(? AS DATE)' in sql
    # rank (SELECT), match, the two where values, rank (ORDER BY)
    assert sql.count("?") == 5
    assert "((" not in sql


# --- the store -------------------------------------------------------------------

STF_TEXT = {
    **STAFETA,
    "filter_columns": ["PatientRef", "AuthoredOn"],
    "required_filters": ["PatientRef"],
    "text_index": "DocChunkText",
    "text_language": "cs",
    "text_terms": "prefix",
}


class TextCursor:
    def __init__(self, hits=(), docs=()):
        self.executed = []
        self._hits, self._docs, self._result = list(hits), list(docs), []

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        self._result = list(self._hits if "SEARCH_INDEX" in sql.upper() else self._docs)

    def fetchall(self):
        return self._result

    def close(self):
        pass


def test_the_store_searches_the_named_index_with_prefix_terms_and_where():
    cursor = TextCursor(
        hits=[("DocumentReference/11#1", 2.0)],
        docs=[
            (
                "DocumentReference/11#1",
                "apixaban vysazen",
                "Dokument 11",
                "DocumentReference/11",
            )
        ],
    )
    store = team_store(STF_TEXT, cursor)
    store.schema_manager.has_text_index.return_value = True

    docs = store.text_search(
        "Kdy byla vysazena antikoagulace?",
        top_k=3,
        where={"PatientRef": "Patient/2", "AuthoredOn": ("<", "2026-10-05")},
    )

    sql, params = cursor.executed[0]
    assert "search_index(DocChunkText, ?)" in sql
    assert "BYOTStf.DocChunk_DocChunkTextRank(%ID, ?)" in sql
    terms = "kdy OR byla* OR vysaz* OR antikoa*"
    assert params == [terms, terms, "Patient/2", "2026-10-05", terms]
    store.schema_manager.has_text_index.assert_called_with(
        "BYOTStf.DocChunk", index_name="DocChunkText"
    )
    assert [d.id for d in docs] == ["DocumentReference/11#1"]
    assert docs[0].metadata["text_score"] == 2.0


def test_an_attached_table_without_a_named_index_is_an_error_and_nothing_is_built():
    cursor = TextCursor()
    store = team_store({**STAFETA}, cursor)

    with pytest.raises(TextIndexMissingError) as caught:
        store.text_search("antikoagulace")

    message = str(caught.value)
    assert "text_index" in message and "does not own" in message
    store.schema_manager.ensure_text_index.assert_not_called()
    assert cursor.executed == []


def test_a_named_index_the_table_lacks_is_an_error():
    cursor = TextCursor()
    store = team_store({**STAFETA, "text_index": "Missing"}, cursor)
    store.schema_manager.has_text_index.return_value = False
    with pytest.raises(TextIndexMissingError, match="Missing"):
        store.text_search("antikoagulace")
    store.schema_manager.ensure_text_index.assert_not_called()
