"""The text language is a setting, not a constant.

One setting, ``storage:iris:text_language`` (default ``en``), reaches every place that
used to assume English: the iFind index DDL, the stop words dropped from a text query,
and the default polarity detector of phrase search. The note-search detectors take a
``language`` or a caller's own lexicon; a language with no lexicon is an error, because
English triggers on another language's text give confident wrong answers.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from iris_vector_rag.note_search.context import (
    CURRENT,
    HISTORICAL,
    OTHER,
    PATIENT,
    Context,
    ContextLexicon,
    ReferenceContextDetector,
    context_lexicon,
)
from iris_vector_rag.note_search.polarity import (
    AFFIRMED,
    NEGATED,
    UNKNOWN,
    NegationLexicon,
    ReferenceNegExDetector,
    negation_lexicon,
)
from iris_vector_rag.storage.text_search import (
    TEXT_INDEX_NAME,
    create_text_index_statement,
    search_terms,
    text_language,
)
from tests.unit.test_text_search import (
    RecordedIndexes,
    Recorder,
    TextCursor,
    manager_over,
    store_with_index,
)


def span_of(text: str, mention: str) -> tuple[int, int]:
    start = text.index(mention)
    return start, start + len(mention)


# --- the index DDL ----------------------------------------------------------


def test_the_index_ddl_carries_the_language_it_is_given():
    assert create_text_index_statement(
        "RAG", "SourceDocuments", "text_content", language="cs"
    ) == (
        f"CREATE INDEX {TEXT_INDEX_NAME} ON RAG.SourceDocuments (text_content) "
        "AS %iFind.Index.Basic(LANGUAGE='cs', LOWER=1)"
    )


def test_the_index_ddl_is_english_by_default():
    assert "LANGUAGE='en'" in create_text_index_statement("RAG", "T", "text_content")


@pytest.mark.parametrize("unset", [None, "", "  "])
def test_an_unset_language_is_english(unset):
    assert text_language(unset) == "en"


def test_a_language_code_is_lower_cased():
    assert "LANGUAGE='de'" in create_text_index_statement(
        "RAG", "T", "text_content", language="DE"
    )


@pytest.mark.parametrize("bad", ["english", "e'n", "c s", "en'); DROP", "e1"])
def test_the_ddl_refuses_anything_but_a_two_letter_code(bad):
    with pytest.raises(ValueError, match="language"):
        create_text_index_statement("RAG", "T", "text_content", language=bad)


# --- stop words -------------------------------------------------------------


def test_english_stop_words_are_dropped_only_for_english():
    assert search_terms("the pain") == "pain"
    assert search_terms("the pain", language="en") == "pain"
    assert search_terms("the bolest hlavy", language="cs") == ("the OR bolest OR hlavy")


def test_a_language_without_stop_words_still_drops_the_operators():
    assert search_terms("bolest and horečka or not", language="cs") == (
        "bolest OR horečka"
    )


# --- the schema manager and the store read the setting ----------------------


def _config_with_language(language):
    config = MagicMock()

    def get(key, default=None):
        if key == "storage:iris:text_language":
            return language
        return default

    config.get.side_effect = get
    return config


def test_ensure_text_index_builds_the_index_in_the_configured_language(
    recorded_catalog,
):
    cursor = Recorder()
    manager, _, table = manager_over(recorded_catalog, cursor)
    manager.config_manager = _config_with_language("cs")
    with patch(
        "iris_vector_rag.storage.schema_manager.IRISCatalogReader", RecordedIndexes()
    ):
        manager.ensure_text_index(table)
    assert cursor.statements == [
        create_text_index_statement("ShapeTest", table, "text_content", language="cs")
    ]


def test_text_search_uses_the_configured_language_for_stop_words(recorded_catalog):
    cursor = TextCursor(hits=[], docs=[])
    store = store_with_index(recorded_catalog, cursor)
    store.storage_config = dict(store.storage_config, text_language="cs")

    store.text_search("the bolest", top_k=2)

    assert cursor.executed[0][1] == ["the OR bolest"] * 3


def test_text_search_is_english_when_nothing_is_configured(recorded_catalog):
    cursor = TextCursor(hits=[], docs=[])
    store = store_with_index(recorded_catalog, cursor)

    store.text_search("the pain", top_k=2)

    assert cursor.executed[0][1] == ["pain"] * 3


# --- polarity ---------------------------------------------------------------


def test_the_english_detector_is_unchanged_by_the_setting():
    text = "Patient denies chest pain."
    span = span_of(text, "chest pain")
    assert ReferenceNegExDetector()(text, span) == NEGATED
    assert ReferenceNegExDetector(language="en")(text, span) == NEGATED
    assert negation_lexicon("EN") is negation_lexicon("en")


def test_a_language_with_no_negation_lexicon_is_an_error_naming_the_ones_there_are():
    with pytest.raises(ValueError) as caught:
        ReferenceNegExDetector(language="cs")
    message = str(caught.value)
    assert "'cs'" in message and "en" in message and "lexicon=" in message


CZECH_LIKE = NegationLexicon(
    pre_negation=(("bez",), ("neguje",)),
    post_negation=(),
    near_post_negation=frozenset({"negativní"}),
    uncertainty=(("možná",),),
    terminators=frozenset({"ale"}),
)


def test_a_callers_lexicon_negates_in_its_own_language():
    detector = ReferenceNegExDetector(lexicon=CZECH_LIKE)
    text = "Pacient bez bolesti na hrudi."
    assert detector(text, span_of(text, "bolesti")) == NEGATED


def test_a_callers_lexicon_brings_its_own_terminators():
    detector = ReferenceNegExDetector(lexicon=CZECH_LIKE)
    text = "Pacient bez horečky, ale bolest hlavy trvá."
    assert detector(text, span_of(text, "bolest hlavy")) == AFFIRMED
    assert detector(text, span_of(text, "horečky")) == NEGATED


def test_a_callers_lexicon_brings_its_own_hedges_and_near_post_words():
    detector = ReferenceNegExDetector(lexicon=CZECH_LIKE)
    hedged = "Možná zápal plic."
    assert detector(hedged, span_of(hedged, "zápal plic")) == UNKNOWN
    after = "Těhotenský test negativní."
    assert detector(after, span_of(after, "Těhotenský test")) == NEGATED


def test_english_triggers_do_not_fire_through_a_callers_lexicon():
    detector = ReferenceNegExDetector(lexicon=CZECH_LIKE)
    text = "Patient denies chest pain."
    assert detector(text, span_of(text, "chest pain")) == AFFIRMED


# --- experiencer and temporality --------------------------------------------


def test_the_english_context_detector_is_unchanged_by_the_setting():
    text = "Mother had a stroke."
    span = span_of(text, "stroke")
    assert ReferenceContextDetector()(text, span).experiencer == OTHER
    assert ReferenceContextDetector(language="en")(text, span).experiencer == OTHER
    assert context_lexicon("en") is context_lexicon("EN")


def test_a_language_with_no_context_lexicon_is_an_error():
    with pytest.raises(ValueError, match="'cs'"):
        ReferenceContextDetector(language="cs")


def test_a_callers_context_lexicon_is_used():
    lexicon = ContextLexicon(
        family_terms=(("matka",),),
        family_history_cue=("rodinná", "anamnéza"),
        hypothetical=(("pokud",),),
        historical_before=(("prodělal",),),
        historical_after=(),
        terminators=frozenset({"ale"}),
    )
    detector = ReferenceContextDetector(lexicon=lexicon)
    text = "Matka měla mrtvici."
    assert detector(text, span_of(text, "mrtvici")) == Context(OTHER, CURRENT)
    past = "Prodělal zápal plic."
    assert detector(past, span_of(past, "zápal plic")) == Context(PATIENT, HISTORICAL)
