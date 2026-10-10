"""Concept retrieval contract, no IRIS: the `concept` mode, QueryOptions.polarity, and the
engine's refusals. A fake concept index records what the engine asks of it.
"""

import pytest

from iris_vector_rag.core.query_options import QueryOptions, normalize_query_params
from iris_vector_rag.note_search.concepts import DictionaryTagger
from iris_vector_rag.retrieval.engine import RetrievalEngine
from iris_vector_rag.retrieval.modes import get_mode, list_modes

DICTIONARY = {
    "C-CHEST-PAIN": ["chest pain", "chest discomfort"],
    "C-SOB": ["shortness of breath", "sob", "dyspnea"],
}


class FakeConceptIndex:
    def __init__(self):
        self.calls = []

    def search(self, concept_ids, polarity=None, section=None, top_k=5):
        self.calls.append(
            {
                "concept_ids": list(concept_ids),
                "polarity": polarity,
                "section": section,
                "top_k": top_k,
            }
        )
        return ["sentinel"]


def _engine(**kwargs):
    return RetrievalEngine(object(), **kwargs)


# --- the mode -----------------------------------------------------------------------


def test_the_concept_mode_is_registered_over_the_concept_source():
    mode = get_mode("concept")

    assert mode.name == "concept"
    assert mode.sources == ["concept"]
    assert mode.requires == ["concept_index"]
    assert mode.fusion is None


def test_the_concept_mode_is_listed():
    assert "concept" in list_modes()


# --- QueryOptions.polarity ----------------------------------------------------------


def test_query_options_takes_an_optional_polarity_defaulting_to_none():
    assert QueryOptions(query="chest pain").polarity is None
    assert QueryOptions(query="chest pain", polarity="negated").polarity == "negated"


def test_normalize_query_params_accepts_polarity():
    opts = normalize_query_params(
        "chest pain", retrieval="concept", polarity="affirmed"
    )

    assert opts.polarity == "affirmed"
    assert opts.retrieval == "concept"


# --- the engine's refusals and dispatch ----------------------------------------------


def test_concept_retrieval_without_an_index_names_the_missing_index():
    engine = _engine(concept_tagger=DictionaryTagger(DICTIONARY))

    with pytest.raises(NotImplementedError) as caught:
        engine.retrieve(QueryOptions(query="chest pain", retrieval="concept"))

    assert "concept_index" in str(caught.value)


def test_concept_retrieval_without_a_tagger_names_the_missing_tagger():
    engine = _engine(concept_index=FakeConceptIndex())

    with pytest.raises(NotImplementedError) as caught:
        engine.retrieve(QueryOptions(query="chest pain", retrieval="concept"))

    assert "concept_tagger" in str(caught.value)


def test_a_query_naming_no_known_concept_is_a_value_error_not_no_match():
    index = FakeConceptIndex()
    engine = _engine(concept_index=index, concept_tagger=DictionaryTagger(DICTIONARY))

    with pytest.raises(ValueError, match="no known concept"):
        engine.retrieve(QueryOptions(query="headache", retrieval="concept"))
    assert index.calls == []


def test_a_similarity_threshold_is_refused_for_concept():
    engine = _engine(
        concept_index=FakeConceptIndex(),
        concept_tagger=DictionaryTagger(DICTIONARY),
    )

    with pytest.raises(ValueError):
        engine.retrieve(
            QueryOptions(
                query="chest pain", retrieval="concept", similarity_threshold=0.5
            )
        )


def test_a_metadata_filter_is_refused_for_concept():
    engine = _engine(
        concept_index=FakeConceptIndex(),
        concept_tagger=DictionaryTagger(DICTIONARY),
    )

    with pytest.raises(ValueError):
        engine.retrieve(
            QueryOptions(
                query="chest pain",
                retrieval="concept",
                metadata_filter={"section": "hpi"},
            )
        )


def test_concept_retrieval_passes_the_query_concepts_and_options_to_the_index():
    index = FakeConceptIndex()
    engine = _engine(concept_index=index, concept_tagger=DictionaryTagger(DICTIONARY))

    result = engine.retrieve(
        QueryOptions(
            query="shortness of breath with chest pain, chest discomfort too",
            retrieval="concept",
            polarity="negated",
            section="hpi",
            top_k=3,
        )
    )

    assert result == ["sentinel"]
    assert index.calls == [
        {
            "concept_ids": ["C-SOB", "C-CHEST-PAIN"],
            "polarity": "negated",
            "section": "hpi",
            "top_k": 3,
        }
    ]
