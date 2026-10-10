"""Tokens and phrases (spec 088, contracts/query-semantics.md sections 3.1 to 3.8)."""

import pytest

from iris_vector_rag.note_search.phrase import (
    find_mentions,
    phrase_search_string,
    phrase_tokens,
)
from iris_vector_rag.note_search.tokens import tokenize


def words(text):
    return [t.text for t in tokenize(text)]


def test_tokens_are_lower_cased_whitespace_delimited_words():
    assert words("Chest   PAIN\tnoted\nhere") == ["chest", "pain", "noted", "here"]


def test_punctuation_at_the_edges_of_a_word_is_removed():
    assert words("(Chest) pain, noted.") == ["chest", "pain", "noted"]


def test_punctuation_inside_a_word_stays():
    assert words("Chest-pain r/o patient's 3.5") == [
        "chest-pain",
        "r/o",
        "patient's",
        "3.5",
    ]


def test_a_lone_punctuation_mark_is_a_token_of_its_own():
    assert words("chest - pain") == ["chest", "-", "pain"]


def test_token_offsets_point_at_the_word_without_its_edge_punctuation():
    text = "x (Chest) pain."
    spans = [(t.start, t.end) for t in tokenize(text)]
    assert [text[s:e] for s, e in spans] == ["x", "Chest", "pain"]


def test_empty_text_has_no_tokens():
    assert tokenize("") == []
    assert tokenize("  \n ") == []


def test_phrase_tokens_drop_double_quotes_and_case():
    assert phrase_tokens('"Chest"   PAIN') == ["chest", "pain"]


@pytest.mark.parametrize("phrase", ["", "   ", "...", '""', "- - -"])
def test_a_phrase_without_a_letter_or_digit_is_an_error(phrase):
    with pytest.raises(ValueError, match="letter or digit"):
        phrase_tokens(phrase)


def test_the_search_string_is_one_quoted_phrase():
    assert phrase_search_string("Chest   PAIN") == '"chest pain"'


def test_operators_are_plain_text_in_the_search_string():
    assert (
        phrase_search_string('"chest pain" OR penicillin')
        == '"chest pain or penicillin"'
    )
    assert phrase_search_string("chest* AND (pain)") == '"chest and pain"'


def test_the_search_string_never_holds_a_quote_inside_the_phrase():
    assert phrase_search_string('say "hi" now').count('"') == 2


def test_find_mentions_returns_every_occurrence_with_its_span():
    text = "No chest pain at rest. Chest pain with exertion."
    spans = find_mentions(text, "chest pain")
    assert [text[s:e] for s, e in spans] == ["chest pain", "Chest pain"]


def test_a_phrase_matches_across_a_sentence_end():
    text = "Chest pain. Chest pain is worse."
    assert [text[s:e] for s, e in find_mentions(text, "pain chest")] == ["pain. Chest"]


def test_hyphenated_text_is_not_the_unhyphenated_phrase():
    assert find_mentions("Chest-pain noted.", "chest pain") == []
    assert len(find_mentions("Chest-pain noted.", "chest-pain")) == 1


def test_no_stemming_and_stop_words_are_kept():
    assert find_mentions("chest pains here", "chest pain") == []
    assert (
        len(find_mentions("No known drug allergies.", "no known drug allergies")) == 1
    )
