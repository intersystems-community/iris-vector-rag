"""The detectors work on a section once, not once per mention (spec 088, speed).

Baystate judged every candidate section in embedded Python and saw 10 to 14 s on a 2,300
note result. Profiling showed the cost grew faster than the text: each mention made two
detectors re-tokenise the whole section and walk it character by character to find the
sentence. These tests pin the cheaper behaviour without timing anything, and prove the
answers did not change by checking against the slow versions kept here as references.
"""

import random

import pytest

from iris_vector_rag.note_search import context, phrase, polarity
from iris_vector_rag.note_search.api import judge_section
from iris_vector_rag.note_search.tokens import tokenize

ALPHABET = list("ab c.\n;?!,-x ") + ["chest pain", " no ", "denies ", ". "]


def _fuzz(seed: int, count: int = 200):
    rng = random.Random(seed)
    for _ in range(count):
        yield "".join(rng.choice(ALPHABET) for _ in range(rng.randint(0, 60)))


def _slow_sentence_bounds(text: str, pos: int):
    """The original: a character walk from the start of the text on every call."""
    start = 0
    for index in range(pos):
        if polarity._is_boundary(text, index):
            start = index + 1
    end = len(text)
    for index in range(pos, len(text)):
        if polarity._is_boundary(text, index):
            end = index
            break
    return start, end


def _slow_sentence_tokens(text: str, pos: int):
    start, end = _slow_sentence_bounds(text, pos)
    return [t for t in tokenize(text) if t.start >= start and t.end <= end]


@pytest.mark.parametrize("seed", range(5))
def test_sentence_bounds_are_what_the_character_walk_gave(seed):
    for text in _fuzz(seed):
        for pos in range(len(text) + 1):
            assert polarity._sentence_bounds(text, pos) == _slow_sentence_bounds(
                text, pos
            ), (text, pos)


@pytest.mark.parametrize("seed", range(5))
def test_sentence_tokens_are_the_tokens_the_filter_gave(seed):
    for text in _fuzz(seed):
        for pos in range(len(text) + 1):
            assert polarity._sentence_tokens(text, pos) == _slow_sentence_tokens(
                text, pos
            ), (text, pos)


def test_a_section_with_many_mentions_is_tokenised_once_by_the_detectors(monkeypatch):
    calls = []
    real = polarity.tokenize

    def counting(text):
        calls.append(len(text))
        return real(text)

    polarity._tokens.cache_clear()
    monkeypatch.setattr(polarity, "tokenize", counting)
    text = " ".join("Denies chest pain. Reports chest pain." for _ in range(40))
    result = judge_section(text, "chest pain")

    assert len(result["mentions"]) == 80
    assert len(calls) == 1, "polarity and context share one tokenisation per text"


def test_phrase_and_detectors_agree_on_the_same_mentions():
    text = "No chest pain at rest. Chest pain with exertion. Mother had chest pain."
    result = judge_section(text, "chest pain")
    assert [(m["polarity"], m["experiencer"]) for m in result["mentions"]] == [
        ("negated", "patient"),
        ("affirmed", "patient"),
        ("affirmed", "other"),
    ]
    assert context.section_context is not None and phrase.find_mentions is not None
