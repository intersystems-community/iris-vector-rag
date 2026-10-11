"""The detectors work on a section once, not once per mention (spec 088, speed).

Baystate judged every candidate section in embedded Python and saw 10 to 14 s on a 2,300
note result. Profiling showed the cost grew faster than the text: each mention made two
detectors re-tokenise the whole section and walk it character by character to find the
sentence. These tests pin the cheaper behaviour without timing anything, and prove the
answers did not change by checking against the slow versions kept here as references.
"""

import random

import pytest

from iris_vector_rag.note_search import context, phrase, polarity, tokens
from iris_vector_rag.note_search.api import judge_section
from iris_vector_rag.note_search.tokens import Token, tokenize

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


def _count_tokenize(monkeypatch):
    calls = []
    real = tokens.tokenize

    def counting(text):
        calls.append(text)
        return real(text)

    tokens.tokenize_cached.cache_clear()
    phrase._phrase_tokens.cache_clear()
    polarity._token_starts.cache_clear()
    monkeypatch.setattr(tokens, "tokenize", counting)
    monkeypatch.setattr(phrase, "tokenize", counting)
    return calls


def test_a_section_is_tokenised_once_however_many_mentions_it_has(monkeypatch):
    calls = _count_tokenize(monkeypatch)
    text = " ".join("Denies chest pain. Reports chest pain." for _ in range(40))
    result = judge_section(text, "chest pain")

    assert len(result["mentions"]) == 80
    assert calls.count(text) == 1, "phrase, polarity and context share one tokenisation"


def test_the_phrase_is_tokenised_once_for_a_whole_batch(monkeypatch):
    from iris_vector_rag.note_search.api import judge_sections

    calls = _count_tokenize(monkeypatch)
    texts = [f"Visit {n}: denies chest pain." for n in range(50)]
    judge_sections(texts, "chest pain")

    assert calls.count("chest pain") == 1
    assert all(calls.count(t) == 1 for t in texts)


def _old_tokenize(text: str):
    """The original body: strips the edge characters one set at a time per word."""
    import re

    out = []
    for match in re.finditer(r"\S+", text or ""):
        word, start = match.group(), match.start()
        edge = "".join({c for c in word if not c.isalnum()})
        lead = len(word) - len(word.lstrip(edge))
        trail = len(word) - len(word.rstrip(edge))
        if lead + trail >= len(word):
            out.append(Token(word.lower(), start, match.end()))
            continue
        out.append(
            Token(
                word[lead : len(word) - trail].lower(),
                start + lead,
                match.end() - trail,
            )
        )
    return out


TOKEN_ALPHABET = list("ab12 _-./,'\"()\t\n") + [
    "é",
    "ß",
    "²",
    "—",
    "...",
    "chest-pain",
    "P_Q",
]


@pytest.mark.parametrize("seed", range(5))
def test_tokenize_is_what_the_original_body_gave(seed):
    rng = random.Random(seed)
    for _ in range(300):
        text = "".join(rng.choice(TOKEN_ALPHABET) for _ in range(rng.randint(0, 40)))
        assert tokenize(text) == _old_tokenize(text), repr(text)


def test_phrase_and_detectors_agree_on_the_same_mentions():
    text = "No chest pain at rest. Chest pain with exertion. Mother had chest pain."
    result = judge_section(text, "chest pain")
    assert [(m["polarity"], m["experiencer"]) for m in result["mentions"]] == [
        ("negated", "patient"),
        ("affirmed", "patient"),
        ("affirmed", "other"),
    ]
    assert context.section_context is not None and phrase.find_mentions is not None


def _slow_triggered(words, triggers, before_mention):
    """The original: every trigger against every position of the sentence."""
    for trigger in triggers:
        size = len(trigger)
        for index in range(len(words) - size + 1):
            if tuple(words[index : index + size]) != trigger:
                continue
            reach = words[index + size :] if before_mention else words[:index]
            if not polarity.TERMINATORS.intersection(reach):
                return True
    return False


TRIGGER_WORDS = [
    "no",
    "not",
    "denies",
    "negative",
    "for",
    "ruled",
    "out",
    "but",
    "x",
    "if",
]


@pytest.mark.parametrize("seed", range(5))
def test_triggered_gives_the_answers_of_the_exhaustive_scan(seed):
    rng = random.Random(seed)
    lists = [
        polarity.PRE_NEGATION_TRIGGERS,
        polarity.POST_NEGATION_TRIGGERS,
        polarity.UNCERTAINTY_TRIGGERS,
    ]
    for _ in range(400):
        words = [rng.choice(TRIGGER_WORDS) for _ in range(rng.randint(0, 12))]
        for triggers in lists:
            for before in (True, False):
                assert polarity._triggered(words, triggers, before) == _slow_triggered(
                    words, triggers, before
                ), (words, before)
