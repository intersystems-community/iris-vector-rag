"""Unit tests for parse_keywords() — Feature 081, Phase 4."""

import pytest


def parse(raw):
    from iris_vector_rag.retrieval.keyword_extractor import parse_keywords
    return parse_keywords(raw)


def test_valid_json_returns_correct_lists():
    raw = '{"high_level_keywords": ["a", "b"], "low_level_keywords": ["c"]}'
    high, low = parse(raw)
    assert high == ["a", "b"]
    assert low == ["c"]


def test_json_with_extra_whitespace():
    raw = '  { "high_level_keywords" : [ "x" ] , "low_level_keywords" : [ "y" ] }  '
    high, low = parse(raw)
    assert high == ["x"]
    assert low == ["y"]


def test_empty_arrays_returns_empty_lists():
    raw = '{"high_level_keywords": [], "low_level_keywords": []}'
    high, low = parse(raw)
    assert high == []
    assert low == []


def test_completely_invalid_string_returns_empty():
    high, low = parse("not json at all")
    assert high == []
    assert low == []


def test_missing_keys_returns_empty_for_missing():
    raw = '{"high_level_keywords": ["only_high"]}'
    high, low = parse(raw)
    assert high == ["only_high"]
    assert low == []


def test_markdown_fence_stripped():
    raw = "```json\n{\"high_level_keywords\":[\"t\"],\"low_level_keywords\":[]}\n```"
    high, low = parse(raw)
    assert high == ["t"]
    assert low == []
