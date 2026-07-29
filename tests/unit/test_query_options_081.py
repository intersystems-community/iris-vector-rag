"""Unit tests for QueryOptions extensions — Feature 081 (dual-level retrieval)."""

import pytest

from iris_vector_rag.core.query_options import QueryOptions, normalize_query_params


def test_normalize_mix_mode_accepted():
    opts = normalize_query_params(query="q", retrieval="mix")
    assert opts.retrieval == "mix"


def test_normalize_global_mode_accepted():
    opts = normalize_query_params(query="q", retrieval="global")
    assert opts.retrieval == "global"


def test_query_options_accepts_high_level_keywords():
    opts = QueryOptions(query="q", high_level_keywords=["systemic risk", "Basel III"])
    assert opts.high_level_keywords == ["systemic risk", "Basel III"]


def test_query_options_accepts_low_level_keywords():
    opts = QueryOptions(query="q", low_level_keywords=["LIBOR", "Tier 1"])
    assert opts.low_level_keywords == ["LIBOR", "Tier 1"]


def test_query_options_keywords_default_none():
    opts = QueryOptions(query="q")
    assert opts.high_level_keywords is None
    assert opts.low_level_keywords is None


def test_normalize_mix_with_weights_no_error():
    # mix is a valid fusion mode — weights should not raise
    opts = normalize_query_params(query="q", retrieval="mix", weights={"relation": 0.6, "vector": 0.4})
    assert opts.weights == {"relation": 0.6, "vector": 0.4}


def test_normalize_global_with_pre_supplied_keywords():
    opts = normalize_query_params(
        query="q",
        retrieval="global",
        high_level_keywords=["theme1"],
        low_level_keywords=["entity1"],
    )
    assert opts.high_level_keywords == ["theme1"]
    assert opts.low_level_keywords == ["entity1"]
