"""LLMConfigError (spec 093, FR-007): one error for every way a ConfigStore entry fails."""

from __future__ import annotations

import pytest

from iris_vector_rag.exceptions import LLMConfigError, RAGException


def test_carries_entry_and_fix():
    err = LLMConfigError("opsreview", "has no `model` key", "add `model` to the entry")

    assert err.entry == "opsreview"
    assert err.fix == "add `model` to the entry"
    assert isinstance(err, RAGException)


def test_message_names_the_full_entry_the_problem_and_the_fix():
    message = str(LLMConfigError("opsreview", "is disabled", "enable it"))

    assert "AI.LLM.opsreview" in message
    assert "is disabled" in message
    assert "enable it" in message


def test_caught_as_rag_exception():
    with pytest.raises(RAGException):
        raise LLMConfigError("x", "problem", "fix")


def test_message_holds_only_what_it_is_given():
    """The error is built from a reference, never a value: a resolved key cannot leak."""
    err = LLMConfigError(
        "opsreview",
        "cannot read `secret://ivrprobe.openai#api_key` from the Wallet",
        "grant %Admin_Wallet:USE",
    )

    assert "secret://ivrprobe.openai#api_key" in str(err)
    assert "sk-" not in str(err)
