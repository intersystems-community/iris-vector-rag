"""create_pipeline picks the answer model: llm_func, then the ConfigStore entry, then YAML.

Spec 093, FR-001, FR-005 and FR-010. Building the pipeline itself is replaced by a
recorder, so the tests see exactly which answer function each path hands on.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

import iris_vector_rag
from iris_vector_rag.config.manager import ConfigurationManager


def explicit(prompt: str) -> str:
    return "explicit"


@pytest.fixture
def recorder(monkeypatch):
    calls: Dict[str, List[Any]] = {"built": [], "configstore": [], "yaml": []}

    def legacy(pipeline_type, connection_manager, config_manager, llm_func, **kwargs):
        calls["built"].append(llm_func)
        return "pipeline"

    def validated(self, pipeline_type, llm_func=None, **kwargs):
        calls["built"].append(llm_func)
        return "pipeline"

    def from_configstore(name, connection_manager):
        calls["configstore"].append((name, connection_manager))
        return f"configstore:{name}"

    def from_yaml(**kwargs):
        calls["yaml"].append(kwargs)
        return "yaml"

    monkeypatch.setattr(iris_vector_rag, "_create_pipeline_legacy", legacy)
    monkeypatch.setattr(
        iris_vector_rag.ValidatedPipelineFactory, "create_pipeline", validated
    )
    monkeypatch.setattr(iris_vector_rag, "configstore_llm", from_configstore)
    monkeypatch.setattr(iris_vector_rag, "get_llm_func", from_yaml)
    monkeypatch.delenv("RAG_LLM__CONFIG_NAME", raising=False)
    return calls


class FakeEngine:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.connection_manager = "engine-connections"


@pytest.mark.parametrize("validate", [False, True])
def test_explicit_llm_func_wins_and_the_entry_is_not_read(recorder, validate):
    iris_vector_rag.create_pipeline(
        "basic",
        llm_func=explicit,
        llm_config="opsreview",
        validate_requirements=validate,
    )

    assert recorder["built"] == [explicit]
    assert recorder["configstore"] == []
    assert recorder["yaml"] == []


@pytest.mark.parametrize("validate", [False, True])
def test_llm_config_reads_the_entry(recorder, validate):
    iris_vector_rag.create_pipeline(
        "basic", llm_config="opsreview", validate_requirements=validate
    )

    assert recorder["built"] == ["configstore:opsreview"]
    assert [name for name, _ in recorder["configstore"]] == ["opsreview"]
    assert recorder["yaml"] == []


def test_llm_config_is_not_passed_on_to_the_pipeline(recorder, monkeypatch):
    seen: Dict[str, Any] = {}

    def legacy(pipeline_type, connection_manager, config_manager, llm_func, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(iris_vector_rag, "_create_pipeline_legacy", legacy)
    iris_vector_rag.create_pipeline("basic", llm_config="opsreview")

    assert "llm_config" not in seen


def test_config_name_setting_reads_the_entry(recorder, monkeypatch):
    monkeypatch.setenv("RAG_LLM__CONFIG_NAME", "opsreview")

    iris_vector_rag.create_pipeline("basic")

    assert recorder["built"] == ["configstore:opsreview"]


def test_llm_config_wins_over_the_config_name_setting(recorder, monkeypatch):
    monkeypatch.setenv("RAG_LLM__CONFIG_NAME", "from-settings")

    iris_vector_rag.create_pipeline("basic", llm_config="from-code")

    assert recorder["built"] == ["configstore:from-code"]


def test_entry_wins_over_the_yaml_model_settings(recorder, monkeypatch):
    monkeypatch.setenv("RAG_LLM__MODEL_NAME", "gpt-yaml")

    iris_vector_rag.create_pipeline("basic", llm_config="opsreview")

    assert recorder["built"] == ["configstore:opsreview"]
    assert recorder["yaml"] == []


def test_entry_is_read_over_the_pipeline_connection_manager(recorder):
    iris_vector_rag.create_pipeline(
        "basic", engine=FakeEngine(), llm_config="opsreview"
    )

    assert recorder["configstore"] == [("opsreview", "engine-connections")]


def test_nothing_set_calls_get_llm_func_exactly_as_before(recorder, monkeypatch):
    """FR-010: provider, model and limits as the YAML gives them; no entry is read."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    iris_vector_rag.create_pipeline("basic")

    assert recorder["configstore"] == []
    assert recorder["built"] == ["yaml"]
    kwargs = recorder["yaml"][0]
    assert kwargs["provider"] == ConfigurationManager().get("llm.provider", "openai")
    assert kwargs["model_name"] == ConfigurationManager().get(
        "llm.model_name", "gpt-4.1-mini"
    )


def test_nothing_set_with_validation_passes_no_llm_func_as_before(recorder):
    iris_vector_rag.create_pipeline("basic", validate_requirements=True)

    assert recorder["built"] == [None]
    assert recorder["configstore"] == []


def test_nothing_set_through_an_engine_calls_get_llm_func(recorder):
    iris_vector_rag.create_pipeline("basic", engine=FakeEngine())

    assert recorder["built"] == ["yaml"]
    assert recorder["configstore"] == []
