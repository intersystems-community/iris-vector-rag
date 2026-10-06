"""`CRAGPipeline()` (what the MCP server builds) must have a connection manager:
`_enhance_retrieval` and the knowledge-base expansion use it, and with None they
failed with "'NoneType' object has no attribute 'get_connection'", which the
evaluator guard turned into a silent `fallback_vector` status."""

from unittest.mock import Mock, patch

import pytest

from iris_vector_rag.pipelines.crag import CRAGPipeline


@pytest.mark.usefixtures("use_test_embedder")
def test_no_arguments_builds_a_default_connection_manager():
    manager = Mock(name="connection_manager")
    with patch(
        "iris_vector_rag.pipelines.crag.ConnectionManager", return_value=manager
    ):
        pipeline = CRAGPipeline(vector_store=Mock(), llm_func=lambda p: "a")
    assert pipeline.connection_manager is manager


@pytest.mark.usefixtures("use_test_embedder")
def test_a_given_connection_manager_is_kept():
    given = Mock(name="given")
    pipeline = CRAGPipeline(
        connection_manager=given, vector_store=Mock(), llm_func=lambda p: "a"
    )
    assert pipeline.connection_manager is given


@pytest.mark.parametrize(
    "status, strategy",
    [
        ("confident", None),
        ("ambiguous", "chunk_enhancement"),
        ("disoriented", "knowledge_base_expansion"),
        ("fallback_vector", "vector_fallback"),
    ],
)
def test_the_correction_applied_is_named(status, strategy):
    assert CRAGPipeline._correction_strategy(status) == strategy
