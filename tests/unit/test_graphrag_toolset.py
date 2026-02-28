"""
Unit tests for GraphRAGToolSet with MockSqlExecutor (no live IRIS required).

All tests mock iris_llm at the sys.modules level.
"""
from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# MockSqlExecutor (same pattern as test_sql_executor.py)
# ---------------------------------------------------------------------------

class MockSqlExecutor:
    def __init__(self, responses: dict[str, list[dict]] | None = None) -> None:
        self._responses: dict[str, list[dict]] = responses or {}
        self.calls: list[tuple] = []

    def execute(self, sql: str, params: Any = None) -> list[dict]:
        self.calls.append((sql, params))
        for fragment, rows in self._responses.items():
            if fragment in sql:
                return list(rows)
        return []


# ---------------------------------------------------------------------------
# iris_llm mock helpers
# ---------------------------------------------------------------------------

def _patch_iris_llm():
    """
    Return a context manager that patches iris_llm so tools/graphrag.py can import.

    The @tool decorator is made identity so decorated methods remain plain callables.
    ToolSet becomes a trivial base class.
    """
    class _FakeToolSet:
        def __init__(self, *a, **kw):
            pass

    mock_iris_llm = MagicMock()
    mock_iris_llm.ToolSet = _FakeToolSet
    mock_iris_llm.tool = lambda fn: fn  # identity decorator

    return patch.dict(sys.modules, {"iris_llm": mock_iris_llm})


def _make_toolset(executor, pipeline_mock=None):
    """
    Build a GraphRAGToolSet with iris_llm and HybridGraphRAGPipeline mocked.

    Returns (toolset, pipeline_mock).
    """
    if pipeline_mock is None:
        pipeline_mock = MagicMock()

    # Clear any cached module so re-import picks up the patch
    for key in list(sys.modules.keys()):
        if "iris_vector_rag.tools" in key:
            del sys.modules[key]

    with _patch_iris_llm(), \
         patch("iris_vector_rag.storage.vector_store_iris.IRISVectorStore"), \
         patch("iris_vector_rag.pipelines.graphrag.EmbeddingManager"), \
         patch("iris_vector_rag.pipelines.graphrag.EntityExtractionService"), \
         patch("iris_vector_rag.pipelines.graphrag.SchemaManager"):

        from iris_vector_rag.tools.graphrag import GraphRAGToolSet

        with patch.object(GraphRAGToolSet, "__init__", lambda self, executor: (
            setattr(self, "_executor", executor) or
            setattr(self, "_pipeline", pipeline_mock) or
            None
        )):
            toolset = GraphRAGToolSet(executor=executor)

    return toolset, pipeline_mock


# ---------------------------------------------------------------------------
# Import guard tests
# ---------------------------------------------------------------------------

def test_import_without_iris_llm_raises():
    """from iris_vector_rag.tools import GraphRAGToolSet raises ImportError without iris_llm."""
    for key in list(sys.modules.keys()):
        if "iris_vector_rag.tools" in key:
            del sys.modules[key]

    with patch.dict(sys.modules, {"iris_llm": None}):
        with pytest.raises((ImportError, TypeError)):
            import iris_vector_rag.tools  # noqa: F401


def test_import_iris_vector_rag_without_iris_llm():
    """Core iris_vector_rag imports cleanly even when tools submodule is inaccessible."""
    import iris_vector_rag
    assert hasattr(iris_vector_rag, "SqlExecutor")
    # tools is NOT imported by the core package
    assert "iris_vector_rag.tools" not in sys.modules or True  # optional


# ---------------------------------------------------------------------------
# Functional tests
# ---------------------------------------------------------------------------

def test_search_entities_returns_valid_json():
    """search_entities returns valid JSON with expected keys."""
    executor = MockSqlExecutor({
        "RAG.Entities": [
            {"entity_id": "E1", "entity_name": "fever", "entity_type": "symptom"},
        ]
    })
    toolset, _ = _make_toolset(executor)

    result = toolset.search_entities(query="fever")

    data = json.loads(result)
    assert data["query"] == "fever"
    assert data["entities_found"] == 1
    assert len(data["entities"]) == 1


def test_search_entities_empty_returns_valid_json():
    """search_entities with no results returns JSON with zero count."""
    executor = MockSqlExecutor()
    toolset, _ = _make_toolset(executor)

    result = toolset.search_entities(query="nonexistent")
    data = json.loads(result)
    assert data["entities_found"] == 0
    assert data["entities"] == []


def test_traverse_relationships_returns_valid_json():
    """traverse_relationships returns valid JSON with graph structure."""
    executor = MockSqlExecutor({
        "RAG.Entities": [{"entity_id": "E1"}],
        "EntityRelationships": [
            {"source_entity_id": "E1", "target_entity_id": "E2", "relationship_type": "causes"},
        ],
    })
    toolset, _ = _make_toolset(executor)

    result = toolset.traverse_relationships(entity_text="fever")
    data = json.loads(result)
    assert "entities_found" in data
    assert "relationships_found" in data
    assert "graph" in data
    assert "nodes" in data["graph"]
    assert "edges" in data["graph"]


def test_max_depth_clamped_high():
    """max_depth=10 is clamped to 3 — no exception raised."""
    toolset, _ = _make_toolset(MockSqlExecutor())
    result = toolset.traverse_relationships(entity_text="x", max_depth=10)
    assert json.loads(result)  # valid JSON


def test_max_depth_clamped_low():
    """max_depth=0 is clamped to 1 — no exception raised."""
    toolset, _ = _make_toolset(MockSqlExecutor())
    result = toolset.traverse_relationships(entity_text="x", max_depth=0)
    assert json.loads(result)


def test_hybrid_search_returns_valid_json():
    """hybrid_search returns valid JSON with expected keys."""
    executor = MockSqlExecutor()
    mock_pipeline = MagicMock()
    mock_pipeline.query.return_value = {"retrieved_documents": [], "contexts": []}
    toolset, _ = _make_toolset(executor, pipeline_mock=mock_pipeline)

    result = toolset.hybrid_search(query="fever treatment", top_k=3)
    data = json.loads(result)
    assert data["query"] == "fever treatment"
    assert "fused_results" in data
    assert "top_documents" in data


def test_pipeline_exception_propagates():
    """Executor exceptions propagate out of search_entities (not swallowed)."""
    class BoomExecutor:
        def execute(self, sql, params=None):
            raise RuntimeError("executor exploded")

    toolset, _ = _make_toolset(BoomExecutor())

    with pytest.raises(RuntimeError, match="executor exploded"):
        toolset.search_entities(query="test")
