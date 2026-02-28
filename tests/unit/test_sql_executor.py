"""
Unit tests for SqlExecutor protocol, MockSqlExecutor, and GraphRAGPipeline injection.

All tests run without a live IRIS connection (constitution P3: .DAT fixture-first;
unit tests may use MockSqlExecutor instead of fixtures when no DB state is needed).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from iris_vector_rag.executor import SqlExecutor

# ---------------------------------------------------------------------------
# MockSqlExecutor (T015)
# ---------------------------------------------------------------------------


class MockSqlExecutor:
    """
    In-memory SqlExecutor for testing.

    Responses are keyed by SQL fragments (substring match).  The first matching
    fragment wins.  Unmatched queries return ``[]``.
    """

    def __init__(self, responses: dict[str, list[dict]] | None = None) -> None:
        self._responses: dict[str, list[dict]] = responses or {}
        self.calls: list[tuple[str, Any]] = []

    def execute(self, sql: str, params: Any = None) -> list[dict]:
        self.calls.append((sql, params))
        for fragment, rows in self._responses.items():
            if fragment in sql:
                return list(rows)  # return a copy so callers can't mutate state
        return []


# ---------------------------------------------------------------------------
# Protocol tests (T016 — test_sql_executor_protocol_isinstance)
# ---------------------------------------------------------------------------


def test_sql_executor_protocol_isinstance():
    """MockSqlExecutor satisfies the SqlExecutor runtime-checkable protocol."""
    mock = MockSqlExecutor()
    assert isinstance(mock, SqlExecutor)


def test_plain_object_not_sql_executor():
    """An object without .execute() does not satisfy SqlExecutor."""
    assert not isinstance(object(), SqlExecutor)


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


def test_import_from_package():
    """SqlExecutor is importable from the top-level package."""
    from iris_vector_rag import SqlExecutor as SE  # noqa: F401

    assert SE is SqlExecutor


def test_import_from_executor_module():
    """SqlExecutor is importable directly from iris_vector_rag.executor."""
    from iris_vector_rag.executor import SqlExecutor as SE  # noqa: F401

    assert SE is SqlExecutor


# ---------------------------------------------------------------------------
# Pipeline constructor tests (T016 — test_pipeline_no_executor_unchanged)
# ---------------------------------------------------------------------------


def _make_pipeline(executor=None):
    """Create a GraphRAGPipeline with mocked collaborators and optional executor."""
    from iris_vector_rag.pipelines.graphrag import GraphRAGPipeline

    mock_cm = MagicMock()
    mock_cfg = MagicMock()
    mock_cfg.get.return_value = {}

    with patch("iris_vector_rag.pipelines.graphrag.EmbeddingManager"), patch(
        "iris_vector_rag.pipelines.graphrag.EntityExtractionService"
    ), patch("iris_vector_rag.pipelines.graphrag.SchemaManager"), patch(
        "iris_vector_rag.storage.vector_store_iris.IRISVectorStore"
    ):
        pipeline = GraphRAGPipeline(
            connection_manager=mock_cm,
            config_manager=mock_cfg,
            executor=executor,
        )
    return pipeline


def test_pipeline_no_executor_unchanged():
    """Pipeline constructed without executor stores None — existing path unchanged."""
    pipeline = _make_pipeline()
    assert pipeline._executor is None


def test_pipeline_stores_injected_executor():
    """Pipeline stores the injected executor on _executor."""
    mock_exec = MockSqlExecutor()
    pipeline = _make_pipeline(executor=mock_exec)
    assert pipeline._executor is mock_exec


# ---------------------------------------------------------------------------
# _execute_sql dispatch tests (T016 — test_pipeline_routes_through_executor)
# ---------------------------------------------------------------------------


def test_pipeline_routes_through_executor():
    """_execute_sql delegates to the injected executor and records the call."""
    mock_exec = MockSqlExecutor({"SELECT 1": [{"n": 1}]})
    pipeline = _make_pipeline(executor=mock_exec)

    result = pipeline._execute_sql("SELECT 1")

    assert result == [{"n": 1}]
    assert len(mock_exec.calls) == 1
    assert mock_exec.calls[0][0] == "SELECT 1"


def test_execute_sql_empty_returns_list():
    """Executor returning [] does not raise."""
    mock_exec = MockSqlExecutor()  # no responses configured
    pipeline = _make_pipeline(executor=mock_exec)

    result = pipeline._execute_sql("SELECT something FROM nowhere")
    assert result == []


def test_execute_sql_passes_params():
    """Parameters are forwarded to the executor."""
    mock_exec = MockSqlExecutor({"WHERE": [{"entity_id": "E1"}]})
    pipeline = _make_pipeline(executor=mock_exec)

    pipeline._execute_sql(
        "SELECT entity_id FROM RAG.Entities WHERE entity_name LIKE ?", ["%fever%"]
    )

    assert mock_exec.calls[0][1] == ["%fever%"]


# ---------------------------------------------------------------------------
# Exception propagation (T016 — test_execute_sql_empty_returns_list variant)
# ---------------------------------------------------------------------------


def test_executor_exception_propagates():
    """Exceptions from the executor propagate out of _execute_sql (not swallowed)."""

    class BoomExecutor:
        def execute(self, sql, params=None):
            raise RuntimeError("DB is on fire")

    pipeline = _make_pipeline(executor=BoomExecutor())

    with pytest.raises(RuntimeError, match="DB is on fire"):
        pipeline._execute_sql("SELECT 1")


# ---------------------------------------------------------------------------
# Phase 3 — US1: query-path tests (T018–T021)
# ---------------------------------------------------------------------------


def test_search_entities_via_executor(monkeypatch):
    """
    _find_seed_entities routes through executor; returned entity IDs come from mock rows.
    """
    mock_exec = MockSqlExecutor(
        {
            "RAG.Entities": [
                {"entity_id": "E1"},
                {"entity_id": "E2"},
            ]
        }
    )
    pipeline = _make_pipeline(executor=mock_exec)

    # Stub entity extraction to return one entity with .text = "fever"
    fake_entity = MagicMock()
    fake_entity.text = "fever"
    pipeline.entity_extraction_service.extract_entities = MagicMock(
        return_value=[fake_entity]
    )

    entity_ids = pipeline._find_seed_entities("patient has fever")

    assert "E1" in entity_ids
    assert "E2" in entity_ids
    # executor was called
    assert any("RAG.Entities" in call[0] for call in mock_exec.calls)


def test_traverse_relationships_via_executor():
    """
    _expand_neighborhood routes through executor; neighbors from mock rows are collected.

    The SQL is a UNION of two selects; the MockSqlExecutor returns a combined
    flat list (one column per row, mirroring what IRIS returns after UNION).
    """
    mock_exec = MockSqlExecutor(
        {
            "EntityRelationships": [
                {"target_entity_id": "E2"},
                {"target_entity_id": "E3"},
            ]
        }
    )
    pipeline = _make_pipeline(executor=mock_exec)

    visited = pipeline._expand_neighborhood({"E1"}, depth=1)

    assert "E1" in visited  # seed always present
    assert "E2" in visited  # returned by executor
    assert "E3" in visited  # second neighbor
    assert any("EntityRelationships" in call[0] for call in mock_exec.calls)


def test_empty_graph_returns_gracefully():
    """Executor returning [] for all queries → pipeline returns empty results, no exception."""
    mock_exec = MockSqlExecutor()  # all queries return []
    pipeline = _make_pipeline(executor=mock_exec)

    # _expand_neighborhood with empty result set
    visited = pipeline._expand_neighborhood({"E1"}, depth=2)
    assert "E1" in visited  # seed always retained
    assert len(visited) == 1  # no neighbors added

    # _get_documents_from_entities with empty result
    docs = pipeline._get_documents_from_entities({"E1"}, top_k=5)
    assert docs == []


def test_validate_knowledge_graph_empty(monkeypatch):
    """_validate_knowledge_graph returns False when executor returns empty count."""
    mock_exec = MockSqlExecutor({"COUNT": [{"COUNT(*)": 0}]})
    pipeline = _make_pipeline(executor=mock_exec)

    assert pipeline._validate_knowledge_graph() is False


def test_validate_knowledge_graph_populated():
    """_validate_knowledge_graph returns True when executor returns nonzero count."""
    mock_exec = MockSqlExecutor({"COUNT": [{"COUNT(*)": 42}]})
    pipeline = _make_pipeline(executor=mock_exec)

    assert pipeline._validate_knowledge_graph() is True
