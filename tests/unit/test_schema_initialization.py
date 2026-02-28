from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from iris_vector_rag.storage.schema_manager import SchemaManager


def _manager_without_init() -> SchemaManager:
    return SchemaManager.__new__(SchemaManager)


def test_initialization_requires_package_for_graphrag():
    manager = _manager_without_init()
    manager._detect_iris_vector_graph = lambda: False

    with pytest.raises(ImportError):
        manager.ensure_iris_vector_graph_tables(pipeline_type="graphrag")


def test_initialization_skips_for_non_graphrag():
    manager = _manager_without_init()
    manager._detect_iris_vector_graph = lambda: False

    result = manager.ensure_iris_vector_graph_tables(pipeline_type="rag")
    assert result.package_detected is False
    assert result.tables_attempted == []
    assert result.tables_created == {}


def test_initialization_records_per_table_errors():
    manager = _manager_without_init()
    manager._detect_iris_vector_graph = lambda: True

    def mock_ensure_table_schema(table_name, pipeline_type=None):
        if table_name == "rdf_edges":
            raise RuntimeError("permission denied")
        return True

    manager.ensure_table_schema = mock_ensure_table_schema

    result = manager.ensure_iris_vector_graph_tables(pipeline_type="graphrag")
    assert result.tables_created["rdf_edges"] is False
    assert "rdf_edges" in result.error_messages
    assert "permission denied" in result.error_messages["rdf_edges"]


def test_initialization_records_timing():
    manager = _manager_without_init()
    manager._detect_iris_vector_graph = lambda: True
    manager.ensure_table_schema = lambda table_name, pipeline_type=None: True

    times = iter([100.0, 100.4])
    with patch.object(time, "time", side_effect=lambda: next(times)):
        result = manager.ensure_iris_vector_graph_tables(pipeline_type="graphrag")

    assert 0 <= result.total_time_seconds <= 1.0
