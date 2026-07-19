from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from iris_vector_rag.integrations.ivg import IVG_REQUIRED_TABLES
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
    manager._initialize_ivg_schema = lambda: {"tables_created": True}

    def mock_ivg_table_exists(table_name):
        if table_name == "rdf_edges":
            raise RuntimeError("permission denied")
        return True

    manager._ivg_table_exists = mock_ivg_table_exists

    with patch(
        "iris_vector_rag.storage.schema_manager.assert_ivg_compatible",
        return_value="2.0.0",
    ):
        result = manager.ensure_iris_vector_graph_tables(pipeline_type="graphrag")
        assert result.tables_attempted == list(IVG_REQUIRED_TABLES)
        assert result.tables_created["rdf_edges"] is False
        assert "rdf_edges" in result.error_messages
        assert "permission denied" in result.error_messages["rdf_edges"]


def test_initialization_records_timing():
    manager = _manager_without_init()
    manager._detect_iris_vector_graph = lambda: True
    manager._initialize_ivg_schema = lambda: {"tables_created": True}
    manager._ivg_table_exists = lambda table_name: True

    times = iter([100.0, 100.4])
    with patch(
        "iris_vector_rag.storage.schema_manager.assert_ivg_compatible",
        return_value="2.0.0",
    ), patch.object(time, "time", side_effect=lambda: next(times)):
        result = manager.ensure_iris_vector_graph_tables(pipeline_type="graphrag")

    assert 0 <= result.total_time_seconds <= 1.0
