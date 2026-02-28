from __future__ import annotations

import importlib.util
from unittest.mock import patch

from iris_vector_rag.storage.schema_manager import SchemaManager


def test_detect_iris_vector_graph_true():
    manager = SchemaManager.__new__(SchemaManager)
    with patch.object(importlib.util, "find_spec", return_value=object()):
        assert manager._detect_iris_vector_graph() is True


def test_detect_iris_vector_graph_false():
    manager = SchemaManager.__new__(SchemaManager)
    with patch.object(importlib.util, "find_spec", return_value=None):
        assert manager._detect_iris_vector_graph() is False
