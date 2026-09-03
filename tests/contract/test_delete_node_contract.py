"""Contract tests for HybridGraphRAGPipeline.delete_node (Feature 062).

Uses __new__ to bypass __init__ (which requires iris_vector_graph) and tests
the method's behavior against mocked stores.
"""

from unittest.mock import MagicMock, patch

import pytest


def _make_pipeline():
    from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

    p = HybridGraphRAGPipeline.__new__(HybridGraphRAGPipeline)
    p.iris_engine = None
    p.vector_store = None
    return p


class TestDeleteNodeContract:
    def test_delete_node_calls_all_stores(self):
        p = _make_pipeline()
        p.iris_engine = MagicMock()
        p.iris_engine.delete_node.return_value = True
        p.vector_store = MagicMock()

        p.delete_node("node-123")

        p.iris_engine.delete_node.assert_called_once_with("node-123")
        p.vector_store.delete_documents.assert_called_once_with(["node-123"])

    def test_delete_node_idempotent_missing(self):
        p = _make_pipeline()
        p.iris_engine = MagicMock()
        p.iris_engine.delete_node.return_value = False
        p.vector_store = MagicMock()
        p.vector_store.delete_documents.return_value = False

        result = p.delete_node("nonexistent-node")

        assert result is None

    def test_delete_node_rejects_empty_string(self):
        p = _make_pipeline()
        with pytest.raises(ValueError, match="non-empty string"):
            p.delete_node("")

    def test_delete_node_rejects_none(self):
        p = _make_pipeline()
        with pytest.raises(ValueError, match="non-empty string"):
            p.delete_node(None)

    def test_delete_node_partial_failure_logs_warning(self):
        p = _make_pipeline()
        p.iris_engine = MagicMock()
        p.iris_engine.delete_node.side_effect = RuntimeError("KG store down")
        p.vector_store = MagicMock()

        with patch(
            "iris_vector_rag.pipelines.hybrid_graphrag.logger"
        ) as mock_logger:
            with pytest.raises(RuntimeError, match="KG store down"):
                p.delete_node("node-456")
            mock_logger.warning.assert_called_once()

    def test_delete_node_no_iris_engine(self):
        p = _make_pipeline()
        p.iris_engine = None
        p.vector_store = MagicMock()

        p.delete_node("node-789")

        p.vector_store.delete_documents.assert_called_once_with(["node-789"])

    def test_delete_node_no_vector_store(self):
        p = _make_pipeline()
        p.iris_engine = MagicMock()
        p.iris_engine.delete_node.return_value = True
        p.vector_store = None

        p.delete_node("node-abc")

        p.iris_engine.delete_node.assert_called_once_with("node-abc")

    def test_delete_node_returns_none(self):
        p = _make_pipeline()
        p.iris_engine = MagicMock()
        p.iris_engine.delete_node.return_value = True
        p.vector_store = MagicMock()

        result = p.delete_node("node-xyz")

        assert result is None

    @pytest.mark.skip(
        reason="IRISGraphRAGBridge lives in opsreview repo; deferred to Feature 062 opsreview task"
    )
    def test_bridge_delete_node_delegates(self):
        pass
