"""Writes through ivg carry the active model's identity (G6 point 7)."""

from unittest.mock import MagicMock

import pytest

from iris_vector_rag.embeddings.manager import EmbeddingManager
from iris_vector_rag.integrations import ivg


class _Cfg:
    def __init__(self, embeddings):
        self._embeddings = embeddings

    def get(self, key, default=None):
        return self._embeddings if key == "embeddings" else default


def test_store_embedding_passes_the_model_key():
    engine = MagicMock()
    manager = EmbeddingManager(_Cfg({"provider": "stub", "dimension": 3}))
    ivg.store_embedding(engine, "n1", [0.1, 0.2, 0.3], manager, metadata={"a": 1})
    engine.store_embedding.assert_called_once_with(
        "n1", [0.1, 0.2, 0.3], {"a": 1}, model_key="stub"
    )


def test_store_embeddings_passes_the_model_key_and_graph():
    engine = MagicMock()
    manager = EmbeddingManager(_Cfg({"provider": "stub", "dimension": 3}))
    items = [{"node_id": "n1", "embedding": [0.1, 0.2, 0.3]}]
    ivg.store_embeddings(engine, items, manager, graph="g")
    engine.store_embeddings.assert_called_once_with(items, model_key="stub", graph="g")


def test_a_real_model_is_named_by_its_model_name(monkeypatch):
    class Model:
        def __init__(self, name, device="cpu"):
            pass

        def get_sentence_embedding_dimension(self):
            return 4

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", Model)
    manager = EmbeddingManager(
        _Cfg(
            {
                "primary_backend": "sentence_transformers",
                "fallback_backends": [],
                "sentence_transformers": {"model_name": "org/some-model"},
            }
        )
    )
    engine = MagicMock()
    ivg.store_embedding(engine, "n", [0.0] * 4, manager)
    assert engine.store_embedding.call_args.kwargs["model_key"] == "org/some-model"


def test_the_caller_cannot_override_the_model_key():
    manager = EmbeddingManager(_Cfg({"provider": "stub", "dimension": 3}))
    with pytest.raises(TypeError):
        ivg.store_embedding(MagicMock(), "n", [0.0] * 3, manager, model_key="other")


def test_ivgs_refusal_is_not_swallowed():
    engine = MagicMock()
    engine.store_embedding.side_effect = ValueError("dimension mismatch: recorded 384")
    manager = EmbeddingManager(_Cfg({"provider": "stub", "dimension": 3}))
    with pytest.raises(ValueError, match="dimension mismatch"):
        ivg.store_embedding(engine, "n", [0.0] * 3, manager)
