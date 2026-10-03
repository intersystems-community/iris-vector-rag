"""Nothing stands in silently for an embedder that failed (G6).

`get_embedding_func` keeps its explicit stub, now the test embedder; CRAG no longer
falls back to it when the real one cannot be built.
"""

from unittest.mock import MagicMock, patch

import pytest

from iris_vector_rag.common import utils
from iris_vector_rag.exceptions import EmbeddingError


def test_explicit_stub_is_the_test_embedder():
    func = utils.get_embedding_func(provider="stub")
    vectors = func(["a", "b"])
    assert len(vectors) == 2
    assert len(vectors[0]) == utils.DEFAULT_EMBEDDING_DIMENSION
    assert sum(x * x for x in vectors[0]) == pytest.approx(1.0)
    assert (
        func("a") == vectors[0]
    ), "a single string returns one vector, same as in a list"


def test_the_hf_embedder_does_not_return_zeros_when_it_gets_nothing_back():
    with patch.object(utils, "build_hf_embedder", return_value=lambda texts: []):
        func = utils.get_embedding_func(provider="huggingface")
        with pytest.raises(EmbeddingError):
            func("text")


def test_crag_does_not_fall_back_to_a_stub_embedder():
    from iris_vector_rag.pipelines.crag import CRAGPipeline

    def real_fails_stub_works(model_name_override=None, provider=None, mock=False):
        if provider == "stub":
            return lambda texts: [[0.0]]
        raise RuntimeError("no model")

    with (
        patch.object(utils, "get_embedding_func", side_effect=real_fails_stub_works),
        patch("iris_vector_rag.embeddings.manager.EmbeddingManager"),
    ):
        with pytest.raises(RuntimeError, match="no model"):
            CRAGPipeline(
                connection_manager=MagicMock(),
                config_manager=MagicMock(),
                vector_store=MagicMock(),
                llm_func=lambda prompt: "",
            )
