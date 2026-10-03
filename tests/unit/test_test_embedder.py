"""The test embedder (G6 point 3): deterministic, normalised, sized by its caller, loud once."""

import logging
import math

import pytest

from iris_vector_rag.testing import embedder as embedder_module
from iris_vector_rag.testing.embedder import MODEL_KEY, TestEmbedder


@pytest.fixture(autouse=True)
def _fresh_warning_state(monkeypatch):
    monkeypatch.setattr(embedder_module, "_warned", False)


def test_vectors_have_the_requested_dimension():
    vectors = TestEmbedder(dimension=7).embed_texts(["a", "b"])
    assert [len(v) for v in vectors] == [7, 7]


def test_dimension_is_not_a_default():
    with pytest.raises(TypeError):
        TestEmbedder()  # type: ignore[call-arg]


@pytest.mark.parametrize("bad", [0, -3, 2.5, "384", None, True])
def test_dimension_must_be_a_positive_integer(bad):
    with pytest.raises(ValueError, match="dimension"):
        TestEmbedder(dimension=bad)


def test_vectors_are_unit_length():
    (vector,) = TestEmbedder(dimension=64).embed_texts(["anything"])
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-9)


def test_same_text_same_vector_whatever_the_batch():
    embedder = TestEmbedder(dimension=16)
    alone = embedder.embed_texts(["needle"])[0]
    batched = embedder.embed_texts(["hay", "needle", "stack"])[1]
    assert alone == batched


def test_different_texts_differ():
    a, b = TestEmbedder(dimension=16).embed_texts(["one", "two"])
    assert a != b


def test_encode_matches_the_sentence_transformers_shape():
    embedder = TestEmbedder(dimension=5)
    assert len(embedder.encode("one text")) == 1
    assert len(embedder.encode(["a", "b", "c"], convert_to_tensor=False)) == 3
    assert embedder.get_sentence_embedding_dimension() == 5


def test_identity_is_stub():
    assert MODEL_KEY == "stub"
    assert TestEmbedder(dimension=3).model_key == "stub"


def test_warns_once_per_process(caplog):
    with caplog.at_level(logging.WARNING, logger=embedder_module.__name__):
        TestEmbedder(dimension=3).embed_texts(["x"])
        TestEmbedder(dimension=3).embed_texts(["y"])
        TestEmbedder(dimension=4)
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "not a real embedding model" in warnings[0].getMessage()
