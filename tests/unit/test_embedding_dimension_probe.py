"""embedding_dimension_of: read a model's width without tripping sentence-transformers 6.x.

sentence-transformers 6 renamed get_sentence_embedding_dimension() to
get_embedding_dimension() and warns on the old name. pyproject sets filterwarnings=error,
so calling the old name when the new one exists turns into an exception.
"""

import warnings

import pytest

from iris_vector_rag.embeddings.dimension import embedding_dimension_of


class _NewNameOnly:
    def get_embedding_dimension(self):
        return 384


class _OldNameOnly:
    def get_sentence_embedding_dimension(self):
        return 384


class _BothNames:
    """Old name warns on every call, the way 6.x does."""

    def get_embedding_dimension(self):
        return 384

    def get_sentence_embedding_dimension(self):
        warnings.warn(
            "get_sentence_embedding_dimension is deprecated; use get_embedding_dimension",
            FutureWarning,
            stacklevel=2,
        )
        return 384


@pytest.mark.parametrize(
    "model", [_NewNameOnly(), _OldNameOnly(), _BothNames()], ids=["new", "old", "both"]
)
def test_returns_the_dimension_for_every_model_shape(model):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        assert embedding_dimension_of(model) == 384


def test_never_calls_the_old_name_when_the_new_one_exists():
    calls = []

    class _Recording(_BothNames):
        def get_sentence_embedding_dimension(self):
            calls.append("old")
            raise FutureWarning("the old name must not be called")

        def get_embedding_dimension(self):
            calls.append("new")
            return 12

    assert embedding_dimension_of(_Recording()) == 12
    assert calls == ["new"]


def test_a_model_with_neither_method_is_an_error():
    with pytest.raises(AttributeError):
        embedding_dimension_of(object())
