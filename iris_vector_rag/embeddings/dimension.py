"""Read the embedding width from a model, without calling a deprecated name.

sentence-transformers 6 renamed get_sentence_embedding_dimension() to
get_embedding_dimension() and warns on the old name. The package's filterwarnings=error
turns that warning into an exception, so the new name is preferred whenever it exists.
"""

from typing import Any, Optional


def embedding_dimension_of(model: Any) -> Optional[int]:
    """Return the model's output width, or None if it reports none.

    Raises AttributeError if the model has neither method.
    """
    getter = getattr(model, "get_embedding_dimension", None)
    if getter is None:
        getter = model.get_sentence_embedding_dimension
    reported = getter()
    return int(reported) if reported else None
