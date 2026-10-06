"""The real LLM and the real embedder, for integration and e2e tests.

The LLM is OpenAI's, reached through the project's own `get_llm_func` with the
`OPENAI_API_KEY` of the environment. The embedder is the real sentence-transformer the
project ships as its default. Nothing here fakes either; fakes belong in tests/unit.
"""

from __future__ import annotations

from typing import Callable

from iris_vector_rag.common.utils import get_embedding_func, get_llm_func

LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_WIDTH = 384  # the width of EMBEDDING_MODEL's vectors


def real_llm() -> Callable[[str], str]:
    """The real OpenAI model as a `prompt -> text` function."""
    return get_llm_func(provider="openai", model_name=LLM_MODEL)


def real_embedder() -> Callable:
    """The real sentence-transformer: a string gives a vector, a list gives a list."""
    return get_embedding_func(model_name_override=EMBEDDING_MODEL)
