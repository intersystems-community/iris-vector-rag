"""GraphRAG answer generation accepts the custom prompt its callers pass.

`HybridGraphRAGPipeline.query()` calls
`self._generate_answer(query_text, retrieved_documents, custom_prompt)`, but the
inherited `GraphRAGPipeline._generate_answer` took only two arguments. Every
hybrid query with `generate_answer=True` therefore raised TypeError, was caught by
the surrounding `except Exception`, and returned the string "Error generating
answer" — answer generation was dead on that pipeline, silently.

The signature and the prompt formatting follow BasicRAG's `_generate_answer`,
which is the pipeline the `custom_prompt` option is documented against.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from iris_vector_rag.core.models import Document
from iris_vector_rag.pipelines.graphrag import GraphRAGPipeline
from iris_vector_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline


@pytest.fixture
def documents():
    return [
        Document(id="d1", page_content="Insulin resistance blunts glucose uptake."),
        Document(id="d2", page_content="Beta cells compensate by secreting more."),
    ]


def _recorder():
    """A pipeline reduced to what _generate_answer touches: llm_func."""
    prompts: list[str] = []

    def llm_func(prompt: str) -> str:
        prompts.append(prompt)
        return "an answer"

    return SimpleNamespace(llm_func=llm_func), prompts


def test_accepts_custom_prompt_positionally(documents):
    """The three-positional-argument call the hybrid pipeline makes must work."""
    pipeline, prompts = _recorder()

    answer = GraphRAGPipeline._generate_answer(
        pipeline, "How does insulin resistance develop?", documents, "Q: {query}"
    )

    assert answer == "an answer"
    assert prompts == ["Q: How does insulin resistance develop?"]


def test_custom_prompt_receives_query_and_context(documents):
    """Same template fields as BasicRAG: {query} and {context}."""
    pipeline, prompts = _recorder()

    GraphRAGPipeline._generate_answer(
        pipeline, "why?", documents, "{context}\n---\n{query}"
    )

    assert prompts == [
        "Insulin resistance blunts glucose uptake.\n\n"
        "Beta cells compensate by secreting more.\n---\nwhy?"
    ]


def test_custom_prompt_is_optional(documents):
    """Callers that pass no prompt still get the default one."""
    pipeline, prompts = _recorder()

    GraphRAGPipeline._generate_answer(pipeline, "why?", documents)

    assert len(prompts) == 1
    assert "why?" in prompts[0]
    assert "Insulin resistance blunts glucose uptake." in prompts[0]


def test_hybrid_call_site_matches_the_signature():
    """Bind the hybrid pipeline's call shape against the inherited signature.

    A signature check, not a mock: the bug was arity, and arity is what an
    inherited method quietly gets wrong when only one subclass calls it.
    """
    signature = inspect.signature(HybridGraphRAGPipeline._generate_answer)

    signature.bind(
        object(), "query text", [Document(id="d1", page_content="body")], "{query}"
    )
