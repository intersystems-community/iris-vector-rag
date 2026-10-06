"""The retrieval_method labels HybridGraphRAG reports are a documented, closed set."""

import ast
import inspect

from iris_vector_rag.pipelines import (
    graphrag,
    hybrid_graphrag,
    hybrid_graphrag_retrieval,
)
from iris_vector_rag.pipelines.hybrid_graphrag import RETRIEVAL_METHODS


def _returned_labels(module):
    """String literals in the second slot of every `return docs, "<label>"`."""
    labels = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if (
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Tuple)
            and len(node.value.elts) == 2
            and isinstance(node.value.elts[1], ast.Constant)
            and isinstance(node.value.elts[1].value, str)
        ):
            labels.add(node.value.elts[1].value)
    return labels


def test_every_label_the_pipeline_can_return_is_documented():
    emitted = (
        _returned_labels(hybrid_graphrag)
        | _returned_labels(hybrid_graphrag_retrieval)
        | _returned_labels(graphrag)
    )
    # Reported as "vector_fallback" by HybridGraphRAGPipeline.query, never as itself.
    emitted.discard("no_matching_entities")

    assert emitted <= RETRIEVAL_METHODS, sorted(emitted - RETRIEVAL_METHODS)


def test_the_documented_set_names_the_engine_labels():
    assert {
        "hybrid_fusion",
        "rrf_fusion",
        "enhanced_text",
        "hnsw_vector",
        "vector_fallback",
    } <= RETRIEVAL_METHODS
