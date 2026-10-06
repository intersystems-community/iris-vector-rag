"""GraphRAG's load_documents follows the base contract: the path is optional and
documents arrive as `documents=`, which is how `ingest()` calls it."""

from unittest.mock import Mock

import pytest

from iris_vector_rag.pipelines.graphrag import GraphRAGPipeline


def _pipeline():
    pipeline = GraphRAGPipeline.__new__(GraphRAGPipeline)
    pipeline.vector_store = Mock()
    pipeline.entity_extraction_enabled = False
    return pipeline


def test_documents_without_a_path_are_loaded():
    pipeline = _pipeline()
    result = pipeline.load_documents(documents=[{"doc_id": "a", "content": "text"}])
    assert result["documents_loaded"] == 1
    (sent,), _ = pipeline.vector_store.add_documents.call_args
    assert sent[0].id == "a" and sent[0].page_content == "text"


def test_a_path_alone_is_refused_not_ignored():
    with pytest.raises(ValueError, match="documents=\\["):
        _pipeline().load_documents("some/dir")


def test_nothing_at_all_loads_nothing():
    assert _pipeline().load_documents()["documents_loaded"] == 0
