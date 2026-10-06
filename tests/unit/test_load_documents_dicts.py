"""`load_documents` is documented to take "dicts or Document objects". A dict used to
reach the vector store unconverted and fail there with
"'dict' object has no attribute 'page_content'"."""

from unittest.mock import Mock

import pytest

from iris_vector_rag.core.models import Document, coerce_documents
from iris_vector_rag.pipelines.basic import BasicRAGPipeline


def test_a_document_passes_through_untouched():
    doc = Document(page_content="x", metadata={"a": 1}, id="d1")
    assert coerce_documents([doc]) == [doc]


def test_a_dict_with_content_and_doc_id_becomes_a_document():
    (doc,) = coerce_documents(
        [
            {
                "doc_id": "PMC001",
                "title": "Risk",
                "content": "text",
                "metadata": {"year": 2023},
            }
        ]
    )
    assert isinstance(doc, Document)
    assert doc.id == "PMC001"
    assert doc.page_content == "text"
    assert doc.metadata == {"year": 2023, "title": "Risk"}


def test_page_content_and_id_spellings_are_accepted():
    (doc,) = coerce_documents([{"id": "a", "page_content": "p"}])
    assert (doc.id, doc.page_content) == ("a", "p")
    (doc,) = coerce_documents([{"text": "t"}])
    assert doc.page_content == "t" and doc.id


def test_the_callers_metadata_dict_is_not_mutated():
    meta = {"year": 1}
    coerce_documents([{"content": "c", "title": "T", "metadata": meta}])
    assert meta == {"year": 1}


def test_a_dict_without_text_is_refused_and_named():
    with pytest.raises(ValueError, match=r"document 1 .*content"):
        coerce_documents([{"content": "ok"}, {"doc_id": "empty"}])


def test_basic_pipeline_hands_documents_to_the_store():
    store = Mock()
    pipeline = BasicRAGPipeline(vector_store=store)
    pipeline.embedding_manager = None
    result = pipeline.load_documents(
        documents=[{"doc_id": "a", "content": "alpha", "title": "T"}],
        generate_embeddings=True,
    )
    (sent,), _ = store.add_documents.call_args
    assert all(isinstance(d, Document) for d in sent)
    assert sent[0].id == "a"
    assert result["documents_loaded"] == 1
