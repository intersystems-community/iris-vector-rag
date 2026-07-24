"""US5 contract: predictable search return types (T032 — TDD, must fail before T033–T034).

FR-014: explicit entry points each return one documented shape.
- search_by_text(query, top_k, filter) -> List[Document]
- search_by_vector(embedding, top_k, filter) -> List[Tuple[Document, float]]
- similarity_search (legacy polymorphic) behavior unchanged.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from iris_vector_rag.core.models import Document


def _make_store():
    from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

    with patch.object(IRISVectorStore, "__init__", lambda self, *a, **kw: None):
        store = IRISVectorStore.__new__(IRISVectorStore)
    store.config_manager = MagicMock()
    store.config_manager._spec = None  # not a mock in the "is Mock" sense
    store.vector_dimension = 3
    store.logger = MagicMock()

    stub_doc = Document(id="1", page_content="hello", metadata={})
    store.similarity_search_by_embedding = MagicMock(
        return_value=[(stub_doc, 0.9)]
    )
    return store, stub_doc


# ---------------------------------------------------------------------------
# search_by_text — must return List[Document] (plain, no score tuples)
# ---------------------------------------------------------------------------

class TestSearchByText:
    def test_method_exists(self):
        """search_by_text must exist on IRISVectorStore (FR-014)."""
        from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
        assert hasattr(IRISVectorStore, "search_by_text"), (
            "FR-014: IRISVectorStore must have search_by_text() method"
        )

    def test_returns_list_of_documents(self):
        """search_by_text returns List[Document], never tuples (FR-014)."""
        store, stub_doc = _make_store()
        embedding = [0.1, 0.2, 0.3]
        with patch.object(type(store), "_embed_query", return_value=embedding, create=True):
            results = store.search_by_text("hello", top_k=1)

        assert isinstance(results, list), "Must return a list"
        assert len(results) > 0
        first = results[0]
        assert isinstance(first, Document), (
            f"search_by_text must return List[Document]; got {type(first)}"
        )

    def test_does_not_return_tuples(self):
        """search_by_text must NOT return (doc, score) tuples."""
        store, stub_doc = _make_store()
        embedding = [0.1, 0.2, 0.3]
        with patch.object(type(store), "_embed_query", return_value=embedding, create=True):
            results = store.search_by_text("hello", top_k=1)

        for item in results:
            assert not isinstance(item, tuple), (
                "search_by_text must return plain Documents, not (doc, score) tuples"
            )

    def test_passes_filter_through(self):
        """search_by_text forwards metadata_filter to the underlying search."""
        store, stub_doc = _make_store()
        embedding = [0.1, 0.2, 0.3]
        filt = {"source": "pubmed"}
        with patch.object(type(store), "_embed_query", return_value=embedding, create=True):
            store.search_by_text("hello", top_k=3, metadata_filter=filt)

        call_args = store.similarity_search_by_embedding.call_args
        assert call_args is not None
        # filter must have been forwarded (positional arg[2] or kwarg)
        passed_filter = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("filter")
        assert passed_filter == filt, (
            f"search_by_text must forward metadata_filter; got {passed_filter}"
        )


# ---------------------------------------------------------------------------
# search_by_vector — must return List[Tuple[Document, float]]
# ---------------------------------------------------------------------------

class TestSearchByVector:
    def test_method_exists(self):
        """search_by_vector must exist on IRISVectorStore (FR-014)."""
        from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
        assert hasattr(IRISVectorStore, "search_by_vector"), (
            "FR-014: IRISVectorStore must have search_by_vector() method"
        )

    def test_returns_list_of_tuples(self):
        """search_by_vector returns List[Tuple[Document, float]] (FR-014)."""
        store, stub_doc = _make_store()
        results = store.search_by_vector([0.1, 0.2, 0.3], top_k=1)

        assert isinstance(results, list), "Must return a list"
        assert len(results) > 0
        first = results[0]
        assert isinstance(first, tuple) and len(first) == 2, (
            f"search_by_vector must return (Document, float) tuples; got {type(first)}"
        )
        doc, score = first
        assert isinstance(doc, Document)
        assert isinstance(score, float)

    def test_passes_filter_through(self):
        """search_by_vector forwards filter to underlying search."""
        store, _ = _make_store()
        filt = {"source": "arxiv"}
        store.search_by_vector([0.1, 0.2, 0.3], top_k=2, metadata_filter=filt)

        call_args = store.similarity_search_by_embedding.call_args
        passed_filter = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("filter")
        assert passed_filter == filt


# ---------------------------------------------------------------------------
# Legacy similarity_search — behavior unchanged (backward compat)
# ---------------------------------------------------------------------------

class TestLegacySimilaritySearch:
    def test_vector_branch_still_returns_tuples(self):
        """similarity_search(embedding_list, top_k) still returns (doc, score) tuples."""
        store, stub_doc = _make_store()
        results = store.similarity_search([0.1, 0.2, 0.3], 1)
        assert isinstance(results, list)
        # vector branch returns List[Tuple[Document, float]]
        for item in results:
            assert isinstance(item, tuple) and len(item) == 2

    def test_text_branch_returns_enriched_documents(self):
        """similarity_search('text', k) still returns List[Document] with score in metadata."""
        store, stub_doc = _make_store()
        store.config_manager._spec = "mocked"  # triggers mock-detection branch
        results = store.similarity_search("hello world", 1)
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, Document)
