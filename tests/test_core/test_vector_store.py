"""
Defines the contract tests for any VectorStore implementation.
"""
import abc
import pytest
from typing import Any, Dict

# Placeholder for Document and VectorStoreError until actual definitions are available
# These would typically be imported from iris_rag.core.models and iris_rag.storage.vector_store.exceptions
class Document:
    def __init__(self, page_content: str, metadata: Dict[str, Any], doc_id: str = None):
        self.page_content = page_content
        self.metadata = metadata
        self.doc_id = doc_id

    def __eq__(self, other):
        if not isinstance(other, Document):
            return NotImplemented
        return (self.page_content == other.page_content and
                self.metadata == other.metadata and
                self.doc_id == other.doc_id)

class VectorStoreError(Exception):
    """Base exception for VectorStore errors."""
    pass

class VectorStoreConnectionError(VectorStoreError):
    """Exception for connection issues with the VectorStore."""
    pass

class VectorStoreDataError(VectorStoreError):
    """Exception for data-related issues in the VectorStore."""
    pass


class VectorStoreContractTests(abc.ABC):
    """
    Abstract base class for testing any VectorStore implementation.
    Subclasses must provide a fixture named `vector_store` that returns an instance
    of the VectorStore implementation to be tested.
    """

    @abc.abstractmethod
    @pytest.fixture
    def vector_store(self) -> Any:
        """
        Fixture to provide an instance of the VectorStore implementation.
        This needs to be implemented by concrete test classes.
        """
        pass

    @abc.abstractmethod
    def test_add_and_fetch_documents_by_ids(self, vector_store: Any):
        """Verifies documents can be added and retrieved by ID."""
        pass

    @abc.abstractmethod
    def test_similarity_search_returns_correct_format(self, vector_store: Any):
        """Verifies search returns List[Tuple[Document, float]]."""
        pass

    @abc.abstractmethod
    def test_delete_documents(self, vector_store: Any):
        """Verifies documents can be deleted."""
        pass

    @abc.abstractmethod
    def test_document_content_is_string_after_retrieval(self, vector_store: Any):
        """
        Verifies that page_content and metadata values (especially title)
        are strings, even if the underlying storage might have CLOBs.
        """
        pass

    @abc.abstractmethod
    def test_empty_results_handled_gracefully_for_search(self, vector_store: Any):
        """Verifies empty results are handled gracefully for similarity search."""
        pass

    @abc.abstractmethod
    def test_empty_results_handled_gracefully_for_fetch(self, vector_store: Any):
        """Verifies empty results are handled gracefully for fetching by IDs."""
        pass

    @abc.abstractmethod
    def test_filter_in_similarity_search(self, vector_store: Any):
        """Verifies filtering in similarity search (if applicable)."""
        pass

    @abc.abstractmethod
    def test_get_document_count(self, vector_store: Any):
        """Verifies getting the total number of documents."""
        pass

    @abc.abstractmethod
    def test_clear_documents(self, vector_store: Any):
        """Verifies clearing all documents from the store."""
        pass

    # --- Error Handling Tests ---
    # These might be more specific to implementations or tested via mocks

    @abc.abstractmethod
    def test_raises_connection_error_on_connection_failure(self, vector_store: Any):
        """Tests that a VectorStoreConnectionError is raised on connection failure."""
        pass

    @abc.abstractmethod
    def test_raises_data_error_on_malformed_data(self, vector_store: Any):
        """Tests that a VectorStoreDataError is raised when adding malformed data."""
        pass