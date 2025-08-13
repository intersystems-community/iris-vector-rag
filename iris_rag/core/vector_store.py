"""
Abstract base class for vector store implementations.

This module defines the VectorStore ABC that all vector store implementations
must follow, ensuring consistent interfaces across different storage backends.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from .models import Document


class VectorStore(ABC):
    """
    Abstract base class for vector store implementations.

    This class defines the interface that all vector store implementations
    must implement to ensure consistency across different storage backends.
    All methods that return Document objects must ensure that page_content
    and relevant metadata (like title) are strings, not CLOB objects.
    """

    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add
            embeddings: Optional pre-computed embeddings for the documents.
                       If None, embeddings should be computed by the implementation.
                       If provided, must have the same length as documents.

        Returns:
            List of document IDs that were added

        Raises:
            VectorStoreDataError: If document data is malformed
            VectorStoreConnectionError: If there are connection issues
        """
        pass

    @abstractmethod
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents from the vector store by their IDs.

        Args:
            ids: List of document IDs to delete

        Returns:
            True if any documents were deleted, False otherwise

        Raises:
            VectorStoreConnectionError: If there are connection issues
        """
        pass

    @abstractmethod
    def similarity_search(
        self, query_embedding: List[float], top_k: int, filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search using a query embedding.

        Args:
            query_embedding: The query vector for similarity search
            top_k: Maximum number of results to return
            filter: Optional metadata filters to apply

        Returns:
            List of tuples containing (Document, similarity_score).
            Document objects MUST have page_content and metadata values
            (especially title) as strings, not CLOB objects.

        Raises:
            VectorStoreConnectionError: If there are connection issues
        """
        pass

    @abstractmethod
    def fetch_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Fetch documents by their IDs.

        Args:
            ids: List of document IDs to fetch

        Returns:
            List of Document objects. Document objects MUST have page_content
            and metadata values (especially title) as strings, not CLOB objects.

        Raises:
            VectorStoreConnectionError: If there are connection issues
        """
        pass

    @abstractmethod
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the vector store.

        Returns:
            Total number of documents

        Raises:
            VectorStoreConnectionError: If there are connection issues
        """
        pass

    @abstractmethod
    def clear_documents(self) -> None:
        """
        Clear all documents from the vector store.

        Warning: This operation is irreversible.

        Raises:
            VectorStoreConnectionError: If there are connection issues
        """
        pass
