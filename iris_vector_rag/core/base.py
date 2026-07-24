import abc
import logging
from typing import Any, Dict, List, Optional, Tuple

from .models import Document
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline(abc.ABC):
    """
    Abstract base class for all RAG (Retrieval Augmented Generation) pipelines.

    This class defines the common interface that all RAG pipeline implementations
    must adhere to. It ensures that different RAG techniques can be used
    interchangeably within the framework.
    """

    def __init__(
        self,
        connection_manager,
        config_manager=None,
        vector_store: Optional[VectorStore] = None,
    ):
        """
        Initialize the RAG pipeline with connection and configuration managers.

        Supports two calling conventions:
        1. Legacy: RAGPipeline(connection_manager, config_manager, vector_store=None)
        2. Engine overload: RAGPipeline(engine, vector_store=None) where engine is IRISVectorEngine

        Args:
            connection_manager: Database connection manager OR IRISVectorEngine instance
            config_manager: Configuration manager (ignored if connection_manager is IRISVectorEngine)
            vector_store: Optional VectorStore instance. If None, IRISVectorStore will be instantiated.
        """
        # engine= overload: accept IRISVectorEngine as first positional arg
        if connection_manager.__class__.__name__ == "IRISVectorEngine":
            engine = connection_manager
            config_manager = engine.config_manager
            if vector_store is None:
                vector_store = engine.vector_store
            connection_manager = engine.connection_manager

        self.connection_manager = connection_manager
        self.config_manager = config_manager

        # Initialize vector store
        if vector_store is None:
            from ..storage.vector_store_iris import IRISVectorStore

            self.vector_store = IRISVectorStore(connection_manager, config_manager)
        else:
            self.vector_store = vector_store

        self._lazy_init_done = False

    def initialize(self) -> None:
        """Explicitly initialize pipeline schema. Idempotent; called lazily on first use."""
        if hasattr(self.vector_store, "schema_manager"):
            self.vector_store.schema_manager.ensure_schema_metadata_table()

    def _ensure_initialized(self) -> None:
        """Lazy init: call initialize() once on first use."""
        if not self._lazy_init_done:
            self.initialize()
            self._lazy_init_done = True

    def _validate_dimensions(self, embedding: List[float], expected_dims: int) -> None:
        actual_dims = len(embedding)
        if actual_dims != expected_dims:
            raise ValueError(
                f"Dimension mismatch: got {actual_dims}-dimensional embedding, "
                f"expected {expected_dims}. Verify your embedding model matches "
                f"the index dimension, or re-index with the correct model."
            )

    @abc.abstractmethod
    def load_documents(self, documents_path: str, **kwargs) -> Dict[str, Any]:
        """
        Loads and processes documents into the RAG pipeline&#x27;s knowledge base.

        This method handles the ingestion, chunking, embedding, and indexing
        of documents.

        Args:
            documents_path: Path to the documents or directory of documents.
            **kwargs: Additional keyword arguments for document loading.

        Returns:
            Dict with ingestion status containing keys:
                - documents_loaded: int (number of documents successfully loaded)
                - documents_failed: int (number of documents that failed to load)
                - embeddings_generated: int (number of embeddings generated)
        """

    @abc.abstractmethod
    def query(
        self, query_text: str, top_k: int = 5, generate_answer: bool = True, **kwargs
    ) -> Dict[str, Any]:
        """
        Unified query method that returns standardized response format.

        Each pipeline should override this method directly as per the pipeline architecture guide.
        The response should be in standardized format with these keys:
        - query: str
        - answer: str
        - retrieved_documents: List[Document]
        - contexts: List[str]
        - execution_time: float
        - metadata: Dict

        Args:
            query_text: The input query string.
            top_k: The number of top relevant documents to retrieve.
            generate_answer: Whether to generate an answer (default: True)
            **kwargs: Additional keyword arguments for the query process.

        Returns:
            Standardized dictionary with keys: query, retrieved_documents, contexts, metadata, answer, execution_time
        """

    # Protected helper methods for vector store operations
    def _retrieve_documents_by_vector(
        self,
        query_embedding: List[float],
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Retrieve documents using vector similarity search.

        Args:
            query_embedding: The query vector for similarity search
            top_k: Maximum number of results to return
            metadata_filter: Optional metadata filters to apply

        Returns:
            List of tuples containing (Document, similarity_score)
        """
        return self.vector_store.similarity_search(
            query_embedding=query_embedding, top_k=top_k, filter=metadata_filter
        )

    def _get_documents_by_ids(self, document_ids: List[str]) -> List[Document]:
        """
        Fetch documents by their IDs.

        Args:
            document_ids: List of document IDs to fetch

        Returns:
            List of Document objects with guaranteed string content
        """
        return self.vector_store.fetch_documents_by_ids(document_ids)

    def _store_documents(
        self, documents: List[Document], embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Store documents in the vector store.

        Args:
            documents: List of Document objects to store
            embeddings: Optional pre-computed embeddings for the documents

        Returns:
            List of document IDs that were stored
        """
        return self.vector_store.add_documents(documents, embeddings)

    # Public methods that all pipelines should have
    def ingest(self, documents: List[Document], **kwargs) -> None:
        """
        Ingest documents into the pipeline's knowledge base.

        This is an alias for load_documents() to maintain compatibility
        with existing test expectations.

        Args:
            documents: List of Document objects to ingest
            **kwargs: Additional arguments passed to load_documents()
        """
        self.load_documents(documents=documents, **kwargs)

    def clear(self) -> None:
        """
        Clear all documents from the pipeline's knowledge base.

        This method removes all stored documents and embeddings from
        the vector store.
        """
        if hasattr(self.vector_store, "clear"):
            self.vector_store.clear()
        else:
            # Fallback for vector stores without clear method
            logger.warning("Vector store does not support clear operation")

    def get_documents(self) -> List[Document]:
        """
        Retrieve all documents from the pipeline's knowledge base.

        Returns:
            List of all Document objects stored in the vector store
        """
        if hasattr(self.vector_store, "get_all_documents"):
            return self.vector_store.get_all_documents()
        else:
            # Fallback for vector stores without get_all_documents method
            logger.warning("Vector store does not support get_all_documents operation")
            return []

    def _store_embeddings(self, documents: List[Document]) -> None:
        """
        Store embeddings for documents in the vector store.

        This method generates embeddings for the provided documents
        and stores them in the vector store.

        Args:
            documents: List of Document objects to generate embeddings for
        """
        # This is typically handled by the vector store's add_documents method
        # but we provide this method for compatibility with existing tests
        self._store_documents(documents)

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        This method performs the retrieval step of the RAG pipeline,
        finding the most relevant documents for the given query.

        Args:
            query: The input query string
            top_k: Number of top relevant documents to retrieve
            **kwargs: Additional arguments for retrieval

        Returns:
            List of relevant Document objects
        """
        # This is typically implemented by calling the query() method
        # but we provide a default implementation for compatibility
        try:
            return self.query(query, top_k, **kwargs)
        except NotImplementedError:
            # If query() is not implemented, return empty list
            logger.warning(
                f"Query method not implemented for {self.__class__.__name__}"
            )
            return []
