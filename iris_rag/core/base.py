import abc
from typing import List, Dict, Any, Optional, Tuple
from .models import Document
from .vector_store import VectorStore

class RAGPipeline(abc.ABC):
    """
    Abstract base class for all RAG (Retrieval Augmented Generation) pipelines.

    This class defines the common interface that all RAG pipeline implementations
    must adhere to. It ensures that different RAG techniques can be used
    interchangeably within the framework.
    """
    
    def __init__(self, connection_manager, config_manager, vector_store: Optional[VectorStore] = None):
        """
        Initialize the RAG pipeline with connection and configuration managers.
        
        Args:
            connection_manager: Database connection manager
            config_manager: Configuration manager
            vector_store: Optional VectorStore instance. If None, IRISVectorStore will be instantiated.
        """
        self.connection_manager = connection_manager
        self.config_manager = config_manager
        
        # Initialize vector store
        if vector_store is None:
            from ..storage.vector_store_iris import IRISVectorStore
            self.vector_store = IRISVectorStore(connection_manager, config_manager)
        else:
            self.vector_store = vector_store

    @abc.abstractmethod
    def execute(self, query_text: str, **kwargs) -> dict:
        """
        Executes the full RAG pipeline for a given query.

        This method should orchestrate the retrieval, augmentation, and generation
        steps of the pipeline.

        Args:
            query_text: The input query string.
            **kwargs: Additional keyword arguments specific to the pipeline implementation.

        Returns:
            A dictionary containing the pipeline&#x27;s output, typically including
            the original query, the generated answer, and retrieved documents.
            The exact structure is defined by the `Standard Return Format` rule.
        """
        pass

    @abc.abstractmethod
    def load_documents(self, documents_path: str, **kwargs) -> None:
        """
        Loads and processes documents into the RAG pipeline&#x27;s knowledge base.

        This method handles the ingestion, chunking, embedding, and indexing
        of documents.

        Args:
            documents_path: Path to the documents or directory of documents.
            **kwargs: Additional keyword arguments for document loading.
        """
        pass

    @abc.abstractmethod
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> list:
        """
        Performs the retrieval step of the RAG pipeline.

        Given a query, this method should return the most relevant document
        chunks or passages from the knowledge base.

        Args:
            query_text: The input query string.
            top_k: The number of top relevant documents to retrieve.
            **kwargs: Additional keyword arguments for the query process.

        Returns:
            A list of retrieved document objects or their representations.
        """
        pass
    
    def run(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Run the full RAG pipeline for a query (convenience method).
        
        This method simply calls execute() to maintain backward compatibility.
        
        Args:
            query: The input query
            **kwargs: Additional arguments passed to execute()
            
        Returns:
            Dictionary with query, answer, and retrieved documents
        """
        return self.execute(query, **kwargs)
    
    # Protected helper methods for vector store operations
    def _retrieve_documents_by_vector(
        self,
        query_embedding: List[float],
        top_k: int,
        metadata_filter: Optional[Dict[str, Any]] = None
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
            query_embedding=query_embedding,
            top_k=top_k,
            filter=metadata_filter
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
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
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