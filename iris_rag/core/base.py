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
    
    def __init__(self, config_manager, vector_store: Optional[VectorStore] = None, connection_manager=None):
        """
        Initialize the RAG pipeline with configuration and connection managers.
        
        Args:
            config_manager: Configuration manager
            vector_store: Optional VectorStore instance. If None, IRISVectorStore will be instantiated.
            connection_manager: Optional IRISConnectionManager instance.
        """
        self.config_manager = config_manager
        
        # Initialize schema manager and vector store
        from ..storage.schema_manager import SchemaManager
        from common.iris_connection_manager import IRISConnectionManager
        
        # Use provided connection_manager or create a new one
        self.connection_manager = connection_manager or IRISConnectionManager(config_manager)
        self.schema_manager = SchemaManager(self.connection_manager, config_manager)
        
        if vector_store is None:
            from ..storage.vector_store_iris import IRISVectorStore
            # Pass embedding manager if available (for automatic embedding generation)
            embedding_manager = getattr(self, 'embedding_manager', None)
            self.vector_store = IRISVectorStore(
                config_manager,
                schema_manager=self.schema_manager,
                connection_manager=self.connection_manager,
                embedding_manager=embedding_manager
            )
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

    def load_documents(self, documents_path: str, **kwargs) -> None:
        """
        Default implementation with automatic chunking via vector store.
        
        Pipelines can override this method if they need special processing,
        but most should use this default implementation.
        
        Args:
            documents_path: Path to the documents or directory of documents.
            **kwargs: Additional keyword arguments including:
                - documents: List of Document objects (if providing directly)
                - auto_chunk: Override automatic chunking setting
                - chunking_strategy: Override chunking strategy
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Load documents from path or direct input
        documents = self._get_documents(documents_path, **kwargs)
        
        # Get pipeline-specific chunking configuration
        pipeline_name = self._get_pipeline_name()
        chunking_config = self._get_chunking_config(pipeline_name)
        
        # Apply any pipeline-specific document preprocessing
        # Extract documents from kwargs to avoid conflicts
        preprocessing_kwargs = {k: v for k, v in kwargs.items() if k != 'documents'}
        documents = self._preprocess_documents(documents, **preprocessing_kwargs)
        
        # Override chunking config with kwargs if provided
        auto_chunk = kwargs.get("auto_chunk", chunking_config.get("enabled", True))
        chunking_strategy = kwargs.get("chunking_strategy", chunking_config.get("strategy", "fixed_size"))
        
        # Delegate to vector store with automatic chunking
        document_ids = self.vector_store.add_documents(
            documents=documents,
            auto_chunk=auto_chunk,
            chunking_strategy=chunking_strategy
        )
        
        logger.info(f"{pipeline_name}: Loaded {len(documents)} documents with chunking (auto_chunk={auto_chunk}, strategy={chunking_strategy}), generated {len(document_ids)} chunks/documents")

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
    
    def _get_documents(self, documents_path: str, **kwargs) -> List[Document]:
        """Load documents from path or kwargs."""
        if "documents" in kwargs:
            return kwargs["documents"]
        return self._load_documents_from_path(documents_path)
    
    def _get_pipeline_name(self) -> str:
        """Extract pipeline name from class name for configuration lookup."""
        class_name = self.__class__.__name__.lower()
        # Remove common suffixes to get clean pipeline name
        for suffix in ['pipeline', 'ragpipeline', 'rag']:
            if class_name.endswith(suffix):
                class_name = class_name[:-len(suffix)]
                break
        return class_name
    
    def _get_chunking_config(self, pipeline_name: str) -> dict:
        """Get pipeline-specific chunking configuration."""
        config_key = f"pipeline_overrides:{pipeline_name}:chunking"
        return self.config_manager.get(config_key, {})
    
    def _preprocess_documents(self, documents: List[Document], **kwargs) -> List[Document]:
        """
        Hook for pipeline-specific document preprocessing.
        Override in subclasses if needed.
        """
        return documents
    
    def _load_documents_from_path(self, documents_path: str) -> List[Document]:
        """Load documents from file or directory path."""
        import os
        
        documents = []
        
        if os.path.isfile(documents_path):
            # Single file
            documents.append(self._load_single_file(documents_path))
        elif os.path.isdir(documents_path):
            # Directory of files
            for filename in os.listdir(documents_path):
                file_path = os.path.join(documents_path, filename)
                if os.path.isfile(file_path):
                    try:
                        doc = self._load_single_file(file_path)
                        documents.append(doc)
                    except Exception as e:
                        import logging
                        logger = logging.getLogger(__name__)
                        logger.warning(f"Failed to load file {file_path}: {e}")
        else:
            raise ValueError(f"Path does not exist: {documents_path}")
        
        return documents
    
    def _load_single_file(self, file_path: str) -> Document:
        """Load a single file as a Document."""
        import os
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        metadata = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "file_size": os.path.getsize(file_path)
        }
        
        return Document(
            page_content=content,
            metadata=metadata
        )