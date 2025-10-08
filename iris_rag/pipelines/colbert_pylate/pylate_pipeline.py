"""
PyLate ColBERT Pipeline with Consistent Configuration

Simple PyLate-based ColBERT implementation that follows the same configuration
patterns as BasicRAGReranking for consistency across the evaluation framework.
"""

import logging
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

from ...core.models import Document
from ..basic import BasicRAGPipeline

logger = logging.getLogger(__name__)


class PyLateColBERTPipeline(BasicRAGPipeline):
    """
    PyLate-based ColBERT pipeline with native reranking.

    Maintains configuration consistency with BasicRAGReranking while using
    PyLate's native rank.rerank method for ColBERT-style late interaction.
    """

    def __init__(
        self,
        connection_manager,
        config_manager,
        **kwargs,
    ):
        """Initialize PyLate ColBERT pipeline with consistent configuration."""
        # Initialize parent pipeline
        super().__init__(connection_manager, config_manager, **kwargs)

        # Use same config pattern as BasicRAGReranking for consistency
        self.colbert_config = self.config_manager.get(
            "pipelines:colbert_pylate",
            self.config_manager.get("pipelines:basic_reranking", {}),
        )

        # Configuration parameters (consistent naming with BasicRAGReranking)
        self.rerank_factor = self.colbert_config.get("rerank_factor", 2)
        self.model_name = self.colbert_config.get(
            "model_name", "lightonai/GTE-ModernColBERT-v1"
        )
        self.batch_size = self.colbert_config.get("batch_size", 32)

        # PyLate-specific parameters
        self.use_native_reranking = self.colbert_config.get(
            "use_native_reranking", True
        )
        self.cache_embeddings = self.colbert_config.get("cache_embeddings", True)
        self.max_doc_length = self.colbert_config.get("max_doc_length", 4096)

        # Initialize components
        self.model = None
        self.is_initialized = False
        self._document_store = {}
        self._embedding_cache = {}
        self.index_folder = None

        # Statistics
        self.stats = {
            "queries_processed": 0,
            "documents_indexed": 0,
            "reranking_operations": 0,
        }

        self._initialize_components()

        logger.info(
            f"Initialized PyLateColBERT with rerank_factor={self.rerank_factor}, model={self.model_name}"
        )

    def _initialize_components(self):
        """Initialize PyLate components with graceful fallback if PyLate is unavailable."""
        if not self._import_pylate():
            self.is_initialized = False
            logger.warning(
                "PyLate not available; running in fallback mode without native reranking"
            )
            return
        self._setup_index_folder()
        self._initialize_model()
        self.is_initialized = True
        logger.info("PyLate ColBERT pipeline initialized successfully")

    def _import_pylate(self):
        """Import PyLate components; return True on success, else False."""
        try:
            global models, rank
            from pylate import models, rank

            logger.debug("PyLate library imported successfully")
            return True
        except Exception as e:
            logger.warning(f"PyLate import failed: {e}")
            self.use_native_reranking = False
            return False

    def _setup_index_folder(self):
        """Setup temporary index folder."""
        self.index_folder = tempfile.mkdtemp(prefix="pylate_index_")
        logger.debug(f"Created index folder: {self.index_folder}")

    def _initialize_model(self):
        """Initialize PyLate ColBERT model."""
        self.model = models.ColBERT(model_name_or_path=self.model_name)
        logger.info(f"PyLate model '{self.model_name}' loaded")

    def load_documents(self, documents=None, documents_path: str = None, **kwargs) -> Dict[str, Any]:
        """
        Load documents and prepare for retrieval.

        Args:
            documents: List of documents (dicts or Document objects) to load directly
            documents_path: Optional path to documents file or directory
            **kwargs: Additional arguments

        Returns:
            Dict with load status:
                - documents_loaded: Number of documents successfully loaded
                - embeddings_generated: Number of embeddings generated
                - documents_failed: Number of documents that failed to load
        """
        # Validation: require either documents or documents_path
        if documents is None and documents_path is None:
            raise ValueError(
                "Error: Missing required input\n"
                "Context: PyLateColBERT document loading\n"
                "Expected: Either 'documents' list or 'documents_path' string\n"
                "Actual: Both are None\n"
                "Fix: Provide documents=[...] or documents_path='path/to/docs.json'"
            )

        # Validation: empty documents list
        if documents is not None and isinstance(documents, list) and len(documents) == 0:
            raise ValueError(
                "Error: Empty documents list\n"
                "Context: PyLateColBERT document loading\n"
                "Expected: Non-empty list of documents\n"
                "Actual: Empty list []\n"
                "Fix: Provide at least one document in the list"
            )

        # Handle both file paths and Document objects
        if documents_path is not None:
            # File path - delegate to parent
            result = super().load_documents(documents_path=documents_path, **kwargs)
            if result and "documents" in result:
                docs = result["documents"]
                # Store documents for PyLate reranking
                for i, doc in enumerate(docs):
                    self._document_store[str(i)] = doc
                self.stats["documents_indexed"] = len(docs)
            return result
        else:
            # List of Document objects
            # Store documents for PyLate reranking with metadata
            for i, doc in enumerate(documents):
                self._document_store[str(i)] = doc
                # Validate metadata exists
                if not hasattr(doc, "metadata") or not doc.metadata:
                    logger.warning(f"Document {i} has no metadata")

            # Call parent to handle vector store indexing
            result = super().load_documents(documents=documents, **kwargs)
            self.stats["documents_indexed"] = result.get("documents_loaded", len(documents))

            logger.info(
                f"Loaded {self.stats['documents_indexed']} documents for PyLate ColBERT"
            )
            logger.debug(
                f"Stored {len(self._document_store)} docs with metadata in document store"
            )

            return result

    def query(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Execute ColBERT query with PyLate native reranking.

        Follows the same pattern as BasicRAGReranking:
        1. Retrieve initial candidates (rerank_factor * top_k)
        2. Apply PyLate native reranking
        3. Return top_k documents with consistent response format

        Args:
            query: The query text
            top_k: Number of documents to return after reranking (must be between 1 and 100)
            **kwargs: Additional arguments

        Returns:
            Dictionary with complete RAG response including reranked documents
        """
        # Validate query
        if not query or query.strip() == "":
            raise ValueError(
                "Error: Query parameter is required and cannot be empty\n"
                "Context: PyLateColBERT pipeline query operation\n"
                "Expected: Non-empty query string\n"
                "Actual: Empty or whitespace-only string\n"
                "Fix: Provide a valid query string, e.g., query='What is diabetes?'"
            )

        # Validate top_k
        if top_k < 1 or top_k > 100:
            raise ValueError(
                f"Error: top_k parameter out of valid range\n"
                f"Context: PyLateColBERT pipeline query operation\n"
                f"Expected: Integer between 1 and 100 (inclusive)\n"
                f"Actual: {top_k}\n"
                f"Fix: Set top_k to a value between 1 and 100, e.g., top_k=5"
            )

        # Calculate initial retrieval size
        initial_k = min(top_k * self.rerank_factor, 100)

        # Get initial candidates using parent pipeline
        parent_kwargs = kwargs.copy()
        parent_kwargs["generate_answer"] = False  # Generate after reranking

        parent_result = super().query(query, top_k=initial_k, **parent_kwargs)
        candidate_documents = parent_result.get("retrieved_documents", [])

        # Apply PyLate native reranking if available and beneficial
        if (
            len(candidate_documents) > 1
            and self.use_native_reranking
            and self.is_initialized
        ):
            final_documents = self._pylate_rerank(
                query, candidate_documents, top_k
            )
            reranked = True
            self.stats["reranking_operations"] += 1
            logger.debug(
                f"PyLate reranked {len(candidate_documents)} → {len(final_documents)} documents"
            )
        else:
            final_documents = candidate_documents[:top_k]
            reranked = False
            logger.debug(
                f"No reranking applied, returning {len(final_documents)} documents"
            )

        # Restore metadata from document store
        final_documents = self._restore_metadata(final_documents)

        # Debug: Verify metadata restoration
        for i, doc in enumerate(final_documents):
            if not hasattr(doc, "metadata") or not doc.metadata:
                logger.warning(f"Document {i} missing metadata after restoration")

        logger.debug(
            f"Query returned {len(final_documents)} docs with restored metadata"
        )

        # Generate answer if requested (same as BasicRAGReranking)
        generate_answer = kwargs.get("generate_answer", True)
        if generate_answer and self.llm_func and final_documents:
            try:
                custom_prompt = kwargs.get("custom_prompt")
                answer = self._generate_answer(
                    query, final_documents, custom_prompt
                )
            except Exception as e:
                logger.warning(f"Answer generation failed: {e}")
                answer = "Error generating answer"
        elif not generate_answer:
            answer = None
        elif not final_documents:
            answer = "No relevant documents found to answer the query."
        else:
            answer = "No LLM function provided. Retrieved documents only."

        # Build response with consistent format
        contexts_list = [doc.page_content for doc in final_documents]
        sources = self._extract_sources(final_documents) if kwargs.get("include_sources", True) else []
        retrieval_method = kwargs.get("method", "colbert_pylate")

        response = {
            "query": query,
            "answer": answer,
            "retrieved_documents": final_documents,
            "contexts": contexts_list,
            "execution_time": parent_result.get("execution_time", 0.0),
            "metadata": {
                "num_retrieved": len(final_documents),
                "processing_time": parent_result.get("execution_time", 0.0),
                "pipeline_type": "colbert_pylate",
                "reranked": reranked,
                "initial_candidates": len(candidate_documents),
                "rerank_factor": self.rerank_factor,
                "generated_answer": generate_answer and answer is not None,
                "model_name": self.model_name,
                "native_reranking": self.use_native_reranking,
                "retrieval_method": retrieval_method,  # FR-003: Include retrieval method
                "context_count": len(contexts_list),  # FR-003: Include context count
                "sources": sources,  # FR-003: Include sources in metadata
            },
        }

        # Add sources to top level if requested
        if kwargs.get("include_sources", True):
            response["sources"] = sources

        self.stats["queries_processed"] += 1
        logger.info(
            f"PyLate ColBERT query completed - {len(final_documents)} docs returned (reranked: {reranked})"
        )
        return response

    def _pylate_rerank(
        self, query_text: str, documents: List[Document], top_k: int
    ) -> List[Document]:
        """Apply PyLate native reranking using rank.rerank method."""
        # Prepare documents for PyLate
        doc_texts = [doc.page_content for doc in documents]
        doc_ids = list(range(len(documents)))

        # Generate embeddings
        query_embeddings = self.model.encode([query_text], is_query=True)
        doc_embeddings = self.model.encode(doc_texts, is_query=False)

        # Apply PyLate native reranking
        reranked_results = rank.rerank(
            documents_ids=[doc_ids],  # Nested list as required by PyLate
            queries_embeddings=query_embeddings,
            documents_embeddings=doc_embeddings,
        )

        # Extract reranked document order (first query results)
        if reranked_results and len(reranked_results) > 0:
            reranked_ids = reranked_results[0][:top_k]  # Top k from first query
            return [documents[doc_id] for doc_id in reranked_ids]
        else:
            raise RuntimeError("PyLate reranking returned empty results")

    def _restore_metadata(self, retrieved_docs: List[Document]) -> List[Document]:
        """
        Restore metadata from document store to retrieved documents.

        Matches retrieved documents to stored documents by page_content
        and re-attaches complete metadata from the original documents.
        """
        restored_docs = []
        for doc in retrieved_docs:
            # Find matching document in store by content
            metadata_restored = False
            for _stored_id, stored_doc in self._document_store.items():
                if stored_doc.page_content == doc.page_content:
                    # Create new Document with restored metadata (Document is frozen)
                    restored_doc = Document(
                        page_content=doc.page_content,
                        metadata=stored_doc.metadata.copy(),
                    )
                    restored_docs.append(restored_doc)
                    metadata_restored = True
                    break

            # If no match found, keep original document
            if not metadata_restored:
                restored_docs.append(doc)

        logger.debug(f"Restored metadata for {len(restored_docs)} documents")
        return restored_docs

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get pipeline information with consistent format."""
        info = (
            super().get_pipeline_info() if hasattr(super(), "get_pipeline_info") else {}
        )

        info.update(
            {
                "pipeline_type": "colbert_pylate",
                "rerank_factor": self.rerank_factor,
                "model_name": self.model_name,
                "use_native_reranking": self.use_native_reranking,
                "batch_size": self.batch_size,
                "is_initialized": self.is_initialized,
                "stats": self.stats.copy(),
            }
        )

        return info
