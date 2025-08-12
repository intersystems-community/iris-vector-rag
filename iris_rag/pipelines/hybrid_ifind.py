"""
Hybrid IFind RAG Pipeline - Single Unified Implementation.

This module provides the single, unified RAG implementation that combines:
1. Vector similarity search for semantic matching
2. IRIS IFind keyword search for precise text matching
3. Reciprocal Rank Fusion (RRF) for optimal result combination

This is the only active Hybrid IFind implementation in the codebase.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable
from ..core.base import RAGPipeline
from ..core.models import Document
from ..core.connection import ConnectionManager
from ..config.manager import ConfigurationManager
from ..storage.enterprise_storage import IRISStorage
from ..embeddings.manager import EmbeddingManager

logger = logging.getLogger(__name__)


class HybridIFindRAGPipeline(RAGPipeline):
    """
    Hybrid IFind RAG Pipeline - Single Unified Implementation.

    This is the only active Hybrid IFind implementation in the codebase.
    It provides a unified pipeline that combines:
    1. Vector similarity search for semantic matching
    2. IRIS IFind text search for precise keyword matching
    3. Reciprocal Rank Fusion (RRF) for optimal result combination

    The pipeline integrates these search methods seamlessly to provide
    superior retrieval performance compared to individual approaches.
    """

    def __init__(
        self,
        connection_manager: ConnectionManager,
        config_manager: ConfigurationManager,
        vector_store=None,
        llm_func: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the Hybrid IFind RAG Pipeline.

        Args:
            connection_manager: Manager for database connections
            config_manager: Manager for configuration settings
            vector_store: Optional VectorStore instance
            llm_func: Optional LLM function for answer generation
        """
        super().__init__(connection_manager, config_manager, vector_store)
        self.llm_func = llm_func

        # Initialize components
        self.storage = IRISStorage(connection_manager, config_manager)
        self.embedding_manager = EmbeddingManager(config_manager)

        # Initialize vector store if not provided (like BasicRAG does)
        if self.vector_store is None:
            from ..storage.vector_store_iris import IRISVectorStore

            self.vector_store = IRISVectorStore(connection_manager, config_manager)

        # Get pipeline configuration
        self.pipeline_config = self.config_manager.get("pipelines:hybrid_ifind", {})
        self.top_k = self.pipeline_config.get("top_k", 5)
        self.vector_weight = self.pipeline_config.get("vector_weight", 0.6)
        self.ifind_weight = self.pipeline_config.get("ifind_weight", 0.4)
        self.min_ifind_score = self.pipeline_config.get("min_ifind_score", 0.1)

        # Set table name for LIKE search fallback
        self.table_name = "RAG.SourceDocumentsIFind"

        logger.info(f"Initialized HybridIFindRAGPipeline with vector_weight={self.vector_weight}")

    def load_documents(self, documents_path: str, **kwargs) -> None:
        """
        Load documents into the knowledge base (required abstract method).

        Args:
            documents_path: Path to documents or directory
            **kwargs: Additional keyword arguments including:
                - documents: List of Document objects (if providing directly)
                - chunk_documents: Whether to chunk documents (default: True)
                - generate_embeddings: Whether to generate embeddings (default: True)
        """
        # Handle direct document input
        if "documents" in kwargs:
            documents = kwargs["documents"]
            if not isinstance(documents, list):
                raise ValueError("Documents must be provided as a list")
        else:
            # Load documents from path - basic implementation
            import os

            documents = []

            if os.path.isfile(documents_path):
                with open(documents_path, "r", encoding="utf-8") as f:
                    content = f.read()
                doc = Document(page_content=content, metadata={"source": documents_path})
                documents.append(doc)
            elif os.path.isdir(documents_path):
                for filename in os.listdir(documents_path):
                    file_path = os.path.join(documents_path, filename)
                    if os.path.isfile(file_path):
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                            doc = Document(page_content=content, metadata={"source": file_path, "filename": filename})
                            documents.append(doc)
                        except Exception as e:
                            logger.warning(f"Failed to load file {file_path}: {e}")

        # Use the ingest_documents method
        result = self.ingest_documents(documents)
        logger.info(f"Hybrid IFind: Loaded {len(documents)} documents - {result}")

    def ingest_documents(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Ingest documents with IFind indexing support.

        Args:
            documents: List of documents to ingest

        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        logger.info(f"Starting Hybrid IFind ingestion of {len(documents)} documents")

        try:
            # Store documents (this will also create IFind indexes if configured)
            result = self.storage.store_documents(documents)

            # Ensure IFind indexes are built
            self._ensure_ifind_indexes()

            end_time = time.time()
            result.update({"processing_time": end_time - start_time, "pipeline_type": "hybrid_ifind_rag"})

            logger.info(f"Hybrid IFind ingestion completed: {result}")
            return result

        except Exception as e:
            logger.error(f"Hybrid IFind ingestion failed: {e}")
            return {"status": "error", "error": str(e), "pipeline_type": "hybrid_ifind_rag"}

    def query(self, query_text: str, top_k: int = 5, generate_answer: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Execute a query using hybrid vector + IFind search with standardized response format.

        Args:
            query_text: The query string
            top_k: Number of top documents to retrieve
            generate_answer: Whether to generate an answer (default: True)
            **kwargs: Additional keyword arguments

        Returns:
            Standardized dictionary with query, retrieved_documents, contexts, metadata, answer, execution_time
        """
        start_time = time.time()
        logger.info(f"Processing Hybrid IFind query: {query_text}")

        try:
            # Perform vector search using vector store (like BasicRAG)
            vector_results = self._vector_search(query_text, top_k)

            # Perform IFind search
            ifind_results = self._ifind_search(query_text, top_k)

            # Fuse results using reciprocal rank fusion
            fused_results = self._fuse_results(vector_results, ifind_results, top_k)

            # Convert to Document objects
            retrieved_documents = []
            for result in fused_results:
                doc = Document(
                    id=result["doc_id"],
                    page_content=result["content"],
                    metadata={
                        "title": result.get("title", ""),
                        "search_type": result.get("search_type", "hybrid"),
                        "vector_score": result.get("vector_score", 0.0),
                        "ifind_score": result.get("ifind_score", 0.0),
                        "hybrid_score": result.get("hybrid_score", 0.0),
                        "has_vector": result.get("has_vector", False),
                        "has_ifind": result.get("has_ifind", False),
                    },
                )
                retrieved_documents.append(doc)

            # Generate answer if requested and LLM function is available
            answer = None
            if generate_answer and self.llm_func and retrieved_documents:
                context = self._build_context_from_documents(retrieved_documents)
                prompt = self._build_prompt(query_text, context)
                answer = self.llm_func(prompt)
            elif generate_answer and not self.llm_func:
                answer = "No LLM function available for answer generation."

            execution_time = time.time() - start_time

            # Return standardized response format
            result = {
                "query": query_text,
                "answer": answer,
                "retrieved_documents": retrieved_documents,
                "contexts": [doc.page_content for doc in retrieved_documents],
                "execution_time": execution_time,
                "metadata": {
                    "num_retrieved": len(retrieved_documents),
                    "pipeline_type": "hybrid_ifind",
                    "generated_answer": generate_answer and answer is not None,
                    "vector_results_count": len(vector_results),
                    "ifind_results_count": len(ifind_results),
                    "vector_weight": self.vector_weight,
                    "ifind_weight": self.ifind_weight,
                },
            }

            logger.info(f"Hybrid IFind query completed in {execution_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Hybrid IFind query failed: {e}")
            return {
                "query": query_text,
                "answer": None,
                "retrieved_documents": [],
                "contexts": [],
                "execution_time": 0.0,
                "metadata": {
                    "num_retrieved": 0,
                    "pipeline_type": "hybrid_ifind",
                    "generated_answer": False,
                    "error": str(e),
                },
            }

    def _ensure_ifind_indexes(self):
        """Ensure IFind indexes are created for text search."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Check if IFind index exists
            check_sql = """
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.INDEXES 
            WHERE TABLE_NAME = 'SourceDocuments' 
            AND TABLE_SCHEMA = 'RAG' 
            AND INDEX_NAME LIKE '%IFIND%'
            """

            cursor.execute(check_sql)
            result = cursor.fetchone()

            if result[0] == 0:
                logger.info("Creating IFind indexes for text search")

                # Create IFind index on content
                create_index_sql = """
                CREATE INDEX IF NOT EXISTS idx_sourcedocs_ifind_content
                ON RAG.SourceDocuments (text_content)
                WITH (TYPE = 'IFIND')
                """

                cursor.execute(create_index_sql)
                connection.commit()
                logger.info("IFind index created successfully")

        except Exception as e:
            logger.error(
                f"HybridIFind: Could not create IFind index - {e}. HybridIFind requires working IFind functionality."
            )
            # FAIL instead of silent fallback
            raise RuntimeError(
                f"HybridIFind pipeline failed: Cannot create IFind indexes. Please use BasicRAG or fix IFind configuration. Error: {e}"
            )
        finally:
            cursor.close()

    def _vector_search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform vector similarity search using vector store like BasicRAG."""
        try:
            # Use vector store for retrieval (same as BasicRAG)
            if hasattr(self, "vector_store") and self.vector_store:
                # Use similarity_search_with_score to get both documents and scores
                if hasattr(self.vector_store, "similarity_search_with_score"):
                    retrieved_documents_with_scores = self.vector_store.similarity_search_with_score(
                        query_text, k=top_k
                    )

                    documents = []
                    for doc, score in retrieved_documents_with_scores:
                        documents.append(
                            {
                                "doc_id": getattr(doc, "id", f"doc_{len(documents)}"),
                                "title": doc.metadata.get("title", ""),
                                "content": doc.page_content,
                                "vector_score": float(score),
                                "search_type": "vector",
                            }
                        )

                    return documents
                else:
                    # Fallback to regular similarity search
                    retrieved_documents = self.vector_store.similarity_search(query_text, k=top_k)

                    documents = []
                    for i, doc in enumerate(retrieved_documents):
                        documents.append(
                            {
                                "doc_id": getattr(doc, "id", f"doc_{i}"),
                                "title": doc.metadata.get("title", ""),
                                "content": doc.page_content,
                                "vector_score": doc.metadata.get("score", 0.8 - (i * 0.1)),  # Mock decreasing scores
                                "search_type": "vector",
                            }
                        )

                    return documents
            else:
                logger.warning("No vector store available for vector search")
                return []

        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

    def _ifind_search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform IFind text search."""
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Try IFind search first using proper IRIS IFind syntax
            # Match vector search structure: doc_id, text_content, score
            ifind_sql = f"""
            SELECT TOP {top_k}
                doc_id, text_content,
                $SCORE(text_content) as ifind_score
            FROM RAG.SourceDocumentsIFind
            WHERE $FIND(text_content, ?)
            ORDER BY $SCORE(text_content) DESC
            """

            try:
                cursor.execute(ifind_sql, [query_text])
                results = cursor.fetchall()

                documents = []
                for row in results:
                    ifind_score = float(row[2]) if row[2] is not None else 0.0  # Score is now row[2]
                    if ifind_score >= self.min_ifind_score:
                        documents.append(
                            {
                                "doc_id": row[0],
                                "title": "",  # No title in this structure
                                "content": row[1],  # text_content is now row[1]
                                "ifind_score": ifind_score,
                                "search_type": "ifind",
                            }
                        )

                return documents

            except Exception as ifind_error:
                # Handle empty error messages from mocks
                error_msg = str(ifind_error) if str(ifind_error).strip() else "IFind query execution failed"
                logger.warning(f"HybridIFind: IFind search failed - {error_msg}. Falling back to LIKE search.")

                # Fallback to LIKE search
                # Match vector search structure: doc_id, text_content, score
                try:
                    like_sql = f"""
                    SELECT TOP {top_k}
                        doc_id, text_content, 1.0 as like_score
                    FROM {self.table_name}
                    WHERE text_content LIKE ?
                    ORDER BY LENGTH(text_content) ASC
                    """

                    like_params = [f"%{query_text}%"]
                    cursor.execute(like_sql, like_params)
                    results = cursor.fetchall()

                    logger.debug(f"LIKE search returned {len(results)} results")

                    documents = []
                    for row in results:
                        documents.append(
                            {
                                "doc_id": row[0],
                                "title": "",  # No title in this structure
                                "content": row[1],  # text_content is now row[1]
                                "ifind_score": 1.0,  # LIKE search gives uniform score
                                "search_type": "text_fallback",
                            }
                        )

                    return documents

                except Exception as like_error:
                    # Handle empty error messages from mocks
                    like_error_msg = str(like_error) if str(like_error).strip() else "LIKE query execution failed"
                    logger.error(f"HybridIFind: Both IFind and LIKE search failed - {like_error_msg}")
                    # Return empty results rather than crashing
                    return []

        finally:
            cursor.close()

    def _normalize_scores(self, results: List[Dict[str, Any]], score_field: str) -> List[Dict[str, Any]]:
        """Normalize scores to 0-1 range."""
        if not results:
            return results

        scores = [doc[score_field] for doc in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            # All scores are the same
            for doc in results:
                doc[score_field] = 1.0
        else:
            # Normalize to 0-1
            for doc in results:
                doc[score_field] = (doc[score_field] - min_score) / (max_score - min_score)

        return results

    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents (legacy method)."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get("title", "Untitled")
            content = doc.get("content", "")
            score = doc.get("hybrid_score", 0.0)
            context_parts.append(f"[Document {i}: {title} (Score: {score:.3f})]\n{content}")

        return "\n\n".join(context_parts)

    def _build_context_from_documents(self, documents: List[Document]) -> str:
        """Build context string from Document objects."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get("title", "Untitled")
            content = doc.page_content
            score = doc.metadata.get("hybrid_score", 0.0)
            context_parts.append(f"[Document {i}: {title} (Score: {score:.3f})]\n{content}")

        return "\n\n".join(context_parts)

    def _fuse_results(
        self, vector_results: List[Dict[str, Any]], ifind_results: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Fuse vector and IFind results using reciprocal rank fusion."""
        # Normalize scores before fusion
        vector_results = self._normalize_scores(vector_results, "vector_score")
        ifind_results = self._normalize_scores(ifind_results, "ifind_score")

        # Create a dictionary to combine results by doc_id
        doc_scores = {}

        # Add vector results with rank-based scoring
        for rank, result in enumerate(vector_results):
            doc_id = result["doc_id"]
            vector_rank_score = 1.0 / (rank + 1)  # Reciprocal rank fusion
            doc_scores[doc_id] = {
                "doc_id": doc_id,
                "title": result.get("title", ""),
                "content": result["content"],
                "vector_score": result.get("vector_score", 0.0),
                "ifind_score": 0.0,
                "vector_rank_score": vector_rank_score,
                "ifind_rank_score": 0.0,
                "search_type": "vector",
                "has_vector": True,
                "has_ifind": False,
            }

        # Add IFind results with rank-based scoring
        for rank, result in enumerate(ifind_results):
            doc_id = result["doc_id"]
            ifind_rank_score = 1.0 / (rank + 1)  # Reciprocal rank fusion

            if doc_id in doc_scores:
                # Document found in both searches - combine scores
                doc_scores[doc_id]["ifind_score"] = result.get("ifind_score", 0.0)
                doc_scores[doc_id]["ifind_rank_score"] = ifind_rank_score
                doc_scores[doc_id]["search_type"] = "hybrid"
                doc_scores[doc_id]["has_ifind"] = True
            else:
                # Document only found in IFind search - preserve original search_type
                doc_scores[doc_id] = {
                    "doc_id": doc_id,
                    "title": result.get("title", ""),
                    "content": result["content"],
                    "vector_score": 0.0,
                    "ifind_score": result.get("ifind_score", 0.0),
                    "vector_rank_score": 0.0,
                    "ifind_rank_score": ifind_rank_score,
                    "search_type": result.get("search_type", "text_search"),  # Preserve original search_type
                    "has_vector": False,
                    "has_ifind": True,
                }

        # Calculate hybrid scores and sort
        for doc_id, doc_data in doc_scores.items():
            # Combine rank scores with weights
            hybrid_score = (
                self.vector_weight * doc_data["vector_rank_score"] + self.ifind_weight * doc_data["ifind_rank_score"]
            )
            doc_data["hybrid_score"] = hybrid_score

        # Sort by hybrid score and return top_k
        sorted_results = sorted(doc_scores.values(), key=lambda x: x["hybrid_score"], reverse=True)
        return sorted_results[:top_k]

    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM generation."""
        return f"""Based on the following retrieved documents (ranked by hybrid vector + text search), please answer the question.

Context:
{context}

Question: {query}

Answer:"""
