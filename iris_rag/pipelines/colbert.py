"""
ColBERT RAG Pipeline implementation for iris_rag package.

This pipeline implements ColBERT (Contextualized Late Interaction over BERT) approach:
1. Token-level embeddings for both queries and documents
2. MaxSim operation for fine-grained matching
3. Late interaction between query and document tokens
"""

import logging
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from ..core.base import RAGPipeline
from ..core.models import Document
from ..core.connection import ConnectionManager
from ..config.manager import ConfigurationManager

logger = logging.getLogger(__name__)


class ColBERTRAGPipeline(RAGPipeline):
    """
    ColBERT RAG pipeline implementation using iris_rag architecture.

    This pipeline uses token-level embeddings and MaxSim operations for
    fine-grained query-document matching.
    """

    def __init__(
        self,
        connection_manager: Optional[ConnectionManager] = None,
        config_manager: Optional[ConfigurationManager] = None,
        colbert_query_encoder: Optional[Callable[[str], List[List[float]]]] = None,
        llm_func: Optional[Callable[[str], str]] = None,
        embedding_func: Optional[Callable] = None,
        vector_store=None,
    ):
        """
        Initialize ColBERT RAG pipeline.

        Args:
            connection_manager: Database connection manager (optional, will create default if None)
            config_manager: Configuration manager (optional, will create default if None)
            colbert_query_encoder: Function to encode queries into token embeddings
            llm_func: Function for answer generation
            embedding_func: Function for document-level embeddings (used for candidate retrieval)
            vector_store: Optional VectorStore instance
        """
        # Handle None arguments by creating default instances
        if connection_manager is None:
            from ..core.connection import ConnectionManager

            connection_manager = ConnectionManager()

        if config_manager is None:
            from ..config.manager import ConfigurationManager

            config_manager = ConfigurationManager()

        super().__init__(connection_manager, config_manager, vector_store)

        # Initialize schema manager for dimension management
        from ..storage.schema_manager import SchemaManager

        self.schema_manager = SchemaManager(connection_manager, config_manager)

        # Get dimensions from schema manager
        self.doc_embedding_dim = self.schema_manager.get_vector_dimension("SourceDocuments")
        self.token_embedding_dim = self.schema_manager.get_vector_dimension("DocumentTokenEmbeddings")

        logger.info(
            f"ColBERT: Document embeddings = {self.doc_embedding_dim}D, Token embeddings = {self.token_embedding_dim}D"
        )

        # Initialize embedding manager for compatibility with tests
        from ..embeddings.manager import EmbeddingManager

        self.embedding_manager = EmbeddingManager(config_manager)

        # Store embedding functions with proper naming
        self.doc_embedding_func = embedding_func  # 384D for document-level retrieval
        self.colbert_query_encoder = colbert_query_encoder  # 768D for token-level scoring
        self.llm_func = llm_func

        # Get ColBERT interface from config if not provided
        if not self.colbert_query_encoder:
            from ..embeddings.colbert_interface import get_colbert_interface_from_config

            self.colbert_interface = get_colbert_interface_from_config(config_manager, connection_manager)
            # Wrap interface methods for backwards compatibility
            self.colbert_query_encoder = self.colbert_interface.encode_query
        else:
            # If custom encoder provided, use RAG Templates interface for other operations
            from ..embeddings.colbert_interface import RAGTemplatesColBERTInterface

            self.colbert_interface = RAGTemplatesColBERTInterface(self.token_embedding_dim)

        if not self.llm_func:
            from common.utils import get_llm_func

            self.llm_func = get_llm_func()

        if not self.doc_embedding_func:
            from common.utils import get_embedding_func

            self.doc_embedding_func = get_embedding_func()

        # Validate dimensions match expectations
        self._validate_embedding_dimensions()

        logger.info("ColBERTRAGPipeline initialized with proper dimension handling")

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Simple tokenization method for compatibility with tests.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        # Simple whitespace tokenization for test compatibility
        return text.lower().split()

    def _validate_embedding_dimensions(self):
        """
        Validate that embedding functions produce the expected dimensions.
        """
        try:
            # Test document embedding function
            if self.doc_embedding_func:
                test_doc_embedding = self.doc_embedding_func("test")
                if len(test_doc_embedding) != self.doc_embedding_dim:
                    logger.warning(
                        f"ColBERT: Document embedding function produces {len(test_doc_embedding)}D, expected {self.doc_embedding_dim}D"
                    )

            # Test ColBERT query encoder
            if self.colbert_query_encoder:
                test_token_embeddings = self.colbert_query_encoder("test")
                if test_token_embeddings and len(test_token_embeddings[0]) != self.token_embedding_dim:
                    logger.warning(
                        f"ColBERT: Token embedding function produces {len(test_token_embeddings[0])}D, expected {self.token_embedding_dim}D"
                    )

        except Exception as e:
            logger.warning(f"ColBERT: Could not validate embedding dimensions: {e}")

    def _format_vector_for_sql(self, vector: List[float]) -> str:
        """Formats a vector list into a comma-separated string for IRIS SQL."""
        if not vector:
            # Return a string representation of an empty list,
            # or handle as an error if empty vectors are not expected.
            # For TO_VECTOR, an empty list might be valid for an empty vector.
            return "[]"
        return "[" + ",".join(f"{x:.15g}" for x in vector) + "]"

    def validate_setup(self) -> bool:
        """
        Validate that ColBERT pipeline is properly set up with token embeddings.

        Returns:
            bool: True if setup is valid, False otherwise
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Check if DocumentTokenEmbeddings table exists
            check_table_sql = """
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentTokenEmbeddings'
            """
            cursor.execute(check_table_sql)
            table_exists = cursor.fetchone()[0] > 0

            if not table_exists:
                logger.error("ColBERT validation failed: DocumentTokenEmbeddings table does not exist")
                return False

            # Check if we have token embeddings
            check_tokens_sql = """
                SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings
            """
            cursor.execute(check_tokens_sql)
            token_count = cursor.fetchone()[0]

            if token_count == 0:
                logger.error("ColBERT validation failed: No token embeddings found in database")
                return False

            logger.info(f"ColBERT validation passed: Found {token_count} token embeddings")
            return True

        except Exception as e:
            logger.error(f"ColBERT validation failed with error: {e}")
            return False
        finally:
            cursor.close()

    def execute(self, query_text: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the ColBERT RAG pipeline (required abstract method).

        Args:
            query_text: The input query string
            **kwargs: Additional parameters including top_k

        Returns:
            Dictionary containing query, answer, and retrieved documents
        """
        top_k = kwargs.get("top_k", 5)
        return self.run(query_text, top_k, **kwargs)

    def run(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Execute the ColBERT RAG pipeline.

        Args:
            query: The input query string
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters

        Returns:
            Dictionary containing query, answer, and retrieved documents
        """
        logger.info(f"ColBERT: Processing query: '{query[:50]}...'")

        # Validate setup before proceeding
        if not self.validate_setup():
            logger.warning("ColBERT setup validation failed - pipeline may not work correctly")

        start_time = self._get_current_time()

        try:
            # Stage 1: Generate query token embeddings
            query_token_embeddings = self.colbert_query_encoder(query)
            logger.debug(f"ColBERT: Generated {len(query_token_embeddings)} query token embeddings")

            # Validate that we have token embeddings
            if not query_token_embeddings:
                raise ValueError("ColBERT query encoder returned empty token embeddings")

            # Stage 2: Use IRISVectorStore for ColBERT search (with query_text for proper doc-level embedding)
            search_results = self.vector_store.colbert_search(
                query_token_embeddings=query_token_embeddings,
                k=top_k,
                query_text=query,  # Pass query text for proper 384D document embedding generation
            )

            # Convert results to Document list for compatibility
            retrieved_docs = [doc for doc, score in search_results]

            # Stage 3: Generate answer using LLM
            answer = self._generate_answer(query, retrieved_docs)

            execution_time = self._get_current_time() - start_time

            result = {
                "query": query,
                "answer": answer,
                "retrieved_documents": retrieved_docs,
                "execution_time": execution_time,
                "technique": "ColBERT",
                "token_count": len(query_token_embeddings),
            }

            logger.info(f"ColBERT: Completed in {execution_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"ColBERT pipeline failed: {e}")
            raise

    def _retrieve_candidate_documents_hnsw(self, query_text: str, k: int = 30) -> List[int]:
        """
        Stage 1: Retrieve candidate documents using document-level HNSW vector search.

        This method performs a document-level HNSW search on RAG.SourceDocuments using
        the average of query token embeddings as the query vector.

        Args:
            query_text: The input query text
            k: Number of candidate documents to retrieve

        Returns:
            List of candidate document IDs
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # For HNSW stage against RAG.SourceDocuments, use the standard document embedding function (384D)
            if not self.doc_embedding_func:
                logger.error("ColBERT HNSW: Document embedding_func is not initialized.")
                return []

            doc_level_query_embedding = self.doc_embedding_func(query_text)  # This should be 384D

            if (
                not doc_level_query_embedding
                or not isinstance(doc_level_query_embedding, list)
                or not all(isinstance(x, float) for x in doc_level_query_embedding)
            ):
                logger.error(
                    f"ColBERT HNSW: Failed to generate valid List[float] document-level query embedding. Type: {type(doc_level_query_embedding)}"
                )
                return []

            query_vector_dim = len(doc_level_query_embedding)
            # Validate dimension matches expected document embedding dimension
            if query_vector_dim != self.doc_embedding_dim:
                logger.error(
                    f"ColBERT HNSW: Document-level query embedding dimension is {query_vector_dim}, expected {self.doc_embedding_dim}."
                )
                return []

            query_vector_str = self._format_vector_for_sql(doc_level_query_embedding)

            logger.info(
                f"ColBERT HNSW: Searching for candidates with query '{query_text[:50]}...' using {query_vector_dim}D embedding."
            )

            # First, verify we have medical documents in SourceDocuments
            count_sql = """
                SELECT COUNT(*) as total_docs,
                       COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as docs_with_embeddings
                FROM RAG.SourceDocuments
            """
            cursor.execute(count_sql)
            count_result = cursor.fetchone()
            total_docs = count_result[0] if count_result else 0
            docs_with_embeddings = count_result[1] if count_result else 0

            logger.info(f"ColBERT HNSW: Found {total_docs} total documents, {docs_with_embeddings} with embeddings")

            if docs_with_embeddings == 0:
                logger.warning("ColBERT HNSW: No documents with embeddings found in RAG.SourceDocuments")
                return []

            # Perform HNSW search on SourceDocuments with enhanced logging
            # Use vector_sql_utils to construct the query properly
            from common.vector_sql_utils import format_vector_search_sql, execute_vector_search

            # Remove this redundant line - use the already formatted query_vector_str from above

            sql = format_vector_search_sql(
                table_name="RAG.SourceDocuments",
                vector_column="embedding",
                vector_string=query_vector_str,
                embedding_dim=query_vector_dim,
                top_k=k,
                id_column="doc_id",
                content_column="text_content",
            )

            # Use execute_vector_search utility
            results = execute_vector_search(cursor, sql)

            # Extract document IDs and log content for debugging
            candidate_doc_ids = []
            for i, row in enumerate(results):
                doc_id = row[0]
                title = row[1] if len(row) > 1 else "No title"
                content_preview = row[2] if len(row) > 2 else "No preview"
                score = row[3] if len(row) > 3 else 0.0

                candidate_doc_ids.append(doc_id)

                # Log first few candidates for debugging
                if i < 3:
                    logger.info(f"ColBERT HNSW: Candidate {i+1} - Doc ID: {doc_id}, Score: {float(score):.4f}")
                    logger.info(f"ColBERT HNSW: Content preview: {content_preview}...")

            logger.info(f"ColBERT HNSW: Retrieved {len(candidate_doc_ids)} candidate documents")

            # Additional validation: check if candidates are medical-related
            if candidate_doc_ids:
                self._validate_candidate_relevance(candidate_doc_ids[:5], query_text)

            return candidate_doc_ids

        except Exception as e:
            logger.error(f"ColBERT HNSW: Error in candidate document retrieval: {e}")
            return []
        finally:
            cursor.close()

    def _validate_candidate_relevance(self, candidate_doc_ids: List[int], query_text: str) -> None:
        """
        Validate that candidate documents are relevant to the query.
        This helps debug cases where irrelevant documents (e.g., forestry papers for medical queries) are retrieved.

        Args:
            candidate_doc_ids: List of candidate document IDs to validate
            query_text: Original query text for context
        """
        if not candidate_doc_ids:
            return

        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            placeholders = ", ".join(["?"] * len(candidate_doc_ids))
            validation_sql = f"""
                SELECT doc_id, title, SUBSTRING(text_content, 1, 200) AS content_sample
                FROM RAG.SourceDocuments
                WHERE doc_id IN ({placeholders})
            """

            cursor.execute(validation_sql, candidate_doc_ids)
            results = cursor.fetchall()

            logger.info(f"ColBERT HNSW: Validating {len(results)} candidates for query: '{query_text[:50]}...'")

            for i, row in enumerate(results):
                doc_id = row[0]
                title = row[1] if row[1] else "No title"
                content_sample = row[2] if row[2] else "No content"

                logger.info(f"ColBERT HNSW: Candidate {i+1} - ID: {doc_id}")
                logger.info(f"ColBERT HNSW: Title: {title}")
                logger.info(f"ColBERT HNSW: Content: {content_sample}...")

                # Basic relevance check - look for medical terms if query seems medical
                query_lower = query_text.lower()
                content_lower = (title + " " + content_sample).lower()

                medical_terms = [
                    "medical",
                    "health",
                    "disease",
                    "treatment",
                    "patient",
                    "clinical",
                    "therapy",
                    "diagnosis",
                ]
                query_has_medical = any(term in query_lower for term in medical_terms)
                content_has_medical = any(term in content_lower for term in medical_terms)

                if query_has_medical and not content_has_medical:
                    logger.warning(
                        f"ColBERT HNSW: Potential relevance issue - medical query but non-medical document: {doc_id}"
                    )

        except Exception as e:
            logger.error(f"ColBERT HNSW: Error validating candidate relevance: {e}")
        finally:
            cursor.close()

    def _load_token_embeddings_for_candidates(self, candidate_doc_ids: List[int]) -> Dict[int, List[List[float]]]:
        """
        Stage 2: Load token embeddings only for candidate documents.

        This method loads token embeddings from RAG.DocumentTokenEmbeddings
        only for the specified candidate document IDs, avoiding the performance
        bottleneck of loading all token embeddings.

        Args:
            candidate_doc_ids: List of candidate document IDs

        Returns:
            Dictionary mapping doc_id to list of token embeddings
        """
        if not candidate_doc_ids:
            return {}

        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            # Create placeholders for the IN clause
            placeholders = ",".join(["?" for _ in candidate_doc_ids])

            # Query to load token embeddings only for candidate documents
            sql = f"""
                SELECT doc_id, token_index, token_embedding
                FROM RAG.DocumentTokenEmbeddings
                WHERE doc_id IN ({placeholders})
                ORDER BY doc_id, token_index
            """

            cursor.execute(sql, candidate_doc_ids)
            results = cursor.fetchall()

            # Group embeddings by document ID
            doc_embeddings_map = {}
            for row in results:
                doc_id, token_index, embedding_str = row

                # Parse the embedding string (VECTOR format from IRIS)
                parsed_embedding = self._parse_embedding_string(embedding_str)
                if parsed_embedding is None:
                    logger.warning(f"Skipping malformed embedding for doc_id {doc_id}, token_index {token_index}")
                    continue

                if doc_id not in doc_embeddings_map:
                    doc_embeddings_map[doc_id] = []

                doc_embeddings_map[doc_id].append(parsed_embedding)

            logger.debug(f"Loaded token embeddings for {len(doc_embeddings_map)} candidate documents")
            return doc_embeddings_map

        except Exception as e:
            logger.error(f"Error loading token embeddings for candidates: {e}")
            return {}
        finally:
            cursor.close()

    def _fetch_documents_by_ids(self, doc_ids: List[int], table_name: str = "RAG.SourceDocuments") -> List[Document]:
        """
        Fetch full document content and metadata by document IDs with CLOB conversion.

        Args:
            doc_ids: List of document IDs to fetch
            table_name: Name of the table to query (default: RAG.SourceDocuments)

        Returns:
            List of Document objects with full content and metadata
        """
        if not doc_ids:
            return []

        # Create placeholders for parameterized query to avoid SQL injection
        placeholders = ",".join(["?" for _ in doc_ids])

        query = f"""
        SELECT TOP {len(doc_ids)} doc_id, text_content, title, metadata
        FROM {table_name}
        WHERE doc_id IN ({placeholders})
        """

        cursor = self.connection_manager.get_connection().cursor()
        cursor.execute(query, doc_ids)
        rows = cursor.fetchall()

        documents = []
        for row in rows:
            doc_id, text_content, title, metadata_str = row

            # Convert CLOB to string if necessary
            from iris_rag.storage.clob_handler import convert_clob_to_string

            page_content = convert_clob_to_string(text_content)

            # Parse metadata if it exists
            metadata = {}
            if title:
                metadata["title"] = convert_clob_to_string(title)
            if metadata_str:
                try:
                    import json

                    metadata_str_converted = convert_clob_to_string(metadata_str)
                    parsed_metadata = json.loads(metadata_str_converted)
                    metadata.update(parsed_metadata)
                except (json.JSONDecodeError, TypeError):
                    # If metadata parsing fails, just use title
                    pass

            documents.append(Document(page_content=page_content, metadata=metadata, id=str(doc_id)))

        return documents

    def _calculate_maxsim_score(self, query_token_embeddings: np.ndarray, doc_token_embeddings: np.ndarray) -> float:
        """
        Calculate MaxSim score between query and document token embeddings.

        Args:
            query_token_embeddings: Query token embeddings (Q_len, dim)
            doc_token_embeddings: Document token embeddings (D_len, dim)

        Returns:
            MaxSim score (float)
        """
        import numpy as np

        # Convert to numpy arrays if they aren't already
        if not isinstance(query_token_embeddings, np.ndarray):
            query_token_embeddings = np.array(query_token_embeddings)
        if not isinstance(doc_token_embeddings, np.ndarray):
            doc_token_embeddings = np.array(doc_token_embeddings)

        # Handle empty arrays
        if query_token_embeddings.size == 0 or doc_token_embeddings.size == 0:
            return 0.0

        # Ensure arrays are 2D
        if query_token_embeddings.ndim == 1:
            query_token_embeddings = query_token_embeddings.reshape(1, -1)
        if doc_token_embeddings.ndim == 1:
            doc_token_embeddings = doc_token_embeddings.reshape(1, -1)

        # Normalize embeddings for cosine similarity
        query_norm = np.linalg.norm(query_token_embeddings, axis=1, keepdims=True)
        doc_norm = np.linalg.norm(doc_token_embeddings, axis=1, keepdims=True)

        # Avoid division by zero
        query_norm = np.where(query_norm == 0, 1e-8, query_norm)
        doc_norm = np.where(doc_norm == 0, 1e-8, doc_norm)

        query_normalized = query_token_embeddings / query_norm
        doc_normalized = doc_token_embeddings / doc_norm

        # Compute cosine similarity matrix: (Q_len, D_len)
        similarity_matrix = np.dot(query_normalized, doc_normalized.T)

        # For each query token, find the maximum similarity with any document token
        max_similarities = np.max(similarity_matrix, axis=1)

        # MaxSim score is the average of maximum similarities (ColBERT standard)
        maxsim_score = np.mean(max_similarities)

        return float(maxsim_score)

    def _validate_maxsim_scores(self, scores: List[float], query_text: str) -> bool:
        """
        Validate MaxSim scores to detect mock encoder issues.

        Args:
            scores: List of MaxSim scores
            query_text: Original query for context

        Returns:
            True if scores appear valid, False if suspicious
        """
        if not scores:
            return True

        # Check for too many identical or near-identical scores
        unique_scores = set(round(score, 4) for score in scores)
        identical_threshold = 0.8  # 80% of scores are identical

        if len(unique_scores) / len(scores) < (1 - identical_threshold):
            logger.warning(
                f"ColBERT Score Validation: {len(scores) - len(unique_scores)} out of {len(scores)} scores are nearly identical"
            )
            logger.warning(
                f"ColBERT Score Validation: This suggests mock encoder issues for query: '{query_text[:50]}...'"
            )
            return False

        # Check for all perfect scores (1.0)
        perfect_scores = sum(1 for score in scores if score >= 0.99)
        if perfect_scores > len(scores) * 0.5:  # More than 50% perfect scores
            logger.warning(
                f"ColBERT Score Validation: {perfect_scores} out of {len(scores)} documents have perfect scores (≥0.99)"
            )
            logger.warning(f"ColBERT Score Validation: This suggests mock encoder generating identical embeddings")
            return False

        # Log score distribution for debugging
        if scores:
            min_score, max_score = min(scores), max(scores)
            avg_score = sum(scores) / len(scores)
            logger.info(
                f"ColBERT Score Distribution: min={min_score:.4f}, max={max_score:.4f}, avg={avg_score:.4f}, unique={len(unique_scores)}"
            )

        return True

    def _retrieve_documents_with_colbert(
        self, query_text: str, query_token_embeddings: np.ndarray, top_k: int
    ) -> List[Document]:
        """
        Retrieve documents using ColBERT V2 hybrid retrieval approach.

        This method implements the optimized V2 strategy:
        1. Stage 1: Document-level HNSW search for candidate retrieval
        2. Stage 2: Selective token embedding loading for candidates only
        3. Stage 3: MaxSim re-ranking of candidates
        4. Stage 4: Final document retrieval with metadata
        5. Stage 5: Score validation and relevance filtering

        Args:
            query_text: The query text
            query_token_embeddings: Query token embeddings as numpy array (Q_len, dim)
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with ColBERT V2 metadata
        """
        try:
            # Get configuration for number of candidates
            num_candidates = self.config_manager.get("pipelines:colbert:num_candidates", 30)
            logger.debug(f"ColBERT V2: Using {num_candidates} candidates for retrieval")

            # Stage 1: Use vector store for candidate retrieval (replacing broken SQL)
            # Convert query text to document-level embedding (384D) for candidate retrieval
            doc_level_query_embedding = self.doc_embedding_func(query_text)

            # Validate document embedding dimension
            if len(doc_level_query_embedding) != self.doc_embedding_dim:
                logger.error(
                    f"ColBERT: Document embedding dimension mismatch: got {len(doc_level_query_embedding)}, expected {self.doc_embedding_dim}"
                )
                return []

            logger.debug(f"ColBERT: Using {self.doc_embedding_dim}D document embedding for candidate retrieval")

            # Use vector store for reliable candidate retrieval (use specific method to avoid interface confusion)
            candidate_results = self.vector_store.similarity_search_by_embedding(
                query_embedding=doc_level_query_embedding, top_k=num_candidates
            )

            # Extract document IDs from results
            candidate_doc_ids = [int(doc.id) for doc, score in candidate_results if doc.id.isdigit()]

            if not candidate_doc_ids:
                logger.warning("ColBERT V2: No candidate documents found via HNSW search")
                return []

            logger.info(f"ColBERT V2: Found {len(candidate_doc_ids)} candidate documents via HNSW")

            # Stage 2: Load token embeddings only for candidate documents
            doc_embeddings_map = self._load_token_embeddings_for_candidates(candidate_doc_ids)

            if not doc_embeddings_map:
                logger.warning("ColBERT V2: No token embeddings found for candidate documents")
                return []

            logger.info(f"ColBERT V2: Loaded token embeddings for {len(doc_embeddings_map)} documents")

            # Stage 3: Calculate MaxSim scores for candidates using token embeddings
            doc_scores = []
            for doc_id in candidate_doc_ids:
                if doc_id in doc_embeddings_map:
                    doc_token_embeddings = np.array(doc_embeddings_map[doc_id])

                    # Validate token embedding dimensions
                    if doc_token_embeddings.shape[1] != self.token_embedding_dim:
                        logger.warning(
                            f"ColBERT: Token embedding dimension mismatch for doc {doc_id}: got {doc_token_embeddings.shape[1]}, expected {self.token_embedding_dim}"
                        )
                        continue

                    maxsim_score = self._calculate_maxsim_score(query_token_embeddings, doc_token_embeddings)
                    doc_scores.append((doc_id, maxsim_score))
                    logger.debug(
                        f"ColBERT V2: Doc {doc_id} MaxSim score: {maxsim_score:.4f} (token dims: {doc_token_embeddings.shape})"
                    )

            if not doc_scores:
                logger.warning("ColBERT V2: No valid MaxSim scores calculated")
                return []

            logger.info(f"ColBERT V2: Calculated MaxSim scores for {len(doc_scores)} documents")

            # Stage 5: Validate MaxSim scores for mock encoder issues
            scores_only = [score for _, score in doc_scores]
            scores_valid = self._validate_maxsim_scores(scores_only, query_text)
            if not scores_valid:
                logger.warning("ColBERT V2: Score validation failed - proceeding with caution")

            # Sort by MaxSim score in descending order
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            # Stage 6: Apply relevance filtering before final selection
            filtered_doc_scores = self._filter_relevant_documents(doc_scores, query_text)

            # Take the top_k documents from filtered results
            top_doc_scores = filtered_doc_scores[:top_k]
            top_doc_ids = [doc_id for doc_id, _ in top_doc_scores]

            logger.info(f"ColBERT V2: Selected top {len(top_doc_scores)} documents for retrieval")

            # Stage 4: Retrieve full document content with CLOB conversion
            logger.debug(f"ColBERT V2: Fetching documents for IDs: {top_doc_ids}")
            documents = self._fetch_documents_by_ids(top_doc_ids)
            logger.debug(f"ColBERT V2: Retrieved {len(documents)} documents")

            # Add MaxSim scores to metadata
            score_map = {str(doc_id): score for doc_id, score in top_doc_scores}
            updated_documents = []
            for doc in documents:
                doc_id = str(doc.id)
                if doc_id in score_map:
                    # Update metadata with MaxSim score
                    updated_metadata = dict(doc.metadata) if doc.metadata else {}
                    updated_metadata.update(
                        {"maxsim_score": float(score_map[doc_id]), "retrieval_method": "colbert_v2_hybrid"}
                    )
                    # Create new document with updated metadata
                    updated_doc = Document(page_content=doc.page_content, metadata=updated_metadata, id=doc.id)
                    updated_documents.append(updated_doc)
                else:
                    updated_documents.append(doc)

            documents = updated_documents

            logger.debug(f"ColBERT V2: Retrieved {len(documents)} documents using hybrid approach")
            return documents

        except Exception as e:
            logger.error(f"ColBERT V2 retrieval failed: {e}")
            # Return empty list on failure for now (can add fallback later)
            return []

    def _filter_relevant_documents(self, doc_scores: List[tuple], query_text: str) -> List[tuple]:
        """
        Filter documents for relevance based on content analysis.

        This addresses the issue where ColBERT may retrieve documents that are
        technically similar but not contextually relevant (e.g., forestry papers for medical queries).

        Args:
            doc_scores: List of (doc_id, score) tuples
            query_text: Original query text

        Returns:
            Filtered list of (doc_id, score) tuples
        """
        if not doc_scores:
            return doc_scores

        # Extract query keywords and determine domain
        query_lower = query_text.lower()

        # Define domain-specific keywords
        medical_terms = [
            "medical",
            "health",
            "disease",
            "treatment",
            "patient",
            "clinical",
            "therapy",
            "diagnosis",
            "medicine",
            "hospital",
            "doctor",
            "cancer",
            "drug",
            "pharmaceutical",
            "symptom",
            "syndrome",
            "pathology",
        ]

        tech_terms = [
            "technology",
            "software",
            "computer",
            "algorithm",
            "data",
            "system",
            "programming",
            "artificial",
            "intelligence",
            "machine",
            "learning",
        ]

        # Determine query domain
        query_is_medical = any(term in query_lower for term in medical_terms)
        query_is_tech = any(term in query_lower for term in tech_terms)

        if not (query_is_medical or query_is_tech):
            # If we can't determine domain, return all documents
            logger.debug("ColBERT Relevance Filter: Cannot determine query domain, returning all documents")
            return doc_scores

        # Get document content for relevance checking
        doc_ids = [doc_id for doc_id, _ in doc_scores]
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            placeholders = ", ".join(["?"] * len(doc_ids))
            content_sql = f"""
                SELECT doc_id, title, SUBSTRING(text_content, 1, 500) AS content_sample
                FROM RAG.SourceDocuments
                WHERE doc_id IN ({placeholders})
            """

            cursor.execute(content_sql, doc_ids)
            content_results = cursor.fetchall()

            # Create content map
            content_map = {}
            for row in content_results:
                doc_id = row[0]
                title = row[1] if row[1] else ""
                content_sample = row[2] if row[2] else ""
                content_map[doc_id] = (title + " " + content_sample).lower()

            # Filter documents based on relevance
            filtered_scores = []
            filtered_count = 0

            for doc_id, score in doc_scores:
                if doc_id not in content_map:
                    # If we can't get content, keep the document
                    filtered_scores.append((doc_id, score))
                    continue

                doc_content = content_map[doc_id]

                # Check relevance based on query domain
                is_relevant = True

                if query_is_medical:
                    # For medical queries, check if document contains medical terms
                    doc_has_medical = any(term in doc_content for term in medical_terms)
                    if not doc_has_medical:
                        # Additional check for scientific/research terms that might be relevant
                        research_terms = ["research", "study", "analysis", "investigation", "experiment"]
                        doc_has_research = any(term in doc_content for term in research_terms)
                        if not doc_has_research:
                            is_relevant = False
                            filtered_count += 1
                            logger.debug(
                                f"ColBERT Relevance Filter: Filtered out doc {doc_id} - medical query but no medical/research content"
                            )

                elif query_is_tech:
                    # For tech queries, check if document contains tech terms
                    doc_has_tech = any(term in doc_content for term in tech_terms)
                    if not doc_has_tech:
                        is_relevant = False
                        filtered_count += 1
                        logger.debug(
                            f"ColBERT Relevance Filter: Filtered out doc {doc_id} - tech query but no tech content"
                        )

                if is_relevant:
                    filtered_scores.append((doc_id, score))

            logger.info(
                f"ColBERT Relevance Filter: Filtered out {filtered_count} irrelevant documents, kept {len(filtered_scores)}"
            )
            return filtered_scores

        except Exception as e:
            logger.error(f"ColBERT Relevance Filter: Error during filtering: {e}")
            # Return original scores if filtering fails
            return doc_scores
        finally:
            cursor.close()

    def _parse_embedding_string(self, embedding_str: str) -> Optional[List[float]]:
        """
        Parse embedding string into list of floats.
        Handles both bracket format [1.0,2.0,3.0] and comma-separated format 1.0,2.0,3.0

        Args:
            embedding_str: String representation of embedding

        Returns:
            List of float values, or None if parsing fails
        """
        try:
            if embedding_str.startswith("[") and embedding_str.endswith("]"):
                # Bracket format: [1.0,2.0,3.0]
                return [float(x.strip()) for x in embedding_str[1:-1].split(",")]
            else:
                # Comma-separated format: 1.0,2.0,3.0
                return [float(x.strip()) for x in embedding_str.split(",")]
        except (ValueError, AttributeError) as e:
            return None

    def _fallback_to_basic_retrieval(self, query_text: str, top_k: int) -> List[Document]:
        """
        Fallback to basic vector retrieval using proper document-level embeddings.

        Args:
            query_text: Original query text
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents
        """
        try:
            # Use proper document embedding function (384D) instead of averaging token embeddings
            doc_level_embedding = self.doc_embedding_func(query_text)

            # Validate dimension matches expected document dimension
            if len(doc_level_embedding) != self.doc_embedding_dim:
                logger.error(
                    f"ColBERT fallback: Document embedding dimension {len(doc_level_embedding)} doesn't match expected {self.doc_embedding_dim}"
                )
                return []

            logger.debug(f"ColBERT fallback: Using proper {self.doc_embedding_dim}D document embedding")

            # Use vector store for consistent search
            search_results = self.vector_store.similarity_search_by_embedding(
                query_embedding=doc_level_embedding, top_k=top_k
            )

            # Convert results to Document list
            documents = [doc for doc, score in search_results]

            # Add fallback metadata
            for doc in documents:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["retrieval_method"] = "colbert_fallback_basic"

            logger.debug(f"ColBERT fallback: Retrieved {len(documents)} documents using proper document embeddings")
            return documents

        except Exception as e:
            logger.error(f"ColBERT fallback failed: {e}")
            return []

    def _generate_answer(self, query: str, documents: List[Document]) -> str:
        """
        Generate answer using retrieved documents and LLM.

        Args:
            query: Original query
            documents: Retrieved documents

        Returns:
            Generated answer string
        """
        if not documents:
            return "No relevant documents found to answer the query."

        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            # Handle both page_content and content attributes for compatibility
            content = getattr(doc, "page_content", None) or getattr(doc, "content", "")
            context_parts.append(f"Document {i}: {content[:500]}...")

        context = "\n\n".join(context_parts)

        # Create prompt for LLM
        prompt = f"""Based on the following documents, please answer the question.

Question: {query}

Documents:
{context}

Please provide a comprehensive answer based on the information in the context documents. If the context doesn't contain enough information to fully answer the question, please indicate what information is missing.

Answer:"""

        # Generate answer using LLM
        answer = self.llm_func(prompt)

        return answer.strip()

    def _get_current_time(self) -> float:
        """Get current time for performance measurement"""
        import time

        return time.time()

    def setup_database(self) -> bool:
        """
        Set up database tables and indexes required for ColBERT pipeline.

        Returns:
            bool: True if setup successful, False otherwise
        """
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()

        try:
            logger.info("Setting up ColBERT database tables and indexes...")

            # Get token embedding dimension from schema manager
            from ..storage.schema_manager import SchemaManager

            schema_manager = SchemaManager(self.connection_manager, self.config_manager)
            token_dimension = schema_manager.get_vector_dimension("DocumentTokenEmbeddings")

            # Create DocumentTokenEmbeddings table if it doesn't exist
            create_token_table_sql = f"""
                CREATE TABLE IF NOT EXISTS RAG.DocumentTokenEmbeddings (
                    id INTEGER IDENTITY PRIMARY KEY,
                    doc_id VARCHAR(255) NOT NULL,
                    token_position INTEGER NOT NULL,
                    token_text VARCHAR(500),
                    token_embedding VECTOR(DOUBLE, {token_dimension}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES RAG.SourceDocuments(doc_id)
                )
            """
            cursor.execute(create_token_table_sql)
            logger.debug(f"DocumentTokenEmbeddings table created/verified with {token_dimension}-dimensional vectors")

            # Create index on doc_id for faster lookups
            try:
                create_index_sql = """
                    CREATE INDEX IF NOT EXISTS idx_doc_token_embeddings_doc_id
                    ON RAG.DocumentTokenEmbeddings (doc_id)
                """
                cursor.execute(create_index_sql)
                logger.debug("Index on DocumentTokenEmbeddings.doc_id created")
            except Exception as e:
                logger.warning(f"Could not create index on DocumentTokenEmbeddings: {e}")

            # Create vector index for token embeddings if possible
            try:
                create_vector_index_sql = """
                    CREATE INDEX IF NOT EXISTS idx_doc_token_embeddings_vector
                    ON RAG.DocumentTokenEmbeddings (token_embedding)
                """
                cursor.execute(create_vector_index_sql)
                logger.debug("Vector index on DocumentTokenEmbeddings.token_embedding created")
            except Exception as e:
                logger.warning(f"Could not create vector index on DocumentTokenEmbeddings: {e}")

            connection.commit()
            logger.info("ColBERT database setup completed successfully")
            return True

        except Exception as e:
            logger.error(f"ColBERT database setup failed: {e}")
            connection.rollback()
            return False
        finally:
            cursor.close()

    def execute(self, query_text: str, **kwargs) -> dict:
        """
        Execute the ColBERT RAG pipeline (alias for run method).

        Args:
            query_text: The input query string
            **kwargs: Additional parameters

        Returns:
            Dictionary containing pipeline results
        """
        return self.run(query_text, **kwargs)

    def load_documents(self, documents_path: str, **kwargs) -> None:
        """
        Load documents into the ColBERT pipeline's knowledge base.

        Args:
            documents_path: Path to documents or directory
            **kwargs: Additional parameters
        """
        # This would implement document loading and token embedding generation
        # For now, we rely on the setup orchestrator to handle this
        logger.info(f"Document loading for ColBERT pipeline: {documents_path}")
        logger.info("Use the setup orchestrator to generate token embeddings")

    def query(self, query_text: str, top_k: int = 5, generate_answer: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Execute the ColBERT pipeline with standardized response format.

        Args:
            query_text: The input query string
            top_k: Number of documents to retrieve
            generate_answer: Whether to generate an answer (default: True)
            **kwargs: Additional parameters

        Returns:
            Standardized dictionary with query, retrieved_documents, contexts, metadata, answer, execution_time
        """
        import time

        start_time = time.time()

        logger.info(f"ColBERT: Processing query: '{query_text[:50]}...'")

        try:
            # Validate setup before proceeding
            if not self.validate_setup():
                logger.warning("ColBERT setup validation failed - pipeline may not work correctly")

            # Generate query token embeddings
            query_tokens = self.colbert_query_encoder(query_text)
            logger.debug(f"ColBERT: Generated {len(query_tokens)} query token embeddings")

            # Validate that we have token embeddings
            if not query_tokens:
                raise ValueError("ColBERT query encoder returned empty token embeddings")

            # Convert to numpy array for consistency
            import numpy as np

            query_token_embeddings = np.array(query_tokens)

            # Retrieve documents using ColBERT matching
            retrieved_docs = self._retrieve_documents_with_colbert(query_text, query_token_embeddings, top_k)

            # Generate answer if requested
            answer = None
            if generate_answer and self.llm_func and retrieved_docs:
                answer = self._generate_answer(query_text, retrieved_docs)
            elif generate_answer and not self.llm_func:
                answer = "No LLM function available for answer generation."
            elif generate_answer and not retrieved_docs:
                answer = "No relevant documents found to answer the query."

            execution_time = time.time() - start_time

            # Return standardized response format
            result = {
                "query": query_text,
                "answer": answer,
                "retrieved_documents": retrieved_docs,
                "contexts": [getattr(doc, "page_content", str(doc)) for doc in retrieved_docs],
                "execution_time": execution_time,
                "metadata": {
                    "num_retrieved": len(retrieved_docs),
                    "pipeline_type": "colbert",
                    "generated_answer": generate_answer and answer is not None,
                    "token_count": len(query_tokens),
                    "search_method": "colbert_v2_hybrid",
                },
            }

            logger.info(f"ColBERT: Completed in {execution_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"ColBERT pipeline failed: {e}")
            return {
                "query": query_text,
                "answer": None,
                "retrieved_documents": [],
                "contexts": [],
                "execution_time": 0.0,
                "metadata": {
                    "num_retrieved": 0,
                    "pipeline_type": "colbert",
                    "generated_answer": False,
                    "error": str(e),
                },
            }
