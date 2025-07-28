"""
ColBERT Retriever implementation for the RAG pipeline.

This module contains the ColBERTRetriever class, which encapsulates
the logic for retrieving documents using the ColBERT methodology.
"""

import logging
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from ...core.models import Document
from common.iris_connection_manager import get_iris_connection
from ...config.manager import ConfigurationManager
from ...storage.vector_store_iris import IRISVectorStore as VectorStore

logger = logging.getLogger(__name__)

class ColBERTRetriever:
    """
    A class to handle the retrieval logic for the ColBERT RAG pipeline.
    """

    def __init__(self,
                 config_manager: ConfigurationManager,
                 vector_store: VectorStore,
                 doc_embedding_func: Callable,
                 doc_embedding_dim: int,
                 token_embedding_dim: int):
        """
        Initialize the ColBERTRetriever.

        Args:
            config_manager: Configuration manager.
            vector_store: The vector store for similarity searches.
            doc_embedding_func: Function for document-level embeddings.
            doc_embedding_dim: Dimension of document embeddings.
            token_embedding_dim: Dimension of token embeddings.
        """
        self.config_manager = config_manager
        self.vector_store = vector_store
        self.doc_embedding_func = doc_embedding_func
        self.doc_embedding_dim = doc_embedding_dim
        self.token_embedding_dim = token_embedding_dim
        
        # Get column names from schema manager via vector store
        self.schema_manager = getattr(vector_store, 'schema_manager', None)
        if self.schema_manager:
            table_config = self.schema_manager.get_table_config("SourceDocuments")
            self.id_column = table_config.get("id_column", "ID")
            self.embedding_column = table_config.get("embedding_column", "embedding")
            self.content_column = table_config.get("content_column", "TEXT_CONTENT")
        else:
            # Fallback to defaults if schema_manager not available
            self.id_column = "ID"
            self.embedding_column = "embedding"
            self.content_column = "TEXT_CONTENT"

    def _format_vector_for_sql(self, vector: List[float]) -> str:
        """Formats a vector list into a comma-separated string for IRIS SQL."""
        if not vector:
            return "[]"
        return "[" + ",".join(f"{x:.15g}" for x in vector) + "]"

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
            if embedding_str.startswith('[') and embedding_str.endswith(']'):
                return [float(x.strip()) for x in embedding_str[1:-1].split(',')]
            else:
                return [float(x.strip()) for x in embedding_str.split(',')]
        except (ValueError, AttributeError):
            return None

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
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            if not self.doc_embedding_func:
                logger.error("ColBERT HNSW: Document embedding_func is not initialized.")
                return []

            doc_level_query_embedding = self.doc_embedding_func(query_text)
            
            if not doc_level_query_embedding or \
               not isinstance(doc_level_query_embedding, list) or \
               not all(isinstance(x, float) for x in doc_level_query_embedding):
                 logger.error(f"ColBERT HNSW: Failed to generate valid List[float] document-level query embedding. Type: {type(doc_level_query_embedding)}")
                 return []
            
            query_vector_dim = len(doc_level_query_embedding)
            if query_vector_dim != self.doc_embedding_dim:
                 logger.error(f"ColBERT HNSW: Document-level query embedding dimension is {query_vector_dim}, expected {self.doc_embedding_dim}.")
                 return []

            query_vector_str = self._format_vector_for_sql(doc_level_query_embedding)
            
            logger.info(f"ColBERT HNSW: Searching for candidates with query '{query_text[:50]}...' using {query_vector_dim}D embedding.")
            
            from common.vector_sql_utils import format_vector_search_sql, execute_vector_search
            
            sql = format_vector_search_sql(
                table_name="RAG.SourceDocuments",
                vector_column=self.embedding_column,
                vector_string=query_vector_str,
                embedding_dim=query_vector_dim,
                top_k=k,
                id_column=self.id_column,
                content_column=self.content_column
            )
            
            results = execute_vector_search(cursor, sql)
            
            candidate_doc_ids = [row[0] for row in results]
            
            logger.info(f"ColBERT HNSW: Retrieved {len(candidate_doc_ids)} candidate documents")
            
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
            
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            placeholders = ', '.join(['?'] * len(candidate_doc_ids))
            validation_sql = f"""
                SELECT doc_id, title, SUBSTRING(text_content, 1, 200) AS content_sample
                FROM RAG.SourceDocuments
                WHERE doc_id IN ({placeholders})
            """
            
            cursor.execute(validation_sql, candidate_doc_ids)
            results = cursor.fetchall()
            
            logger.info(f"ColBERT HNSW: Validating {len(results)} candidates for query: '{query_text[:50]}...'")
            
            for i, row in enumerate(results):
                doc_id, title, content_sample = row
                logger.info(f"ColBERT HNSW: Candidate {i+1} - ID: {doc_id}, Title: {title or 'No title'}")
                
        except Exception as e:
            logger.error(f"ColBERT HNSW: Error validating candidate relevance: {e}")
        finally:
            cursor.close()

    def _load_token_embeddings_for_candidates(self, candidate_doc_ids: List) -> Dict:
        """
        Stage 2: Load token embeddings only for candidate documents.
        
        Args:
            candidate_doc_ids: List of candidate document IDs (can be int or str)
            
        Returns:
            Dictionary mapping doc_id to list of token embeddings
        """
        if not candidate_doc_ids:
            return {}
            
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            placeholders = ','.join(['?' for _ in candidate_doc_ids])
            
            sql = f"""
                SELECT doc_id, token_index, token_embedding
                FROM RAG.DocumentTokenEmbeddings
                WHERE doc_id IN ({placeholders})
                ORDER BY doc_id, token_index
            """
            
            cursor.execute(sql, candidate_doc_ids)
            results = cursor.fetchall()
            
            doc_embeddings_map = {}
            for row in results:
                doc_id, token_index, embedding_str = row
                
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

    def _fetch_documents_by_ids(self, doc_ids: List, table_name: str = "RAG.SourceDocuments") -> List[Document]:
        """
        Fetch full document content and metadata by document IDs with CLOB conversion.
        
        Args:
            doc_ids: List of document IDs (can be int or str)
            table_name: Name of the table to query
            
        Returns:
            List of Document objects
        """
        if not doc_ids:
            return []
            
        placeholders = ','.join(['?' for _ in doc_ids])
        
        query = f"""
        SELECT TOP {len(doc_ids)} doc_id, text_content, title, metadata
        FROM {table_name}
        WHERE doc_id IN ({placeholders})
        """
        
        cursor = get_iris_connection().cursor()
        cursor.execute(query, doc_ids)
        rows = cursor.fetchall()
        
        documents = []
        for row in rows:
            doc_id, text_content, title, metadata_str = row
            
            from iris_rag.storage.clob_handler import convert_clob_to_string
            page_content = convert_clob_to_string(text_content)
            
            metadata = {}
            if title:
                metadata['title'] = convert_clob_to_string(title)
            if metadata_str:
                try:
                    import json
                    metadata_str_converted = convert_clob_to_string(metadata_str)
                    parsed_metadata = json.loads(metadata_str_converted)
                    metadata.update(parsed_metadata)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            documents.append(Document(
                page_content=page_content,
                metadata=metadata,
                id=str(doc_id)
            ))
        
        return documents

    def _calculate_maxsim_score(self, query_token_embeddings: np.ndarray, doc_token_embeddings: np.ndarray) -> float:
        """
        Calculate MaxSim score between query and document token embeddings.
        """
        import numpy as np
        
        if not isinstance(query_token_embeddings, np.ndarray):
            query_token_embeddings = np.array(query_token_embeddings)
        if not isinstance(doc_token_embeddings, np.ndarray):
            doc_token_embeddings = np.array(doc_token_embeddings)
        
        if query_token_embeddings.size == 0 or doc_token_embeddings.size == 0:
            return 0.0
        
        if query_token_embeddings.ndim == 1:
            query_token_embeddings = query_token_embeddings.reshape(1, -1)
        if doc_token_embeddings.ndim == 1:
            doc_token_embeddings = doc_token_embeddings.reshape(1, -1)
            
        query_norm = np.linalg.norm(query_token_embeddings, axis=1, keepdims=True)
        doc_norm = np.linalg.norm(doc_token_embeddings, axis=1, keepdims=True)
        
        query_norm = np.where(query_norm == 0, 1e-8, query_norm)
        doc_norm = np.where(doc_norm == 0, 1e-8, doc_norm)
        
        query_normalized = query_token_embeddings / query_norm
        doc_normalized = doc_token_embeddings / doc_norm
            
        similarity_matrix = np.dot(query_normalized, doc_normalized.T)
        
        max_similarities = np.max(similarity_matrix, axis=1)
        
        maxsim_score = np.mean(max_similarities)
        
        return float(maxsim_score)

    def _validate_maxsim_scores(self, scores: List[float], query_text: str) -> bool:
        """
        Validate MaxSim scores to detect mock encoder issues.
        """
        if not scores:
            return True
            
        unique_scores = set(round(score, 4) for score in scores)
        identical_threshold = 0.8
        
        if len(unique_scores) / len(scores) < (1 - identical_threshold):
            logger.warning(f"ColBERT Score Validation: {len(scores) - len(unique_scores)} out of {len(scores)} scores are nearly identical")
            return False
            
        perfect_scores = sum(1 for score in scores if score >= 0.99)
        if perfect_scores > len(scores) * 0.5:
            logger.warning(f"ColBERT Score Validation: {perfect_scores} out of {len(scores)} documents have perfect scores (≥0.99)")
            return False
            
        if scores:
            min_score, max_score = min(scores), max(scores)
            avg_score = sum(scores) / len(scores)
            logger.info(f"ColBERT Score Distribution: min={min_score:.4f}, max={max_score:.4f}, avg={avg_score:.4f}, unique={len(unique_scores)}")
            
        return True

    def _retrieve_documents_with_colbert(self, query_text: str, query_token_embeddings: np.ndarray, top_k: int) -> List[Document]:
        """
        Retrieve documents using ColBERT V2 hybrid retrieval approach.
        """
        try:
            num_candidates = self.config_manager.get('pipelines:colbert:num_candidates', 30)
            
            doc_level_query_embedding = self.doc_embedding_func(query_text)
            
            if len(doc_level_query_embedding) != self.doc_embedding_dim:
                logger.error(f"ColBERT: Document embedding dimension mismatch: got {len(doc_level_query_embedding)}, expected {self.doc_embedding_dim}")
                return []
            
            candidate_results = self.vector_store.similarity_search_by_embedding(
                query_embedding=doc_level_query_embedding,
                top_k=num_candidates
            )
            
            # Debug: Log the first few document IDs to understand the format
            if candidate_results:
                sample_ids = [doc.id for doc, score in candidate_results[:5]]
                logger.info(f"ColBERT: Sample document IDs from vector search: {sample_ids}")
            
            # Extract document IDs - handle both numeric and UUID string formats
            candidate_doc_ids = []
            for doc, score in candidate_results:
                if doc.id.isdigit():
                    candidate_doc_ids.append(int(doc.id))
                else:
                    # For UUID strings, use the string directly
                    candidate_doc_ids.append(doc.id)
            
            logger.info(f"ColBERT: Stage 1 - Found {len(candidate_doc_ids)} candidate documents from vector search")
            
            if not candidate_doc_ids:
                logger.warning("ColBERT: No candidate documents found from vector search")
                return []
            
            doc_embeddings_map = self._load_token_embeddings_for_candidates(candidate_doc_ids)
            logger.info(f"ColBERT: Stage 2 - Loaded token embeddings for {len(doc_embeddings_map)} documents")
            
            if not doc_embeddings_map:
                logger.warning("ColBERT: No token embeddings found for any candidate documents - falling back to basic retrieval")
                return self._fallback_to_basic_retrieval(query_text, top_k)
            
            doc_scores = []
            for doc_id in candidate_doc_ids:
                if doc_id in doc_embeddings_map:
                    doc_token_embeddings = np.array(doc_embeddings_map[doc_id])
                    
                    if doc_token_embeddings.shape[1] != self.token_embedding_dim:
                        continue
                    
                    maxsim_score = self._calculate_maxsim_score(query_token_embeddings, doc_token_embeddings)
                    doc_scores.append((doc_id, maxsim_score))
            
            if not doc_scores:
                return []
            
            scores_only = [score for _, score in doc_scores]
            self._validate_maxsim_scores(scores_only, query_text)
            
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            filtered_doc_scores = self._filter_relevant_documents(doc_scores, query_text)
            
            top_doc_scores = filtered_doc_scores[:top_k]
            top_doc_ids = [doc_id for doc_id, _ in top_doc_scores]
            
            documents = self._fetch_documents_by_ids(top_doc_ids)
            
            score_map = {str(doc_id): score for doc_id, score in top_doc_scores}
            updated_documents = []
            for doc in documents:
                doc_id = str(doc.id)
                if doc_id in score_map:
                    updated_metadata = dict(doc.metadata) if doc.metadata else {}
                    updated_metadata.update({
                        "maxsim_score": float(score_map[doc_id]),
                        "retrieval_method": "colbert_v2_hybrid"
                    })
                    updated_doc = Document(
                        page_content=doc.page_content,
                        metadata=updated_metadata,
                        id=doc.id
                    )
                    updated_documents.append(updated_doc)
                else:
                    updated_documents.append(doc)
            
            return updated_documents
                
        except Exception as e:
            logger.error(f"ColBERT V2 retrieval failed: {e}")
            return []

    def _filter_relevant_documents(self, doc_scores: List[tuple], query_text: str) -> List[tuple]:
        """
        Filter documents for relevance based on content analysis.
        """
        if not doc_scores:
            return doc_scores
            
        query_lower = query_text.lower()
        
        medical_terms = ['medical', 'health', 'disease', 'treatment', 'patient', 'clinical',
                        'therapy', 'diagnosis', 'medicine', 'hospital', 'doctor', 'cancer',
                        'drug', 'pharmaceutical', 'symptom', 'syndrome', 'pathology']
        
        tech_terms = ['technology', 'software', 'computer', 'algorithm', 'data', 'system',
                     'programming', 'artificial', 'intelligence', 'machine', 'learning']
        
        query_is_medical = any(term in query_lower for term in medical_terms)
        query_is_tech = any(term in query_lower for term in tech_terms)
        
        if not (query_is_medical or query_is_tech):
            return doc_scores
        
        doc_ids = [doc_id for doc_id, _ in doc_scores]
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            placeholders = ', '.join(['?'] * len(doc_ids))
            content_sql = f"""
                SELECT doc_id, title, SUBSTRING(text_content, 1, 500) AS content_sample
                FROM RAG.SourceDocuments
                WHERE doc_id IN ({placeholders})
            """
            
            cursor.execute(content_sql, doc_ids)
            content_results = cursor.fetchall()
            
            content_map = {}
            for row in content_results:
                doc_id, title, content_sample = row
                content_map[doc_id] = (title + " " + content_sample).lower()
            
            filtered_scores = []
            for doc_id, score in doc_scores:
                if doc_id not in content_map:
                    filtered_scores.append((doc_id, score))
                    continue
                
                doc_content = content_map[doc_id]
                
                is_relevant = True
                if query_is_medical:
                    doc_has_medical = any(term in doc_content for term in medical_terms)
                    if not doc_has_medical:
                        research_terms = ['research', 'study', 'analysis', 'investigation', 'experiment']
                        doc_has_research = any(term in doc_content for term in research_terms)
                        if not doc_has_research:
                            is_relevant = False
                
                elif query_is_tech:
                    doc_has_tech = any(term in doc_content for term in tech_terms)
                    if not doc_has_tech:
                        is_relevant = False
                
                if is_relevant:
                    filtered_scores.append((doc_id, score))
            
            return filtered_scores
            
        except Exception as e:
            logger.error(f"ColBERT Relevance Filter: Error during filtering: {e}")
            return doc_scores
        finally:
            cursor.close()

    def _fallback_to_basic_retrieval(self, query_text: str, top_k: int) -> List[Document]:
        """
        Fallback to basic vector retrieval using proper document-level embeddings.
        """
        try:
            doc_level_embedding = self.doc_embedding_func(query_text)
            
            if len(doc_level_embedding) != self.doc_embedding_dim:
                return []
            
            search_results = self.vector_store.similarity_search_by_embedding(
                query_embedding=doc_level_embedding,
                top_k=top_k
            )
            
            documents = [doc for doc, score in search_results]
            
            for doc in documents:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["retrieval_method"] = "colbert_fallback_basic"
            
            return documents
            
        except Exception as e:
            logger.error(f"ColBERT fallback failed: {e}")
            return []