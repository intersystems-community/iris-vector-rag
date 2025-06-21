"""
Document Service for Reconciliation Controller

This module provides document-related database operations for the reconciliation
system, including CRUD operations on documents, chunks, and embeddings.
"""

import logging
from typing import List, Optional, Tuple
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.validation.embedding_validator import EmbeddingValidator
from common.db_vector_utils import insert_vector

logger = logging.getLogger(__name__)


class DocumentService:
    """
    Service class for document-related database operations.
    
    Handles CRUD operations on documents, chunks, and embeddings tables
    for the reconciliation system.
    """
    
    def __init__(self, connection_manager: ConnectionManager, config_manager: ConfigurationManager):
        """
        Initialize the DocumentService.
        
        Args:
            connection_manager: Connection manager for database access
            config_manager: Configuration manager for settings
        """
        self.connection_manager = connection_manager
        self.config_manager = config_manager
        self.embedding_validator = EmbeddingValidator(config_manager)
        
        logger.debug("DocumentService initialized")
    
    def get_document_ids_by_source(self, source_uri: str) -> List[int]:
        """
        Get document IDs for a specific source URI.
        
        Args:
            source_uri: The source URI to search for
            
        Returns:
            List of document IDs
        """
        try:
            iris_connector = self.connection_manager.get_connection("iris")
            cursor = iris_connector.cursor()
            
            cursor.execute(
                "SELECT id FROM RAG.SourceDocuments WHERE source_uri = ?",
                [source_uri]
            )
            
            results = cursor.fetchall()
            doc_ids = [row[0] for row in results]
            cursor.close()
            
            logger.debug(f"Found {len(doc_ids)} documents for source URI: {source_uri}")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error getting document IDs by source: {e}")
            return []
    
    def get_document_content_by_id(self, doc_id: int) -> Optional[str]:
        """
        Get document content by document ID.
        
        Args:
            doc_id: Document ID to retrieve content for
            
        Returns:
            Document content as string, or None if not found
        """
        try:
            iris_connector = self.connection_manager.get_connection("iris")
            cursor = iris_connector.cursor()
            
            cursor.execute(
                "SELECT text_content FROM RAG.SourceDocuments WHERE id = ?",
                [doc_id]
            )
            
            result = cursor.fetchone()
            cursor.close()
            
            if result and result[0]:
                return str(result[0])
            return None
            
        except Exception as e:
            logger.error(f"Error getting document content for ID {doc_id}: {e}")
            return None
    
    def get_all_source_document_ids(self) -> List[int]:
        """
        Get all source document IDs.
        
        Returns:
            List of all document IDs
        """
        try:
            iris_connector = self.connection_manager.get_connection("iris")
            cursor = iris_connector.cursor()
            
            cursor.execute("SELECT id FROM RAG.SourceDocuments")
            
            results = cursor.fetchall()
            doc_ids = [row[0] for row in results]
            cursor.close()
            
            logger.debug(f"Found {len(doc_ids)} total source documents")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error getting all source document IDs: {e}")
            return []
    
    def get_chunk_ids_for_document(self, doc_id: int) -> List[int]:
        """
        Get chunk IDs for a specific document.
        
        Args:
            doc_id: Document ID to get chunks for
            
        Returns:
            List of chunk IDs
        """
        try:
            iris_connector = self.connection_manager.get_connection("iris")
            cursor = iris_connector.cursor()
            
            cursor.execute(
                "SELECT id FROM RAG.DocumentChunks WHERE doc_id = ?",
                [doc_id]
            )
            
            results = cursor.fetchall()
            chunk_ids = [row[0] for row in results]
            cursor.close()
            
            logger.debug(f"Found {len(chunk_ids)} chunks for document {doc_id}")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Error getting chunk IDs for document {doc_id}: {e}")
            return []
    
    def get_embedding_ids_for_chunk(self, chunk_id: int) -> List[int]:
        """
        Get embedding IDs for a specific chunk.
        
        Args:
            chunk_id: Chunk ID to get embeddings for
            
        Returns:
            List of embedding IDs
        """
        try:
            iris_connector = self.connection_manager.get_connection("iris")
            cursor = iris_connector.cursor()
            
            cursor.execute(
                "SELECT id FROM RAG.DocumentTokenEmbeddings WHERE chunk_id = ?",
                [chunk_id]
            )
            
            results = cursor.fetchall()
            embedding_ids = [row[0] for row in results]
            cursor.close()
            
            logger.debug(f"Found {len(embedding_ids)} embeddings for chunk {chunk_id}")
            return embedding_ids
            
        except Exception as e:
            logger.error(f"Error getting embedding IDs for chunk {chunk_id}: {e}")
            return []
    
    def delete_embeddings_by_ids(self, embedding_ids: List[int]) -> int:
        """
        Delete embeddings by their IDs.
        
        Args:
            embedding_ids: List of embedding IDs to delete
            
        Returns:
            Number of embeddings deleted
        """
        if not embedding_ids:
            return 0
            
        try:
            iris_connector = self.connection_manager.get_connection("iris")
            cursor = iris_connector.cursor()
            
            # Create placeholders for the IN clause
            placeholders = ','.join(['?' for _ in embedding_ids])
            cursor.execute(
                f"DELETE FROM RAG.DocumentTokenEmbeddings WHERE id IN ({placeholders})",
                embedding_ids
            )
            
            deleted_count = cursor.rowcount
            cursor.close()
            iris_connector.commit()
            
            logger.debug(f"Deleted {deleted_count} embeddings")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting embeddings: {e}")
            return 0
    
    def delete_chunks_by_ids(self, chunk_ids: List[int]) -> int:
        """
        Delete chunks by their IDs.
        
        Args:
            chunk_ids: List of chunk IDs to delete
            
        Returns:
            Number of chunks deleted
        """
        if not chunk_ids:
            return 0
            
        try:
            iris_connector = self.connection_manager.get_connection("iris")
            cursor = iris_connector.cursor()
            
            # Create placeholders for the IN clause
            placeholders = ','.join(['?' for _ in chunk_ids])
            cursor.execute(
                f"DELETE FROM RAG.DocumentChunks WHERE id IN ({placeholders})",
                chunk_ids
            )
            
            deleted_count = cursor.rowcount
            cursor.close()
            iris_connector.commit()
            
            logger.debug(f"Deleted {deleted_count} chunks")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return 0
    
    def delete_documents_by_ids(self, doc_ids: List[int]) -> int:
        """
        Delete documents by their IDs.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            Number of documents deleted
        """
        if not doc_ids:
            return 0
            
        try:
            iris_connector = self.connection_manager.get_connection("iris")
            cursor = iris_connector.cursor()
            
            # Create placeholders for the IN clause
            placeholders = ','.join(['?' for _ in doc_ids])
            cursor.execute(
                f"DELETE FROM RAG.SourceDocuments WHERE id IN ({placeholders})",
                doc_ids
            )
            
            deleted_count = cursor.rowcount
            cursor.close()
            iris_connector.commit()
            
            logger.debug(f"Deleted {deleted_count} documents")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return 0
    
    def get_documents_with_mock_embeddings(self) -> List[str]:
        """
        Get document IDs that have mock embeddings.
        
        Returns:
            List of document IDs with mock embeddings
        """
        try:
            # Sample embeddings and use validator to identify mock ones
            sample_embeddings = self.embedding_validator.sample_embeddings_from_database(
                table_name="RAG.DocumentTokenEmbeddings",
                sample_size=200
            )
            
            mock_doc_ids = []
            for doc_id, embedding_str in sample_embeddings:
                embedding_array = self.embedding_validator._parse_embedding_string(embedding_str)
                if embedding_array is not None and self.embedding_validator._is_mock_embedding(embedding_array):
                    if doc_id not in mock_doc_ids:
                        mock_doc_ids.append(doc_id)
            
            logger.info(f"Found {len(mock_doc_ids)} documents with mock embeddings")
            return mock_doc_ids
            
        except Exception as e:
            logger.error(f"Error identifying documents with mock embeddings: {e}")
            return []
    
    def get_documents_with_low_diversity_embeddings(self) -> List[str]:
        """
        Get document IDs that have low diversity embeddings.
        
        Returns:
            List of document IDs with low diversity embeddings
        """
        try:
            # Sample embeddings and use validator to identify low diversity ones
            sample_embeddings = self.embedding_validator.sample_embeddings_from_database(
                table_name="RAG.DocumentTokenEmbeddings",
                sample_size=200
            )
            
            low_diversity_doc_ids = []
            for doc_id, embedding_str in sample_embeddings:
                embedding_array = self.embedding_validator._parse_embedding_string(embedding_str)
                if embedding_array is not None:
                    diversity_score = self.embedding_validator._calculate_embedding_diversity(embedding_array)
                    if diversity_score < self.embedding_validator.diversity_threshold:
                        if doc_id not in low_diversity_doc_ids:
                            low_diversity_doc_ids.append(doc_id)
            
            logger.info(f"Found {len(low_diversity_doc_ids)} documents with low diversity embeddings")
            return low_diversity_doc_ids
            
        except Exception as e:
            logger.error(f"Error identifying documents with low diversity embeddings: {e}")
            return []
    
    def get_documents_without_embeddings(self) -> List[str]:
        """
        Get document IDs that have no token embeddings.
        
        Returns:
            List of document IDs without embeddings
        """
        try:
            iris_connector = self.connection_manager.get_connection("iris")
            cursor = iris_connector.cursor()
            
            # Find documents in SourceDocuments that don't have token embeddings
            cursor.execute("""
                SELECT sd.id
                FROM RAG.SourceDocuments sd
                LEFT JOIN RAG.DocumentTokenEmbeddings dte ON sd.id = dte.doc_id
                WHERE dte.doc_id IS NULL
            """)
            
            results = cursor.fetchall()
            doc_ids = [row[0] for row in results]
            cursor.close()
            
            logger.info(f"Found {len(doc_ids)} documents without token embeddings")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error identifying documents without embeddings: {e}")
            return []
    
    def get_documents_with_incomplete_embeddings(self, min_embeddings_threshold: int) -> List[str]:
        """
        Get document IDs that have incomplete token embeddings.
        
        Args:
            min_embeddings_threshold: Minimum number of embeddings required per document
            
        Returns:
            List of document IDs with incomplete embeddings
        """
        try:
            iris_connector = self.connection_manager.get_connection("iris")
            cursor = iris_connector.cursor()
            
            # Find documents that have some but not complete token embeddings
            cursor.execute(f"""
                SELECT doc_id
                FROM (
                    SELECT sd.id as doc_id, COUNT(dte.id) as embedding_count
                    FROM RAG.SourceDocuments sd
                    JOIN RAG.DocumentTokenEmbeddings dte ON sd.id = dte.doc_id
                    GROUP BY sd.id
                    HAVING COUNT(dte.id) > 0 AND COUNT(dte.id) < {min_embeddings_threshold}
                ) AS subquery
            """)
            
            results = cursor.fetchall()
            doc_ids = [row[0] for row in results]
            cursor.close()
            
            logger.info(f"Found {len(doc_ids)} documents with incomplete token embeddings")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error identifying documents with incomplete embeddings: {e}")
            return []
    
    def save_document_embeddings(self, doc_id: str, tokens: List[str], 
                                embeddings: List[List[float]], target_dimension: int) -> bool:
        """
        Save document embeddings to the database.
        
        Args:
            doc_id: Document ID
            tokens: List of token texts
            embeddings: List of embedding vectors
            target_dimension: Target dimension for embeddings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            iris_connector = self.connection_manager.get_connection("iris")
            cursor = iris_connector.cursor()
            
            # Clear existing token embeddings for this document
            cursor.execute(
                "DELETE FROM RAG.DocumentTokenEmbeddings WHERE doc_id = ?",
                [doc_id]
            )
            
            # Store token embeddings using the utility function
            for token_index, (token_text, embedding_vector) in enumerate(zip(tokens, embeddings)):
                if not isinstance(embedding_vector, list):
                    logger.error(f"Embedding for token '{token_text}' in doc '{doc_id}' is not a list: {type(embedding_vector)}. Skipping.")
                    continue

                key_cols = {"doc_id": doc_id, "token_index": token_index}
                additional_cols = {"token_text": token_text}
                
                success = insert_vector(
                    cursor=cursor,
                    table_name="RAG.DocumentTokenEmbeddings",
                    vector_column_name="token_embedding",
                    vector_data=embedding_vector,
                    target_dimension=target_dimension,
                    key_columns=key_cols,
                    additional_data=additional_cols
                )
                
                if not success:
                    logger.error(f"Failed to insert token embedding for doc '{doc_id}', token_index {token_index}")
                    cursor.close()
                    return False
            
            cursor.close()
            iris_connector.commit()
            logger.debug(f"Successfully saved {len(embeddings)} embeddings for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving document embeddings for doc {doc_id}: {e}")
            return False