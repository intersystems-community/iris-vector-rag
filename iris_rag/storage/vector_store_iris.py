"""
IRIS-specific implementation of the VectorStore abstract base class.

This module provides a concrete implementation of the VectorStore interface
for InterSystems IRIS, including CLOB handling and vector search capabilities.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple

from ..core.vector_store import VectorStore
from ..core.models import Document
from ..core.connection import ConnectionManager
from ..config.manager import ConfigurationManager
from ..core.vector_store_exceptions import (
    VectorStoreError,
    VectorStoreConnectionError,
    VectorStoreDataError,
    VectorStoreCLOBError,
    VectorStoreConfigurationError
)
from .clob_handler import ensure_string_content

logger = logging.getLogger(__name__)


class IRISVectorStore(VectorStore):
    """
    IRIS-specific implementation of the VectorStore interface.
    
    This class provides vector storage and retrieval capabilities using
    InterSystems IRIS as the backend database, with proper CLOB handling
    to ensure all returned content is in string format.
    """
    
    def __init__(self, connection_manager: ConnectionManager, config_manager: ConfigurationManager):
        """
        Initialize IRIS vector store with connection and configuration managers.
        
        Args:
            connection_manager: Manager for database connections
            config_manager: Manager for configuration settings
            
        Raises:
            VectorStoreConnectionError: If connection cannot be established
            VectorStoreConfigurationError: If configuration is invalid
        """
        self.connection_manager = connection_manager
        self.config_manager = config_manager
        self._connection = None
        
        # Get storage configuration
        self.storage_config = self.config_manager.get("storage:iris", {})
        self.table_name = self.storage_config.get("table_name", "RAG.SourceDocuments")
        self.vector_dimension = self.storage_config.get("vector_dimension", 384)
        
        # Validate table name for security
        self._validate_table_name(self.table_name)
        
        # Define allowed filter keys for security
        self._allowed_filter_keys = {
            "category", "year", "source_type", "document_id", "author_name",
            "title", "source", "type", "date", "status", "version", "pmcid",
            "journal", "doi", "publication_date", "keywords", "abstract_type"
        }
        
        # Test connection on initialization
        try:
            self._get_connection()
        except Exception as e:
            raise VectorStoreConnectionError(f"Failed to initialize IRIS connection: {e}")
    
    def _get_connection(self):
        """Get or create database connection."""
        if self._connection is None:
            try:
                self._connection = self.connection_manager.get_connection("iris")
            except Exception as e:
                raise VectorStoreConnectionError(f"Failed to get IRIS connection: {e}")
        return self._connection
    
    def _validate_table_name(self, table_name: str) -> None:
        """
        Validate table name against whitelist to prevent SQL injection.
        
        Args:
            table_name: The table name to validate
            
        Raises:
            VectorStoreConfigurationError: If table name is not in whitelist
        """
        allowed_tables = {
            "RAG.SourceDocuments",
            "RAG.DocumentTokenEmbeddings",
            "RAG.TestDocuments",
            "RAG.BackupDocuments"
        }
        
        if table_name not in allowed_tables:
            logger.error(f"Security violation: Invalid table name attempted: {table_name}")
            raise VectorStoreConfigurationError(f"Invalid table name: {table_name}")
    
    def _validate_filter_keys(self, filter_dict: Dict[str, Any]) -> None:
        """
        Validate filter keys against whitelist to prevent SQL injection.
        
        Args:
            filter_dict: Dictionary of filter key-value pairs
            
        Raises:
            VectorStoreDataError: If any filter key is not in whitelist
        """
        for key in filter_dict.keys():
            if key not in self._allowed_filter_keys:
                logger.warning(f"Security violation: Invalid filter key attempted: {key}")
                raise VectorStoreDataError(f"Invalid filter key: {key}")
    
    def _validate_filter_values(self, filter_dict: Dict[str, Any]) -> None:
        """
        Validate filter values for basic type safety.
        
        Args:
            filter_dict: Dictionary of filter key-value pairs
            
        Raises:
            VectorStoreDataError: If any filter value has invalid type
        """
        for key, value in filter_dict.items():
            if value is None or callable(value) or isinstance(value, (list, dict)):
                logger.warning(f"Security violation: Invalid filter value type for key '{key}': {type(value).__name__}")
                raise VectorStoreDataError(f"Invalid filter value for key '{key}': {type(value).__name__}")
    
    def _sanitize_error_message(self, error: Exception, operation: str) -> str:
        """
        Sanitize error messages to prevent information disclosure.
        
        Args:
            error: The original exception
            operation: Description of the operation that failed
            
        Returns:
            Sanitized error message safe for logging
        """
        # Log full error details at debug level only
        logger.debug(f"Full error details for {operation}: {str(error)}")
        
        # Return generic error message for higher log levels
        error_type = type(error).__name__
        return f"Database operation failed during {operation}: {error_type}"
    
    def _ensure_string_content(self, document_data: Dict[str, Any]) -> Document:
        """
        Process raw database row to ensure string content and create Document object.
        
        Args:
            document_data: Raw data from database query
            
        Returns:
            Document object with guaranteed string content
            
        Raises:
            VectorStoreCLOBError: If CLOB conversion fails
        """
        try:
            processed_data = ensure_string_content(document_data)
            
            # Parse metadata if it's a JSON string
            metadata = processed_data.get('metadata', {})
            if isinstance(metadata, str) and metadata:
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse metadata JSON: {metadata}")
                    metadata = {"raw_metadata": metadata}
            elif not isinstance(metadata, dict):
                metadata = {}
            
            return Document(
                id=processed_data.get('id', processed_data.get('doc_id', '')),
                page_content=processed_data.get('page_content', ''),
                metadata=metadata
            )
        except Exception as e:
            raise VectorStoreCLOBError(f"Failed to process document data: {e}")
    
    def add_documents(
        self, 
        documents: List[Document], 
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Add documents to the IRIS vector store.
        
        Args:
            documents: List of Document objects to add
            embeddings: Optional pre-computed embeddings for the documents
        
        Returns:
            List of document IDs that were added
            
        Raises:
            VectorStoreDataError: If document data is malformed
            VectorStoreConnectionError: If there are connection issues
        """
        if not documents:
            return []
        
        if embeddings and len(embeddings) != len(documents):
            raise VectorStoreDataError("Number of embeddings must match number of documents")
        
        # Validate documents
        for doc in documents:
            if not isinstance(doc.page_content, str):
                raise VectorStoreDataError("Document page_content must be a string")
        
        connection = self._get_connection()
        cursor = connection.cursor()
        
        try:
            added_ids = []
            for i, doc in enumerate(documents):
                metadata_json = json.dumps(doc.metadata)
                
                # Check if document exists
                check_sql = f"SELECT COUNT(*) FROM {self.table_name} WHERE id = ?"
                cursor.execute(check_sql, [doc.id])
                exists = cursor.fetchone()[0] > 0
                
                if exists:
                    # Update existing document
                    if embeddings:
                        update_sql = f"""
                        UPDATE {self.table_name}
                        SET text_content = ?, metadata = ?, embedding = TO_VECTOR(?)
                        WHERE id = ?
                        """
                        embedding_str = json.dumps(embeddings[i])
                        cursor.execute(update_sql, [doc.page_content, metadata_json, embedding_str, doc.id])
                    else:
                        update_sql = f"""
                        UPDATE {self.table_name}
                        SET text_content = ?, metadata = ?
                        WHERE id = ?
                        """
                        cursor.execute(update_sql, [doc.page_content, metadata_json, doc.id])
                else:
                    # Insert new document
                    if embeddings:
                        insert_sql = f"""
                        INSERT INTO {self.table_name} (id, text_content, metadata, embedding)
                        VALUES (?, ?, ?, TO_VECTOR(?))
                        """
                        embedding_str = json.dumps(embeddings[i])
                        cursor.execute(insert_sql, [doc.id, doc.page_content, metadata_json, embedding_str])
                    else:
                        insert_sql = f"""
                        INSERT INTO {self.table_name} (id, text_content, metadata)
                        VALUES (?, ?, ?)
                        """
                        cursor.execute(insert_sql, [doc.id, doc.page_content, metadata_json])
                
                added_ids.append(doc.id)
            
            connection.commit()
            logger.info(f"Added {len(added_ids)} documents to {self.table_name}")
            return added_ids
            
        except Exception as e:
            connection.rollback()
            sanitized_error = self._sanitize_error_message(e, "add_documents")
            logger.error(sanitized_error)
            raise VectorStoreDataError(f"Failed to add documents: {sanitized_error}")
        finally:
            cursor.close()
    
    def delete_documents(self, ids: List[str]) -> bool:
        """
        Delete documents from the IRIS vector store by their IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if any documents were deleted, False otherwise
        """
        if not ids:
            return True
        
        connection = self._get_connection()
        cursor = connection.cursor()
        
        try:
            placeholders = ','.join(['?' for _ in ids])
            delete_sql = f"DELETE FROM {self.table_name} WHERE id IN ({placeholders})"
            cursor.execute(delete_sql, ids)
            
            deleted_count = cursor.rowcount
            connection.commit()
            
            logger.info(f"Deleted {deleted_count} documents from {self.table_name}")
            return deleted_count > 0
            
        except Exception as e:
            connection.rollback()
            sanitized_error = self._sanitize_error_message(e, "delete_documents")
            logger.error(sanitized_error)
            raise VectorStoreConnectionError(f"Failed to delete documents: {sanitized_error}")
        finally:
            cursor.close()
    
    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search using a query embedding.
        
        Args:
            query_embedding: The query vector for similarity search
            top_k: Maximum number of results to return
            filter: Optional metadata filters to apply
            
        Returns:
            List of tuples containing (Document, similarity_score)
        """
        # Validate filter if provided
        if filter:
            self._validate_filter_keys(filter)
            self._validate_filter_values(filter)
        
        connection = self._get_connection()
        cursor = connection.cursor()
        
        try:
            # Build base query - table name is already validated in __init__
            base_sql = f"""
            SELECT TOP {top_k} id, text_content, metadata,
                   VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?)) as similarity_score
            FROM {self.table_name}
            WHERE embedding IS NOT NULL
            """
            
            params = [json.dumps(query_embedding)]
            
            # Add metadata filters if provided - keys are now validated
            if filter:
                filter_conditions = []
                for key, value in filter.items():
                    # Key is already validated, safe to use in f-string
                    filter_conditions.append(f"JSON_EXTRACT(metadata, '$.{key}') = ?")
                    params.append(str(value))
                
                if filter_conditions:
                    base_sql += " AND " + " AND ".join(filter_conditions)
            
            # Order by similarity score descending
            base_sql += " ORDER BY similarity_score DESC"
            
            cursor.execute(base_sql, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                doc_id, text_content, metadata_json, similarity_score = row
                
                # Process row data to ensure string content
                document_data = {
                    'id': doc_id,
                    'text_content': text_content,
                    'metadata': metadata_json
                }
                
                document = self._ensure_string_content(document_data)
                results.append((document, float(similarity_score)))
            
            logger.debug(f"Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            sanitized_error = self._sanitize_error_message(e, "similarity_search")
            logger.error(sanitized_error)
            raise VectorStoreConnectionError(f"Vector search failed: {sanitized_error}")
        finally:
            cursor.close()
    
    def fetch_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """
        Fetch documents by their IDs.
        
        Args:
            ids: List of document IDs to fetch
            
        Returns:
            List of Document objects with guaranteed string content
        """
        if not ids:
            return []
        
        connection = self._get_connection()
        cursor = connection.cursor()
        
        try:
            placeholders = ','.join(['?' for _ in ids])
            select_sql = f"""
            SELECT id, text_content, metadata
            FROM {self.table_name}
            WHERE id IN ({placeholders})
            """
            
            cursor.execute(select_sql, ids)
            rows = cursor.fetchall()
            
            documents = []
            for row in rows:
                doc_id, text_content, metadata_json = row
                
                # Process row data to ensure string content
                document_data = {
                    'id': doc_id,
                    'text_content': text_content,
                    'metadata': metadata_json
                }
                
                document = self._ensure_string_content(document_data)
                documents.append(document)
            
            logger.debug(f"Fetched {len(documents)} documents by IDs")
            return documents
            
        except Exception as e:
            sanitized_error = self._sanitize_error_message(e, "fetch_documents_by_ids")
            logger.error(sanitized_error)
            raise VectorStoreConnectionError(f"Failed to fetch documents by IDs: {sanitized_error}")
        finally:
            cursor.close()
    
    def get_document_count(self) -> int:
        """
        Get the total number of documents in the vector store.
        
        Returns:
            Total number of documents
        """
        connection = self._get_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            count = cursor.fetchone()[0]
            return int(count)
        except Exception as e:
            sanitized_error = self._sanitize_error_message(e, "get_document_count")
            logger.error(sanitized_error)
            raise VectorStoreConnectionError(f"Failed to get document count: {sanitized_error}")
        finally:
            cursor.close()
    
    def clear_documents(self) -> None:
        """
        Clear all documents from the vector store.
        
        Warning: This operation is irreversible.
        """
        connection = self._get_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute(f"DELETE FROM {self.table_name}")
            connection.commit()
            logger.info(f"Cleared all documents from {self.table_name}")
        except Exception as e:
            connection.rollback()
            sanitized_error = self._sanitize_error_message(e, "clear_documents")
            logger.error(sanitized_error)
            raise VectorStoreConnectionError(f"Failed to clear documents: {sanitized_error}")
        finally:
            cursor.close()