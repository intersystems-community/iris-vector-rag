"""
IRIS-specific implementation of the VectorStore abstract base class.

This module provides a concrete implementation of the VectorStore interface
for InterSystems IRIS, including CLOB handling and vector search capabilities.
"""

import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from ..core.vector_store import VectorStore
from ..core.models import Document
from common.iris_connection_manager import get_iris_connection
# Removed circular import - connection sharing handled via dependency injection
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
    
    def __init__(self, config_manager: ConfigurationManager, schema_manager=None, connection_manager=None, embedding_manager=None):
        """
        Initialize IRIS vector store with configuration and connection managers.
        
        Args:
            config_manager: Manager for configuration settings
            schema_manager: Optional SchemaManager instance
            connection_manager: Optional IRISConnectionManager instance
            embedding_manager: Optional EmbeddingManager instance for automatic embedding generation
            
        Raises:
            VectorStoreConnectionError: If connection cannot be established
            VectorStoreConfigurationError: If configuration is invalid
        """
        from common.iris_connection_manager import IRISConnectionManager

        self.config_manager = config_manager
        
        # Use shared connection for transaction consistency
        if connection_manager:
            self.connection_manager = connection_manager
            self._connection = self.connection_manager.get_connection()
        else:
            # Create new connection using standard connection manager
            self._connection = get_iris_connection()
            # Create a mock connection manager for compatibility
            self.connection_manager = type('ConnectionManager', (), {
                'get_connection': lambda: self._connection
            })()
        
        self.embedding_manager = embedding_manager
        
        # Get storage configuration
        self.storage_config = self.config_manager.get("storage:iris", {})
        self.table_name = self.storage_config.get("table_name", "RAG.SourceDocuments")
        
        # Get chunking configuration
        self.chunking_config = self.config_manager.get("storage:chunking", {})
        self.auto_chunk = self.chunking_config.get("enabled", False)
        
        # Initialize chunking service if enabled
        self.chunking_service = None
        if self.auto_chunk:
            try:
                from tools.chunking.chunking_service import DocumentChunkingService
                self.chunking_service = DocumentChunkingService()
                logger.info(f"Chunking service initialized with strategy: {self.chunking_config.get('strategy', 'fixed_size')}")
            except ImportError as e:
                logger.warning(f"Could not import DocumentChunkingService: {e}")
                self.auto_chunk = False
        
        # Get schema info from schema manager (single source of truth)
        if schema_manager:
            self.schema_manager = schema_manager
        else:
            from .schema_manager import SchemaManager
            self.schema_manager = SchemaManager(self.connection_manager, config_manager)
            
        table_short_name = self.table_name.replace("RAG.", "")
        
        table_config = self.schema_manager.get_table_config(table_short_name)
        if not table_config:
            raise VectorStoreConfigurationError(f"No schema configuration found for table: {table_short_name}")
            
        columns = table_config.get("columns", {})
        self.vector_dimension = table_config.get("dimension")
        self.id_column = columns.get("id", "ID")
        self.embedding_column = columns.get("embedding", "embedding")
        self.content_column = columns.get("content", "TEXT_CONTENT")
        
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
                self._connection = self.connection_manager.get_connection()
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
                page_content=processed_data.get('text_content', processed_data.get('page_content', '')),
                metadata=metadata,
                id=processed_data.get('doc_id', processed_data.get('id', ''))
            )
        except Exception as e:
            raise VectorStoreCLOBError(f"Failed to process document data: {e}")
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        auto_chunk: Optional[bool] = None,
        chunking_strategy: Optional[str] = None
    ) -> List[str]:
        """
        Add documents to the IRIS vector store with optional automatic chunking.
        
        Args:
            documents: List of Document objects to add
            embeddings: Optional pre-computed embeddings for the documents
            auto_chunk: Override automatic chunking setting for this operation
            chunking_strategy: Override chunking strategy for this operation
        
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
        
        # Determine if we should chunk documents
        should_chunk = auto_chunk if auto_chunk is not None else self.auto_chunk
        
        # If chunking is enabled, process documents through chunking service
        if should_chunk and self.chunking_service:
            return self._add_documents_with_chunking(
                documents, embeddings, chunking_strategy
            )
        else:
            # Use original add_documents logic
            return self._add_documents_direct(documents, embeddings)
    
    def _add_documents_with_chunking(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        chunking_strategy: Optional[str] = None
    ) -> List[str]:
        """
        Add documents with automatic chunking enabled.
        
        Args:
            documents: List of Document objects to chunk and add
            embeddings: Optional pre-computed embeddings (will be used if available)
            chunking_strategy: Strategy to use for chunking
            
        Returns:
            List of chunk IDs that were added
        """
        strategy = chunking_strategy or self.chunking_config.get("strategy", "fixed_size")
        threshold = self.chunking_config.get("threshold", 1000)
        
        all_chunk_ids = []
        
        for i, doc in enumerate(documents):
            # Get embedding for this document if available
            doc_embedding = embeddings[i] if embeddings and i < len(embeddings) else None
            
            # Check if document needs chunking
            if len(doc.page_content) <= threshold:
                # Document is small enough, add directly
                if doc_embedding:
                    chunk_ids = self._add_documents_direct([doc], [doc_embedding])
                elif hasattr(self, 'embedding_manager') and self.embedding_manager:
                    generated_embedding = self.embedding_manager.embed_text(doc.page_content)
                    chunk_ids = self._add_documents_direct([doc], [generated_embedding])
                else:
                    # Fall back to adding without embeddings
                    logger.warning(f"No embeddings available for document {doc.id}, adding without embeddings")
                    chunk_ids = self._add_documents_direct([doc], None)
                all_chunk_ids.extend(chunk_ids)
            else:
                # Document needs chunking
                try:
                    chunk_records = self.chunking_service.chunk_document(
                        doc_id=doc.id,
                        text=doc.page_content,
                        strategy_name=strategy
                    )
                    
                    # Convert chunk records to Document objects
                    chunk_documents = []
                    for chunk_record in chunk_records:
                        chunk_metadata = doc.metadata.copy()
                        chunk_metadata.update({
                            "chunk_id": chunk_record["chunk_id"],
                            "chunk_index": chunk_record["chunk_index"],
                            "chunk_type": chunk_record["chunk_type"],
                            "original_doc_id": doc.id,
                            "is_chunk": True
                        })
                        
                        chunk_doc = Document(
                            page_content=chunk_record["chunk_text"],
                            metadata=chunk_metadata,
                            id=chunk_record["chunk_id"]
                        )
                        chunk_documents.append(chunk_doc)
                    
                    # Generate embeddings for chunk documents
                    if hasattr(self, 'embedding_manager') and self.embedding_manager:
                        chunk_texts = [chunk_doc.page_content for chunk_doc in chunk_documents]
                        chunk_embeddings = self.embedding_manager.embed_texts(chunk_texts)
                        chunk_ids = self._add_documents_direct(chunk_documents, chunk_embeddings)
                    elif doc_embedding:
                        # Use the original document embedding for all chunks (not ideal but better than no embeddings)
                        logger.warning(f"Using original document embedding for all chunks of {doc.id}")
                        chunk_embeddings = [doc_embedding] * len(chunk_documents)
                        chunk_ids = self._add_documents_direct(chunk_documents, chunk_embeddings)
                    else:
                        # Fall back to adding without embeddings
                        logger.warning(f"No embeddings available for chunks of document {doc.id}, adding without embeddings")
                        chunk_ids = self._add_documents_direct(chunk_documents, None)
                    all_chunk_ids.extend(chunk_ids)
                    
                    logger.info(f"Document {doc.id} chunked into {len(chunk_documents)} chunks using {strategy} strategy")
                    
                except Exception as e:
                    logger.error(f"Failed to chunk document {doc.id}: {e}")
                    # Fall back to adding the original document
                    if doc_embedding:
                        chunk_ids = self._add_documents_direct([doc], [doc_embedding])
                    elif hasattr(self, 'embedding_manager') and self.embedding_manager:
                        generated_embedding = self.embedding_manager.embed_text(doc.page_content)
                        chunk_ids = self._add_documents_direct([doc], [generated_embedding])
                    else:
                        # Fall back to adding without embeddings
                        logger.warning(f"No embeddings available for document {doc.id}, adding without embeddings")
                        chunk_ids = self._add_documents_direct([doc], None)
                    all_chunk_ids.extend(chunk_ids)
        
        return all_chunk_ids
    
    def _add_documents_direct(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Add documents directly without chunking (original implementation).
        
        Args:
            documents: List of Document objects to add
            embeddings: Optional pre-computed embeddings for the documents
            
        Returns:
            List of document IDs that were added
        """
        
        connection = self._get_connection()
        cursor = connection.cursor()
        
        try:
            added_ids = []
            for i, doc in enumerate(documents):
                metadata_json = json.dumps(doc.metadata)
                
                # Check if document exists using the correct ID column
                check_sql = f"SELECT COUNT(*) FROM {self.table_name} WHERE {self.id_column} = ?"
                cursor.execute(check_sql, [doc.id])
                exists = cursor.fetchone()[0] > 0
                
                if exists:
                    # Update existing document
                    if embeddings:
                        update_sql = f"""
                        UPDATE {self.table_name}
                        SET {self.content_column} = ?, metadata = ?, {self.embedding_column} = TO_VECTOR(?)
                        WHERE {self.id_column} = ?
                        """
                        embedding_str = json.dumps(embeddings[i])
                        cursor.execute(update_sql, [doc.page_content, metadata_json, embedding_str, doc.id])
                    else:
                        update_sql = f"""
                        UPDATE {self.table_name}
                        SET {self.content_column} = ?, metadata = ?
                        WHERE {self.id_column} = ?
                        """
                        cursor.execute(update_sql, [doc.page_content, metadata_json, doc.id])
                else:
                    # Insert new document with explicit ID
                    if embeddings:
                        insert_sql = f"""
                        INSERT INTO {self.table_name} ({self.id_column}, {self.content_column}, metadata, {self.embedding_column})
                        VALUES (?, ?, ?, TO_VECTOR(?))
                        """
                        embedding_str = json.dumps(embeddings[i])
                        cursor.execute(insert_sql, [doc.id, doc.page_content, metadata_json, embedding_str])
                    else:
                        insert_sql = f"""
                        INSERT INTO {self.table_name} ({self.id_column}, {self.content_column}, metadata)
                        VALUES (?, ?, ?)
                        """
                        cursor.execute(insert_sql, [doc.id, doc.page_content, metadata_json])
                
                added_ids.append(doc.id)
            
            # With autocommit enabled, no need for explicit commit
            logger.info(f"Added {len(added_ids)} documents to {self.table_name}")
            return added_ids
            
        except Exception as e:
            # With autocommit enabled, no need for explicit rollback
            sanitized_error = self._sanitize_error_message(e, "add_documents")
            print(e)
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
    
    def similarity_search_by_embedding(
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
            # Use the driver-aware vector search utility
            from common.vector_sql_utils import execute_driver_aware_vector_search
            
            # Format embedding as bracket-delimited string for IRIS
            embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Build metadata filter clause if needed
            additional_where = None
            if filter:
                filter_conditions = []
                for key, value in filter.items():
                    # Key is already validated, safe to use in f-string
                    filter_conditions.append(f"JSON_EXTRACT(metadata, '$.{key}') = '{str(value)}'")
                
                if filter_conditions:
                    additional_where = " AND ".join(filter_conditions)
            
            # Get current vector dimension from schema manager
            table_short_name = self.table_name.replace("RAG.", "")
            expected_dimension = self.schema_manager.get_vector_dimension(table_short_name)
            
            # Validate query embedding dimension matches expected
            if len(query_embedding) != expected_dimension:
                error_msg = f"Query embedding dimension {len(query_embedding)} doesn't match expected {expected_dimension} for table {table_short_name}"
                logger.error(error_msg)
                raise VectorStoreDataError(error_msg)
            
            logger.debug(f"Vector search: query={len(query_embedding)}D, expected={expected_dimension}D, table={table_short_name}")
            
            # Execute the search using the driver-aware utility
            # Try to get driver type from connection manager, fall back to auto-detection
            driver_type = None
            if hasattr(self.connection_manager, 'get_driver_type'):
                try:
                    driver_type = self.connection_manager.get_driver_type()
                except Exception as e:
                    logger.debug(f"Failed to get driver type from connection manager: {e}")
                    driver_type = None
            
            rows = execute_driver_aware_vector_search(
                cursor=cursor,
                table_name=self.table_name,
                vector_column=self.embedding_column,
                vector_string=embedding_str,
                embedding_dim=expected_dimension,
                top_k=top_k,
                id_column=self.id_column,
                content_column=self.content_column,
                additional_where=additional_where,
                driver_type=driver_type  # Will auto-detect if None
            )
            
            # Now fetch metadata for the returned documents
            if rows:
                doc_ids = [row[0] for row in rows]
                placeholders = ','.join(['?' for _ in doc_ids])
                metadata_sql = f"SELECT ID, metadata FROM {self.table_name} WHERE ID IN ({placeholders})"
                cursor.execute(metadata_sql, doc_ids)
                metadata_map = {row[0]: row[1] for row in cursor.fetchall()}
            
            results = []
            for row in rows:
                doc_id, text_content, similarity_score = row
                
                # Get metadata from the map
                metadata_json = metadata_map.get(doc_id, None) if 'metadata_map' in locals() else None
                
                # Process row data to ensure string content
                document_data = {
                    'doc_id': doc_id,
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
            SELECT {self.id_column}, {self.content_column}, metadata
            FROM {self.table_name}
            WHERE {self.id_column} IN ({placeholders})
            """
            
            cursor.execute(select_sql, ids)
            rows = cursor.fetchall()
            
            documents = []
            for row in rows:
                doc_id, text_content, metadata_json = row
                
                # Process row data to ensure string content
                document_data = {
                    'doc_id': doc_id,
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
    
    # Implementation of abstract base class method for compatibility
    def similarity_search(self, *args, **kwargs):
        """
        Overloaded similarity_search method that handles both:
        1. Base class signature: similarity_search(query_embedding, top_k, filter)
        2. LangChain signature: similarity_search(query, k, filter)
        """
        # Check if first argument is a string (LangChain interface)
        if args and isinstance(args[0], str):
            # LangChain interface: similarity_search(query, k, filter)
            query = args[0]
            k = args[1] if len(args) > 1 else kwargs.get('k', 4)
            filter_param = args[2] if len(args) > 2 else kwargs.get('filter')
            
            # Get embedding function for text query
            embedding_func = kwargs.get('embedding_func')
            if not embedding_func:
                from ..embeddings.manager import EmbeddingManager
                embedding_manager = EmbeddingManager(self.config_manager)
                query_embedding = embedding_manager.embed_text(query)
            else:
                query_embedding = embedding_func.embed_query(query)
            
            # Call our implementation and return just documents for LangChain
            results = self.similarity_search_by_embedding(query_embedding, k, filter_param)
            return [doc for doc, score in results]
            
        else:
            # Base class interface: similarity_search(query_embedding, top_k, filter)
            query_embedding = args[0] if args else kwargs['query_embedding']
            top_k = args[1] if len(args) > 1 else kwargs.get('top_k', 5)
            filter_param = args[2] if len(args) > 2 else kwargs.get('filter')
            
            return self.similarity_search_by_embedding(query_embedding, top_k, filter_param)
    
    # =============================================================================
    # LangChain VectorStore Compatibility Interface
    # =============================================================================
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[str]:
        """
        Add texts to the vector store (LangChain interface).
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs
            **kwargs: Additional arguments
            
        Returns:
            List of document IDs that were added
        """
        # Convert texts to Document objects
        documents = []
        for i, text in enumerate(texts):
            # Use provided ID or generate one
            doc_id = ids[i] if ids and i < len(ids) else f"doc_{i}_{hash(text) % 1000000}"
            
            # Use provided metadata or empty dict
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            
            # Create Document object using positional arguments
            doc = Document(
                text,  # page_content (positional)
                metadata,  # metadata (positional)
                doc_id  # id (positional)
            )
            documents.append(doc)
        
        # Use our existing add_documents method
        return self.add_documents(documents)
    
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores (LangChain interface).
        
        Args:
            query: Query text to search for
            k: Number of results to return
            filter: Optional metadata filters
            **kwargs: Additional arguments
            
        Returns:
            List of tuples containing (Document, similarity_score)
        """
        # We need an embedding function - get it from kwargs or use default
        embedding_func = kwargs.get('embedding_func')
        if not embedding_func:
            # Import default embedding function
            from ..embeddings.manager import EmbeddingManager
            embedding_manager = EmbeddingManager(self.config_manager)
            query_embedding = embedding_manager.embed_text(query)
        else:
            query_embedding = embedding_func.embed_query(query)
        
        # Use our existing similarity_search method (returns tuples)
        return self.similarity_search_by_vector(query_embedding, k, filter)
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search using embedding vector (LangChain interface).
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            filter: Optional metadata filters
            **kwargs: Additional arguments
            
        Returns:
            List of tuples containing (Document, similarity_score)
        """
        # Use our existing similarity_search method
        return self.similarity_search_by_embedding(
            query_embedding=embedding,
            top_k=k,
            filter=filter
        )
    
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """
        Delete documents by IDs (LangChain interface).
        
        Args:
            ids: List of document IDs to delete
            **kwargs: Additional arguments
            
        Returns:
            True if deletion was successful, False otherwise
        """
        if not ids:
            return True
        
        try:
            # Use our existing delete functionality
            self.delete_documents(ids)
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents {ids}: {e}")
            return False
    
    # =============================================================================
    # Advanced RAG Methods (RAG Templates Extensions)
    # =============================================================================
    
    def colbert_search(
        self,
        query_token_embeddings: List[List[float]],
        k: int = 5,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Perform real ColBERT-style search with token-level embeddings.
        
        This implements the proper ColBERT approach:
        1. Generate document-level query embedding for candidate retrieval (384D)
        2. Load token embeddings for candidate documents (768D) 
        3. Calculate MaxSim scores between query tokens and document tokens
        4. Return top-k documents ranked by MaxSim scores
        
        Args:
            query_token_embeddings: List of 768D token embeddings for the query
            k: Number of results to return
            **kwargs: Additional arguments
            
        Returns:
            List of tuples containing (Document, maxsim_score)
        """
        import numpy as np
        
        # Step 1: Get document-level query embedding for candidate retrieval
        # We need a 384D embedding for SourceDocuments search, NOT averaged token embeddings
        query_text = kwargs.get('query_text', '')
        if not query_text:
            # If no query text provided, we can't generate proper document embedding
            # Fall back to averaging (this shouldn't happen in normal usage)
            logger.warning("ColBERT: No query_text provided, falling back to token averaging")
            query_token_array = np.array(query_token_embeddings)
            avg_embedding = np.mean(query_token_array, axis=0).tolist()
            # This will still cause dimension mismatch - need proper 384D embedding
        else:
            # Generate proper 384D document-level embedding for candidate retrieval
            from ..embeddings.manager import EmbeddingManager
            embedding_manager = EmbeddingManager(self.config_manager)
            avg_embedding = embedding_manager.embed_text(query_text)
        
        # Validate that document embedding is correct dimension
        doc_dim = self.schema_manager.get_vector_dimension("SourceDocuments")
        if len(avg_embedding) != doc_dim:
            error_msg = f"ColBERT: Document embedding dimension {len(avg_embedding)} doesn't match expected {doc_dim}"
            logger.error(error_msg)
            raise VectorStoreDataError(error_msg)
        
        logger.debug(f"ColBERT: Using {len(avg_embedding)}D document embedding for candidate retrieval")
        
        # Step 2: Get candidate documents using document-level search
        candidates = self.similarity_search_by_embedding(
            query_embedding=avg_embedding,
            top_k=k * 3,  # Get more candidates for MaxSim reranking
            filter=kwargs.get('filter')
        )
        
        if not candidates:
            return []
        
        # Step 3: Load token embeddings for candidate documents
        candidate_ids = [doc.id for doc, _ in candidates]
        token_embeddings_map = self._load_token_embeddings_for_candidates(candidate_ids)
        
        if not token_embeddings_map:
            logger.warning("ColBERT: No token embeddings found for candidates, returning document-level results")
            # Return candidates with ColBERT metadata
            enhanced_results = []
            for doc, score in candidates[:k]:
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata['retrieval_method'] = 'colbert_no_tokens'
                doc.metadata['token_count'] = len(query_token_embeddings)
                enhanced_results.append((doc, score))
            return enhanced_results
        
        # Step 4: Calculate MaxSim scores for candidates with token embeddings
        maxsim_results = []
        query_token_array = np.array(query_token_embeddings)
        
        for doc, _ in candidates:
            doc_id = doc.id
            if doc_id not in token_embeddings_map:
                continue
                
            doc_token_embeddings = np.array(token_embeddings_map[doc_id])
            
            # Calculate MaxSim score using interface if available
            if hasattr(self, 'colbert_interface'):
                maxsim_score = self.colbert_interface.calculate_maxsim(
                    query_token_embeddings.tolist(), doc_token_embeddings.tolist()
                )
            else:
                maxsim_score = self._calculate_maxsim_score(query_token_array, doc_token_embeddings)
            maxsim_results.append((doc, maxsim_score))
        
        # Step 5: Sort by MaxSim score and return top-k
        maxsim_results.sort(key=lambda x: x[1], reverse=True)
        
        # Add ColBERT metadata to results
        enhanced_results = []
        for doc, score in maxsim_results[:k]:
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata['retrieval_method'] = 'colbert_real'
            doc.metadata['maxsim_score'] = float(score)
            doc.metadata['token_count'] = len(query_token_embeddings)
            enhanced_results.append((doc, score))
        
        logger.info(f"ColBERT: Processed {len(candidates)} candidates, calculated MaxSim for {len(maxsim_results)}, returning top {len(enhanced_results)}")
        return enhanced_results
    
    def _load_token_embeddings_for_candidates(self, candidate_ids: List[str]) -> Dict[str, List[List[float]]]:
        """
        Load token embeddings for candidate documents from DocumentTokenEmbeddings table.
        
        Args:
            candidate_ids: List of document IDs to load token embeddings for
            
        Returns:
            Dictionary mapping doc_id to list of token embeddings
        """
        if not candidate_ids:
            return {}
            
        connection = self._get_connection()
        cursor = connection.cursor()
        
        try:
            # Use doc_ids directly as strings (no need to convert to int)
            valid_ids = [str(doc_id) for doc_id in candidate_ids if doc_id]
            
            if not valid_ids:
                return {}
            
            placeholders = ','.join(['?' for _ in valid_ids])
            sql = f"""
                SELECT doc_id, token_index, token_embedding
                FROM RAG.DocumentTokenEmbeddings
                WHERE doc_id IN ({placeholders})
                ORDER BY doc_id, token_index
            """
            
            cursor.execute(sql, valid_ids)
            results = cursor.fetchall()
            
            # Group embeddings by document ID
            doc_embeddings_map = {}
            for row in results:
                doc_id, token_index, embedding_data = row
                doc_id_str = str(doc_id)
                
                # Parse embedding data
                if isinstance(embedding_data, str):
                    # Parse string representation
                    embedding = self._parse_embedding_string(embedding_data)
                else:
                    # Handle IRIS VECTOR type
                    from .clob_handler import convert_clob_to_string
                    embedding_str = convert_clob_to_string(embedding_data)
                    embedding = self._parse_embedding_string(embedding_str)
                
                if embedding is None:
                    continue
                
                if doc_id_str not in doc_embeddings_map:
                    doc_embeddings_map[doc_id_str] = []
                
                doc_embeddings_map[doc_id_str].append(embedding)
            
            logger.debug(f"ColBERT: Loaded token embeddings for {len(doc_embeddings_map)} documents")
            return doc_embeddings_map
            
        except Exception as e:
            logger.error(f"ColBERT: Error loading token embeddings: {e}")
            return {}
        finally:
            cursor.close()
    
    def _parse_embedding_string(self, embedding_str: str) -> Optional[List[float]]:
        """Parse embedding string into list of floats."""
        try:
            if embedding_str.startswith('[') and embedding_str.endswith(']'):
                # Bracket format: [1.0,2.0,3.0]
                return [float(x.strip()) for x in embedding_str[1:-1].split(',')]
            else:
                # Comma-separated format: 1.0,2.0,3.0
                return [float(x.strip()) for x in embedding_str.split(',')]
        except (ValueError, AttributeError):
            return None
    
    def _calculate_maxsim_score(self, query_token_embeddings: np.ndarray, doc_token_embeddings: np.ndarray) -> float:
        """
        Calculate MaxSim score between query and document token embeddings.
        
        Args:
            query_token_embeddings: Query token embeddings (Q_len, dim)
            doc_token_embeddings: Document token embeddings (D_len, dim)
            
        Returns:
            MaxSim score (float)
        """
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
    
    def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        k: int = 5,
        vector_weight: float = 0.6,
        ifind_weight: float = 0.4,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid vector + IFind text search.
        
        Args:
            query_embedding: Query vector for similarity search
            query_text: Query text for IFind search
            k: Number of results to return
            vector_weight: Weight for vector similarity scores
            ifind_weight: Weight for IFind text scores
            **kwargs: Additional arguments
            
        Returns:
            List of tuples containing (Document, hybrid_score)
        """
        # For now, simplify to just vector search until IFind is properly configured
        # This gets HybridIFind working with our reliable vector search
        
        # Get more candidates for fusion
        vector_results = self.similarity_search_by_embedding(
            query_embedding=query_embedding,
            top_k=k * 2,  # Get more for potential fusion
            filter=kwargs.get('filter')
        )
        
        # For now, just return vector results with hybrid metadata
        # Future: implement actual IFind text search and fusion
        enhanced_results = []
        for doc, score in vector_results[:k]:
            # Update metadata to indicate this was a hybrid search
            if doc.metadata is None:
                doc.metadata = {}
            doc.metadata['retrieval_method'] = 'hybrid_vector_only'
            doc.metadata['vector_weight'] = vector_weight
            doc.metadata['ifind_weight'] = ifind_weight
            
            # Weighted score (for now just the vector score)
            hybrid_score = score * vector_weight
            enhanced_results.append((doc, hybrid_score))
        
        return enhanced_results
    
    def graph_search(
        self,
        query_entities: List[str],
        k: int = 5,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Perform graph-based entity search.
        
        Args:
            query_entities: List of extracted entities from query
            k: Number of results to return
            **kwargs: Additional arguments
            
        Returns:
            List of tuples containing (Document, entity_match_score)
        """
        # TODO: GraphRAG already works, this is for future enhancement
        raise NotImplementedError("Graph search can be implemented for enhanced GraphRAG")