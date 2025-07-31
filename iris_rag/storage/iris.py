"""
IRIS-specific storage implementation for RAG templates.

This module provides storage operations specifically designed for InterSystems IRIS,
including document insertion, vector search, and metadata queries.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from ..core.models import Document
from ..core.connection import ConnectionManager
from ..config.manager import ConfigurationManager

logger = logging.getLogger(__name__)

def _convert_clob_to_string(value: Any) -> str:
    """
    Convert CLOB/IRISInputStream objects to strings.
    
    Args:
        value: The value to convert, potentially an IRISInputStream
        
    Returns:
        String representation of the value
    """
    if hasattr(value, 'read') and callable(getattr(value, 'read')):
        try:
            # This is likely an IRISInputStream (CLOB)
            stream_bytes = value.read()
            if isinstance(stream_bytes, bytes):
                return stream_bytes.decode('utf-8', errors='replace')
            else:
                return str(stream_bytes)
        except Exception as e:
            logger.warning(f"Could not read stream value: {e}")
            return "[Error Reading Stream]"
    elif isinstance(value, str):
        return value
    elif value is None:
        return ""
    else:
        return str(value)


def _process_db_row_for_document(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a database row dictionary to convert CLOB fields to strings.
    
    Args:
        row_dict: Dictionary representing a database row
        
    Returns:
        Processed dictionary with CLOB fields converted to strings
    """
    processed_row = {}
    for key, value in row_dict.items():
        processed_row[key] = _convert_clob_to_string(value)
    return processed_row


class IRISStorage:
    """
    Storage layer implementation for InterSystems IRIS.
    
    Provides document storage, retrieval, and vector search capabilities
    specifically optimized for IRIS database operations.
    """
    
    def __init__(self, connection_manager: ConnectionManager, config_manager: ConfigurationManager):
        """
        Initialize IRIS storage with connection and configuration managers.
        
        Args:
            connection_manager: Manager for database connections
            config_manager: Manager for configuration settings
        """
        self.connection_manager = connection_manager
        self.config_manager = config_manager
        self._connection = None
        
        # Get storage configuration
        self.storage_config = self.config_manager.get("storage:iris", {})
        self.table_name = self.storage_config.get("table_name", "RAG.SourceDocuments")
        self.vector_dimension = self.storage_config.get("vector_dimension", 384)
        
    def _get_connection(self):
        """Get or create database connection."""
        if self._connection is None:
            self._connection = self.connection_manager.get_connection("iris")
        return self._connection
    
    def initialize_schema(self) -> None:
        """
        Initialize the database schema for document storage.
        
        Creates the necessary tables and indexes if they don't exist.
        """
        connection = self._get_connection()
        cursor = connection.cursor()
        
        try:
            # Check if table exists and print columns for diagnostics
            try:
                required_columns = {
                    "DOC_ID": "VARCHAR(255)",
                    "TEXT_CONTENT": "LONGVARCHAR",
                    "METADATA": "LONGVARCHAR",
                    "EMBEDDING": f"VECTOR(DOUBLE, {self.vector_dimension})",
                }

                # Check existing columns
                existing_columns = []
                cursor.execute(f"SELECT * FROM {self.table_name} WHERE 1=0")
                existing_columns = [desc[0].upper() for desc in cursor.description]
                logger.info(f"Table {self.table_name} already exists. Columns: {existing_columns}")

                # Add any missing columns
                for col, definition in required_columns.items():
                    if col not in existing_columns:
                        try:
                            cursor.execute(f"ALTER TABLE {self.table_name} ADD {col} {definition}")
                            logger.info(f"Added missing column '{col}' to {self.table_name}")
                        except Exception as e:
                            logger.warning(f"Failed to add column '{col}': {e}")
            except Exception:
                logger.info(f"Table {self.table_name} does not exist or query failed, will attempt to create.")

            # Create main documents table
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id VARCHAR(255) PRIMARY KEY,
                doc_id VARCHAR(255),
                text_content LONGVARCHAR,
                metadata LONGVARCHAR,
                embedding VECTOR(DOUBLE, {self.vector_dimension}),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_sql)
            
            # Create vector index for similarity search with configurable HNSW parameters
            try:
                # Get vector index configuration from config manager
                vector_config = self.config_manager.get_vector_index_config()
                
                # Use IRISSQLUtils for proper HNSW index creation
                from common.iris_sql_utils import IRISSQLUtils
                utils = IRISSQLUtils()
                
                # Create optimized vector index with HNSW parameters
                index_success = utils.optimize_vector_table(
                    connection,
                    self.table_name,
                    'embedding',
                    vector_config
                )
                
                if index_success:
                    logger.info(f"Created optimized vector index for table: {self.table_name}")
                    logger.info(f"HNSW parameters: M={vector_config.get('M')}, "
                              f"efConstruction={vector_config.get('efConstruction')}, "
                              f"Distance={vector_config.get('Distance')}")
                else:
                    logger.warning(f"Could not create optimized vector index for table: {self.table_name}")
                    
            except Exception as e:
                logger.warning(f"Could not create vector index: {e}")
                # Continue without vector index - basic functionality will still work
            
            connection.commit()
            logger.info(f"Schema initialized for table: {self.table_name}")
            
        except Exception as e:
            connection.rollback()
            logger.error(f"Failed to initialize schema: {e}")
            raise
        finally:
            cursor.close()
    
    def store_document(self, document: Document, embedding: Optional[List[float]] = None) -> None:
        """
        Store a single document with optional embedding.
        
        Args:
            document: Document to store
            embedding: Optional vector embedding for the document
        """
        self.store_documents([document], [embedding] if embedding else None)
    
    def store_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> None:
        """
        Store multiple documents with optional embeddings.
        
        Args:
            documents: List of documents to store
            embeddings: Optional list of vector embeddings for the documents
        """
        if embeddings and len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        connection = self._get_connection()
        cursor = connection.cursor()
        
        try:
            # Use IRIS-compatible check-then-insert/update pattern
            # Map Document.id to doc_id column in RAG.SourceDocuments
            for i, doc in enumerate(documents):
                metadata_json = json.dumps(doc.metadata)
                
                # First, check if document exists (using doc_id column)
                check_sql = f"SELECT COUNT(*) FROM {self.table_name} WHERE doc_id = ?"
                cursor.execute(check_sql, [doc.id])
                exists = cursor.fetchone()[0] > 0
                
                if exists:
                    # Update existing document
                    if embeddings:
                        update_sql = f"""
                        UPDATE {self.table_name}
                        SET text_content = ?, metadata = ?, embedding = TO_VECTOR(?)
                        WHERE doc_id = ?
                        """
                        embedding_str = json.dumps(embeddings[i])
                        cursor.execute(update_sql, [doc.page_content, metadata_json, embedding_str, doc.id])
                    else:
                        update_sql = f"""
                        UPDATE {self.table_name}
                        SET text_content = ?, metadata = ?
                        WHERE doc_id = ?
                        """
                        cursor.execute(update_sql, [doc.page_content, metadata_json, doc.id])
                else:
                    # Insert new document (using doc_id column and available columns)
                    if embeddings:
                        insert_sql = f"""
                        INSERT INTO {self.table_name} (doc_id, text_content, metadata, embedding)
                        VALUES (?, ?, ?, TO_VECTOR(?))
                        """
                        embedding_str = json.dumps(embeddings[i])
                        cursor.execute(insert_sql, [doc.id, doc.page_content, metadata_json, embedding_str])
                    else:
                        insert_sql = f"""
                        INSERT INTO {self.table_name} (doc_id, text_content, metadata)
                        VALUES (?, ?, ?)
                        """
                        cursor.execute(insert_sql, [doc.id, doc.page_content, metadata_json])
            
            connection.commit()
            
            logger.info(f"Stored {len(documents)} documents in {self.table_name}")
            
        except Exception as e:
            connection.rollback()
            logger.error(f"Failed to store documents: {e}")
            raise
        finally:
            cursor.close()
    
    def retrieve_documents_by_ids(self, document_ids: List[str]) -> List[Document]:
        """
        Retrieve documents by their IDs.
        
        Args:
            document_ids: List of document IDs to retrieve
            
        Returns:
            List of retrieved documents
        """
        if not document_ids:
            return []
        
        connection = self._get_connection()
        cursor = connection.cursor()
        
        try:
            # Create placeholders for IN clause
            placeholders = ','.join(['?' for _ in document_ids])
            # Ensure selecting 'text_content' which is the actual column name
            select_sql = f"""
            SELECT id, text_content, metadata
            FROM {self.table_name}
            WHERE id IN ({placeholders})
            """
            
            cursor.execute(select_sql, document_ids)
            rows = cursor.fetchall()
            
            documents = []
            for row in rows:
                doc_id, actual_content_column_value, metadata_json = row
                logger.debug(f"IRISStorage.retrieve_documents_by_ids - Fetched doc_id: {doc_id}, type: {type(doc_id)}")
                
                # Convert CLOB to string if necessary
                page_content = _convert_clob_to_string(actual_content_column_value)
                metadata_str = _convert_clob_to_string(metadata_json)
                metadata = json.loads(metadata_str) if metadata_str else {}
                
                documents.append(Document(
                    id=str(doc_id), # Ensure id is string
                    page_content=page_content, # Ensure this is a string
                    metadata=metadata
                ))
            
            logger.debug(f"Retrieved {len(documents)} documents by IDs")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents by IDs: {e}")
            raise
        finally:
            cursor.close()
    
    def vector_search(self, query_embedding: List[float], top_k: int = 5, 
                     metadata_filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        Perform vector similarity search.
        
        Args:
            query_embedding: Query vector for similarity search
            top_k: Number of top results to return
            metadata_filter: Optional metadata filters to apply
            
        Returns:
            List of tuples containing (Document, similarity_score)
        """
        connection = self._get_connection()
        cursor = connection.cursor()
        
        try:
            # Build base query
            # Ensure selecting 'text_content'
            base_sql = f"""
            SELECT TOP {top_k} id, text_content, metadata,
                   VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(?)) as similarity_score
            FROM {self.table_name}
            WHERE embedding IS NOT NULL
            """
            
            params = [json.dumps(query_embedding)]
            
            # Add metadata filters if provided
            if metadata_filter:
                filter_conditions = []
                for key, value in metadata_filter.items():
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
                doc_id, actual_content_column_value, metadata_json, similarity_score = row
                logger.debug(f"IRISStorage.vector_search - Fetched doc_id: {doc_id}, type: {type(doc_id)}")
                
                # Convert CLOB to string if necessary
                page_content = _convert_clob_to_string(actual_content_column_value)
                metadata_str = _convert_clob_to_string(metadata_json)
                metadata = json.loads(metadata_str) if metadata_str else {}
                
                document = Document(
                    id=str(doc_id), # Ensure id is string
                    page_content=page_content, # Ensure this is a string
                    metadata=metadata
                )
                results.append((document, float(similarity_score)))
            
            logger.debug(f"Vector search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            # Fallback to text-based search if vector search fails
            return self._fallback_text_search(query_embedding, top_k, metadata_filter)
        finally:
            cursor.close()
    
    def _fallback_text_search(self, query_embedding: List[float], top_k: int, 
                             metadata_filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        Fallback text-based search when vector search is not available.
        
        This is a simple implementation that returns documents without similarity scoring.
        """
        connection = self._get_connection()
        cursor = connection.cursor()
        
        try:
            # Ensure selecting 'text_content'
            base_sql = f"""
            SELECT TOP {top_k} id, text_content, metadata
            FROM {self.table_name}
            """
            
            params = []
            
            # Add metadata filters if provided
            if metadata_filter:
                filter_conditions = []
                for key, value in metadata_filter.items():
                    filter_conditions.append(f"JSON_EXTRACT(metadata, '$.{key}') = ?")
                    params.append(str(value))
                
                if filter_conditions:
                    base_sql += " WHERE " + " AND ".join(filter_conditions)
            
            # Order by creation time as fallback
            base_sql += " ORDER BY created_at DESC"
            
            cursor.execute(base_sql, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                doc_id, actual_content_column_value, metadata_json = row
                logger.debug(f"IRISStorage._fallback_text_search - Fetched doc_id: {doc_id}, type: {type(doc_id)}")
                
                # Convert CLOB to string if necessary
                page_content = _convert_clob_to_string(actual_content_column_value)
                metadata_str = _convert_clob_to_string(metadata_json)
                metadata = json.loads(metadata_str) if metadata_str else {}
                
                document = Document(
                    id=str(doc_id), # Ensure id is string
                    page_content=page_content, # Ensure this is a string
                    metadata=metadata
                )
                # Use a default similarity score for fallback
                results.append((document, 0.5))
            
            logger.warning(f"Used fallback text search, returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Fallback text search failed: {e}")
            return []
        finally:
            cursor.close()
    
    def get_document_count(self) -> int:
        """
        Get the total number of documents in storage.
        
        Returns:
            Total document count
        """
        connection = self._get_connection()
        cursor = connection.cursor()
        
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            count = cursor.fetchone()[0]
            return int(count)
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
        finally:
            cursor.close()
    
    def clear_documents(self) -> None:
        """
        Clear all documents from storage.
        
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
            logger.error(f"Failed to clear documents: {e}")
            raise
        finally:
            cursor.close()
    
    def search_entity_embeddings(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[str, str, float]]:
        """
        Search for similar entities using vector embeddings.
        
        Args:
            query_embedding: Query vector embedding
            top_k: Number of top results to return
            
        Returns:
            List of tuples (entity_id, entity_text, similarity_score)
        """
        connection = self._get_connection()
        cursor = connection.cursor()
        
        try:
            from common.vector_format_fix import format_vector_for_iris
            
            # Format query embedding for IRIS using TO_VECTOR approach
            query_vector = format_vector_for_iris(query_embedding)
            query_vector_str = '[' + ','.join(f'{x:.8f}' for x in query_vector) + ']'
            
            # Use VECTOR_COSINE for similarity search on DocumentEntities table
            search_sql = f"""
            SELECT TOP {top_k}
                entity_id,
                entity_text,
                VECTOR_COSINE(embedding, TO_VECTOR(?)) as similarity_score
            FROM RAG.DocumentEntities
            WHERE embedding IS NOT NULL
            ORDER BY similarity_score DESC
            """
            
            cursor.execute(search_sql, [query_vector_str])
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                entity_id, entity_text, similarity_score = row
                results.append((entity_id, entity_text, float(similarity_score)))
            
            logger.debug(f"Entity embedding search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Entity embedding search failed: {e}")
            return []
        finally:
            cursor.close()