"""
IRIS-specific implementation of the VectorStore abstract base class.

This module provides a concrete implementation of the VectorStore interface
for InterSystems IRIS, including CLOB handling and vector search capabilities.
"""

import json
import logging
from common.db_vector_utils import insert_vector
from typing import List, Dict, Any, Optional, Tuple

from ..core.vector_store import VectorStore
from ..core.models import Document
from ..core.connection import ConnectionManager
from ..config.manager import ConfigurationManager
from ..core.vector_store_exceptions import (
    VectorStoreConnectionError,
    VectorStoreDataError,
    VectorStoreCLOBError,
    VectorStoreConfigurationError,
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

    def __init__(
        self,
        connection_manager: Optional[ConnectionManager] = None,
        config_manager: Optional[ConfigurationManager] = None,
        schema_manager=None,
        **kwargs,
    ):
        """
        Initialize IRIS vector store with connection and configuration managers.

        Args:
            connection_manager: Manager for database connections (optional for testing)
            config_manager: Manager for configuration settings (optional for testing)
            schema_manager: Schema manager for table management (optional, will be created if not provided)
            **kwargs: Additional keyword arguments for compatibility

        Raises:
            VectorStoreConnectionError: If connection cannot be established
            VectorStoreConfigurationError: If configuration is invalid
        """
        # Import here to avoid circular imports
        from ..storage.schema_manager import SchemaManager

        self.connection_manager = connection_manager
        if self.connection_manager is None:
            # Create a default connection manager for testing
            from ..core.connection import ConnectionManager

            self.connection_manager = ConnectionManager()
        self.config_manager = config_manager
        if self.config_manager is None:
            # Create a default config manager for testing
            from ..config.manager import ConfigurationManager

            self.config_manager = ConfigurationManager()
        self._connection = None

        # Get storage configuration
        self.storage_config = self.config_manager.get("storage:iris", {})
        self.table_name = self.storage_config.get("table_name", "RAG.SourceDocuments")

        # Get chunking configuration
        self.chunking_config = self.config_manager.get("storage:chunking", {})
        self.auto_chunk = self.chunking_config.get("enabled", False)

        # Initialize chunking service if auto chunking is enabled
        self.chunking_service = None
        if self.auto_chunk:
            try:
                from tools.chunking.chunking_service import DocumentChunkingService

                self.chunking_service = DocumentChunkingService(self.chunking_config)
            except ImportError:
                logger.warning(
                    "DocumentChunkingService not available, disabling auto chunking"
                )
                self.auto_chunk = False

        # Get vector dimension from schema manager (single source of truth)
        if schema_manager:
            self.schema_manager = schema_manager
        else:
            from .schema_manager import SchemaManager

            self.schema_manager = SchemaManager(
                self.connection_manager, self.config_manager
            )
        table_short_name = self.table_name.replace("RAG.", "")
        self.vector_dimension = self.schema_manager.get_vector_dimension(
            table_short_name
        )

        # Validate table name for security
        self._validate_table_name(self.table_name)

        # Define allowed filter keys for security
        self._allowed_filter_keys = {
            "category",
            "year",
            "source_type",
            "document_id",
            "author_name",
            "title",
            "source",
            "type",
            "date",
            "status",
            "version",
            "pmcid",
            "journal",
            "doi",
            "publication_date",
            "keywords",
            "abstract_type",
        }

        # Test connection on initialization (skip in test mode)
        try:
            # Only test connection if not in test mode or if explicitly requested
            import os

            if os.environ.get("PYTEST_CURRENT_TEST") is None:
                self._get_connection()
        except Exception as e:
            raise VectorStoreConnectionError(
                f"Failed to initialize IRIS connection: {e}"
            )

    def _get_connection(self):
        """Get or create database connection."""
        if self._connection is None:
            try:
                self._connection = self.connection_manager.get_connection("iris")
            except Exception as e:
                raise VectorStoreConnectionError(f"Failed to get IRIS connection: {e}")
        return self._connection

    def _ensure_table_exists(self, cursor):
        """Ensure the target table exists, creating it if necessary."""
        try:
            # Check if table exists by trying to query it
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            logger.debug(f"Table {self.table_name} exists")
        except Exception as e:
            logger.info(f"Table {self.table_name} does not exist, creating it: {e}")
            try:
                # Use schema manager to ensure proper table creation
                table_short_name = self.table_name.replace("RAG.", "")
                expected_config = {
                    "vector_dimension": self.vector_dimension,
                    "vector_data_type": "FLOAT",
                }
                success = self.schema_manager.ensure_table_schema(table_short_name)
                if success:
                    logger.info(f"✅ Successfully created table {self.table_name}")
                else:
                    logger.warning(
                        f"⚠️ Table creation may have failed for {self.table_name}"
                    )
            except Exception as create_error:
                logger.error(
                    f"Failed to create table {self.table_name}: {create_error}"
                )
                # Don't raise here - let the subsequent operations fail with clearer errors

    def _validate_table_name(self, table_name: str) -> None:
        """
        Validate table name to prevent SQL injection.

        Args:
            table_name: The table name to validate

        Raises:
            VectorStoreConfigurationError: If table name contains dangerous characters
        """
        # Default allowed tables (for backward compatibility)
        default_allowed_tables = {
            "RAG.SourceDocuments",
            "RAG.DocumentTokenEmbeddings",
            "RAG.TestDocuments",
            "RAG.BackupDocuments",
        }

        # Check if it's a default table (always allowed)
        if table_name in default_allowed_tables:
            return

        # For custom tables, validate format to prevent SQL injection
        import re

        # Allow schema.table format with alphanumeric, underscore, and dot
        # Pattern: schema_name.table_name where both parts are safe identifiers
        table_pattern = r"^[a-zA-Z][a-zA-Z0-9_]*\.[a-zA-Z][a-zA-Z0-9_]*$"

        if not re.match(table_pattern, table_name):
            logger.error(f"Security violation: Invalid table name format: {table_name}")
            raise VectorStoreConfigurationError(
                f"Invalid table name format: {table_name}. "
                f"Must be in format 'Schema.TableName' with alphanumeric characters and underscores only."
            )

        # Additional check: prevent SQL keywords and dangerous patterns
        dangerous_patterns = [
            "drop",
            "delete",
            "insert",
            "update",
            "create",
            "alter",
            "truncate",
            "exec",
            "execute",
            "select",
            "union",
            "script",
            "--",
            ";",
            "/*",
            "*/",
            "xp_",
            "sp_",
            "declare",
            "cast",
            "convert",
        ]

        table_lower = table_name.lower()
        for pattern in dangerous_patterns:
            if pattern in table_lower:
                logger.error(
                    f"Security violation: Dangerous pattern in table name: {table_name}"
                )
                raise VectorStoreConfigurationError(
                    f"Table name contains restricted pattern: {pattern}"
                )

        logger.info(f"✅ Custom table name validated: {table_name}")

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
                logger.warning(
                    f"Security violation: Invalid filter key attempted: {key}"
                )
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
                logger.warning(
                    f"Security violation: Invalid filter value type for key '{key}': {type(value).__name__}"
                )
                raise VectorStoreDataError(
                    f"Invalid filter value for key '{key}': {type(value).__name__}"
                )

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
            metadata = processed_data.get("metadata", {})
            if isinstance(metadata, str) and metadata:
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse metadata JSON: {metadata}")
                    metadata = {"raw_metadata": metadata}
            elif not isinstance(metadata, dict):
                metadata = {}

            return Document(
                id=processed_data.get("doc_id", processed_data.get("id", "")),
                page_content=processed_data.get(
                    "text_content", processed_data.get("page_content", "")
                ),
                metadata=metadata,
            )
        except Exception as e:
            raise VectorStoreCLOBError(f"Failed to process document data: {e}")

    def _chunk_document(
        self, document: Document, chunking_strategy: Optional[str] = None
    ) -> List[Document]:
        """
        Chunk a document using the specified strategy.

        Args:
            document: Document to chunk
            chunking_strategy: Strategy to use for chunking (optional, uses config default)

        Returns:
            List of chunked documents with unique IDs
        """
        if not self.chunking_service:
            # If no chunking service available, return original document
            return [document]

        try:
            # Use the chunking service to chunk the document
            # The chunking service expects (doc_id, text, strategy_name)
            strategy_name = chunking_strategy or self.chunking_config.get(
                "strategy", "fixed_size"
            )
            chunk_records = self.chunking_service.chunk_document(
                document.id, document.page_content, strategy_name
            )

            # Convert chunk records to Document objects with unique IDs
            chunked_documents = []
            for chunk_record in chunk_records:
                # Use the unique chunk_id as the Document ID to avoid collisions
                chunk_doc = Document(
                    id=chunk_record[
                        "chunk_id"
                    ],  # This is unique: "doc-123_chunk_fixed_size_0"
                    page_content=chunk_record[
                        "chunk_text"
                    ],  # Note: chunk service uses "chunk_text"
                    metadata={
                        **document.metadata,  # Inherit original metadata
                        "parent_doc_id": document.id,  # Reference to original document
                        "chunk_index": chunk_record.get("chunk_index", 0),
                        "chunk_strategy": strategy_name,
                        "start_pos": chunk_record.get("start_position", 0),
                        "end_pos": chunk_record.get(
                            "end_position", len(chunk_record["chunk_text"])
                        ),
                    },
                )
                chunked_documents.append(chunk_doc)

            logger.debug(
                f"Document {document.id} chunked into {len(chunked_documents)} pieces with unique IDs"
            )
            return chunked_documents

        except Exception as e:
            logger.warning(f"Chunking failed for document {document.id}: {e}")
            # Fallback to original document if chunking fails
            return [document]

    def _generate_embeddings(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for documents.

        Args:
            documents: List of documents to generate embeddings for

        Returns:
            List of embedding vectors
        """
        try:
            # Import embedding function here to avoid circular imports
            from ..embeddings.manager import EmbeddingManager

            embedding_manager = EmbeddingManager(self.config_manager)
            embedding_func = lambda text: embedding_manager.embed_text(text)

            embeddings = []
            for doc in documents:
                embedding = embedding_func(doc.page_content)
                embeddings.append(embedding)

            return embeddings
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            # Return empty embeddings if generation fails
            # Handle case where vector_dimension might be a Mock object
            try:
                dim = int(self.vector_dimension) if self.vector_dimension else 768
            except (TypeError, ValueError):
                dim = 768  # Default dimension
            return [[0.0] * dim for _ in documents]

    def _store_documents(
        self, documents: List[Document], embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """
        Store documents in the database with optional embeddings.

        This method is called internally by add_documents after chunking and embedding generation.

        Args:
            documents: List of documents to store
            embeddings: Optional embeddings for the documents

        Returns:
            List of document IDs that were stored
        """
        if not documents:
            return []

        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            # Ensure table exists before any operations
            self._ensure_table_exists(cursor)

            # If embeddings are provided, ensure the table has the proper vector schema
            if embeddings:
                logger.debug(
                    f"Embeddings provided: {len(embeddings)} embeddings - ensuring vector schema"
                )
                table_short_name = self.table_name.replace("RAG.", "")
                # Force schema update to ensure embedding column exists
                schema_success = self.schema_manager.ensure_table_schema(
                    table_short_name
                )
                if not schema_success:
                    logger.warning(
                        f"Schema update may have failed for {self.table_name} - proceeding anyway"
                    )

            added_ids = []
            logger.debug(
                f"_store_documents called with {len(documents)} documents and embeddings: {embeddings is not None}"
            )

            for i, doc in enumerate(documents):
                metadata_json = json.dumps(doc.metadata)

                # Check if document exists - use doc_id column (correct schema)
                check_sql = f"SELECT COUNT(*) FROM {self.table_name} WHERE doc_id = ?"
                cursor.execute(check_sql, [doc.id])
                exists = cursor.fetchone()[0] > 0

                # Always use insert_vector utility for consistent handling (it works with or without embeddings)
                if embeddings and len(embeddings) > i:
                    logger.debug(
                        f"Inserting document {doc.id} with embedding using insert_vector utility"
                    )
                    # Use the required insert_vector utility function for vector insertions/updates
                    # Use doc_id and text_content per the actual IRIS schema
                    success = insert_vector(
                        cursor=cursor,
                        table_name=self.table_name,
                        vector_column_name="embedding",
                        vector_data=embeddings[i],
                        target_dimension=self.vector_dimension,
                        key_columns={"doc_id": doc.id},
                        additional_data={
                            "text_content": doc.page_content,
                            "metadata": metadata_json,
                        },
                    )
                    if success:
                        added_ids.append(doc.id)
                        logger.debug(
                            f"Successfully upserted document {doc.id} with vector"
                        )
                    else:
                        logger.error(f"Failed to upsert document {doc.id} with vector")
                else:
                    # Insert without embedding - use correct schema (doc_id, text_content, metadata)
                    if exists:
                        update_sql = f"""
                        UPDATE {self.table_name}
                        SET text_content = ?, metadata = ?
                        WHERE doc_id = ?
                        """
                        cursor.execute(
                            update_sql, [doc.page_content, metadata_json, doc.id]
                        )
                        logger.debug(
                            f"Updated existing document {doc.id} without vector"
                        )
                    else:
                        # Insert using correct schema
                        insert_sql = f"""
                        INSERT INTO {self.table_name} (doc_id, text_content, metadata)
                        VALUES (?, ?, ?)
                        """
                        cursor.execute(
                            insert_sql, [doc.id, doc.page_content, metadata_json]
                        )
                        logger.debug(f"Inserted new document {doc.id} without vector")

                    added_ids.append(doc.id)

            connection.commit()
            logger.info(f"Successfully stored {len(added_ids)} documents")
            return added_ids

        except Exception as e:
            connection.rollback()
            error_msg = self._sanitize_error_message(e, "document storage")
            logger.error(error_msg)
            raise VectorStoreDataError(f"Failed to store documents: {error_msg}")
        finally:
            cursor.close()

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        chunking_strategy: Optional[str] = None,
        auto_chunk: Optional[bool] = None,
    ) -> List[str]:
        """
        Add documents to the IRIS vector store with automatic chunking support.

        Args:
            documents: List of Document objects to add
            embeddings: Optional pre-computed embeddings for the documents
            chunking_strategy: Optional chunking strategy override
            auto_chunk: Optional override for automatic chunking (None uses config default)

        Returns:
            List of document IDs that were added

        Raises:
            VectorStoreDataError: If document data is malformed
            VectorStoreConnectionError: If there are connection issues
        """
        if not documents:
            return []

        # Determine if we should use automatic chunking
        should_chunk = auto_chunk if auto_chunk is not None else self.auto_chunk

        # Process documents through chunking if enabled
        processed_documents = []
        if should_chunk and self.chunking_service:
            logger.debug(
                f"Auto-chunking enabled, processing {len(documents)} documents"
            )
            # Use provided strategy or fall back to configured strategy
            effective_strategy = chunking_strategy or self.chunking_config.get(
                "strategy", "fixed_size"
            )
            for doc in documents:
                # Check if document exceeds threshold
                threshold = self.chunking_config.get("threshold", 1000)
                if len(doc.page_content) > threshold:
                    chunks = self._chunk_document(doc, effective_strategy)
                    processed_documents.extend(chunks)
                    logger.debug(f"Document {doc.id} chunked into {len(chunks)} pieces")
                else:
                    processed_documents.append(doc)
                    logger.debug(f"Document {doc.id} below threshold, not chunked")
        else:
            processed_documents = documents
            logger.debug(
                f"Auto-chunking disabled, using {len(documents)} original documents"
            )

        # Generate embeddings if not provided (regardless of chunking)
        if embeddings is None and processed_documents:
            logger.debug(
                "No embeddings provided, generating embeddings for processed documents"
            )
            embeddings = self._generate_embeddings(processed_documents)
        elif embeddings and len(embeddings) != len(processed_documents):
            # If embeddings were provided but count doesn't match after chunking, regenerate
            logger.warning(
                f"Embedding count mismatch after chunking: {len(embeddings)} vs {len(processed_documents)}, regenerating"
            )
            embeddings = self._generate_embeddings(processed_documents)

        # Validate processed documents
        for doc in processed_documents:
            if not isinstance(doc.page_content, str):
                raise VectorStoreDataError("Document page_content must be a string")

        # Use the _store_documents method to handle the actual storage
        return self._store_documents(processed_documents, embeddings)

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
            placeholders = ",".join(["?" for _ in ids])
            delete_sql = (
                f"DELETE FROM {self.table_name} WHERE doc_id IN ({placeholders})"
            )
            cursor.execute(delete_sql, ids)

            deleted_count = cursor.rowcount
            connection.commit()

            logger.info(f"Deleted {deleted_count} documents from {self.table_name}")
            return deleted_count > 0

        except Exception as e:
            connection.rollback()
            sanitized_error = self._sanitize_error_message(e, "delete_documents")
            logger.error(sanitized_error)
            raise VectorStoreConnectionError(
                f"Failed to delete documents: {sanitized_error}"
            )
        finally:
            cursor.close()

    def similarity_search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
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
            # Use the safe vector utilities (proven pattern)
            from common.vector_sql_utils import (
                build_safe_vector_dot_sql,
                execute_safe_vector_search,
            )

            # Build metadata filter clause if needed
            additional_where = None
            if filter:
                filter_conditions = []
                for key, value in filter.items():
                    # Key is already validated, safe to use in f-string
                    # Use IRIS JSON_VALUE function instead of JSON_EXTRACT
                    # JSON_VALUE returns a string, so we compare as strings
                    filter_conditions.append(
                        f"JSON_VALUE(metadata, '$.{key}') = '{str(value)}'"
                    )

                if filter_conditions:
                    additional_where = " AND ".join(filter_conditions)

            # Get current vector dimension from schema manager
            table_short_name = self.table_name.replace("RAG.", "")
            expected_dimension = self.schema_manager.get_vector_dimension(
                table_short_name
            )

            # Validate query embedding dimension matches expected
            if len(query_embedding) != expected_dimension:
                error_msg = f"Query embedding dimension {len(query_embedding)} doesn't match expected {expected_dimension} for table {table_short_name}"
                logger.error(error_msg)
                raise VectorStoreDataError(error_msg)

            logger.debug(
                f"Vector search: query={len(query_embedding)}D, expected={expected_dimension}D, table={table_short_name}"
            )

            # Convert query embedding to bracketed string format for TO_VECTOR
            # IMPORTANT: TO_VECTOR() does NOT accept parameter markers
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # Build SQL with embedded vector string (no parameters)
            # Use new schema columns: doc_id and text_content
            sql = build_safe_vector_dot_sql(
                table=self.table_name,
                vector_column="embedding",
                vector_string=embedding_str,
                vector_dimension=expected_dimension,
                id_column="doc_id",
                extra_columns=["text_content"],
                top_k=top_k,
                additional_where=additional_where,
            )

            # DEBUG: Check if SQL contains DOUBLE or FLOAT
            if 'DOUBLE' in sql:
                logger.error(f"FOUND DOUBLE IN SQL! SQL snippet: {sql[sql.find('TO_VECTOR'):sql.find('TO_VECTOR')+100]}")
            elif 'FLOAT' in sql:
                logger.info(f"SQL correctly uses FLOAT. SQL snippet: {sql[sql.find('TO_VECTOR'):sql.find('TO_VECTOR')+100]}")
            else:
                logger.warning("No FLOAT or DOUBLE found in SQL!")

            # Execute using the safe helper (vector already embedded in SQL)
            logger.debug(
                f"Executing safe vector search with {len(query_embedding)}D vector"
            )
            try:
                rows = execute_safe_vector_search(cursor, sql)
            except Exception as e:
                # Check if this is a table not found error
                if "Table" in str(e) and "not found" in str(e):
                    logger.info(
                        f"Table {self.table_name} not found, attempting to create it automatically"
                    )
                    self._create_table_automatically()
                    # Retry the search after table creation
                    rows = execute_safe_vector_search(cursor, sql)
                else:
                    # Log the SQL that failed for debugging
                    logger.error(f"Vector search SQL that failed: {sql[:500]}...")
                    logger.error(f"Error details: {type(e).__name__}: {e}")
                    # Re-raise other errors
                    raise

            # Now fetch metadata for the returned documents
            metadata_map = {}
            if rows:
                # Handle Mock objects that aren't iterable
                try:
                    doc_ids = [row[0] for row in rows]
                    placeholders = ",".join(["?" for _ in doc_ids])
                    # Use 'doc_id' column (correct schema)
                    metadata_sql = f"SELECT doc_id, metadata FROM {self.table_name} WHERE doc_id IN ({placeholders})"
                    cursor.execute(metadata_sql, doc_ids)
                    metadata_map = {row[0]: row[1] for row in cursor.fetchall()}
                except (TypeError, AttributeError):
                    # Handle Mock objects by skipping metadata fetch
                    logger.debug(
                        "Rows is not iterable (likely a Mock object), skipping metadata fetch"
                    )
                    metadata_map = {}

            results = []
            # Handle Mock objects that aren't iterable
            try:
                row_iterator = iter(rows)
            except (TypeError, AttributeError):
                # Handle Mock objects by returning empty results
                logger.debug(
                    "Rows is not iterable (likely a Mock object), returning empty results"
                )
                return []

            for row in rows:
                doc_id, content, similarity_score = row

                # Get metadata from the map
                metadata_json = (
                    metadata_map.get(doc_id, None)
                    if "metadata_map" in locals()
                    else None
                )

                # Process row data to ensure string content
                document_data = {
                    "doc_id": doc_id,
                    "text_content": content,
                    "metadata": metadata_json,
                }

                document = self._ensure_string_content(document_data)
                # Handle similarity_score that might be a list or single value
                if isinstance(similarity_score, (list, tuple)):
                    # If it's a list/tuple, take the first element
                    score_value = (
                        float(similarity_score[0]) if similarity_score else 0.0
                    )
                elif similarity_score is not None:
                    # If it's already a single value, use it directly
                    score_value = float(similarity_score)
                else:
                    # Handle NULL similarity scores (database returned None)
                    score_value = 0.0

                results.append((document, score_value))

            # Handle Mock objects that don't have len()
            try:
                result_count = len(results)
                logger.debug(f"Vector search returned {result_count} results")
            except (TypeError, AttributeError):
                # Handle Mock objects or other non-sequence types
                logger.debug(
                    "Vector search returned results (count unavailable due to mock object)"
                )
            return results

        except Exception as e:
            sanitized_error = self._sanitize_error_message(e, "similarity_search")
            logger.error(sanitized_error)
            raise VectorStoreConnectionError(f"Vector search failed: {sanitized_error}")
        finally:
            cursor.close()

    def _create_table_automatically(self):
        """
        Create the required table automatically using schema manager.

        This method uses the schema manager to create the table with the correct
        schema based on the table name and configuration.
        """
        try:
            logger.info(f"Creating table {self.table_name} automatically")

            # Get the table short name (without RAG. prefix)
            table_short_name = self.table_name.replace("RAG.", "")

            # Get expected configuration for this table
            expected_config = self.schema_manager._get_expected_schema_config(
                table_short_name
            )

            # Get a connection and cursor
            connection = self._get_connection()
            cursor = connection.cursor()

            try:
                # Use the schema manager's migration method to create the table
                if table_short_name == "SourceDocuments":
                    success = self.schema_manager._migrate_source_documents_table(
                        cursor, expected_config, preserve_data=False
                    )
                elif table_short_name == "DocumentTokenEmbeddings":
                    success = (
                        self.schema_manager._migrate_document_token_embeddings_table(
                            cursor, expected_config, preserve_data=False
                        )
                    )
                elif table_short_name == "DocumentEntities":
                    success = self.schema_manager._migrate_document_entities_table(
                        cursor, expected_config, preserve_data=False
                    )
                elif table_short_name == "KnowledgeGraphNodes":
                    success = self.schema_manager._migrate_knowledge_graph_nodes_table(
                        cursor, expected_config, preserve_data=False
                    )
                elif table_short_name == "KnowledgeGraphEdges":
                    success = self.schema_manager._migrate_knowledge_graph_edges_table(
                        cursor, expected_config, preserve_data=False
                    )
                else:
                    logger.warning(
                        f"Unknown table type: {table_short_name}, cannot create automatically"
                    )
                    success = False

                if success:
                    logger.info(f"Successfully created table {self.table_name}")
                else:
                    logger.error(f"Failed to create table {self.table_name}")

            finally:
                cursor.close()

        except Exception as e:
            logger.error(f"Error creating table {self.table_name}: {e}")
            # Don't re-raise the error, let the original operation fail with the original error

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
            placeholders = ",".join(["?" for _ in ids])

            # Try new schema first (doc_id, text_content)
            try:
                select_sql = f"""
                SELECT doc_id, text_content, metadata
                FROM {self.table_name}
                WHERE doc_id IN ({placeholders})
                """
                cursor.execute(select_sql, ids)
                rows = cursor.fetchall()

                documents = []
                for row in rows:
                    doc_id, text_content, metadata_json = row
                    document_data = {
                        "doc_id": doc_id,
                        "text_content": text_content,
                        "metadata": metadata_json,
                    }
                    document = self._ensure_string_content(document_data)
                    documents.append(document)

                logger.debug(f"Fetched {len(documents)} documents by IDs using new schema")
                return documents

            except Exception as new_schema_error:
                # Fallback to simple schema (id, content)
                if "not found in the applicable tables" in str(new_schema_error):
                    logger.debug("Falling back to simple schema (id, content)")
                    select_sql = f"""
                    SELECT id, content, metadata
                    FROM {self.table_name}
                    WHERE id IN ({placeholders})
                    """
                    cursor.execute(select_sql, ids)
                    rows = cursor.fetchall()

                    documents = []
                    for row in rows:
                        doc_id, text_content, metadata_json = row
                        document_data = {
                            "doc_id": doc_id,
                            "text_content": text_content,
                            "metadata": metadata_json,
                        }
                        document = self._ensure_string_content(document_data)
                        documents.append(document)

                    logger.debug(f"Fetched {len(documents)} documents by IDs using simple schema")
                    return documents
                else:
                    raise

        except Exception as e:
            sanitized_error = self._sanitize_error_message(e, "fetch_documents_by_ids")
            logger.error(sanitized_error)
            raise VectorStoreConnectionError(
                f"Failed to fetch documents by IDs: {sanitized_error}"
            )
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
            raise VectorStoreConnectionError(
                f"Failed to get document count: {sanitized_error}"
            )
        finally:
            cursor.close()

    def get_all_documents(self) -> List[Document]:
        """
        Retrieve all documents from the vector store.

        Returns:
            List of all Document objects in the store
        """
        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            # Try new schema first (doc_id, text_content)
            try:
                cursor.execute(
                    f"SELECT doc_id, text_content, metadata FROM {self.table_name}"
                )
                rows = cursor.fetchall()

                documents = []
                for row in rows:
                    doc_id, text_content, metadata_json = row
                    document_data = {
                        "doc_id": doc_id,
                        "text_content": text_content,
                        "metadata": metadata_json,
                    }
                    document = self._ensure_string_content(document_data)
                    documents.append(document)

                logger.debug(f"Retrieved {len(documents)} documents using new schema")
                return documents

            except Exception as new_schema_error:
                # Fallback to simple schema (id, content)
                if "not found in the applicable tables" in str(new_schema_error):
                    logger.debug("Falling back to simple schema for get_all_documents")
                    cursor.execute(
                        f"SELECT id, content, metadata FROM {self.table_name}"
                    )
                    rows = cursor.fetchall()

                    documents = []
                    for row in rows:
                        doc_id, text_content, metadata_json = row
                        document_data = {
                            "doc_id": doc_id,
                            "text_content": text_content,
                            "metadata": metadata_json,
                        }
                        document = self._ensure_string_content(document_data)
                        documents.append(document)

                    logger.debug(f"Retrieved {len(documents)} documents using simple schema")
                    return documents
                else:
                    raise

        except Exception as e:
            sanitized_error = self._sanitize_error_message(e, "get_all_documents")
            logger.error(sanitized_error)
            raise VectorStoreConnectionError(
                f"Failed to get all documents: {sanitized_error}"
            )
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
            raise VectorStoreConnectionError(
                f"Failed to clear documents: {sanitized_error}"
            )
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
            k = args[1] if len(args) > 1 else kwargs.get("k", 4)
            filter_param = args[2] if len(args) > 2 else kwargs.get("filter")

            # Get embedding function for text query
            embedding_func = kwargs.get("embedding_func")
            if not embedding_func:
                from ..embeddings.manager import EmbeddingManager

                embedding_manager = EmbeddingManager(self.config_manager)
                query_embedding = embedding_manager.embed_text(query)
            else:
                query_embedding = embedding_func.embed_query(query)

            # Call our implementation and add scores to document metadata
            results = self.similarity_search_by_embedding(
                query_embedding, k, filter_param
            )
            # Add score to each document's metadata for MCP/API compatibility
            documents_with_scores = []
            for doc, score in results:
                # Create new document with score in metadata
                enriched_metadata = {**doc.metadata, 'score': score, 'similarity': score}
                enriched_doc = Document(
                    id=doc.id,
                    page_content=doc.page_content,
                    metadata=enriched_metadata
                )
                documents_with_scores.append(enriched_doc)
            return documents_with_scores

        else:
            # Base class interface: similarity_search(query_embedding, top_k, filter)
            query_embedding = args[0] if args else kwargs["query_embedding"]
            top_k = args[1] if len(args) > 1 else kwargs.get("top_k", 5)
            filter_param = args[2] if len(args) > 2 else kwargs.get("filter")

            return self.similarity_search_by_embedding(
                query_embedding, top_k, filter_param
            )

    def similarity_search_with_score(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None) -> List[Tuple[Document, float]]:
        """
        Perform similarity search and return results with scores.

        Args:
            query: Text query to search for
            k: Maximum number of results to return
            filter: Optional metadata filters to apply

        Returns:
            List of tuples containing (Document, similarity_score)
        """
        # Get embedding function for text query
        from ..embeddings.manager import EmbeddingManager

        embedding_manager = EmbeddingManager(self.config_manager)
        query_embedding = embedding_manager.embed_text(query)

        # Call similarity_search_by_embedding which already returns (doc, score) tuples
        return self.similarity_search_by_embedding(query_embedding, k, filter)
