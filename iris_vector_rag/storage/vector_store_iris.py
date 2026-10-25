"""
IRIS-specific implementation of the VectorStore abstract base class.

This module provides a concrete implementation of the VectorStore interface
for InterSystems IRIS, including CLOB handling and vector search capabilities.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from iris_vector_rag.common.db_vector_utils import insert_vector, normalize_vector

from ..config.manager import ConfigurationManager
from ..core.connection import ConnectionManager
from ..core.models import Document
from ..core.vector_store import VectorStore
from ..exceptions import (
    AttachedTableReadOnly,
    DimensionMismatchError,
    EmbeddingError,
    RequiredFilterMissing,
    TextIndexMissingError,
    VectorStoreCLOBError,
    VectorStoreConfigurationError,
    VectorStoreConnectionError,
    VectorStoreDataError,
)
from .attached import compile_where, where_kwargs
from .catalog import IRISCatalogReader
from .clob_handler import ensure_string_content
from .shape_resolver import TableShapeResolver
from .table_shape import (
    ColumnFact,
    ColumnRole,
    RoleNotPresent,
    TableNotFound,
    TableShape,
    TableShapeError,
)
from .text_search import (
    TEXT_INDEX_NAME,
    build_text_search_sql,
    json_member_predicate,
    rank_procedure,
    search_terms,
    section_filter_predicate,
    text_language,
)

logger = logging.getLogger(__name__)


def _choice(value, key: str, allowed, default: str) -> str:
    """A setting from ``allowed``, lower-cased, or ``default`` when unset."""
    chosen = str(value).strip().lower() if value not in (None, "") else default
    if chosen not in allowed:
        raise ValueError(
            f"storage:iris:{key} must be one of {', '.join(allowed)}, got {value!r}"
        )
    return chosen


#: An HNSW index's Distance (lower-cased) -> the metric that can use it.
_HNSW_DISTANCES = {"cosine": "cosine", "dotproduct": "dot"}

#: Optional roles an attached table may surface in Document.metadata, by key.
_METADATA_ROLES = (("title", ColumnRole.TITLE), ("parent_id", ColumnRole.PARENT))


def _resolve(resolver, schema: str, table: str, declared) -> TableShape:
    """Resolve, passing a declaration only when there is one.

    A resolver written before feature 092 (a test double, a caller's own) takes
    ``(schema, table)``; it keeps working for every table that declares nothing.
    """
    if declared:
        return resolver.resolve(schema, table, declared=declared)
    return resolver.resolve(schema, table)


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
        shapes: Optional[TableShapeResolver] = None,
        **kwargs,
    ):
        """
        Initialize IRIS vector store with connection and configuration managers.

        Args:
            connection_manager: Manager for database connections (optional for testing)
            config_manager: Manager for configuration settings (optional for testing)
            schema_manager: Schema manager for table management (optional, will be created if not provided)
            shapes: Resolver for this session's table shapes (optional). One is built
                from this store's own connection on first use when not supplied; pass
                a resolver over a `RecordedCatalog` to describe a shape in a test.
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
        # Feature 082: one answer to "which column holds what", read from the
        # database once per table. Resolved lazily so the constructor stays pure.
        self.shapes = shapes
        # Tables already warned about dropping metadata, by qualified name. One line
        # per table is a warning; one line per document is a scrollback (FR-016).
        self._metadata_drop_warned: set = set()
        # Tables whose HNSW index was compared with the metric, by qualified name.
        self._hnsw_checked: set = set()
        self.config_manager = config_manager
        if self.config_manager is None:
            # Create a default config manager for testing
            from ..config.manager import ConfigurationManager

            self.config_manager = ConfigurationManager()
        self._connection = None

        # Get storage configuration
        self.storage_config = self.config_manager.get("storage:iris", {})
        self.table_name = kwargs.get("table_name") or self.storage_config.get(
            "table_name", "RAG.SourceDocuments"
        )

        # Feature 092: a table someone else owns, read through declared columns.
        # Constructor arguments override the `storage:iris:` keys of the same name.
        from .attached import ColumnMapping, FilterSpec

        self.column_mapping = ColumnMapping.from_settings(self.storage_config, kwargs)
        self.filter_spec = FilterSpec.from_settings(self.storage_config, kwargs)
        self.metric = _choice(
            kwargs.get("metric") or self.storage_config.get("metric"),
            "metric",
            ("cosine", "dot"),
            # Attached tables rank by cosine, the industry default; the library's own
            # tables keep dot product, so nothing changes for them.
            default="cosine" if self.column_mapping.attached else "dot",
        )
        self.text_index_name = kwargs.get("text_index") or self.storage_config.get(
            "text_index"
        )
        self.text_terms = _choice(
            kwargs.get("text_terms") or self.storage_config.get("text_terms"),
            "text_terms",
            ("words", "prefix"),
            default="words",
        )
        self._text_language = kwargs.get("text_language")
        #: The caller's embedder for query text (str -> vector), or None for the
        #: configured provider. It must produce vectors of the column's width.
        self.embedder = kwargs.get("embedder")

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
                self.connection_manager,
                self.config_manager,
            )
        self._vector_dimension: Optional[int] = None

        # Validate table name for security
        self._validate_table_name(self.table_name)

        # Initialize MetadataFilterManager for custom filter key management (Feature 051 - User Story 1)
        from .metadata_filter_manager import MetadataFilterManager

        self.metadata_filter_manager = MetadataFilterManager(
            self.config_manager.to_dict()
        )

        # Legacy support: Keep _allowed_filter_keys for backward compatibility
        # but it now uses MetadataFilterManager as source of truth
        self._allowed_filter_keys = set(
            self.metadata_filter_manager.get_allowed_filter_keys()
        )

        # IRIS EMBEDDING support (Feature 051)
        self.embedding_config_name = kwargs.get("embedding_config")
        self.use_iris_embedding = self.embedding_config_name is not None

        if self.use_iris_embedding:
            logger.info(
                f"IRISVectorStore initialized with IRIS EMBEDDING support "
                f"(config: {self.embedding_config_name})"
            )

        # Feature 078: Pure Constructors - no DB connection on init
        # Connection is now established lazily on first use

    def _get_connection(self):
        """Get or create database connection."""
        try:
            self._connection = self.connection_manager.get_connection("iris")
        except Exception as e:
            raise VectorStoreConnectionError(f"Failed to get IRIS connection: {e}")
        return self._connection

    # ========================================================================
    # Table shape (Feature 082)
    # ========================================================================

    @property
    def vector_dimension(self) -> int:
        """The embedding width this store's table is created at: the schema manager's.

        Resolved on first use, because the schema manager takes it from the embedding
        model and constructing a store must not load one.
        """
        if self._vector_dimension is None:
            _, table_short_name = self._schema_and_table()
            self._vector_dimension = self.schema_manager.get_vector_dimension(
                table_short_name
            )
        return self._vector_dimension

    @vector_dimension.setter
    def vector_dimension(self, dimension: int) -> None:
        self._vector_dimension = dimension

    def _schema_and_table(self, table_name: Optional[str] = None) -> Tuple[str, str]:
        """Split a table name; unqualified means the schema this store writes to.

        That is the schema manager's prefix, not the literal "RAG": a caller who
        configured `IRIS_SCHEMA_PREFIX` has one schema, and the shape must be read
        from the table the store is actually writing to.
        """
        name = self.table_name if table_name is None else table_name
        if "." in name:
            schema, _, table = name.partition(".")
            return schema, table
        return getattr(self.schema_manager, "schema_prefix", "RAG") or "RAG", name

    @property
    def text_language(self) -> str:
        """The language of text queries and polarity: ``storage:iris:text_language``.

        Read on each use, so a test or caller that changes ``storage_config`` is heard.
        It must match the language the table's text index was built in.
        """
        return text_language(
            self._text_language or self.storage_config.get("text_language")
        )

    def _shape_resolver(self) -> TableShapeResolver:
        if self.shapes is None:
            self.shapes = TableShapeResolver(IRISCatalogReader(self._get_connection()))
        return self.shapes

    def shape(self, cursor=None) -> TableShape:
        """The physical shape of this store's table, read from the database once.

        Every call site that used to derive a column name from a name list, a
        configuration entry or an environment variable asks this instead, so a table
        is written and read through the same column by construction.

        Passing a cursor allows the table to be created if it does not exist yet,
        which is what the write path wants and no read path does: a search against a
        missing table must say so rather than quietly create one.
        """
        schema, table = self._schema_and_table()
        resolver = self._shape_resolver()
        declared = self.column_mapping.declared
        try:
            return _resolve(resolver, schema, table, declared)
        except TableNotFound:
            if cursor is None or declared:
                # An attached table is never created on the owner's behalf.
                raise
        # `_ensure_table_exists` invalidates what it changed, so the resolve below
        # reads the table it just created. If creation failed this raises
        # TableNotFound naming the table, which is the error the caller needs; the
        # old code let the next statement fail instead.
        self._ensure_table_exists(cursor)
        return _resolve(resolver, schema, table, declared)

    def refresh_shape(self, table: Optional[str] = None) -> TableShape:
        """Read a table's shape again, for a table changed outside this library.

        The store invalidates its own DDL, so this is only needed when something
        else altered the table under a running session. One call, one catalog read;
        the shape it returns is the one every later operation will use.
        """
        schema, name = self._schema_and_table(table)
        resolver = self._shape_resolver()
        resolver.invalidate(schema, name)
        own_table = table is None or name.lower() == self._schema_and_table()[1].lower()
        declared = self.column_mapping.declared if own_table else None
        return _resolve(resolver, schema, name, declared)

    def _get_vector_data_type(self) -> str:
        """The embedding column's element type, e.g. 'FLOAT' or 'DOUBLE'.

        Deprecated since 0.15 and removed in 0.16: ask `shape().vector_spec()`. Kept
        because `iris_vector_rag.pipelines.crag` still calls it.
        """
        return self.shape().vector_spec().element_type

    # ========================================================================
    # IRIS EMBEDDING Support Methods (Feature 051)
    # ========================================================================

    def query_embedding_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Query %Embedding.Config table for configuration details.

        Args:
            config_name: Name of the embedding configuration

        Returns:
            Dictionary with configuration details or None if not found

        Raises:
            VectorStoreConnectionError: If database query fails
        """
        if not self.use_iris_embedding:
            logger.warning("IRIS EMBEDDING support not enabled for this vector store")
            return None

        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            # Query %Embedding.Config table
            sql = """
                SELECT Name, Configuration
                FROM %Embedding.Config
                WHERE Name = ?
            """
            cursor.execute(sql, [config_name])
            row = cursor.fetchone()

            if not row:
                logger.warning(
                    f"EMBEDDING config '{config_name}' not found in %Embedding.Config"
                )
                return None

            # Parse configuration JSON
            config_json = json.loads(row[1]) if row[1] else {}

            return {
                "name": row[0],
                "model_name": config_json.get("modelName", "unknown"),
                "hf_cache_path": config_json.get("hfCachePath", ""),
                "python_path": config_json.get("pythonPath", ""),
                "batch_size": config_json.get("batchSize", 32),
                "device_preference": config_json.get("devicePreference", "auto"),
                "enable_entity_extraction": config_json.get(
                    "enableEntityExtraction", False
                ),
                "entity_types": config_json.get("entityTypes", []),
                "configuration_json": config_json,
            }

        except Exception as e:
            sanitized_error = self._sanitize_error_message(e, "query_embedding_config")
            logger.error(sanitized_error)
            raise VectorStoreConnectionError(
                f"Failed to query EMBEDDING config: {sanitized_error}"
            )
        finally:
            cursor.close()

    def get_embedding_column_metadata(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get metadata about the vector column of this store's table.

        Answered from the resolved shape rather than from `INFORMATION_SCHEMA`: a
        `VECTOR(FLOAT, 384)` column is reported there as `varchar`, so the query this
        replaced (`DATA_TYPE = 'EMBEDDING'`) matched nothing on any real table.

        Args:
            table_name: Name of the table to query. Must be this store's table.

        Returns:
            A single-entry list describing the vector column, or an empty list if the
            table has no vector column.
        """
        if table_name.split(".")[-1].lower() != self._schema_and_table()[1].lower():
            logger.warning(
                "get_embedding_column_metadata asked about %s but this store holds %s",
                table_name,
                self.table_name,
            )
            return []

        shape = self.shape()
        column = shape.optional_column_for(ColumnRole.EMBEDDING)
        if column is None or column.vector is None:
            return []
        return [
            {
                "column_name": column.name,
                "data_type": "VECTOR",
                "element_type": column.vector.element_type,
                "dimension": column.vector.dimension,
                "is_nullable": column.nullable,
            }
        ]

    def search_with_embedding(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search using IRIS EMBEDDING auto-vectorization instead of manual embedding.

        This method leverages IRIS EMBEDDING columns to auto-generate query vectors,
        avoiding the need to manually call embedding functions.

        Args:
            query: The query text (will be auto-vectorized by IRIS)
            top_k: Maximum number of results to return
            filter: Optional metadata filters to apply

        Returns:
            List of tuples containing (Document, similarity_score)

        Raises:
            VectorStoreConnectionError: If EMBEDDING support not enabled or query fails
        """
        # Feature 092: this search takes no `where`, so a table that requires
        # filters cannot be searched through it.
        self._check_where(None)
        if not self.use_iris_embedding:
            raise VectorStoreConnectionError(
                "IRIS EMBEDDING support not enabled. Initialize with embedding_config parameter."
            )

        # Get embedding config details
        config = self.query_embedding_config(self.embedding_config_name)
        if not config:
            raise VectorStoreConnectionError(
                f"EMBEDDING config '{self.embedding_config_name}' not found in database"
            )

        # Use the embedding cache to generate query vector
        from ..embeddings.iris_embedding import embed_texts

        try:
            embedding_result = embed_texts(self.embedding_config_name, [query])
            query_embedding = embedding_result.embeddings[0]

            logger.debug(
                f"Generated query embedding via IRIS EMBEDDING cache "
                f"(cache_hit={embedding_result.cache_hit}, time={embedding_result.embedding_time_ms:.1f}ms)"
            )

        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise VectorStoreConnectionError(f"Query vectorization failed: {e}")

        # Use existing similarity_search_by_embedding method
        return self.similarity_search_by_embedding(query_embedding, top_k, filter)

    # ========================================================================
    # End IRIS EMBEDDING Support Methods
    # ========================================================================

    def _ensure_table_exists(self, cursor):
        """Ensure the target table exists, creating it if necessary.

        Every DDL path in this store goes through here, and so does the one piece of
        bookkeeping DDL owes the rest of the store: the cached shape of a table that
        was just created describes nothing, so it is dropped here rather than at each
        call site (feature 082, FR-006).
        """
        schema, table_short_name = self._schema_and_table()
        try:
            # Check if table exists by trying to query it
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            logger.debug(f"Table {self.table_name} exists")
        except Exception as e:
            logger.info(f"Table {self.table_name} does not exist, creating it: {e}")
            try:
                # Use schema manager to ensure proper table creation
                success = self.schema_manager.ensure_table_schema(table_short_name)
                self._shape_resolver().invalidate(schema, table_short_name)
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

    _RESERVED_TABLE_PARTS = frozenset(
        {
            "DROP",
            "DELETE",
            "INSERT",
            "UPDATE",
            "CREATE",
            "ALTER",
            "TRUNCATE",
            "EXEC",
            "EXECUTE",
            "SELECT",
            "UNION",
            "DECLARE",
            "CONVERT",
            "CAST",
            "SCRIPT",
        }
    )

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

        # The pattern above is the defence: it admits only letters, digits and
        # underscores, so no quote, space, comment marker or semicolon can get through.
        # A substring blacklist on top of it rejected valid names such as
        # `RAG.Descriptions` ("script") and `RAG.Forecasts` ("cast"). What stays is a
        # check that no whole part of the name is itself a SQL keyword.
        for part in table_name.split("."):
            if part.upper() in self._RESERVED_TABLE_PARTS:
                logger.error(
                    f"Security violation: SQL keyword as a table name part: {table_name}"
                )
                raise VectorStoreConfigurationError(
                    f"Table name part '{part}' is a reserved SQL keyword"
                )

        logger.info(f"✅ Custom table name validated: {table_name}")

    def get_allowed_filter_keys(self) -> List[str]:
        """
        Get list of all allowed metadata filter keys (default + custom).

        Returns:
            Sorted list of allowed filter keys

        Example:
            >>> store = IRISVectorStore()
            >>> allowed_keys = store.get_allowed_filter_keys()
            >>> # Returns: ['abstract_type', 'author_name', ..., 'tenant_id', ...]
        """
        return self.metadata_filter_manager.get_allowed_filter_keys()

    def _validate_filter_keys(self, filter_dict: Dict[str, Any]) -> None:
        """
        Validate filter keys against whitelist to prevent SQL injection.

        Uses MetadataFilterManager for validation, which supports custom fields
        configured via storage.iris.custom_filter_keys.

        Args:
            filter_dict: Dictionary of filter key-value pairs

        Raises:
            VectorStoreConfigurationError: If any filter key is not allowed
        """
        if not filter_dict:
            return

        validation_result = self.metadata_filter_manager.validate_filter_keys(
            filter_dict
        )

        if not validation_result.is_valid:
            logger.warning(
                f"Security violation: Invalid filter keys attempted: {validation_result.rejected_keys}"
            )
            raise VectorStoreConfigurationError(validation_result.error_message)

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

            def embedding_func(text):
                return embedding_manager.embed_text(text)

            embeddings = []
            for doc in documents:
                embedding = embedding_func(doc.page_content)
                embeddings.append(embedding)

            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise EmbeddingError(
                f"Failed to generate embeddings: {e}",
                doc_count=len(documents),
                error_type=type(e).__name__,
            ) from e

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
        self._guard_writable()
        if not documents:
            return []

        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            # The shape both proves the table exists (creating it if it does not) and
            # names every column written below. A caller-supplied identifier aimed at
            # a database-generated column fails here, before any SQL (FR-008).
            shape = self.shape(cursor)
            identifier = shape.writable_identifier()
            text = shape.column_for(ColumnRole.TEXT)
            metadata_column = shape.optional_column_for(ColumnRole.METADATA)
            title_column = shape.optional_column_for(ColumnRole.TITLE)
            vector = shape.vector_spec() if embeddings else None
            embedding_column = (
                shape.column_for(ColumnRole.EMBEDDING) if embeddings else None
            )

            # Every width is checked before the first row is written. The session's
            # connection is in autocommit mode, so a rollback cannot take back the rows
            # written before a bad vector: the batch has to be refused whole up front.
            if embeddings and vector is not None:
                for i, embedding in enumerate(embeddings[: len(documents)]):
                    if len(embedding) != vector.dimension:
                        raise DimensionMismatchError(
                            len(embedding),
                            vector.dimension,
                            f"{shape.qualified_name}.{embedding_column.name} "
                            f"(document {documents[i].id})",
                            document_id=documents[i].id,
                        )

            added_ids = []
            logger.debug(
                f"_store_documents called with {len(documents)} documents and embeddings: {embeddings is not None}"
            )

            # The drop stays -- a table with no metadata column has nowhere to put it,
            # and refusing the write would break every caller of such a table. It stops
            # being silent (FR-016). Once per table: warning on every document is how a
            # real warning gets scrolled past during a bulk ingest.
            if (
                metadata_column is None
                and shape.qualified_name not in self._metadata_drop_warned
                and any(document.metadata for document in documents)
            ):
                self._metadata_drop_warned.add(shape.qualified_name)
                logger.warning(
                    f"{shape.qualified_name} has no metadata column: the metadata on "
                    f"these documents is being dropped, and filtering this table by "
                    f"metadata will fail. Warned once per table."
                )

            for i, doc in enumerate(documents):
                # Only the canonical column for each role is written. A table that
                # also carries the older spelling keeps it readable through the
                # per-row fallback on the read path; writing both would create two
                # copies of one fact that can disagree after an update (FR-003).
                values: Dict[str, Any] = {text.name: doc.page_content}
                if metadata_column is not None:
                    values[metadata_column.name] = json.dumps(doc.metadata)
                title = doc.metadata.get("title") or doc.metadata.get("source")
                if title_column is not None and title:
                    values[title_column.name] = title

                if embeddings and len(embeddings) > i:
                    # The store asks IRIS whether the row exists instead of trying
                    # the INSERT and reading its failure: ivg logs a failed INSERT at
                    # ERROR level, and this driver reports a duplicate key as
                    # "<LIST ERROR> Incorrect list format", so every re-ingested
                    # document used to log an error although the call succeeded.
                    if self._row_exists(cursor, identifier.name, doc.id):
                        self._update_with_vector(
                            cursor,
                            identifier.name,
                            doc.id,
                            values,
                            embedding_column.name,
                            embeddings[i],
                            vector,
                        )
                        success = True
                    else:
                        success = insert_vector(
                            cursor=cursor,
                            table_name=self.table_name,
                            vector_column=embedding_column.name,
                            vector_data=embeddings[i],
                            dimension=vector.dimension,
                            key_columns={identifier.name: doc.id},
                            additional_columns=values,
                            dtype=vector.element_type,
                            upsert=False,
                        )
                    if not success:
                        # ivg has logged the cause. Carrying on would commit the rest of
                        # the batch and return a partial id list as if it were whole.
                        raise VectorStoreDataError(
                            f"Failed to store document {doc.id} with its vector; "
                            f"the batch stopped there (rows before it may already be "
                            f"stored if the connection autocommits)",
                            document_id=doc.id,
                            stored_ids=list(added_ids),
                        )
                    added_ids.append(doc.id)
                    logger.debug(f"Successfully upserted document {doc.id} with vector")
                    continue

                check_sql = (
                    f"SELECT COUNT(*) FROM {self.table_name} "
                    f"WHERE {identifier.name} = ?"
                )
                cursor.execute(check_sql, [doc.id])
                exists = cursor.fetchone()[0] > 0

                if exists:
                    columns = list(values)
                    update_sql = (
                        f"UPDATE {self.table_name} SET "
                        f"{', '.join(f'{column} = ?' for column in columns)} "
                        f"WHERE {identifier.name} = ?"
                    )
                    cursor.execute(
                        update_sql, [values[column] for column in columns] + [doc.id]
                    )
                    logger.debug(f"Updated existing document {doc.id} without vector")
                else:
                    columns = [identifier.name] + list(values)
                    insert_sql = (
                        f"INSERT INTO {self.table_name} "
                        f"({', '.join(columns)}) VALUES "
                        f"({', '.join('?' for _ in columns)})"
                    )
                    cursor.execute(
                        insert_sql,
                        [doc.id] + [values[column] for column in list(values)],
                    )
                    logger.debug(f"Inserted new document {doc.id} without vector")

                added_ids.append(doc.id)

            connection.commit()
            logger.info(f"Successfully stored {len(added_ids)} documents")
            return added_ids

        except (TableShapeError, DimensionMismatchError, VectorStoreDataError):
            # The shape errors already name the table and the real cause; wrapping
            # them in a storage failure is what used to send callers looking a
            # layer away for a problem in their own schema.
            connection.rollback()
            raise
        except Exception as e:
            connection.rollback()
            error_msg = self._sanitize_error_message(e, "document storage")
            logger.error(error_msg)
            raise VectorStoreDataError(f"Failed to store documents: {error_msg}")
        finally:
            cursor.close()

    def _row_exists(self, cursor, identifier: str, doc_id: str) -> bool:
        cursor.execute(
            f"SELECT COUNT(*) FROM {self.table_name} WHERE {identifier} = ?", [doc_id]
        )
        row = cursor.fetchone()
        return bool(row and row[0] > 0)

    def _update_with_vector(
        self,
        cursor,
        identifier: str,
        doc_id: str,
        values: Dict[str, Any],
        vector_column: str,
        embedding: List[float],
        vector,
    ) -> None:
        """UPDATE an existing row, its vector included. The width is already checked."""
        self._guard_writable()
        processed = normalize_vector(embedding, vector.dimension)
        assignments = [f"{column} = ?" for column in values]
        assignments.append(
            f"{vector_column} = TO_VECTOR(?, {vector.element_type}, {vector.dimension})"
        )
        cursor.execute(
            f"UPDATE {self.table_name} SET {', '.join(assignments)} "
            f"WHERE {identifier} = ?",
            [*values.values(), "[" + ",".join(map(str, processed)) + "]", doc_id],
        )

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
        self._guard_writable()
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
        self._guard_writable()
        if not ids:
            return True

        id_col = self.shape().column_for(ColumnRole.IDENTIFIER).name

        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            placeholders = ",".join(["?" for _ in ids])
            delete_sql = (
                f"DELETE FROM {self.table_name} WHERE {id_col} IN ({placeholders})"
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

    # Stream (LONGVARCHAR) metadata columns cannot be used with LIKE directly:
    # IRIS silently matches nothing. SUBSTRING projects the stream to a string.
    _STREAM_METADATA_PROJECTION = 32000

    def _metadata_filter_column_expr(self, metadata_column: ColumnFact) -> str:
        """SQL expression for the metadata column that LIKE can evaluate.

        The shape already knows whether the column is filterable, so this needs no
        query of its own; before feature 082 it read the catalog once per column and
        cached the answer itself (FR-005, FR-009).
        """
        if metadata_column.filterable_with_like:
            return metadata_column.name
        return f"SUBSTRING({metadata_column.name},1,{self._STREAM_METADATA_PROJECTION})"

    @staticmethod
    def _metadata_filter_predicate(filter: Dict[str, Any], metadata_expr: str) -> str:
        """
        Build the SQL predicate for a metadata filter over serialized JSON.

        Matches the JSON member `"key": <literal>` with or without a space after
        the colon (json.dumps emits the space; hand-written JSON often does not).
        The literal is the JSON encoding of the value, so strings are quoted and
        numbers/booleans are not. The closing quote of a string literal rules out
        prefix collisions; the remaining approximation (the same member at a
        nested depth, or the pattern inside a longer string) is removed by the
        exact-match backstop in similarity_search_by_embedding.
        """
        conditions = []
        for key, value in filter.items():
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(key)):
                raise VectorStoreDataError(f"Invalid metadata filter key: {key!r}")
            conditions.append(json_member_predicate(key, value, metadata_expr))
        return " AND ".join(conditions)

    @staticmethod
    def _apply_exact_metadata_filter(rows, metadata_map, filter):
        """Drop rows whose parsed metadata does not match exactly; return (kept, dropped)."""
        kept = []
        dropped = 0
        for row in rows:
            meta = metadata_map.get(row[0], {}) or {}
            if all(meta.get(k) == v for k, v in filter.items()):
                kept.append(row)
            else:
                dropped += 1
        return kept, dropped

    def _log_row_counts(self, cursor, embedding_column: str) -> None:
        """Explain an empty result set by counting the rows (LOG-004, feature 033).

        An empty result has two very different causes -- an empty table and a table
        whose embeddings were never written -- and these two counts are what tells
        them apart. They are read only when the search found nothing, because a
        `COUNT(*)` over the whole table on every search is two extra round trips
        buying an explanation nobody needs (SC-002). The embedding column is the one
        the table has, passed in from the shape, so this diagnostic cannot name a
        column the query did not use (FR-001).
        """
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            total = cursor.fetchone()[0]
            logger.debug(f"Total documents in {self.table_name}: {total}")

            cursor.execute(
                f"SELECT COUNT(*) FROM {self.table_name} "
                f"WHERE {embedding_column} IS NOT NULL"
            )
            embedded = cursor.fetchone()[0]
            logger.debug(f"Documents with embeddings: {embedded}")
        except Exception as exc:
            # Diagnostics never fail a search: the counts are here to explain a
            # result, not to produce one.
            logger.debug(f"Unable to count rows in {self.table_name}: {exc}")
            logger.debug("Documents with embeddings: 0")

    def similarity_search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search using a query embedding.

        Args:
            query_embedding: The query vector for similarity search
            top_k: Maximum number of results to return
            filter: Optional metadata filters to apply
            where: Optional filters on real columns (feature 092): ``{col: value}``,
                ``{col: (op, value)}`` or ``{col: [values]}``, on ``filter_columns``

        Returns:
            List of tuples containing (Document, similarity_score)

        Raises:
            RequiredFilterMissing: a ``required_filters`` column is not in ``where``.
            FilterError: ``where`` cannot be applied as written.
        """
        # Validate filter if provided
        if filter:
            self._validate_filter_keys(filter)
            self._validate_filter_values(filter)

        # Resolved before the cursor is opened, so a table that cannot answer a
        # vector search says so instead of failing inside a SQL error several
        # layers away, and so these errors are not caught by the handler below.
        shape = self.shape()
        where_sql, where_params = compile_where(where, shape, self.filter_spec)
        self._warn_on_index_mismatch(shape)
        vector = shape.vector_spec()
        identifier = shape.column_for(ColumnRole.IDENTIFIER)
        text = shape.column_for(ColumnRole.TEXT)
        legacy_text = shape.legacy_column_for(ColumnRole.TEXT)
        metadata_column = shape.optional_column_for(ColumnRole.METADATA)

        if len(query_embedding) != vector.dimension:
            error = DimensionMismatchError(
                len(query_embedding),
                vector.dimension,
                f"{shape.qualified_name}.{shape.column_for(ColumnRole.EMBEDDING).name} "
                f"(query)",
            )
            logger.error(str(error))
            raise error

        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            # Use the safe vector utilities (proven pattern)
            from iris_vector_rag.common.vector_sql_utils import (
                build_bound_vector_dot_sql,
                execute_safe_vector_search,
            )

            additional_where = None
            if filter:
                # A filter this table cannot apply is an error, not an empty page:
                # silence here is indistinguishable from a scope that genuinely holds
                # no rows (FR-016).
                if metadata_column is None:
                    raise RoleNotPresent(
                        "a metadata filter was supplied but this table has no metadata "
                        f"column, so the filter cannot be applied (filter keys: "
                        f"{sorted(filter)})",
                        shape.schema,
                        shape.table,
                    )
                # Push the filter into SQL so recall is scoped to the filtered
                # subset instead of to a global page (opsreview upstream report:
                # docs/upstream/ivr-metadata-filter-pushdown.md).
                additional_where = self._metadata_filter_predicate(
                    filter, self._metadata_filter_column_expr(metadata_column)
                )
            # The real-column predicate goes first: it never starts with "(", so the
            # builder's own parenthesis is never doubled.
            additional_where = (
                " AND ".join(p for p in (where_sql, additional_where) if p) or None
            )

            logger.debug(f"Query embedding dimensions: {len(query_embedding)}")
            logger.debug(f"Top-K parameter: {top_k}")

            embedding_column = shape.sql_name(ColumnRole.EMBEDDING)

            # The filter is applied in SQL; the page is top_k. The exact-match
            # check below is only a backstop for LIKE approximations and widens
            # the page once if it actually drops rows.
            search_top_k = top_k

            # The query vector is bound: TO_VECTOR(?, <type>, <dim>), with the type and
            # dimension as literals, works for FLOAT and DOUBLE columns on IRIS 2026.1
            # and later (see build_bound_vector_dot_sql).
            embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # The older spelling of the text column is selected alongside the
            # canonical one on a mixed table, so a row written before 0.15 can be
            # presented from whichever column actually holds its text (FR-004).
            text_columns = [shape.sql_name(ColumnRole.TEXT)]
            if legacy_text is not None and not shape.attached:
                text_columns.append(legacy_text.name)
            # An attached table's title and parent ride along, for Document.metadata.
            extras = self._metadata_extras(shape)

            def _build_sql(page_size: int) -> str:
                return build_bound_vector_dot_sql(
                    table=self.table_name,
                    vector_column=embedding_column,
                    vector_dimension=vector.dimension,
                    id_column=identifier.name,
                    extra_columns=text_columns + [sql for _, sql in extras],
                    top_k=page_size,
                    additional_where=additional_where,
                    vector_data_type=vector.element_type,
                    metric=self.metric,
                    id_expression=shape.identifier_sql if shape.attached else None,
                )

            def _fetch_page(page_size: int):
                """Run the ranked query for one page and load its metadata."""
                sql = _build_sql(page_size)
                logger.debug(f"SQL query executed: {sql}")
                page_rows = execute_safe_vector_search(
                    cursor, sql, [embedding_str, *where_params]
                )

                page_metadata = {}
                if page_rows and metadata_column is not None:
                    # Handle Mock objects that aren't iterable
                    try:
                        doc_ids = [row[0] for row in page_rows]
                        placeholders = ",".join(["?" for _ in doc_ids])
                        id_sql = shape.identifier_sql
                        metadata_sql = (
                            f"SELECT {id_sql}, {shape.sql_name(ColumnRole.METADATA)} "
                            f"FROM {self.table_name} "
                            f"WHERE {id_sql} IN ({placeholders})"
                        )
                        if where_sql:
                            metadata_sql += f" AND {where_sql}"
                        cursor.execute(metadata_sql, [*doc_ids, *where_params])
                        for doc_id, metadata_json in cursor.fetchall():
                            if isinstance(metadata_json, dict):
                                page_metadata[doc_id] = metadata_json
                                continue
                            if not metadata_json:
                                page_metadata[doc_id] = {}
                                continue
                            try:
                                page_metadata[doc_id] = json.loads(metadata_json)
                            except Exception:
                                page_metadata[doc_id] = {}
                    except (TypeError, AttributeError):
                        logger.debug(
                            "Rows is not iterable (likely a Mock object), skipping metadata fetch"
                        )
                        page_metadata = {}
                return page_rows, page_metadata

            logger.debug(
                f"Executing safe vector search with {len(query_embedding)}D vector"
            )
            rows, metadata_map = _fetch_page(search_top_k)

            if filter:
                try:
                    page_was_full = len(rows) >= search_top_k
                except (TypeError, AttributeError):
                    page_was_full = False
                rows, dropped = self._apply_exact_metadata_filter(
                    rows, metadata_map, filter
                )
                if dropped and page_was_full and len(rows) < top_k:
                    # The LIKE predicate admitted rows the exact check rejected and
                    # the page was full, so matching rows may sit just past it:
                    # fetch one wider page rather than paying an over-fetch always.
                    wider = max(top_k * 5, top_k + dropped)
                    logger.debug(
                        "metadata filter backstop dropped %d row(s); re-fetching top %d",
                        dropped,
                        wider,
                    )
                    rows, metadata_map = _fetch_page(wider)
                    rows, _ = self._apply_exact_metadata_filter(
                        rows, metadata_map, filter
                    )
                    rows = rows[:top_k]

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
                doc_id, similarity_score = row[0], row[-1]
                # One presented form regardless of which column holds the text:
                # the canonical column when it is populated, the older spelling
                # otherwise (FR-004). Column order follows `text_columns`.
                text_end = 1 + len(text_columns)
                content = next(
                    (value for value in row[1:text_end] if value not in (None, "")),
                    "",
                )

                # Get metadata from the map
                metadata_json = metadata_map.get(doc_id, None)

                # Process row data to ensure string content
                document_data = {
                    "doc_id": doc_id,
                    "text_content": content,
                    "metadata": metadata_json,
                }

                document = self._ensure_string_content(document_data)
                for (key, _), value in zip(extras, row[text_end:-1]):
                    if value is not None:
                        document.metadata[key] = value
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
                if result_count == 0:
                    logger.info("Vector search returned 0 results")
                    logger.debug("Similarity scores: []")
                    self._log_row_counts(cursor, embedding_column)
                else:
                    logger.debug(f"Vector search returned {result_count} results")
                    similarity_scores = [score for _, score in results]
                    logger.debug(f"Similarity scores: {similarity_scores}")
            except (TypeError, AttributeError):
                # Handle Mock objects or other non-sequence types
                logger.debug(
                    "Vector search returned results (count unavailable due to mock object)"
                )
            return results

        except TableShapeError:
            # A shape problem is a statement about the table, not about the
            # connection. Wrapping it in VectorStoreConnectionError would rename the
            # one thing the caller needs to read (FR-016).
            raise
        except Exception as e:
            logger.info("Vector search returned 0 results")
            logger.debug("Similarity scores: []")
            sanitized_error = self._sanitize_error_message(e, "similarity_search")
            logger.error(sanitized_error)
            raise VectorStoreConnectionError(f"Vector search failed: {sanitized_error}")
        finally:
            cursor.close()

    def _metadata_extras(self, shape: TableShape) -> List[Tuple[str, str]]:
        """(metadata key, SQL column) for an attached table's title and parent.

        Empty for the library's own tables, so their statements are unchanged.
        """
        if not shape.attached:
            return []
        return [
            (key, shape.sql_name(role))
            for key, role in _METADATA_ROLES
            if shape.has(role)
        ]

    def _warn_on_index_mismatch(self, shape: TableShape) -> None:
        """Warn once per table when its HNSW index was built for another measure.

        IRIS uses an HNSW index only for the query function its ``Distance`` names,
        so a mismatch means every search scans the table (FR-013).
        """
        if shape.qualified_name in self._hnsw_checked or not shape.has(
            ColumnRole.EMBEDDING
        ):
            return
        self._hnsw_checked.add(shape.qualified_name)
        resolver = self._shape_resolver()
        ask = getattr(resolver, "hnsw_index", None)
        found = (
            ask(shape.class_name, shape.column_for(ColumnRole.EMBEDDING).name)
            if ask
            else None
        )
        if not found:
            return
        index, distance = found
        if _HNSW_DISTANCES.get(distance.lower()) != self.metric:
            logger.warning(
                f"HNSW index {index} on {shape.qualified_name} was built with "
                f"Distance='{distance}' but this store ranks by metric={self.metric!r}, "
                "so IRIS will not use the index and every search scans the table. "
                f"Set metric to match the index, or rebuild it."
            )

    def _guard_writable(self) -> None:
        """Refuse to write a table the library reads through declared columns (092)."""
        if self.column_mapping.attached:
            raise AttachedTableReadOnly(
                f"{self.table_name} is an attached table: the library reads it through "
                "declared columns and never writes, alters or deletes from it. Write "
                "through the code that owns the table.",
                table=self.table_name,
            )

    def _read_columns(self) -> List[str]:
        """The columns a plain read selects, in order.

        Identifier, then the text column and the older spelling of it where the
        table carries both, then metadata where the table has it, then an attached
        table's title and parent.
        """
        shape = self.shape()
        columns = [shape.identifier_sql, shape.sql_name(ColumnRole.TEXT)]
        legacy_text = shape.legacy_column_for(ColumnRole.TEXT)
        if legacy_text is not None and not shape.attached:
            columns.append(legacy_text.name)
        if shape.has(ColumnRole.METADATA):
            columns.append(shape.sql_name(ColumnRole.METADATA))
        columns.extend(sql for _, sql in self._metadata_extras(shape))
        return columns

    def _documents_from_rows(self, rows, columns: List[str]) -> List[Document]:
        """Rows selected through `_read_columns`, presented as documents."""
        shape = self.shape()
        extras = self._metadata_extras(shape)
        data_end = len(columns) - len(extras)
        metadata_index = data_end - 1 if shape.has(ColumnRole.METADATA) else None
        text_end = data_end if metadata_index is None else metadata_index

        documents = []
        for row in rows:
            text = next(
                (value for value in row[1:text_end] if value not in (None, "")), ""
            )
            document = self._ensure_string_content(
                {
                    "doc_id": row[0],
                    "text_content": text,
                    "metadata": (
                        None if metadata_index is None else row[metadata_index]
                    ),
                }
            )
            for (key, _), value in zip(extras, row[data_end:]):
                if value is not None:
                    document.metadata[key] = value
            documents.append(document)
        return documents

    def fetch_documents_by_ids(
        self,
        ids: List[str],
        where_sql: str = "",
        where_params: Sequence[Any] = (),
    ) -> List[Document]:
        """
        Fetch documents by their IDs.

        Args:
            ids: List of document IDs to fetch
            where_sql: A predicate from `compile_where`, ANDed with the id list, so a
                search hydrates its hits under the scope it ranked them in: an id
                need only be unique within the required filters (feature 092).
            where_params: That predicate's parameters.

        Returns:
            List of Document objects with guaranteed string content
        """
        if not ids:
            return []

        columns = self._read_columns()

        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            placeholders = ",".join(["?" for _ in ids])
            # One statement, built from the shape. The canonical-then-legacy retry
            # this replaces guessed at the schema by letting a statement fail.
            select_sql = (
                f"SELECT {', '.join(columns)} FROM {self.table_name} "
                f"WHERE {columns[0]} IN ({placeholders})"
            )
            if where_sql:
                select_sql += f" AND {where_sql}"
            cursor.execute(select_sql, [*ids, *where_params])
            documents = self._documents_from_rows(cursor.fetchall(), columns)
            logger.debug(f"Fetched {len(documents)} documents by IDs")
            return documents

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
        columns = self._read_columns()

        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            cursor.execute(f"SELECT {', '.join(columns)} FROM {self.table_name}")
            documents = self._documents_from_rows(cursor.fetchall(), columns)
            logger.debug(f"Retrieved {len(documents)} documents")
            return documents

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
        self._guard_writable()
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
        # Check if first argument is a string OR if 'query' kwarg is a string (LangChain interface)
        if (args and isinstance(args[0], str)) or (
            not args and "query" in kwargs and isinstance(kwargs["query"], str)
        ):
            # LangChain interface: similarity_search(query, k, filter)
            query = args[0] if args else kwargs["query"]
            k = args[1] if len(args) > 1 else kwargs.get("k", 4)
            # Support both 'filter' and 'metadata_filter' parameter names for backward compatibility
            filter_param = (
                args[2]
                if len(args) > 2
                else kwargs.get("filter") or kwargs.get("metadata_filter")
            )

            # Get embedding function for text query
            where = kwargs.get("where")
            self._check_where(where)
            embedding_func = kwargs.get("embedding_func")
            if not embedding_func:
                query_embedding = self._embed_query(query)
            else:
                query_embedding = embedding_func.embed_query(query)

            # Call our implementation and add scores to document metadata
            results = self._by_embedding(query_embedding, k, filter_param, where)
            # Add score to each document's metadata for MCP/API compatibility
            documents_with_scores = []
            for doc, score in results:
                # Create new document with score in metadata
                enriched_metadata = {
                    **doc.metadata,
                    "score": score,
                    "similarity": score,
                }
                enriched_doc = Document(
                    id=doc.id, page_content=doc.page_content, metadata=enriched_metadata
                )
                documents_with_scores.append(enriched_doc)
            return documents_with_scores

        else:
            # Base class interface: similarity_search(query_embedding, top_k, filter)
            query_embedding = args[0] if args else kwargs["query_embedding"]
            top_k = args[1] if len(args) > 1 else kwargs.get("top_k", 5)
            filter_param = args[2] if len(args) > 2 else kwargs.get("filter")

            return self._by_embedding(
                query_embedding, top_k, filter_param, kwargs.get("where")
            )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search and return results with scores.

        Args:
            query: Text query to search for
            k: Maximum number of results to return
            filter: Optional metadata filters to apply

        Returns:
            List of tuples containing (Document, similarity_score)
        """
        self._check_where(where)
        query_embedding = self._embed_query(query)

        # Call similarity_search_by_embedding which already returns (doc, score) tuples
        return self._by_embedding(query_embedding, k, filter, where)

    def search_by_text(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Return List[Document] for a text query (FR-014).

        Embeds ``query`` and delegates to similarity_search_by_embedding.
        Always returns plain Documents — scores are available via search_by_vector.
        """
        self._check_where(where)
        embedding = self._embed_query(query)
        results = self._by_embedding(embedding, top_k, metadata_filter, where)
        return [doc for doc, _ in results]

    def search_by_text_with_score(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return (Document, similarity score) pairs for a text query, best first.

        The scored sibling of `search_by_text`, which stays score-free (FR-014).
        """
        self._check_where(where)
        embedding = self._embed_query(query)
        return self._by_embedding(embedding, top_k, metadata_filter, where)

    def _text_index(self, shape: TableShape) -> Optional[str]:
        """The iFind index the text leg reads: the configured one, else the library's.

        An attached table has no default: the library never builds one there, so
        only an index the owner named can be read.
        """
        if self.text_index_name:
            return self.text_index_name
        return None if shape.attached else TEXT_INDEX_NAME

    def _require_text_index(self, shape: TableShape) -> str:
        """The text index name; `TextIndexMissingError` when the table has none."""
        index = self._text_index(shape)
        if index is None:
            raise TextIndexMissingError(
                f"{shape.qualified_name} is an attached table with no text index named, "
                "so text, hybrid and rrf retrieval cannot search it. Set text_index to "
                "the name of its iFind index; the library does not create one on a "
                "table it does not own.",
                table=shape.qualified_name,
            )
        if index == TEXT_INDEX_NAME:
            present = self.schema_manager.has_text_index(shape.qualified_name)
        else:
            present = self.schema_manager.has_text_index(
                shape.qualified_name, index_name=index
            )
        if not present:
            if shape.attached:
                fix = "Check text_index: the library will not create an index here."
            else:
                fix = (
                    f"Create it once with SchemaManager(...).ensure_text_index("
                    f"{shape.table!r}); it is built over the rows already in the table."
                )
            raise TextIndexMissingError(
                f"{shape.qualified_name} has no full-text index {index}, so text, "
                f"phrase, hybrid and rrf retrieval cannot search it. {fix}",
                table=shape.qualified_name,
                index=index,
            )
        return index

    def _ranked_text_rows(
        self,
        shape: TableShape,
        search_string: str,
        page: int,
        extra_where: Optional[str],
        index: str = TEXT_INDEX_NAME,
        where_params: Sequence[Any] = (),
    ) -> List[tuple]:
        """``(id, rank)`` for the best ``page`` rows matching ``search_string``."""
        sql = build_text_search_sql(
            table=self.table_name,
            id_column=shape.column_for(ColumnRole.IDENTIFIER).name,
            index_name=index,
            rank_proc=rank_procedure(shape.schema, shape.class_name, index),
            top_k=page,
            extra_where=extra_where,
            id_expression=shape.identifier_sql if shape.attached else None,
        )
        connection = self._get_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(
                sql, [search_string, search_string, *where_params, search_string]
            )
            return [(str(row[0]), float(row[1])) for row in cursor.fetchall()]
        except Exception as exc:
            raise VectorStoreConnectionError(
                f"Text search failed: {self._sanitize_error_message(exc, 'text_search')}"
            ) from exc
        finally:
            cursor.close()

    def text_search(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Full-text search over the document table, best match first.

        Searches through the table's iFind index (`SchemaManager.ensure_text_index`),
        on whichever column the catalog says holds the text. Each hit is a complete
        document whose ``metadata["text_score"]`` is the index's TF-IDF rank. A metadata
        filter is applied in SQL, like the vector search, and again on the parsed
        metadata; if that drops rows the page is widened once.

        A query with no searchable words returns ``[]``: nothing in it can match.

        Raises:
            TextIndexMissingError: the table has no full-text index. An empty answer
                would read as "no document matches", so the absence is an error.
            RoleNotPresent: a filter was given and the table has no metadata column.
            VectorStoreConnectionError: the database rejected the search.
        """
        if metadata_filter:
            self._validate_filter_keys(metadata_filter)
            self._validate_filter_values(metadata_filter)

        shape = self.shape()
        shape.column_for(ColumnRole.TEXT)  # a table with no text cannot be searched
        metadata_column = shape.optional_column_for(ColumnRole.METADATA)
        if metadata_filter and metadata_column is None:
            raise RoleNotPresent(
                "a metadata filter was supplied but this table has no metadata "
                f"column, so the filter cannot be applied (filter keys: "
                f"{sorted(metadata_filter)})",
                shape.schema,
                shape.table,
            )

        where_sql, where_params = compile_where(where, shape, self.filter_spec)
        index = self._require_text_index(shape)

        terms = search_terms(query, language=self.text_language, terms=self.text_terms)
        if not terms:
            return []

        extra_where = None
        if metadata_filter:
            extra_where = self._metadata_filter_predicate(
                metadata_filter, self._metadata_filter_column_expr(metadata_column)
            )
        # The real-column predicate first: it never starts with "(" (see R2).
        extra_where = " AND ".join(p for p in (where_sql, extra_where) if p) or None
        page = top_k
        for _ in range(2):
            rows = self._ranked_text_rows(
                shape, terms, page, extra_where, index, where_params
            )

            by_id = {
                str(d.id): d
                for d in self.fetch_documents_by_ids(
                    [r[0] for r in rows], where_sql, where_params
                )
            }
            hits = []
            for doc_id, score in rows:
                doc = by_id.get(doc_id)
                if doc is None:
                    continue
                doc.metadata["text_score"] = score
                hits.append(doc)
            if not metadata_filter:
                return hits[:top_k]
            kept = [
                d
                for d in hits
                if all(d.metadata.get(k) == v for k, v in metadata_filter.items())
            ]
            # Widen once when the exact check dropped rows and the page was full.
            if len(kept) == len(hits) or len(rows) < page:
                return kept[:top_k]
            page = top_k * 10
        return kept[:top_k]

    #: Most section rows one phrase search reads while looking for ``top_k`` parents.
    PHRASE_READ_BOUND = 2000

    def phrase_search(
        self,
        phrase: str,
        top_k: int = 5,
        section: Optional[str] = None,
        detector: Optional[Any] = None,
    ) -> List[Document]:
        """Exact-phrase search over the table, rolled up to parent documents.

        The phrase must match words adjacent and in order
        (specs/088-note-search-first-slice/contracts/query-semantics.md). Rows that carry
        a ``parent_id`` in their metadata are sections: their hits roll up to one
        document per parent, scored by the best section. Other rows are their own
        parent. Each hit's ``metadata`` holds ``text_score``, ``polarity`` (from
        ``detector``, the reference detector by default) and ``matched_sections``.

        ``section`` limits the search to sections of that label, in SQL, before the
        page is cut. Sections are read past ``top_k`` until ``top_k`` parents are found,
        the matches run out, or `PHRASE_READ_BOUND` is reached; in the last case every
        hit carries ``metadata["search_truncated"] = True``.

        Raises:
            ValueError: the phrase has no letter or digit.
            TextIndexMissingError: the table has no full-text index.
        """
        required = self.filter_spec.required_filters
        if required:
            raise RequiredFilterMissing(
                f"{self.table_name} requires a filter on {', '.join(required)}, and "
                "phrase search takes no where=, so it would read every row. Use "
                "retrieval 'text', 'hybrid' or 'rrf' with where=.",
                rejected_keys=list(required),
            )
        from iris_vector_rag.note_search.phrase import (
            find_mentions,
            phrase_search_string,
        )
        from iris_vector_rag.note_search.polarity import (
            ReferenceNegExDetector,
            section_polarity,
        )
        from iris_vector_rag.note_search.rollup import SectionHit, roll_up
        from iris_vector_rag.note_search.store import SECTION_COLUMN

        search_string = phrase_search_string(phrase)  # raises ValueError first
        detector = detector or ReferenceNegExDetector(language=self.text_language)

        shape = self.shape()
        shape.column_for(ColumnRole.TEXT)
        metadata_column = shape.optional_column_for(ColumnRole.METADATA)
        if section is not None and metadata_column is None:
            raise RoleNotPresent(
                "a section filter was supplied but this table has no metadata column",
                shape.schema,
                shape.table,
            )
        self._require_text_index(shape)

        extra_where = None
        if section is not None:
            # The opt-in section_label column (bitmap-indexed) when the table has it;
            # the metadata LIKE otherwise. Both are approximations that the parsed
            # metadata check below makes exact, and they agree on every section row.
            if SECTION_COLUMN in shape.columns:
                extra_where = section_filter_predicate(section, column=SECTION_COLUMN)
            else:
                extra_where = section_filter_predicate(
                    section,
                    metadata_expr=self._metadata_filter_column_expr(metadata_column),
                )

        page = max(int(top_k) * 4, 20)
        while True:
            rows = self._ranked_text_rows(shape, search_string, page, extra_where)
            by_id = {
                str(d.id): d for d in self.fetch_documents_by_ids([r[0] for r in rows])
            }
            hits = []
            for row_id, score in rows:
                doc = by_id.get(row_id)
                if doc is None:
                    continue
                meta = doc.metadata or {}
                if section is not None and meta.get("section") != section:
                    continue  # the LIKE approximated; the parsed metadata decides
                mentions = find_mentions(doc.page_content, phrase)
                if not mentions:
                    continue  # the index proposed it; the re-scan found no phrase
                polarity = section_polarity(
                    detector(doc.page_content, m) for m in mentions
                )
                hits.append(
                    SectionHit(
                        id=row_id,
                        parent_id=str(meta.get("parent_id") or row_id),
                        label=str(meta.get("section", "")),
                        text=doc.page_content,
                        score=score,
                        polarity=polarity,
                        ordinal=int(meta.get("ordinal", 0) or 0),
                    )
                )
            parents = roll_up(hits, k=top_k)
            exhausted = len(rows) < page
            enough = len({h.parent_id for h in hits}) >= top_k
            if exhausted or enough:
                truncated = False
                break
            if page >= self.PHRASE_READ_BOUND:
                truncated = True
                logger.warning(
                    "phrase_search read %d sections and found %d of %d parents",
                    page,
                    len(parents),
                    top_k,
                )
                break
            page = min(page * 4, self.PHRASE_READ_BOUND)

        documents = []
        for parent in parents:
            metadata = {
                "text_score": parent.score,
                "polarity": parent.polarity,
                "matched_sections": [
                    {
                        "id": s.id,
                        "label": s.label,
                        "text_score": s.score,
                        "polarity": s.polarity,
                    }
                    for s in parent.matched_sections
                ],
            }
            if truncated:
                metadata["search_truncated"] = True
            documents.append(
                Document(page_content=parent.text, metadata=metadata, id=parent.id)
            )
        return documents

    def search_by_vector(
        self,
        embedding: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Return List[Tuple[Document, float]] for a pre-computed embedding (FR-014).

        Always returns (Document, score) tuples — use search_by_text for plain docs.
        """
        return self._by_embedding(embedding, top_k, metadata_filter, where)

    def _check_where(self, where: Optional[Dict[str, Any]]) -> None:
        """Validate ``where`` (required filters first) before anything is embedded.

        A store with no required filters and no ``where`` checks nothing, so the
        search path is the one it was before feature 092.
        """
        spec = getattr(self, "filter_spec", None)
        if where or (spec is not None and spec.required_filters):
            compile_where(where, self.shape(), spec)

    def _by_embedding(self, embedding, top_k, metadata_filter, where):
        """`similarity_search_by_embedding`, with ``where`` passed as `where_kwargs`."""
        return self.similarity_search_by_embedding(
            embedding, top_k, metadata_filter, **where_kwargs(where)
        )

    def _embed_query(self, query: str) -> List[float]:
        """Embed a text query: the caller's embedder, else the configured provider."""
        if self.embedder is not None:
            return list(self.embedder(query))
        from ..embeddings.manager import EmbeddingManager

        embedding_manager = EmbeddingManager(self.config_manager)
        return embedding_manager.embed_text(query)
