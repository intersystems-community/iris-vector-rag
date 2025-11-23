# Feature Specification: Enterprise Enhancements for iris-vector-rag

**Feature ID**: F-ENT-001
**Version**: 1.0.0
**Status**: Draft
**Date**: 2025-11-22
**Author**: Based on AI Hub integration experience
**Priority**: High (P0 for items 1-3, P1 for items 4-5, P2 for items 6-8)

---

## Executive Summary

This specification consolidates eight enterprise enhancement proposals for iris-vector-rag, identified during real-world integration with the AI Hub platform. These enhancements address production deployment requirements while maintaining backward compatibility.

**Key Enhancements**:
1. **Configurable Metadata Filter Keys** - Extend security allowlist for custom fields
2. **Multi-Collection Management API** - Explicit virtual collection lifecycle
3. **RBAC Integration Hooks** - Enterprise security integration
4. **OpenTelemetry Instrumentation** - Production observability
5. **Flexible Connection Parameters** - Dependency injection support
6. **Metadata Schema Discovery** - Runtime schema inspection
7. **JSON_VALUE Compatibility** - Auto-setup for IRIS compatibility
8. **Batch Operations** - Optimized bulk indexing

**Implementation Approach**: Phased rollout with zero breaking changes, all features optional and backward compatible.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Goals and Non-Goals](#goals-and-non-goals)
3. [Enhancement 1: Configurable Metadata Filter Keys](#enhancement-1-configurable-metadata-filter-keys)
4. [Enhancement 2: Multi-Collection Management API](#enhancement-2-multi-collection-management-api)
5. [Enhancement 3: RBAC Integration Hooks](#enhancement-3-rbac-integration-hooks)
6. [Enhancement 4: OpenTelemetry Instrumentation](#enhancement-4-opentelemetry-instrumentation)
7. [Enhancement 5: Flexible Connection Parameters](#enhancement-5-flexible-connection-parameters)
8. [Enhancement 6: Metadata Schema Discovery](#enhancement-6-metadata-schema-discovery)
9. [Enhancement 7: JSON_VALUE Compatibility](#enhancement-7-json_value-compatibility)
10. [Enhancement 8: Batch Operations Support](#enhancement-8-batch-operations-support)
11. [Implementation Plan](#implementation-plan)
12. [Testing Strategy](#testing-strategy)
13. [Migration Guide](#migration-guide)
14. [Success Metrics](#success-metrics)

---

## Problem Statement

### Current State

iris-vector-rag is a production-ready RAG framework with excellent core functionality. However, enterprise integration revealed gaps in:

1. **Extensibility**: Hardcoded allowlists prevent custom metadata fields
2. **Management**: No explicit API for virtual collection lifecycle
3. **Security**: No built-in RBAC integration points
4. **Observability**: No instrumentation for production monitoring
5. **Flexibility**: Configuration-only connection management
6. **Usability**: Manual schema discovery required
7. **Setup**: IRIS-specific SQL functions need manual creation
8. **Performance**: One-by-one document indexing inefficient for bulk loads

### Pain Points from AI Hub Integration

**Pain Point 1: Custom Metadata Fields Blocked**
> "We needed `tenant_id` and `collection_id` for multi-tenancy, but iris-vector-rag's allowlist rejected them. Had to fork and modify `vector_store_iris.py` line 113."

**Pain Point 2: No Collection Management**
> "Virtual collections work via metadata filtering, but there's no API to list/delete/manage them. Had to write raw SQL queries."

**Pain Point 3: Security Integration Gap**
> "Our RBAC system controls document access, but iris-vector-rag has no hooks. Had to wrap every call with permission checks."

**Pain Point 4: Production Observability**
> "Can't monitor RAG query performance in production. No visibility into embedding latency, search time, or reranking duration."

**Pain Point 5: Connection Injection Needed**
> "We use connection pooling, but iris-vector-rag requires YAML config. Had to maintain parallel configurations."

---

## Goals and Non-Goals

### Goals

**Primary Goals**:
1. ✅ Enable custom metadata filter keys via configuration (security maintained)
2. ✅ Provide explicit virtual collection management API
3. ✅ Add optional RBAC integration hooks (no built-in policy)
4. ✅ Instrument with OpenTelemetry (disabled by default)
5. ✅ Support direct connection injection (config still works)
6. ✅ Add metadata schema discovery utility
7. ✅ Auto-create JSON_VALUE compatibility function
8. ✅ Optimize batch document indexing

**Secondary Goals**:
- Maintain 100% backward compatibility
- All features optional (opt-in)
- Zero performance impact when disabled
- Clear migration documentation

### Non-Goals

- ❌ Built-in RBAC policy implementation (users provide their own)
- ❌ Real-time alerting (use external monitoring tools)
- ❌ Custom vector distance functions (IRIS handles this)
- ❌ Multi-database support (IRIS-specific enhancements)
- ❌ Breaking API changes

---

## Enhancement 1: Configurable Metadata Filter Keys

### Current Behavior

**File**: `iris_vector_rag/storage/vector_store_iris.py:113-131`

```python
self._allowed_filter_keys = {
    "category", "year", "source_type", "document_id",
    "author_name", "title", "source", "type", "date",
    "status", "version", "pmcid", "journal", "doi",
    "publication_date", "keywords", "abstract_type"
}
```

**Problem**: Hardcoded allowlist blocks enterprise use cases (tenant_id, collection_id, security_level, department).

### Proposed Solution

#### Configuration Schema

**File**: `iris_vector_rag/config/default_config.yaml`

```yaml
storage:
  iris:
    # Base allowlist (maintained for security)
    allowed_filter_keys:
      - category
      - year
      - source_type
      - document_id
      - author_name
      - title
      - source
      - type
      - date
      - status
      - version
      - pmcid
      - journal
      - doi
      - publication_date
      - keywords
      - abstract_type

    # User-defined extensions (merged with base)
    custom_filter_keys:
      - tenant_id        # Multi-tenancy
      - collection_id    # Virtual collections
      - security_level   # Classification
      - department       # Organizational units
      - project_id       # Project isolation
```

#### Implementation

**File**: `iris_vector_rag/storage/vector_store_iris.py`

```python
class IRISVectorStore(VectorStore):
    def __init__(self, connection_manager, config_manager, **kwargs):
        # Get base allowed keys
        base_keys = config_manager.get(
            "storage:iris:allowed_filter_keys",
            self._DEFAULT_ALLOWED_KEYS  # Fallback to constant
        )

        # Get custom keys (user-defined extensions)
        custom_keys = config_manager.get(
            "storage:iris:custom_filter_keys",
            []
        )

        # Merge: base + custom
        self._allowed_filter_keys = set(base_keys) | set(custom_keys)

        logger.info(
            f"Initialized with {len(self._allowed_filter_keys)} allowed filter keys "
            f"({len(custom_keys)} custom)"
        )

    # Default keys (for backward compatibility if config missing)
    _DEFAULT_ALLOWED_KEYS = {
        "category", "year", "source_type", "document_id",
        "author_name", "title", "source", "type", "date",
        "status", "version", "pmcid", "journal", "doi",
        "publication_date", "keywords", "abstract_type"
    }
```

#### Security Validation

**Maintain SQL Injection Protection**:

```python
def _validate_filter_keys(self, filter: Dict[str, Any]):
    """
    Validate filter keys against allowlist.

    Raises:
        VectorStoreConfigurationError: If filter key not allowed
    """
    for key in filter.keys():
        if key not in self._allowed_filter_keys:
            raise VectorStoreConfigurationError(
                f"Filter key '{key}' not in allowed list. "
                f"Add to storage:iris:custom_filter_keys in config. "
                f"Allowed keys: {sorted(self._allowed_filter_keys)}"
            )
```

### Benefits

1. **Security Maintained**: SQL injection prevention still enforced
2. **Enterprise Use Cases**: Supports multi-tenancy, RBAC, custom taxonomies
3. **Backward Compatible**: Default keys unchanged, custom keys optional
4. **Self-Documenting**: Configuration explicitly lists allowed fields
5. **Clear Errors**: Helpful error messages when filter key rejected

### Migration

**Existing Code**: No changes needed (default keys unchanged)

**New Use Case**:
```python
# config/custom_config.yaml
storage:
  iris:
    custom_filter_keys:
      - tenant_id
      - collection_id

# Application code
pipeline = create_pipeline('basic', config_file='config/custom_config.yaml')
results = pipeline.query(
    query="...",
    metadata_filter={
        "collection_id": "medical_pmc",  # ✅ Now allowed
        "tenant_id": "tenant_123"        # ✅ Now allowed
    }
)
```

---

## Enhancement 2: Multi-Collection Management API

### Current Behavior

iris-vector-rag uses metadata-based virtual collections (all documents in `RAG.SourceDocuments`, differentiated by `metadata.collection_id`). However, there's **no explicit collection management API**.

**Gaps**:
- Can't list all collections
- Can't get collection statistics
- Can't delete all documents in a collection
- Can't validate collection existence

### Proposed Solution

#### New API Methods

**File**: `iris_vector_rag/storage/vector_store_iris.py`

```python
class IRISVectorStore(VectorStore):
    # ========================================================================
    # Collection Management API (Virtual Collections)
    # ========================================================================

    def create_collection(
        self,
        collection_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a virtual collection (metadata-based).

        This is a logical operation - no physical table is created.
        Collections are differentiated by metadata.collection_id field.

        Args:
            collection_id: Unique collection identifier
            metadata: Optional collection-level metadata (stored in registry)

        Returns:
            True if created, False if collection already exists

        Raises:
            VectorStoreConfigurationError: If collection_id invalid
        """
        # Validate collection_id
        if not collection_id or not collection_id.replace("_", "").replace("-", "").isalnum():
            raise VectorStoreConfigurationError(
                f"Invalid collection_id: {collection_id}. "
                f"Must be alphanumeric with underscores/hyphens only."
            )

        # Check if collection exists
        if self.collection_exists(collection_id):
            logger.info(f"Collection '{collection_id}' already exists")
            return False

        # Optionally: Store collection metadata in registry table
        # For now, just validate the ID is usable
        logger.info(f"✅ Virtual collection '{collection_id}' registered (uses RAG.SourceDocuments)")
        return True

    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all virtual collections with statistics.

        Queries distinct collection_id values from metadata column.

        Returns:
            List of dicts with keys:
            - collection_id: Collection identifier
            - document_count: Number of documents
            - created_at: First document timestamp (if available)
            - last_updated: Most recent document timestamp
            - total_size_bytes: Approximate size
        """
        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            # Query distinct collection IDs with stats
            sql = f"""
                SELECT
                    JSON_VALUE(metadata, '$.collection_id') AS collection_id,
                    COUNT(*) AS document_count,
                    MIN(created_at) AS created_at,
                    MAX(updated_at) AS last_updated,
                    SUM(LENGTH(page_content)) AS total_size_bytes
                FROM {self.table_name}
                WHERE JSON_VALUE(metadata, '$.collection_id') IS NOT NULL
                GROUP BY JSON_VALUE(metadata, '$.collection_id')
                ORDER BY document_count DESC
            """

            cursor.execute(sql)
            results = cursor.fetchall()

            collections = []
            for row in results:
                collections.append({
                    "collection_id": row[0],
                    "document_count": row[1],
                    "created_at": row[2],
                    "last_updated": row[3],
                    "total_size_bytes": row[4] or 0
                })

            return collections

        except Exception as e:
            sanitized_error = self._sanitize_error_message(e, "list_collections")
            logger.error(sanitized_error)
            raise VectorStoreConnectionError(f"Failed to list collections: {sanitized_error}")
        finally:
            cursor.close()

    def get_collection_info(self, collection_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific collection.

        Args:
            collection_id: Collection identifier

        Returns:
            Dict with keys:
            - collection_id: Collection identifier
            - document_count: Number of documents
            - created_at: First document timestamp
            - last_updated: Most recent document timestamp
            - total_size_bytes: Approximate size
            - avg_embedding_dim: Average embedding dimension
            - exists: Boolean indicating if collection has documents

        Raises:
            VectorStoreConnectionError: If query fails
        """
        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            sql = f"""
                SELECT
                    COUNT(*) AS document_count,
                    MIN(created_at) AS created_at,
                    MAX(updated_at) AS last_updated,
                    SUM(LENGTH(page_content)) AS total_size_bytes,
                    AVG(VECTOR_DIM(embedding)) AS avg_embedding_dim
                FROM {self.table_name}
                WHERE JSON_VALUE(metadata, '$.collection_id') = ?
            """

            cursor.execute(sql, [collection_id])
            row = cursor.fetchone()

            if not row or row[0] == 0:
                return {
                    "collection_id": collection_id,
                    "document_count": 0,
                    "exists": False
                }

            return {
                "collection_id": collection_id,
                "document_count": row[0],
                "created_at": row[1],
                "last_updated": row[2],
                "total_size_bytes": row[3] or 0,
                "avg_embedding_dim": int(row[4]) if row[4] else None,
                "exists": True
            }

        except Exception as e:
            sanitized_error = self._sanitize_error_message(e, "get_collection_info")
            logger.error(sanitized_error)
            raise VectorStoreConnectionError(
                f"Failed to get collection info: {sanitized_error}"
            )
        finally:
            cursor.close()

    def delete_collection(self, collection_id: str) -> int:
        """
        Delete all documents in a collection.

        WARNING: This is a destructive operation. All documents with
        metadata.collection_id matching the given ID will be deleted.

        Args:
            collection_id: Collection identifier to delete

        Returns:
            Number of documents deleted

        Raises:
            VectorStoreConnectionError: If delete fails
        """
        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            # Count documents before deletion
            count_sql = f"""
                SELECT COUNT(*)
                FROM {self.table_name}
                WHERE JSON_VALUE(metadata, '$.collection_id') = ?
            """
            cursor.execute(count_sql, [collection_id])
            count_before = cursor.fetchone()[0]

            if count_before == 0:
                logger.info(f"Collection '{collection_id}' has no documents to delete")
                return 0

            # Delete documents
            delete_sql = f"""
                DELETE FROM {self.table_name}
                WHERE JSON_VALUE(metadata, '$.collection_id') = ?
            """
            cursor.execute(delete_sql, [collection_id])
            connection.commit()

            logger.info(f"✅ Deleted {count_before} documents from collection '{collection_id}'")
            return count_before

        except Exception as e:
            connection.rollback()
            sanitized_error = self._sanitize_error_message(e, "delete_collection")
            logger.error(sanitized_error)
            raise VectorStoreConnectionError(
                f"Failed to delete collection: {sanitized_error}"
            )
        finally:
            cursor.close()

    def collection_exists(self, collection_id: str) -> bool:
        """
        Check if collection has any documents.

        Args:
            collection_id: Collection identifier

        Returns:
            True if collection has at least one document
        """
        info = self.get_collection_info(collection_id)
        return info["exists"]
```

### Benefits

1. **Explicit API**: Makes virtual collection pattern first-class
2. **Lifecycle Management**: Create, list, info, delete operations
3. **Statistics**: Document counts, sizes, timestamps
4. **Cleanup**: Easy deletion of test/temporary collections
5. **Discovery**: List all collections in database

### Usage Examples

```python
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

vector_store = IRISVectorStore(config_manager=config)

# Create virtual collection
vector_store.create_collection("medical_pmc")

# List all collections
collections = vector_store.list_collections()
for coll in collections:
    print(f"{coll['collection_id']}: {coll['document_count']} documents")

# Get collection stats
info = vector_store.get_collection_info("medical_pmc")
print(f"Collection size: {info['total_size_bytes'] / 1024 / 1024:.2f} MB")

# Delete collection (cleanup)
deleted_count = vector_store.delete_collection("test_collection")
print(f"Deleted {deleted_count} documents")
```

---

## Enhancement 3: RBAC Integration Hooks

### Current Behavior

iris-vector-rag has no authorization checks. All operations (search, index, delete) are unrestricted at the library level.

**Enterprise Requirement**: Multi-tenant environments need:
- Collection-level access control
- Row-level security for sensitive documents
- Operation-level permissions (read vs. write)

### Proposed Solution

#### RBAC Policy Interface

**File**: `iris_vector_rag/security/rbac.py` (new)

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from ..core.models import Document

class RBACPolicy(ABC):
    """
    Abstract interface for RBAC policy enforcement.

    Users implement this interface to integrate with their authorization systems.
    """

    @abstractmethod
    def check_collection_access(
        self,
        user: str,
        collection_id: Optional[str],
        operation: str  # 'read', 'write', 'delete'
    ) -> bool:
        """
        Check if user can perform operation on collection.

        Args:
            user: User identifier (email, username, etc.)
            collection_id: Target collection (None = all collections)
            operation: Operation type ('read', 'write', 'delete')

        Returns:
            True if access allowed, False otherwise
        """
        pass

    @abstractmethod
    def filter_documents(
        self,
        user: str,
        documents: List[Document]
    ) -> List[Document]:
        """
        Filter documents based on user permissions (row-level security).

        Args:
            user: User identifier
            documents: List of documents to filter

        Returns:
            Filtered list of documents user can access
        """
        pass


class PermissiveRBACPolicy(RBACPolicy):
    """
    Default RBAC policy that allows all operations.

    Use this when RBAC is not required (single-tenant, development).
    """

    def check_collection_access(self, user, collection_id, operation):
        return True  # Allow all

    def filter_documents(self, user, documents):
        return documents  # No filtering
```

#### Integration Points

**File**: `iris_vector_rag/storage/vector_store_iris.py`

```python
class IRISVectorStore(VectorStore):
    def __init__(
        self,
        connection_manager,
        config_manager,
        rbac_policy: Optional[RBACPolicy] = None,  # NEW: Optional RBAC
        **kwargs
    ):
        super().__init__(connection_manager, config_manager, **kwargs)

        # Initialize RBAC policy (default: permissive)
        if rbac_policy is None:
            from ..security.rbac import PermissiveRBACPolicy
            self.rbac_policy = PermissiveRBACPolicy()
        else:
            self.rbac_policy = rbac_policy

        logger.info(
            f"Initialized with RBAC policy: {self.rbac_policy.__class__.__name__}"
        )

    def similarity_search_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int,
        filter: Optional[Dict[str, Any]] = None,
        user: Optional[str] = None  # NEW: Optional user context
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with optional RBAC enforcement.

        Args:
            query_embedding: Query vector
            top_k: Maximum results
            filter: Metadata filters
            user: User context for RBAC (optional)

        Returns:
            List of (Document, score) tuples

        Raises:
            PermissionError: If user lacks read access to collection
        """
        # RBAC: Check collection-level access
        if user:
            collection_id = filter.get("collection_id") if filter else None
            if not self.rbac_policy.check_collection_access(user, collection_id, "read"):
                raise PermissionError(
                    f"User '{user}' does not have read access to collection '{collection_id}'"
                )

        # Perform search (existing logic)
        results = self._do_similarity_search(query_embedding, top_k, filter)

        # RBAC: Row-level filtering
        if user:
            documents = [doc for doc, score in results]
            filtered_docs = self.rbac_policy.filter_documents(user, documents)

            # Rebuild results with filtered documents
            filtered_doc_ids = {doc.id for doc in filtered_docs}
            results = [
                (doc, score) for doc, score in results
                if doc.id in filtered_doc_ids
            ]

        return results

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        user: Optional[str] = None  # NEW: Optional user context
    ) -> List[str]:
        """
        Add documents with optional RBAC enforcement.

        Args:
            documents: Documents to add
            embeddings: Pre-computed embeddings
            user: User context for RBAC (optional)

        Returns:
            List of document IDs added

        Raises:
            PermissionError: If user lacks write access to collection
        """
        # RBAC: Check collection-level write access
        if user:
            # Extract collection_id from first document's metadata
            collection_id = None
            if documents and documents[0].metadata:
                collection_id = documents[0].metadata.get("collection_id")

            if not self.rbac_policy.check_collection_access(user, collection_id, "write"):
                raise PermissionError(
                    f"User '{user}' does not have write access to collection '{collection_id}'"
                )

        # Add documents (existing logic)
        return self._do_add_documents(documents, embeddings)

    def delete_collection(
        self,
        collection_id: str,
        user: Optional[str] = None  # NEW: Optional user context
    ) -> int:
        """
        Delete collection with optional RBAC enforcement.

        Args:
            collection_id: Collection to delete
            user: User context for RBAC (optional)

        Returns:
            Number of documents deleted

        Raises:
            PermissionError: If user lacks delete access to collection
        """
        # RBAC: Check delete access
        if user:
            if not self.rbac_policy.check_collection_access(user, collection_id, "delete"):
                raise PermissionError(
                    f"User '{user}' does not have delete access to collection '{collection_id}'"
                )

        # Delete collection (existing logic)
        return self._do_delete_collection(collection_id)
```

### Example Implementation

**User-Defined RBAC Policy**:

```python
from iris_vector_rag.security.rbac import RBACPolicy

class IRISSecurityRBACPolicy(RBACPolicy):
    """
    RBAC policy that integrates with IRIS Security.Resources table.
    """

    def __init__(self, iris_connection):
        self.connection = iris_connection

    def check_collection_access(self, user, collection_id, operation):
        """Check IRIS %Security.Resources for permission."""
        cursor = self.connection.cursor()
        try:
            # Query IRIS security table
            sql = """
                SELECT COUNT(*)
                FROM Security.Resources
                WHERE Username = ?
                  AND Resource = ?
                  AND Permission LIKE ?
            """
            resource = f"COLLECTION_{collection_id}"
            permission = f"%{operation}%"  # 'read', 'write', 'delete'

            cursor.execute(sql, [user, resource, permission])
            count = cursor.fetchone()[0]

            return count > 0
        finally:
            cursor.close()

    def filter_documents(self, user, documents):
        """Filter by security_level metadata field."""
        # Get user's security clearance
        cursor = self.connection.cursor()
        try:
            sql = "SELECT ClearanceLevel FROM Users WHERE Username = ?"
            cursor.execute(sql, [user])
            row = cursor.fetchone()
            user_clearance = row[0] if row else 0

            # Filter documents
            return [
                doc for doc in documents
                if doc.metadata.get("security_level", 0) <= user_clearance
            ]
        finally:
            cursor.close()


# Usage
policy = IRISSecurityRBACPolicy(iris_connection)
vector_store = IRISVectorStore(
    config_manager=config,
    rbac_policy=policy  # ✅ RBAC enabled
)

# Search with user context (RBAC enforced)
results = vector_store.similarity_search_by_embedding(
    query_embedding=[...],
    top_k=10,
    filter={"collection_id": "medical_records"},
    user="john.doe@hospital.com"  # ✅ User context
)
```

### Benefits

1. **Optional**: No impact if RBAC not needed (default permissive policy)
2. **Flexible**: Users implement their own authorization logic
3. **Collection-Level**: Check access before operations
4. **Row-Level**: Filter results based on document metadata
5. **Auditable**: User context passed through all operations
6. **Integration-Ready**: Works with IRIS Security, LDAP, OAuth, etc.

---

## Enhancement 4: OpenTelemetry Instrumentation

**Note**: This enhancement is already specified in detail in `specs/feature_opentelemetry_integration.md`. This section provides a summary and integration points with other enhancements.

### Summary

Add OpenTelemetry instrumentation following GenAI semantic conventions:
- Span creation for all RAG operations (embedding, search, rerank, LLM)
- Dual storage: IRIS TelemetrySpan table + OTLP export
- Token usage and cost tracking
- Configurable (disabled by default, zero overhead when off)

### Integration with Other Enhancements

**RBAC Integration**:
```python
# Add user context to telemetry spans
span.set_attribute("user.id", user)
span.set_attribute("rbac.collection_id", collection_id)
span.set_attribute("rbac.operation", operation)
```

**Collection Management**:
```python
# Track collection operations
with tracer.start_as_current_span("collection.create") as span:
    span.set_attribute("collection.id", collection_id)
    vector_store.create_collection(collection_id)
```

**Batch Operations**:
```python
# Track batch performance
with tracer.start_as_current_span("documents.add_batch") as span:
    span.set_attribute("batch.size", len(documents))
    span.set_attribute("batch.duration_ms", duration)
```

### Configuration

```yaml
telemetry:
  enabled: false  # Disabled by default
  service_name: "iris-rag-api"
  otlp:
    endpoint: "http://localhost:4318"
  sampling:
    ratio: 0.1  # 10% in production
```

**See**: `specs/feature_opentelemetry_integration.md` for complete specification.

---

## Enhancement 5: Flexible Connection Parameters

### Current Behavior

`ConnectionManager` requires configuration via YAML. This prevents:
- Dependency injection patterns
- Using existing connection pools
- Dynamic connection management
- Testability with mock connections

### Proposed Solution

#### Support Three Connection Patterns

**File**: `iris_vector_rag/core/connection.py`

```python
from typing import Optional, Callable, Any

class ConnectionManager:
    """
    Manages IRIS database connections with flexible initialization.

    Supports three patterns:
    1. Config-based (current): Pass config_manager
    2. Direct injection: Pass existing connection
    3. Factory pattern: Pass connection factory function
    """

    def __init__(
        self,
        config_manager: Optional[ConfigurationManager] = None,
        connection: Optional[Any] = None,          # NEW: Direct injection
        connection_factory: Optional[Callable] = None  # NEW: Factory
    ):
        """
        Initialize connection manager.

        Args:
            config_manager: Configuration manager (pattern 1)
            connection: Existing connection object (pattern 2)
            connection_factory: Connection factory callable (pattern 3)

        Raises:
            ValueError: If no initialization method provided
        """
        # Validate: exactly one initialization method
        init_methods = [
            config_manager is not None,
            connection is not None,
            connection_factory is not None
        ]
        if sum(init_methods) != 1:
            raise ValueError(
                "Must provide exactly one of: config_manager, connection, or connection_factory"
            )

        # Pattern 1: Config-based (current behavior)
        if config_manager:
            self._config_manager = config_manager
            self._connection = None
            self._connection_factory = None
            self._config_managed = True
            logger.info("ConnectionManager: config-based initialization")

        # Pattern 2: Direct injection
        elif connection:
            self._connection = connection
            self._config_manager = None
            self._connection_factory = None
            self._config_managed = False
            logger.info("ConnectionManager: direct connection injection")

        # Pattern 3: Factory pattern
        elif connection_factory:
            self._connection_factory = connection_factory
            self._connection = None
            self._config_manager = None
            self._config_managed = False
            logger.info("ConnectionManager: factory-based initialization")

    def get_connection(self, connection_name: str = "iris") -> Any:
        """
        Get database connection.

        Args:
            connection_name: Connection identifier (used in config mode)

        Returns:
            Database connection object
        """
        # Pattern 1: Config-based
        if self._config_managed:
            if self._connection is None:
                self._connection = self._create_from_config(connection_name)
            return self._connection

        # Pattern 2: Direct injection
        elif self._connection is not None:
            return self._connection

        # Pattern 3: Factory pattern
        elif self._connection_factory is not None:
            if self._connection is None:
                self._connection = self._connection_factory()
            return self._connection

    def _create_from_config(self, connection_name: str):
        """Create connection from configuration (existing logic)."""
        # Existing implementation unchanged
        pass
```

### Usage Examples

**Pattern 1: Config-Based (Current)**
```python
# Backward compatible - no changes needed
config = ConfigurationManager()
connection_manager = ConnectionManager(config_manager=config)
vector_store = IRISVectorStore(connection_manager=connection_manager)
```

**Pattern 2: Direct Injection (AI Hub)**
```python
# Use existing connection from pool
iris_conn = aigw_connection_pool.get_connection()

connection_manager = ConnectionManager(connection=iris_conn)
vector_store = IRISVectorStore(connection_manager=connection_manager)

# Vector store uses injected connection
results = vector_store.similarity_search(...)

# Return connection to pool
aigw_connection_pool.return_connection(iris_conn)
```

**Pattern 3: Factory Pattern (Lazy)**
```python
# Provide connection factory for lazy initialization
def create_iris_connection():
    import iris
    return iris.connect(
        hostname="localhost",
        port=1972,
        namespace="USER",
        username="_SYSTEM",
        password="SYS"
    )

connection_manager = ConnectionManager(connection_factory=create_iris_connection)
vector_store = IRISVectorStore(connection_manager=connection_manager)

# Connection created on first use
results = vector_store.similarity_search(...)  # Factory called here
```

### Benefits

1. **Dependency Injection**: Integrate with DI frameworks
2. **Connection Pooling**: Use existing pool managers
3. **Testability**: Inject mock connections for testing
4. **Flexibility**: Choose pattern based on architecture
5. **Backward Compatible**: Config-based pattern still default

---

## Enhancement 6: Metadata Schema Discovery

### Current Behavior

Users must manually know what metadata fields exist in their documents. No built-in schema discovery.

**Problem**: When integrating with iris-vector-rag, users ask:
- "What metadata fields can I filter on?"
- "What values does the 'category' field contain?"
- "Which documents have the 'pmcid' field?"

### Proposed Solution

#### Metadata Schema Sampling

**File**: `iris_vector_rag/storage/vector_store_iris.py`

```python
class IRISVectorStore(VectorStore):
    def sample_metadata_schema(
        self,
        collection_id: Optional[str] = None,
        sample_size: int = 100
    ) -> Dict[str, Dict[str, Any]]:
        """
        Discover metadata schema by sampling documents.

        Analyzes metadata JSON fields to infer types and statistics.

        Args:
            collection_id: Optional collection to sample (None = all documents)
            sample_size: Number of documents to sample (default: 100)

        Returns:
            Dict mapping field names to metadata:
            {
                "category": {
                    "type": "string",
                    "example_values": ["medical", "legal", "technical"],
                    "unique_count": 8,
                    "frequency": 0.95,  # % of sampled docs with this field
                    "null_count": 5
                },
                "priority": {
                    "type": "integer",
                    "min": 1,
                    "max": 5,
                    "avg": 3.2,
                    "frequency": 0.80
                },
                "publication_date": {
                    "type": "date",
                    "min": "2020-01-01",
                    "max": "2024-12-31",
                    "frequency": 0.70
                }
            }

        Raises:
            VectorStoreConnectionError: If sampling fails
        """
        connection = self._get_connection()
        cursor = connection.cursor()

        try:
            # Build sample query
            if collection_id:
                where_clause = "WHERE JSON_VALUE(metadata, '$.collection_id') = ?"
                params = [collection_id, sample_size]
            else:
                where_clause = ""
                params = [sample_size]

            sql = f"""
                SELECT TOP {sample_size} metadata
                FROM {self.table_name}
                {where_clause}
            """

            cursor.execute(sql, params if collection_id else [sample_size])
            rows = cursor.fetchall()

            if not rows:
                return {}

            # Parse metadata JSON and analyze
            schema = {}
            total_docs = len(rows)

            for row in rows:
                metadata_json = json.loads(row[0]) if row[0] else {}

                for field, value in metadata_json.items():
                    # Initialize field stats if first encounter
                    if field not in schema:
                        schema[field] = {
                            "type": None,
                            "example_values": set(),
                            "count": 0,
                            "null_count": 0,
                            "numeric_values": []  # For min/max/avg
                        }

                    field_stats = schema[field]
                    field_stats["count"] += 1

                    if value is None:
                        field_stats["null_count"] += 1
                        continue

                    # Infer type
                    value_type = self._infer_metadata_type(value)
                    if field_stats["type"] is None:
                        field_stats["type"] = value_type

                    # Collect examples (limit to 5)
                    if len(field_stats["example_values"]) < 5:
                        field_stats["example_values"].add(str(value))

                    # Collect numeric values for stats
                    if isinstance(value, (int, float)):
                        field_stats["numeric_values"].append(value)

            # Finalize schema (calculate stats)
            final_schema = {}
            for field, stats in schema.items():
                field_info = {
                    "type": stats["type"] or "unknown",
                    "frequency": stats["count"] / total_docs,
                    "null_count": stats["null_count"]
                }

                # Add example values
                if stats["example_values"]:
                    field_info["example_values"] = sorted(list(stats["example_values"]))[:5]
                    field_info["unique_count"] = len(stats["example_values"])

                # Add numeric stats
                if stats["numeric_values"]:
                    field_info["min"] = min(stats["numeric_values"])
                    field_info["max"] = max(stats["numeric_values"])
                    field_info["avg"] = sum(stats["numeric_values"]) / len(stats["numeric_values"])

                final_schema[field] = field_info

            return final_schema

        except Exception as e:
            sanitized_error = self._sanitize_error_message(e, "sample_metadata_schema")
            logger.error(sanitized_error)
            raise VectorStoreConnectionError(
                f"Failed to sample metadata schema: {sanitized_error}"
            )
        finally:
            cursor.close()

    def _infer_metadata_type(self, value: Any) -> str:
        """Infer metadata field type from value."""
        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            # Try to detect dates
            import re
            if re.match(r"\d{4}-\d{2}-\d{2}", value):
                return "date"
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "unknown"
```

### Usage Examples

```python
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

vector_store = IRISVectorStore(config_manager=config)

# Discover schema for all documents
schema = vector_store.sample_metadata_schema(sample_size=200)

# Print schema
import json
print(json.dumps(schema, indent=2))

# Output:
# {
#   "category": {
#     "type": "string",
#     "example_values": ["medical", "legal", "technical"],
#     "unique_count": 8,
#     "frequency": 0.95,
#     "null_count": 10
#   },
#   "pmcid": {
#     "type": "string",
#     "example_values": ["PMC12345", "PMC67890"],
#     "frequency": 0.60
#   },
#   "publication_year": {
#     "type": "integer",
#     "min": 2015,
#     "max": 2024,
#     "avg": 2021.5,
#     "frequency": 0.85
#   }
# }

# Discover schema for specific collection
schema = vector_store.sample_metadata_schema(
    collection_id="medical_pmc",
    sample_size=100
)
```

### Benefits

1. **Self-Documenting**: Discover what metadata exists
2. **Query Building**: Know available filter fields
3. **Data Validation**: Check metadata consistency
4. **Integration Helper**: Understand collection structure
5. **Type Inference**: Automatic type detection

---

## Enhancement 7: JSON_VALUE Compatibility

### Current Behavior

iris-vector-rag uses `JSON_VALUE()` function for metadata filtering, but:
- IRIS uses `JSON_TABLE()` by default (JSON_VALUE not built-in)
- Users must manually create JSON_VALUE function via ObjectScript
- Setup friction for new users

### Proposed Solution

#### Auto-Detection and Creation

**File**: `iris_vector_rag/storage/vector_store_iris.py`

```python
class IRISVectorStore(VectorStore):
    def __init__(self, connection_manager, config_manager, **kwargs):
        super().__init__(connection_manager, config_manager, **kwargs)

        # Auto-setup JSON_VALUE compatibility if needed
        auto_setup = config_manager.get("storage:iris:auto_setup_json_value", True)
        if auto_setup:
            self._ensure_json_value_exists()

    def _ensure_json_value_exists(self):
        """
        Ensure JSON_VALUE function exists, create if missing.

        This function provides SQL Server / MySQL compatibility for IRIS.
        """
        try:
            connection = self._get_connection()
            cursor = connection.cursor()

            # Test if JSON_VALUE exists
            try:
                cursor.execute("SELECT JSON_VALUE('{}', '$')")
                cursor.fetchone()
                logger.debug("JSON_VALUE function exists")
                cursor.close()
                return  # Already exists
            except Exception:
                # Function doesn't exist, create it
                logger.info("JSON_VALUE function not found, creating...")

            # Create JSON_VALUE function using IRIS %ASQ.Parser
            sql = """
            CREATE FUNCTION SQLUSER.JSON_VALUE(
                json VARCHAR(32000),
                json_path VARCHAR(1000)
            )
            RETURNS VARCHAR(5000)
            LANGUAGE OBJECTSCRIPT
            {
                // Parse JSON and extract value at path
                set parser = ##class(%ASQ.Parser).%New()
                set value = parser.JSONValue(json, json_path)
                return value
            }
            """

            cursor.execute(sql)
            connection.commit()
            logger.info("✅ JSON_VALUE function created successfully")
            cursor.close()

        except Exception as e:
            logger.warning(
                f"Could not auto-create JSON_VALUE function: {e}. "
                f"You may need to run setup script manually."
            )
            # Don't raise - let user proceed with manual setup
```

#### Manual Setup Script

**File**: `scripts/setup/create_json_value.sql`

```sql
-- Create JSON_VALUE function for IRIS
-- Alternative to auto-setup if ObjectScript function creation restricted

CREATE FUNCTION SQLUSER.JSON_VALUE(
    json VARCHAR(32000),
    json_path VARCHAR(1000)
)
RETURNS VARCHAR(5000)
LANGUAGE OBJECTSCRIPT
{
    // Parse JSON using IRIS %ASQ.Parser
    set parser = ##class(%ASQ.Parser).%New()

    // Extract value at JSON path
    set value = parser.JSONValue(json, json_path)

    // Return value as string
    if (value = "") {
        return ""
    }
    return value
}
;

-- Test JSON_VALUE function
SELECT JSON_VALUE('{"name": "test", "age": 30}', '$.name') AS name,
       JSON_VALUE('{"name": "test", "age": 30}', '$.age') AS age;

-- Expected output:
-- name | age
-- -----+----
-- test | 30
```

#### CLI Command

```bash
# Setup JSON_VALUE function
python -m iris_vector_rag.setup create-json-value \
    --host localhost \
    --port 1972 \
    --namespace USER \
    --username _SYSTEM \
    --password SYS

# Test JSON_VALUE function
python -m iris_vector_rag.setup test-json-value
```

### Configuration

```yaml
storage:
  iris:
    # Auto-create JSON_VALUE function if missing (default: true)
    auto_setup_json_value: true

    # Skip auto-setup if function creation restricted
    # auto_setup_json_value: false
```

### Benefits

1. **Reduced Friction**: Automatic setup for new users
2. **Self-Healing**: Detects and fixes missing function
3. **Manual Fallback**: SQL script provided if auto-setup fails
4. **CLI Tool**: Easy setup via command line
5. **Configurable**: Can disable auto-setup if needed

---

## Enhancement 8: Batch Operations Support

### Current Behavior

Document indexing processes documents one-at-a-time or in small batches. For bulk indexing (10K+ documents), this is inefficient:
- High network round-trips
- Slow commit overhead
- No progress tracking

### Proposed Solution

#### Optimized Batch Operations

**File**: `iris_vector_rag/storage/vector_store_iris.py`

```python
from tqdm import tqdm

class IRISVectorStore(VectorStore):
    def add_documents_batch(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        batch_size: int = 1000,
        show_progress: bool = False,
        error_handling: str = "continue"  # 'continue', 'stop', 'rollback'
    ) -> Dict[str, Any]:
        """
        Add documents in optimized batches.

        Uses:
        - Batch INSERT statements (reduced round-trips)
        - Transaction management per batch
        - Progress tracking with tqdm
        - Error recovery strategies

        Args:
            documents: Documents to add
            embeddings: Pre-computed embeddings (must match document count)
            batch_size: Documents per batch (default: 1000)
            show_progress: Show progress bar (default: False)
            error_handling: Error strategy ('continue', 'stop', 'rollback')

        Returns:
            Dict with keys:
            - doc_ids: List of successfully added document IDs
            - success_count: Number of documents added
            - error_count: Number of failed documents
            - errors: List of error dicts (if any)
            - duration_seconds: Total execution time

        Raises:
            VectorStoreDataError: If data validation fails
            VectorStoreConnectionError: If connection fails
        """
        import time
        start_time = time.time()

        # Validate inputs
        if not documents:
            return {
                "doc_ids": [],
                "success_count": 0,
                "error_count": 0,
                "errors": [],
                "duration_seconds": 0
            }

        if embeddings and len(embeddings) != len(documents):
            raise VectorStoreDataError(
                f"Embedding count ({len(embeddings)}) must match document count ({len(documents)})"
            )

        # Split into batches
        batches = [
            documents[i:i + batch_size]
            for i in range(0, len(documents), batch_size)
        ]

        # Track results
        doc_ids = []
        errors = []

        # Progress bar
        iterator = tqdm(batches, desc="Indexing documents") if show_progress else batches

        connection = self._get_connection()

        for batch_idx, batch in enumerate(iterator):
            batch_start = batch_idx * batch_size
            batch_end = batch_start + len(batch)
            batch_embeddings = embeddings[batch_start:batch_end] if embeddings else None

            try:
                # Add batch with transaction
                batch_ids = self._add_batch_transactional(
                    batch,
                    batch_embeddings,
                    connection
                )
                doc_ids.extend(batch_ids)

                # Update progress
                if show_progress:
                    iterator.set_postfix({
                        "success": len(doc_ids),
                        "errors": len(errors)
                    })

            except Exception as e:
                # Error handling strategy
                error_info = {
                    "batch_idx": batch_idx,
                    "batch_start": batch_start,
                    "batch_size": len(batch),
                    "error": str(e)
                }
                errors.append(error_info)

                if error_handling == "stop":
                    # Stop immediately
                    logger.error(f"Batch {batch_idx} failed, stopping: {e}")
                    break
                elif error_handling == "rollback":
                    # Rollback all changes
                    logger.error(f"Batch {batch_idx} failed, rolling back all changes: {e}")
                    connection.rollback()
                    return {
                        "doc_ids": [],
                        "success_count": 0,
                        "error_count": len(documents),
                        "errors": errors,
                        "duration_seconds": time.time() - start_time
                    }
                else:  # continue
                    # Log error and continue with next batch
                    logger.warning(f"Batch {batch_idx} failed, continuing: {e}")
                    continue

        duration = time.time() - start_time

        return {
            "doc_ids": doc_ids,
            "success_count": len(doc_ids),
            "error_count": len(errors),
            "errors": errors,
            "duration_seconds": duration
        }

    def _add_batch_transactional(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]],
        connection
    ) -> List[str]:
        """
        Add a batch of documents within a transaction.

        Args:
            documents: Batch of documents
            embeddings: Corresponding embeddings
            connection: Database connection

        Returns:
            List of document IDs added

        Raises:
            Exception: If batch insertion fails
        """
        cursor = connection.cursor()

        try:
            # Start transaction
            cursor.execute("START TRANSACTION")

            # Build batch INSERT statement
            # Using VALUES clause with multiple rows
            placeholders = []
            values = []

            for i, doc in enumerate(documents):
                embedding = embeddings[i] if embeddings else None

                # Prepare values
                doc_id = doc.id or str(uuid.uuid4())
                content = doc.page_content
                metadata_json = json.dumps(doc.metadata) if doc.metadata else "{}"
                embedding_str = self._vector_to_sql(embedding) if embedding else None

                values.extend([doc_id, content, metadata_json, embedding_str])
                placeholders.append("(?, ?, ?, TO_VECTOR(?))")

            # Execute batch INSERT
            sql = f"""
                INSERT INTO {self.table_name}
                    (document_id, page_content, metadata, embedding)
                VALUES {', '.join(placeholders)}
            """

            cursor.execute(sql, values)

            # Commit transaction
            connection.commit()

            # Return document IDs
            doc_ids = [doc.id or str(uuid.uuid4()) for doc in documents]
            return doc_ids

        except Exception as e:
            # Rollback transaction on error
            connection.rollback()
            logger.error(f"Batch insertion failed: {e}")
            raise
        finally:
            cursor.close()
```

### Usage Examples

```python
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore
from iris_vector_rag.core.models import Document

vector_store = IRISVectorStore(config_manager=config)

# Load 10K documents efficiently
documents = [
    Document(
        page_content=f"Document {i} content...",
        metadata={"doc_num": i, "collection_id": "bulk_load"}
    )
    for i in range(10000)
]

# Batch insert with progress bar
result = vector_store.add_documents_batch(
    documents=documents,
    batch_size=1000,           # 1K documents per batch
    show_progress=True,        # Show tqdm progress bar
    error_handling="continue"  # Skip failed batches
)

print(f"✅ Indexed {result['success_count']} documents in {result['duration_seconds']:.2f}s")
print(f"❌ Failed: {result['error_count']} batches")

# Expected output:
# Indexing documents: 100%|██████████| 10/10 [00:05<00:00,  1.85it/s]
# ✅ Indexed 10000 documents in 5.42s
# ❌ Failed: 0 batches
```

### Performance Comparison

| Method | 10K Documents | 100K Documents |
|--------|---------------|----------------|
| `add_documents()` (one-by-one) | ~120 seconds | ~1200 seconds (20 min) |
| `add_documents_batch()` (batch_size=1000) | ~5 seconds | ~50 seconds |
| **Speedup** | **24x** | **24x** |

### Benefits

1. **10-100x Faster**: Batch INSERT reduces round-trips
2. **Progress Tracking**: tqdm progress bar for long operations
3. **Error Recovery**: Continue/stop/rollback strategies
4. **Transaction Safety**: Per-batch commits
5. **Production-Ready**: Handles bulk data migrations

---

## Implementation Plan

### Phase 1: Core Enhancements (Week 1) - **P0**

**Goals**: Non-breaking changes that unblock enterprise use cases

**Tasks**:
1. ✅ **Configurable Metadata Filter Keys**
   - Add `custom_filter_keys` configuration parameter
   - Update `_validate_filter_keys()` to merge base + custom
   - Add configuration validation
   - Write unit tests
   - Update documentation

2. ✅ **Multi-Collection Management API**
   - Implement `create_collection()`, `list_collections()`, `get_collection_info()`
   - Implement `delete_collection()`, `collection_exists()`
   - Add SQL queries for statistics
   - Write integration tests
   - Update API documentation

3. ✅ **JSON_VALUE Compatibility**
   - Implement `_ensure_json_value_exists()` auto-setup
   - Create `scripts/setup/create_json_value.sql`
   - Add CLI command `create-json-value`
   - Write compatibility tests
   - Update installation docs

**Deliverables**:
- Configuration schema updated
- New API methods tested
- JSON_VALUE auto-setup working
- Documentation updated

**Success Criteria**:
- All existing tests pass (backward compatibility)
- 3 new contract tests passing per enhancement
- Documentation includes migration examples

---

### Phase 2: Integration Hooks (Week 2) - **P1**

**Goals**: Add optional integration points for enterprise systems

**Tasks**:
4. ✅ **RBAC Integration Hooks**
   - Create `iris_vector_rag/security/rbac.py` with `RBACPolicy` interface
   - Implement `PermissiveRBACPolicy` (default)
   - Add optional `rbac_policy` parameter to `IRISVectorStore.__init__()`
   - Add `user` parameter to `similarity_search_by_embedding()`, `add_documents()`, `delete_collection()`
   - Implement permission checks and row-level filtering
   - Write RBAC integration tests
   - Document custom policy implementation

5. ✅ **Flexible Connection Parameters**
   - Update `ConnectionManager.__init__()` to accept `connection` and `connection_factory`
   - Implement pattern detection (config vs. injection vs. factory)
   - Update `get_connection()` to support all patterns
   - Write connection injection tests
   - Document all three patterns

**Deliverables**:
- `RBACPolicy` interface with example implementations
- `ConnectionManager` supports 3 initialization patterns
- Integration tests for RBAC and connection injection
- Migration guide for existing users

**Success Criteria**:
- RBAC tests pass with mock policy
- Connection injection works with mock connections
- Zero impact on existing code (optional features)

---

### Phase 3: Observability (Week 3) - **P1**

**Goals**: Add OpenTelemetry instrumentation (already specified)

**Tasks**:
6. ✅ **OpenTelemetry Instrumentation**
   - Implement span creation for all RAG operations
   - Add dual storage (IRIS TelemetrySpan + OTLP)
   - Implement cost tracking
   - Add `configure_telemetry()` function
   - Write telemetry integration tests
   - **Reference**: `specs/feature_opentelemetry_integration.md`

**Deliverables**:
- OpenTelemetry spans working (disabled by default)
- Configuration for OTLP export
- Cost tracking functional
- Integration tests with mock tracer

**Success Criteria**:
- Zero performance impact when disabled
- Spans follow GenAI semantic conventions
- IRIS TelemetrySpan table populated
- Documentation includes OTel configuration

---

### Phase 4: Utilities and Performance (Week 4) - **P2**

**Goals**: Add convenience utilities and performance optimizations

**Tasks**:
7. ✅ **Metadata Schema Discovery**
   - Implement `sample_metadata_schema()` method
   - Add type inference logic
   - Add statistics calculation (frequency, min/max/avg)
   - Write schema discovery tests
   - Document usage examples

8. ✅ **Batch Operations Support**
   - Implement `add_documents_batch()` method
   - Add transaction management per batch
   - Implement progress tracking with tqdm
   - Add error recovery strategies
   - Write batch performance benchmarks
   - Document performance gains

**Deliverables**:
- `sample_metadata_schema()` working with type inference
- `add_documents_batch()` 10-100x faster than one-by-one
- Progress tracking with tqdm
- Benchmark results documented

**Success Criteria**:
- Schema discovery accurate for test data
- Batch operations achieve 10x+ speedup
- Error handling robust (continue/stop/rollback)
- Documentation includes performance comparison

---

### Phase 5: Documentation and Release (Week 5)

**Goals**: Comprehensive documentation and release preparation

**Tasks**:
1. Update README.md with new features
2. Create migration guide for existing users
3. Add cookbook examples for each enhancement
4. Update API reference documentation
5. Create blog post announcing enhancements
6. Prepare CHANGELOG for v2.0.0 release

**Deliverables**:
- Complete documentation update
- Migration guide published
- Cookbook with 8 examples (one per enhancement)
- Release notes ready

**Success Criteria**:
- All enhancements documented
- Migration guide tested by QA
- Zero breaking changes confirmed
- Release announcement approved

---

## Testing Strategy

### Unit Tests (Per Enhancement)

**Coverage Target**: 90%+ for new code

**Test Files**:
- `tests/unit/storage/test_configurable_filter_keys.py` - 8 tests
- `tests/unit/storage/test_collection_management.py` - 12 tests
- `tests/unit/security/test_rbac_integration.py` - 10 tests
- `tests/unit/telemetry/test_otel_instrumentation.py` - 15 tests
- `tests/unit/core/test_connection_patterns.py` - 9 tests
- `tests/unit/storage/test_metadata_schema.py` - 7 tests
- `tests/unit/storage/test_json_value_compat.py` - 6 tests
- `tests/unit/storage/test_batch_operations.py` - 11 tests

**Total**: 78 new unit tests

### Integration Tests

**Coverage**: End-to-end workflows with real IRIS database

**Test Files**:
- `tests/integration/test_enterprise_metadata_filtering.py` - Custom filter keys with real data
- `tests/integration/test_collection_lifecycle.py` - Create, list, info, delete collections
- `tests/integration/test_rbac_enforcement.py` - RBAC with mock policy
- `tests/integration/test_telemetry_spans.py` - OpenTelemetry with mock tracer
- `tests/integration/test_connection_injection.py` - All 3 connection patterns
- `tests/integration/test_bulk_indexing.py` - 10K document batch performance

**Total**: 6 integration tests (12-15 scenarios each)

### Contract Tests (TDD Approach)

**Purpose**: Validate API contracts before implementation

**Test Files**:
- `tests/contract/test_configurable_filter_keys_contract.py`
- `tests/contract/test_collection_management_contract.py`
- `tests/contract/test_rbac_hooks_contract.py`
- `tests/contract/test_connection_flexibility_contract.py`

**Methodology**:
1. Write contract tests first (API expected behavior)
2. Run tests (should fail initially)
3. Implement enhancement
4. Run tests (should pass)
5. Refactor with test safety net

### Performance Benchmarks

**Batch Operations Benchmark**:
```python
import time
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

vector_store = IRISVectorStore(config_manager=config)

# Generate test documents
documents = generate_test_documents(count=10000)

# Benchmark: one-by-one
start = time.time()
for doc in documents:
    vector_store.add_documents([doc])
one_by_one_time = time.time() - start

# Benchmark: batch
start = time.time()
vector_store.add_documents_batch(documents, batch_size=1000)
batch_time = time.time() - start

speedup = one_by_one_time / batch_time
print(f"Speedup: {speedup:.1f}x")  # Expected: 10-100x
```

**Expected Results**:
- 10K documents: 24x speedup
- 100K documents: 24x speedup
- Batch size 500: 18x speedup
- Batch size 1000: 24x speedup
- Batch size 2000: 26x speedup (diminishing returns)

---

## Migration Guide

### For Existing Users

**Zero Breaking Changes**: All enhancements are opt-in and backward compatible.

**No Code Changes Needed**: Existing code continues to work unchanged.

### Adopting New Features

#### 1. Custom Metadata Filter Keys

**Before** (v1.x):
```python
# Had to fork and modify vector_store_iris.py to add custom keys
```

**After** (v2.0):
```yaml
# config/custom_config.yaml
storage:
  iris:
    custom_filter_keys:
      - tenant_id
      - collection_id
      - security_level
```

```python
# Use new config
pipeline = create_pipeline('basic', config_file='config/custom_config.yaml')

# Filter by custom metadata
results = pipeline.query(
    query="...",
    metadata_filter={"tenant_id": "tenant_123"}  # ✅ Now allowed
)
```

#### 2. Collection Management

**Before** (v1.x):
```python
# No API - had to write raw SQL
cursor.execute("SELECT DISTINCT JSON_VALUE(metadata, '$.collection_id') FROM RAG.SourceDocuments")
```

**After** (v2.0):
```python
from iris_vector_rag.storage.vector_store_iris import IRISVectorStore

vector_store = IRISVectorStore(config_manager=config)

# Explicit collection API
collections = vector_store.list_collections()
for coll in collections:
    print(f"{coll['collection_id']}: {coll['document_count']} docs")
```

#### 3. RBAC Integration

**Before** (v1.x):
```python
# Had to wrap every call with permission checks
if not rbac.check_permission(user, collection_id, "read"):
    raise PermissionError()
results = vector_store.similarity_search(...)
```

**After** (v2.0):
```python
# Implement RBACPolicy once
class MyRBACPolicy(RBACPolicy):
    def check_collection_access(self, user, collection_id, operation):
        return rbac.check_permission(user, collection_id, operation)

    def filter_documents(self, user, documents):
        return [doc for doc in documents if user_can_access(user, doc)]

# Use RBAC-enabled vector store
vector_store = IRISVectorStore(config_manager=config, rbac_policy=MyRBACPolicy())

# RBAC enforced automatically
results = vector_store.similarity_search_by_embedding(
    query_embedding=[...],
    top_k=10,
    filter={"collection_id": "medical"},
    user="john.doe@example.com"  # ✅ RBAC enforced
)
```

#### 4. Connection Injection

**Before** (v1.x):
```python
# Had to use config-based connections only
# No way to inject existing connections
```

**After** (v2.0):
```python
# Use existing connection from pool
conn = connection_pool.get_connection()

connection_manager = ConnectionManager(connection=conn)
vector_store = IRISVectorStore(connection_manager=connection_manager)

# Use vector store with injected connection
results = vector_store.similarity_search(...)

# Return connection to pool
connection_pool.return_connection(conn)
```

#### 5. OpenTelemetry

**Before** (v1.x):
```python
# No observability - operations were "black boxes"
```

**After** (v2.0):
```yaml
# config/custom_config.yaml
telemetry:
  enabled: true
  service_name: "my-rag-service"
  otlp:
    endpoint: "http://localhost:4318"
```

```python
import iris_vector_rag

# Enable telemetry (once at startup)
iris_vector_rag.configure_telemetry(service_name="my-rag-service")

# All operations now instrumented
pipeline = create_pipeline('basic')
results = pipeline.query("...")  # ✅ Spans created automatically
```

---

## Success Metrics

### Adoption Metrics

**Week 1 Post-Release**:
- 10% of iris-vector-rag users adopt at least one enhancement
- GitHub stars increase by 50+

**Month 1 Post-Release**:
- 50% of new integrations use custom metadata filter keys
- 30% enable OpenTelemetry instrumentation
- 5+ community-contributed RBAC policy examples

**Month 3 Post-Release**:
- 80% of enterprise users adopt collection management API
- 10+ blog posts/tutorials about enhancements
- Zero regression bugs reported

### Performance Metrics

**Batch Operations**:
- 10K documents: 20x+ speedup measured
- 100K documents: 20x+ speedup measured
- CI/CD pipeline build time reduced by 50% (bulk test data loading)

**OpenTelemetry Overhead**:
- Disabled: 0ms overhead (measured)
- Enabled (10% sampling): <5ms per query overhead
- Span storage: <1GB per 1M spans

### Business Metrics

**Enterprise Adoption**:
- 5+ Fortune 500 companies deploy enhancements
- 10+ consulting partners build on enhancements
- 20+ customer success stories documented

**Community Engagement**:
- 100+ GitHub discussions about enhancements
- 50+ pull requests improving enhancements
- 10+ third-party tools integrate with RBAC hooks

---

## Risks and Mitigations

### Risk 1: Backward Compatibility Breakage

**Risk**: Enhancements accidentally break existing code

**Mitigation**:
- All features opt-in (default behavior unchanged)
- Comprehensive backward compatibility tests
- Alpha/beta releases with community testing
- Deprecation warnings for any future breaking changes

**Validation**:
- Run full test suite against v1.x behavior
- Test with 3 real-world integrations (AI Hub, etc.)
- Community beta period (2 weeks)

---

### Risk 2: Performance Regression

**Risk**: New features slow down existing operations

**Mitigation**:
- Lazy initialization (RBAC, telemetry only active when configured)
- Benchmarks for all enhancements
- Performance regression tests in CI/CD
- Profiling of hot paths

**Validation**:
- Benchmark before/after for core operations
- Zero overhead when features disabled (measured)
- Performance regression tests fail CI if >5% slowdown

---

### Risk 3: RBAC Policy Misuse

**Risk**: Users implement incorrect RBAC policies (security gaps)

**Mitigation**:
- Clear documentation with security warnings
- Example implementations (fail-secure patterns)
- Audit logging of RBAC decisions (via telemetry)
- Security review checklist in docs

**Validation**:
- Security audit of example policies
- Penetration testing with common misconfigurations
- Documentation review by security team

---

### Risk 4: JSON_VALUE Auto-Setup Failure

**Risk**: Auto-setup fails in restricted environments (permissions)

**Mitigation**:
- Graceful degradation (log warning, don't crash)
- Manual setup SQL script provided
- CLI command for setup
- Configuration option to disable auto-setup

**Validation**:
- Test in restricted IRIS environment
- Test with non-admin user
- Test with SQL script fallback

---

### Risk 5: Batch Operation Memory Usage

**Risk**: Large batches consume excessive memory

**Mitigation**:
- Configurable batch size (default: 1000)
- Streaming mode for very large datasets
- Memory usage monitoring
- Documentation of memory requirements

**Validation**:
- Test with 1M documents (monitor memory)
- Benchmark different batch sizes
- Document memory requirements per batch size

---

## Appendix A: Configuration Schema

### Complete Configuration Example

**File**: `config/enterprise_config.yaml`

```yaml
# Enterprise Enhancements Configuration

storage:
  iris:
    table_name: "RAG.SourceDocuments"
    vector_dimension: 1536

    # Enhancement 1: Configurable Metadata Filter Keys
    allowed_filter_keys:
      - category
      - year
      - source_type
      - document_id
      - author_name
      - title
      - source
      - type
      - date
      - status
      - version
      - pmcid
      - journal
      - doi
      - publication_date
      - keywords
      - abstract_type

    custom_filter_keys:
      - tenant_id        # Multi-tenancy
      - collection_id    # Virtual collections
      - security_level   # Classification
      - department       # Organizational
      - project_id       # Project isolation
      - owner            # Document ownership

    # Enhancement 7: JSON_VALUE Compatibility
    auto_setup_json_value: true  # Auto-create if missing

# Enhancement 4: OpenTelemetry Instrumentation
telemetry:
  enabled: false  # Disabled by default
  service_name: "iris-rag-api"
  service_version: "2.0.0"

  otlp:
    endpoint: "http://localhost:4318"
    protocol: "grpc"

  sampling:
    type: "traceidratio"
    ratio: 0.1  # 10% in production

  storage:
    iris_enabled: true
    otlp_enabled: true

  cost_tracking:
    enabled: true
    pricing_config: "config/llm_pricing.yaml"

# Enhancement 8: Batch Operations
batch_operations:
  default_batch_size: 1000
  show_progress: false  # Enable for CLI usage
  error_handling: "continue"  # 'continue', 'stop', 'rollback'
```

---

## Appendix B: API Reference

### IRISVectorStore New Methods

```python
class IRISVectorStore:
    # Enhancement 2: Collection Management
    def create_collection(collection_id: str, metadata: Optional[Dict] = None) -> bool
    def list_collections() -> List[Dict[str, Any]]
    def get_collection_info(collection_id: str) -> Dict[str, Any]
    def delete_collection(collection_id: str) -> int
    def collection_exists(collection_id: str) -> bool

    # Enhancement 6: Metadata Schema Discovery
    def sample_metadata_schema(
        collection_id: Optional[str] = None,
        sample_size: int = 100
    ) -> Dict[str, Dict[str, Any]]

    # Enhancement 8: Batch Operations
    def add_documents_batch(
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        batch_size: int = 1000,
        show_progress: bool = False,
        error_handling: str = "continue"
    ) -> Dict[str, Any]
```

### ConnectionManager New Patterns

```python
class ConnectionManager:
    def __init__(
        config_manager: Optional[ConfigurationManager] = None,
        connection: Optional[Any] = None,
        connection_factory: Optional[Callable] = None
    )
```

### RBACPolicy Interface

```python
class RBACPolicy(ABC):
    @abstractmethod
    def check_collection_access(user: str, collection_id: Optional[str], operation: str) -> bool

    @abstractmethod
    def filter_documents(user: str, documents: List[Document]) -> List[Document]
```

---

## Appendix C: Contribution Guide

### How to Contribute Enhancements

**Option 1: Contribute to iris-vector-rag Core**

1. Fork `intersystems-community/iris-vector-rag`
2. Create feature branch per enhancement
3. Implement with tests (90%+ coverage)
4. Update documentation
5. Submit pull request with real-world use case

**Option 2: Create Extension Package**

```bash
# Create iris-vector-rag-enterprise package
pip install iris-vector-rag-enterprise

# Extends core with enterprise features
from iris_vector_rag_enterprise import EnterpriseVectorStore

vector_store = EnterpriseVectorStore(
    config_manager=config,
    rbac_policy=MyRBACPolicy(),
    enable_telemetry=True
)
```

### Development Setup

```bash
# Clone repository
git clone https://github.com/intersystems-community/iris-vector-rag.git
cd iris-vector-rag

# Create feature branch
git checkout -b feature/configurable-filter-keys

# Install dependencies
make setup-env
make install
source .venv/bin/activate

# Run tests
make test

# Implement enhancement
# ... code changes ...

# Run tests again
make test

# Format code
make format

# Commit and push
git add .
git commit -m "feat: add configurable metadata filter keys"
git push origin feature/configurable-filter-keys
```

---

## Conclusion

These eight enterprise enhancements emerged from real-world integration challenges with the AI Hub platform. They address critical gaps in extensibility, security, observability, and usability while maintaining 100% backward compatibility.

**Implementation Approach**:
- Phased rollout over 5 weeks
- All features optional (opt-in)
- Comprehensive testing (78+ unit tests, 6+ integration tests)
- Zero breaking changes

**Expected Impact**:
- **Extensibility**: Custom metadata fields unblocked
- **Management**: Explicit collection lifecycle API
- **Security**: RBAC integration hooks
- **Observability**: OpenTelemetry instrumentation
- **Flexibility**: Connection injection support
- **Usability**: Schema discovery and JSON_VALUE auto-setup
- **Performance**: 10-100x speedup for bulk operations

**Next Steps**:
1. Community feedback on specification
2. Prioritization based on user needs
3. Alpha implementation (Phase 1)
4. Beta testing with AI Hub and other integrations
5. v2.0.0 release with all enhancements

---

**Feature Specification Status**: ✅ **READY FOR REVIEW**
**Target Release**: v2.0.0
**Estimated Timeline**: 5 weeks (implementation) + 2 weeks (testing/docs)
**Backward Compatibility**: 100% (all features opt-in)
