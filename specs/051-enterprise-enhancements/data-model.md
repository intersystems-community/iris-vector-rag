# Data Model: Enterprise Enhancements for RAG System

**Branch**: `051-enterprise-enhancements` | **Date**: 2025-11-22 | **Spec**: [spec.md](spec.md)

## Purpose

This document defines the entity models for all 6 enterprise enhancements identified in the specification. Each entity includes fields, relationships, validation rules, and lifecycle management.

---

## Entity 1: CustomMetadataField

**Purpose**: Represents an administrator-configured metadata field that extends default filtering capabilities for multi-tenant and domain-specific use cases.

### Fields

| Field | Type | Required | Description | Validation Rules |
|-------|------|----------|-------------|------------------|
| `field_name` | string | Yes | Name of the custom metadata field | - Must be alphanumeric + underscores<br>- Cannot start with digit<br>- Max length: 64 characters<br>- Cannot duplicate default field names |
| `is_enabled` | boolean | Yes | Whether field is active for filtering | Default: `true` |
| `added_at` | datetime | Yes | When field was configured | Auto-generated on creation |
| `added_by` | string | No | Administrator who added field | Optional audit trail |
| `description` | string | No | Human-readable purpose | Max length: 255 characters |

### Relationships

- **Merged With**: `DEFAULT_FILTER_KEYS` (hardcoded list of 17 default fields)
- **Used By**: Queries via `metadata_filter` parameter

### Lifecycle

1. **Configuration**: Administrator adds field to `storage.iris.custom_filter_keys` in YAML
2. **Initialization**: System merges custom fields with defaults on startup
3. **Validation**: System validates field names before allowing queries
4. **Usage**: Users pass custom field in `metadata_filter` parameter

### Example

```yaml
# Configuration
storage:
  iris:
    custom_filter_keys:
      - tenant_id         # Added for multi-tenancy
      - security_level    # Added for access control
      - department        # Added for organizational filtering
```

```python
# Runtime representation
CustomMetadataField(
    field_name="tenant_id",
    is_enabled=True,
    added_at=datetime(2025, 11, 22, 10, 30, 0),
    added_by="admin@company.com",
    description="Tenant identifier for multi-tenant isolation"
)
```

### Validation Logic

```python
import re

def validate_field_name(field_name: str) -> bool:
    """Validate custom metadata field name."""
    # Must be alphanumeric + underscores, not start with digit
    pattern = r'^[a-zA-Z_][a-zA-Z0-9_]{0,63}$'
    if not re.match(pattern, field_name):
        raise ValueError(
            f"Invalid field name '{field_name}'. "
            f"Must be alphanumeric + underscores, max 64 chars, "
            f"cannot start with digit."
        )

    # Cannot duplicate default fields
    if field_name in DEFAULT_FILTER_KEYS:
        raise ValueError(
            f"Field name '{field_name}' conflicts with default field. "
            f"Choose a different name."
        )

    return True
```

---

## Entity 2: Collection

**Purpose**: Represents a logical grouping of documents with shared metadata (e.g., collection_id). Collections provide organizational structure for document management.

### Fields

| Field | Type | Required | Description | Validation Rules |
|-------|------|----------|-------------|------------------|
| `collection_id` | string | Yes | Unique identifier for collection | - Max length: 128 characters<br>- Alphanumeric + hyphens/underscores<br>- Case-sensitive |
| `document_count` | integer | Yes | Number of documents in collection | - Auto-calculated<br>- Min: 0 |
| `total_size_bytes` | integer | Yes | Total size of all documents (content + embeddings) | - Auto-calculated<br>- Min: 0 |
| `created_at` | datetime | Yes | When first document was added | Auto-generated |
| `last_updated` | datetime | Yes | When collection was last modified | Auto-updated on document add/delete |
| `metadata` | object | No | Custom collection-level metadata | - JSON object<br>- Max size: 10 KB |

### Relationships

- **Contains**: List of `Document` entities (via `collection_id` metadata field)
- **Accessed By**: Users subject to RBAC policy checks

### Lifecycle

1. **Implicit Creation**: Collection created automatically when first document with `collection_id` is added
2. **Growth**: Document count and size updated as documents added
3. **Discovery**: Administrators list collections to see statistics
4. **Deletion**: Explicit deletion removes all documents with matching `collection_id`

### Example

```python
Collection(
    collection_id="medical-docs-2024",
    document_count=15432,
    total_size_bytes=524288000,  # ~500 MB
    created_at=datetime(2024, 1, 15, 9, 0, 0),
    last_updated=datetime(2025, 11, 20, 14, 30, 0),
    metadata={
        "department": "Clinical Research",
        "owner": "research-team@company.com",
        "retention_years": 7
    }
)
```

### Query Operations

**List Collections**:
```python
collections = store.list_collections()
# Returns list of Collection entities with statistics
```

**Get Collection Info**:
```python
info = store.get_collection_info("medical-docs-2024")
# Returns single Collection entity with detailed stats
```

**Delete Collection**:
```python
deleted_count = store.delete_collection("medical-docs-2024")
# Returns number of documents deleted
```

**Check Existence**:
```python
exists = store.collection_exists("medical-docs-2024")
# Returns boolean
```

---

## Entity 3: PermissionPolicy

**Purpose**: Represents an authorization decision made by the RBAC policy interface. This is a runtime entity (not persisted) that encapsulates access control logic.

### Fields

| Field | Type | Required | Description | Validation Rules |
|-------|------|----------|-------------|------------------|
| `user` | string | Yes | User identifier (email, username, or account ID) | - Non-empty string<br>- Max length: 255 characters |
| `resource` | string | Yes | Resource being accessed (collection_id or "all") | - Non-empty string<br>- Max length: 128 characters |
| `operation` | string | Yes | Operation being attempted | - Must be one of: `read`, `write`, `delete`, `admin` |
| `decision` | boolean | Yes | Whether access is granted | - `true` = allowed<br>- `false` = denied |
| `reason` | string | No | Explanation for decision (for audit logs) | - Max length: 255 characters |
| `evaluated_at` | datetime | Yes | When policy was evaluated | Auto-generated |
| `policy_version` | string | No | Version of policy that made decision | - For audit trail |

### Relationships

- **Evaluated By**: `RBACPolicy` implementation (user-provided class)
- **Applies To**: `Collection` entities and `Document` entities
- **Logged In**: Audit trail (if enabled)

### Lifecycle

1. **Invocation**: User attempts operation (query, add document, delete collection)
2. **Evaluation**: RBAC policy checks permissions (calls `check_collection_access()` or `filter_documents()`)
3. **Decision**: Policy returns boolean decision
4. **Enforcement**: System allows or denies operation
5. **Audit**: Decision logged with context (optional)

### Example

```python
# Collection-level permission check
PermissionPolicy(
    user="john.doe@company.com",
    resource="medical-docs-2024",
    operation="read",
    decision=True,
    reason="User has 'reader' role in LDAP group",
    evaluated_at=datetime(2025, 11, 22, 10, 45, 0),
    policy_version="1.2.0"
)

# Document-level filtering result
PermissionPolicy(
    user="jane.smith@company.com",
    resource="doc-12345",
    operation="read",
    decision=False,
    reason="Document security_level=5 exceeds user clearance=3",
    evaluated_at=datetime(2025, 11, 22, 10, 46, 0),
    policy_version="1.2.0"
)
```

### RBAC Policy Interface

```python
from abc import ABC, abstractmethod

class RBACPolicy(ABC):
    @abstractmethod
    def check_collection_access(
        self,
        user: str,
        collection_id: Optional[str],
        operation: str
    ) -> bool:
        """Return True if user can perform operation on collection."""
        pass

    @abstractmethod
    def filter_documents(
        self,
        user: str,
        documents: List[Document]
    ) -> List[Document]:
        """Return filtered list of documents user can access."""
        pass

    def get_audit_context(self, user: str) -> dict:
        """Optional: Return additional context for audit logs."""
        return {"user": user}
```

### Error Handling

```python
class PermissionDeniedError(Exception):
    """Raised when RBAC policy denies access."""
    def __init__(self, user, resource, operation):
        super().__init__(
            f"User '{user}' denied '{operation}' access to '{resource}'. "
            f"Contact your administrator if you believe this is an error."
        )
```

---

## Entity 4: MonitoringMetric

**Purpose**: Represents a telemetry data point captured during RAG system operations. Enables production observability and cost tracking.

### Fields

| Field | Type | Required | Description | Validation Rules |
|-------|------|----------|-------------|------------------|
| `metric_id` | string | Yes | Unique identifier for metric | Auto-generated UUID |
| `operation_type` | string | Yes | Type of operation | - Must be one of: `query`, `indexing`, `embedding`, `generation` |
| `timestamp` | datetime | Yes | When operation occurred | Auto-generated |
| `duration_ms` | float | Yes | Operation duration in milliseconds | - Min: 0 |
| `status` | string | Yes | Operation outcome | - One of: `success`, `error`, `timeout` |
| `user` | string | No | User who initiated operation | - Max length: 255 characters |
| `collection_id` | string | No | Collection accessed (if applicable) | - Max length: 128 characters |
| `query_text` | string | No | Query text (for search operations) | - Max length: 1024 characters<br>- Truncated if longer |
| `top_k` | integer | No | Number of documents requested | - Min: 1<br>- Max: 100 |
| `documents_retrieved` | integer | No | Actual documents returned | - Min: 0 |
| `embedding_model` | string | No | Model used for embeddings | - Max length: 64 characters |
| `embedding_dimensions` | integer | No | Vector dimension count | - Common: 384, 768, 1536 |
| `llm_model` | string | No | LLM model used (if generation) | - Max length: 64 characters<br>- e.g., "gpt-4", "claude-3-opus" |
| `prompt_tokens` | integer | No | Input tokens to LLM | - Min: 0 |
| `completion_tokens` | integer | No | Output tokens from LLM | - Min: 0 |
| `estimated_cost_usd` | float | No | Estimated operation cost | - Min: 0<br>- Calculated from token counts + pricing |
| `error_message` | string | No | Error details (if status=error) | - Max length: 512 characters |
| `span_id` | string | No | OpenTelemetry span ID | - For distributed tracing |
| `trace_id` | string | No | OpenTelemetry trace ID | - Links related operations |

### Relationships

- **Belongs To**: OpenTelemetry trace (via `trace_id`)
- **Child Of**: Parent span (via OpenTelemetry context)
- **Exported To**: External observability systems (Prometheus, Grafana, Datadog)

### Lifecycle

1. **Instrumentation**: Code wrapped in `telemetry.trace_operation()` context manager
2. **Capture**: Metric created when operation starts
3. **Enrichment**: Fields populated as operation progresses
4. **Completion**: Duration and status recorded when operation ends
5. **Export**: Metric sent to OpenTelemetry collector (if enabled)
6. **Retention**: Stored in external system per retention policy

### Example

```python
# Query operation metric
MonitoringMetric(
    metric_id="550e8400-e29b-41d4-a716-446655440000",
    operation_type="query",
    timestamp=datetime(2025, 11, 22, 10, 50, 0),
    duration_ms=1456.3,
    status="success",
    user="john.doe@company.com",
    collection_id="medical-docs-2024",
    query_text="What are the symptoms of diabetes?",
    top_k=5,
    documents_retrieved=5,
    embedding_model="all-MiniLM-L6-v2",
    embedding_dimensions=384,
    llm_model="gpt-4",
    prompt_tokens=1234,
    completion_tokens=567,
    estimated_cost_usd=0.0234,
    error_message=None,
    span_id="7a9f2c8b3d1e",
    trace_id="4b8e1d6c2f9a3e5d"
)
```

### OpenTelemetry Integration

```python
from iris_vector_rag.monitoring.telemetry import telemetry

def query(self, query_text: str, top_k: int = 5):
    with telemetry.trace_operation(
        "rag.query",
        query_length=len(query_text),
        top_k=top_k,
        pipeline="basic"
    ) as span:
        # Operation executes
        # Span automatically captures duration, status, errors
        pass
```

### Cost Calculation

```python
# Pricing table (example, varies by provider)
PRICING = {
    "gpt-4": {"input": 0.00003, "output": 0.00006},  # per token
    "claude-3-opus": {"input": 0.000015, "output": 0.000075}
}

def calculate_cost(llm_model, prompt_tokens, completion_tokens):
    if llm_model not in PRICING:
        return 0.0
    prices = PRICING[llm_model]
    return (prompt_tokens * prices["input"]) + (completion_tokens * prices["output"])
```

---

## Entity 5: BulkOperation

**Purpose**: Represents a batch document loading operation with progress tracking and error handling. Enables efficient large-scale data ingestion.

### Fields

| Field | Type | Required | Description | Validation Rules |
|-------|------|----------|-------------|------------------|
| `operation_id` | string | Yes | Unique identifier for bulk operation | Auto-generated UUID |
| `status` | string | Yes | Current operation status | - One of: `pending`, `in_progress`, `completed`, `failed`, `cancelled` |
| `started_at` | datetime | Yes | When operation started | Auto-generated |
| `completed_at` | datetime | No | When operation finished | Auto-set on completion |
| `total_documents` | integer | Yes | Total documents to process | - Min: 1 |
| `processed_documents` | integer | Yes | Documents processed so far | - Min: 0<br>- Max: `total_documents` |
| `success_count` | integer | Yes | Documents successfully added | - Min: 0 |
| `error_count` | integer | Yes | Documents that failed | - Min: 0 |
| `progress_percentage` | float | Yes | Completion percentage | - Calculated: `(processed / total) * 100`<br>- Range: 0.0 - 100.0 |
| `batch_size` | integer | Yes | Documents per batch | - Min: 1<br>- Default: 1000 |
| `error_handling` | string | Yes | Error handling strategy | - One of: `continue`, `stop`, `rollback` |
| `show_progress` | boolean | Yes | Whether to display progress bar | Default: `false` |
| `errors` | array | No | List of error details | - Max 100 errors stored<br>- Each error: `{index, doc_id, error}` |
| `time_seconds` | float | No | Total operation time | - Calculated on completion |
| `throughput_docs_per_sec` | float | No | Processing throughput | - Calculated: `total_documents / time_seconds` |

### Relationships

- **Processes**: List of `Document` entities
- **Creates**: `Collection` entity (if documents belong to new collection)
- **Logs**: `MonitoringMetric` entities (if telemetry enabled)

### Lifecycle

1. **Initialization**: User calls `add_documents_batch()` with document list
2. **Planning**: System calculates batch count and creates `BulkOperation` entity
3. **Processing**: Documents processed in batches of `batch_size`
4. **Progress Updates**: `processed_documents` and `progress_percentage` updated after each batch
5. **Error Handling**: Errors recorded in `errors` array per strategy
6. **Completion**: Status set to `completed` or `failed`, metrics calculated

### Example

```python
# Bulk operation in progress
BulkOperation(
    operation_id="7c9e6679-7425-40de-944b-e07fc1f90ae7",
    status="in_progress",
    started_at=datetime(2025, 11, 22, 11, 0, 0),
    completed_at=None,
    total_documents=10000,
    processed_documents=4500,
    success_count=4487,
    error_count=13,
    progress_percentage=45.0,
    batch_size=1000,
    error_handling="continue",
    show_progress=True,
    errors=[
        {
            "index": 127,
            "doc_id": "doc-127",
            "error": "Invalid metadata: missing required field 'collection_id'"
        },
        # ... up to 100 errors
    ],
    time_seconds=None,
    throughput_docs_per_sec=None
)

# Completed bulk operation
BulkOperation(
    operation_id="7c9e6679-7425-40de-944b-e07fc1f90ae7",
    status="completed",
    started_at=datetime(2025, 11, 22, 11, 0, 0),
    completed_at=datetime(2025, 11, 22, 11, 0, 8),
    total_documents=10000,
    processed_documents=10000,
    success_count=9987,
    error_count=13,
    progress_percentage=100.0,
    batch_size=1000,
    error_handling="continue",
    show_progress=True,
    errors=[...],  # 13 errors
    time_seconds=8.2,
    throughput_docs_per_sec=1219.5  # 10000 / 8.2
)
```

### Progress Tracking

```python
# With progress bar (CLI usage)
result = store.add_documents_batch(
    documents=docs,
    batch_size=1000,
    show_progress=True
)
# Output:
# Loading documents: 100%|██████████| 10/10 [00:08<00:00,  1.22batch/s]
```

### Error Handling Strategies

**1. Continue (default)**:
```python
# Skip failed documents, continue processing
# Best for: Data migrations, best-effort loading
result = store.add_documents_batch(docs, error_handling="continue")
```

**2. Stop**:
```python
# Stop on first error, commit previous batches
# Best for: Production loads, fail-fast validation
result = store.add_documents_batch(docs, error_handling="stop")
```

**3. Rollback**:
```python
# Stop on first error, rollback all batches
# Best for: Critical operations, all-or-nothing semantics
result = store.add_documents_batch(docs, error_handling="rollback")
```

---

## Entity 6: MetadataSchema

**Purpose**: Represents the discovered structure of metadata fields within a collection. Enables developers to understand available filters without manual documentation.

### Fields

| Field | Type | Required | Description | Validation Rules |
|-------|------|----------|-------------|------------------|
| `field_name` | string | Yes | Metadata field name | - Unique within schema<br>- Max length: 128 characters |
| `field_type` | string | Yes | Inferred JSON type | - One of: `string`, `integer`, `float`, `boolean`, `datetime`, `array`, `object`, `unknown` |
| `frequency` | float | Yes | Field occurrence rate | - Range: 0.0 - 1.0<br>- 1.0 = present in all documents |
| `nullable` | boolean | Yes | Whether field can be null | Default: `false` |
| `unique_value_count` | integer | Yes | Number of distinct values | - Min: 0 |
| `sample_size` | integer | Yes | Documents sampled for discovery | - Default: 100 |
| `examples` | array | No | Example values (for strings/datetimes) | - Max 5 examples<br>- Truncated if longer |
| `min_value` | varies | No | Minimum value (numeric/datetime fields) | - Type matches `field_type` |
| `max_value` | varies | No | Maximum value (numeric/datetime fields) | - Type matches `field_type` |
| `avg_value` | float | No | Average value (numeric fields only) | - Only for `integer` and `float` types |
| `discovered_at` | datetime | Yes | When schema was sampled | Auto-generated |
| `collection_id` | string | No | Collection sampled (if specific) | - `null` = all collections sampled |

### Relationships

- **Describes**: Metadata fields in `Document` entities
- **Derived From**: Sample of documents in `Collection`
- **Used By**: Developers building queries and filters

### Lifecycle

1. **Invocation**: Developer calls `sample_metadata_schema()`
2. **Sampling**: System randomly samples 100-200 documents
3. **Aggregation**: Field names, types, and statistics calculated
4. **Inference**: Types inferred from Python values
5. **Return**: Schema returned as dictionary

### Example

```python
# String field schema
MetadataSchema(
    field_name="tenant_id",
    field_type="string",
    frequency=0.98,  # 98% of documents have this field
    nullable=True,
    unique_value_count=12,
    sample_size=100,
    examples=["tenant_001", "tenant_002", "tenant_003"],
    min_value=None,
    max_value=None,
    avg_value=None,
    discovered_at=datetime(2025, 11, 22, 11, 15, 0),
    collection_id="medical-docs-2024"
)

# Integer field schema
MetadataSchema(
    field_name="priority",
    field_type="integer",
    frequency=0.75,  # 75% of documents have this field
    nullable=False,
    unique_value_count=5,
    sample_size=100,
    examples=[1, 2, 3, 4, 5],
    min_value=1,
    max_value=5,
    avg_value=2.8,
    discovered_at=datetime(2025, 11, 22, 11, 15, 0),
    collection_id="medical-docs-2024"
)

# Datetime field schema
MetadataSchema(
    field_name="created_at",
    field_type="datetime",
    frequency=1.0,  # 100% of documents have this field
    nullable=False,
    unique_value_count=87,
    sample_size=100,
    examples=["2023-06-20T08:15:00Z", "2024-03-15T14:30:00Z"],
    min_value="2023-01-15T10:30:00Z",
    max_value="2024-11-22T15:45:00Z",
    avg_value=None,
    discovered_at=datetime(2025, 11, 22, 11, 15, 0),
    collection_id="medical-docs-2024"
)
```

### Discovery API

```python
# Discover schema for specific collection
schema = store.sample_metadata_schema(
    collection_id="medical-docs-2024",
    sample_size=200
)

# Returns:
{
    "tenant_id": MetadataSchema(...),
    "priority": MetadataSchema(...),
    "created_at": MetadataSchema(...),
    # ... all discovered fields
}
```

### Type Inference Rules

| Python Type | Inferred JSON Type | Notes |
|-------------|-------------------|-------|
| `bool` | `boolean` | Checked before `int` (bool is subclass of int) |
| `int` | `integer` | Whole numbers |
| `float` | `float` | Decimal numbers |
| `str` (ISO 8601) | `datetime` | e.g., "2024-11-22T10:30:00Z" |
| `str` | `string` | Default for strings |
| `list` | `array` | JSON arrays |
| `dict` | `object` | Nested JSON objects |
| Other | `unknown` | Unsupported types |

---

## Entity Relationships Diagram

```
┌─────────────────────┐
│ CustomMetadataField │
│  - field_name       │
│  - is_enabled       │
└─────────────────────┘
          │
          │ merged with
          ▼
┌─────────────────────┐
│ DEFAULT_FILTER_KEYS │ (hardcoded)
│  - collection_id    │
│  - category         │
│  - year, etc.       │
└─────────────────────┘
          │
          │ validates
          ▼
┌─────────────────────┐        contains        ┌─────────────────────┐
│    Collection       │◄─────────────────────►│     Document        │
│  - collection_id    │                        │  - page_content     │
│  - document_count   │                        │  - embeddings       │
│  - total_size_bytes │                        │  - metadata         │
└─────────────────────┘                        └─────────────────────┘
          │                                              │
          │ accessed via                                 │ described by
          ▼                                              ▼
┌─────────────────────┐                        ┌─────────────────────┐
│  PermissionPolicy   │                        │  MetadataSchema     │
│  - user             │                        │  - field_name       │
│  - resource         │                        │  - field_type       │
│  - operation        │                        │  - frequency        │
│  - decision         │                        │  - statistics       │
└─────────────────────┘                        └─────────────────────┘
          │
          │ enforces
          ▼
┌─────────────────────┐        processes       ┌─────────────────────┐
│   RBACPolicy ABC    │                        │   BulkOperation     │
│  - check_access()   │                        │  - operation_id     │
│  - filter_docs()    │                        │  - status           │
└─────────────────────┘                        │  - progress_%       │
                                                │  - errors[]         │
                                                └─────────────────────┘
                                                          │
                                                          │ logs
                                                          ▼
                                                ┌─────────────────────┐
                                                │  MonitoringMetric   │
                                                │  - operation_type   │
                                                │  - duration_ms      │
                                                │  - tokens/cost      │
                                                └─────────────────────┘
```

---

## Storage Considerations

### IRIS Database Schema

All enhancements use the existing `RAG.SourceDocuments` table:

```sql
-- Existing table (no changes required)
CREATE TABLE RAG.SourceDocuments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content VARCHAR(MAX),
    embeddings VECTOR(DOUBLE, 384),  -- or 768, 1536 depending on model
    metadata VARCHAR(MAX),  -- JSON string
    collection_id VARCHAR(128),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Metadata JSON Structure** (examples):
```json
{
  "collection_id": "medical-docs-2024",
  "tenant_id": "tenant_001",
  "security_level": 3,
  "priority": 2,
  "category": "research-paper",
  "year": 2024,
  "author": "Dr. Smith",
  "file_name": "diabetes-study.pdf",
  "page_number": 127,
  "created_at": "2024-11-22T10:30:00Z"
}
```

### In-Memory Entities

The following entities are **not persisted** (runtime only):
- `CustomMetadataField` - Loaded from YAML config on startup
- `PermissionPolicy` - Evaluated on-demand per request
- `BulkOperation` - Transient (returned as response)
- `MetadataSchema` - Calculated on-demand from sampling

### Persisted Entities

The following entities **may be persisted** (optional):
- `Collection` - Statistics can be cached in `RAG.CollectionStats` table (optional optimization)
- `MonitoringMetric` - Exported to external observability systems (OpenTelemetry, Prometheus)

---

## Validation Summary

| Entity | Validation Strategy | Error Handling |
|--------|-------------------|----------------|
| CustomMetadataField | Regex pattern, default field check | Raise `ValueError` with clear message |
| Collection | Alphanumeric check, max length | Raise `ValueError` |
| PermissionPolicy | Enum validation (operation) | Raise `PermissionDeniedError` |
| MonitoringMetric | Type validation, range checks | Log warning, continue (non-blocking) |
| BulkOperation | Status transitions, progress bounds | Raise `RuntimeError` on invalid state |
| MetadataSchema | Sample size bounds, type inference | Return `unknown` type on failure |

---

## Next Steps

1. **Generate contract YAML files** defining API specifications for each enhancement
2. **Generate quickstart.md** with usage examples for all 6 entities
3. **Update agent context** with new entity models
4. **Phase 2: Generate tasks.md** with atomic implementation tasks

**Ready for**: Contract generation (Phase 1 continued)
