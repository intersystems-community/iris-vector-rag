# Data Model: Production-Grade REST API for RAG Pipelines

**Feature**: 042-full-rest-api
**Date**: 2025-10-16
**Purpose**: Define data entities, relationships, validation rules, and state transitions

---

## Entity 1: API Request

### Description
Represents an incoming HTTP request containing query text, pipeline selection, parameters, and authentication credentials. Includes metadata like request ID, timestamp, and client identifier for tracing and monitoring.

### Fields

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `request_id` | UUID | Yes | Auto-generated | Unique identifier for request tracing |
| `timestamp` | DateTime | Yes | Auto-generated (ISO8601) | When request was received |
| `method` | Enum | Yes | `GET`, `POST`, `PUT`, `DELETE` | HTTP method |
| `endpoint` | String | Yes | Max 255 chars, starts with `/` | API endpoint path |
| `query_text` | String | Conditional | 1-10000 chars (for query endpoints) | User's query text |
| `pipeline_type` | Enum | Conditional | `basic`, `basic_rerank`, `crag`, `graphrag`, `pylate_colbert` | Selected RAG pipeline |
| `parameters` | JSON Object | No | Valid JSON | Query parameters (top_k, filters, etc.) |
| `api_key_id` | UUID | Yes | Foreign key to Authentication Token | Which API key was used |
| `client_ip` | IP Address | Yes | Valid IPv4/IPv6 | Client IP address |
| `user_agent` | String | No | Max 500 chars | Client user agent string |
| `status_code` | Integer | Yes | 200-599 | HTTP response status code |
| `response_time_ms` | Integer | Yes | >= 0 | Request execution time in milliseconds |
| `error_type` | String | No | Max 100 chars | Error type if request failed |
| `error_message` | String | No | Max 1000 chars | Error message if request failed |

### Validation Rules
- **FR-003**: Query text must be 1-10000 characters (reject with 422 if exceeded)
- **FR-017**: Status code must be valid HTTP status (200, 400, 401, 422, 429, 500, 503)
- **FR-035**: All required fields must be logged for monitoring
- **Query endpoint validation**: `query_text` and `pipeline_type` required for `POST /{pipeline}/_search`

### State Transitions
```
None → Received (timestamp set) → Processing → Completed (status_code, response_time_ms set)
                                             ↘ Failed (error_type, error_message set)
```

### IRIS Storage
- **Table**: `RAG_API.ApiRequestLog`
- **Indexes**: `request_id` (primary), `api_key_id` (foreign key), `timestamp` (for time-series queries)
- **Retention**: Auto-delete records older than 30 days (FR-036)
- **Partitioning**: By timestamp (monthly partitions for efficient deletion)

### Relationships
- **Many-to-One** with Authentication Token (via `api_key_id`)
- **One-to-Many** with Query Response (one request can have multiple response chunks in streaming mode)

### Example JSON (Pydantic Model)
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-10-16T12:34:56.789Z",
  "method": "POST",
  "endpoint": "/api/v1/graphrag/_search",
  "query_text": "What is diabetes?",
  "pipeline_type": "graphrag",
  "parameters": {
    "top_k": 5,
    "filters": {"domain": "medical"}
  },
  "api_key_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "client_ip": "192.168.1.100",
  "user_agent": "Python-Requests/2.31.0",
  "status_code": 200,
  "response_time_ms": 1456
}
```

---

## Entity 2: Pipeline Instance

### Description
Represents an initialized RAG pipeline (basic, CRAG, GraphRAG, etc.) with its current health status, configuration, and performance statistics. Manages lifecycle from initialization through shutdown.

### Fields

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `pipeline_id` | UUID | Yes | Auto-generated | Unique pipeline instance identifier |
| `pipeline_type` | Enum | Yes | `basic`, `basic_rerank`, `crag`, `graphrag`, `pylate_colbert` | Pipeline type |
| `name` | String | Yes | 1-100 chars, alphanumeric + dash | Human-readable pipeline name |
| `status` | Enum | Yes | `healthy`, `degraded`, `unavailable` | Current health status |
| `initialized_at` | DateTime | Yes | ISO8601 | When pipeline was initialized |
| `last_health_check` | DateTime | Yes | ISO8601 | Last health check timestamp |
| `total_queries` | Integer | Yes | >= 0 | Total queries processed since initialization |
| `avg_latency_ms` | Float | Yes | >= 0 | Average query latency (last 100 queries) |
| `error_count` | Integer | Yes | >= 0 | Number of errors since initialization |
| `error_rate` | Float | Yes | 0.0-1.0 | Error rate (errors / total_queries) |
| `config` | JSON Object | Yes | Valid JSON | Pipeline configuration (model, embeddings, etc.) |
| `capabilities` | List[String] | Yes | Non-empty | Supported features (e.g., ["vector_search", "graph_traversal"]) |

### Validation Rules
- **FR-005**: Pipeline must be initialized before accepting queries
- **FR-006**: Status must be `healthy` for query processing
- **FR-007**: Graceful shutdown must complete in-flight queries before status → `unavailable`
- **Health check interval**: Status updated every 30 seconds

### State Transitions
```
None → Initializing (pipeline_id, initialized_at set)
     ↓
   Healthy (status="healthy", ready for queries)
     ↓
   Degraded (status="degraded", some features unavailable)
     ↓
   Unavailable (status="unavailable", cannot process queries)
     ↓
   Shutdown (removed from active pool)
```

### In-Memory Storage
- **Not persisted to IRIS**: Pipeline instances exist only in memory (FastAPI app state)
- **Registry**: `Dict[str, PipelineInstance]` keyed by `pipeline_type`
- **Lifecycle**: Created on FastAPI startup, destroyed on shutdown

### Relationships
- **One-to-Many** with API Request (one pipeline handles many requests)
- **One-to-One** with RAG framework pipeline object (iris_rag.core.base.RAGPipeline)

### Example JSON (Pydantic Model)
```json
{
  "pipeline_id": "a3c4e5f6-7890-4b2d-9e1f-234567890abc",
  "pipeline_type": "graphrag",
  "name": "graphrag-production",
  "status": "healthy",
  "initialized_at": "2025-10-16T10:00:00.000Z",
  "last_health_check": "2025-10-16T12:34:50.000Z",
  "total_queries": 15234,
  "avg_latency_ms": 1234.5,
  "error_count": 23,
  "error_rate": 0.0015,
  "config": {
    "llm_model": "gpt-4",
    "embedding_model": "text-embedding-3-small",
    "top_k": 10
  },
  "capabilities": ["vector_search", "graph_traversal", "entity_extraction"]
}
```

---

## Entity 3: Authentication Token

### Description
Represents credentials used to authenticate API requests. Includes associated permissions, rate limits, and expiration information. API keys use base64-encoded `id:secret` format.

### Fields

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `key_id` | UUID | Yes | Auto-generated | API key identifier |
| `key_secret_hash` | String | Yes | bcrypt hash | Hashed API key secret (never store plaintext) |
| `name` | String | Yes | 1-100 chars | Human-readable key name |
| `permissions` | List[Enum] | Yes | At least one of `read`, `write`, `admin` | Allowed operations |
| `rate_limit_tier` | Enum | Yes | `basic`, `premium`, `enterprise` | Rate limiting tier |
| `requests_per_minute` | Integer | Yes | 1-10000 | Max requests per minute |
| `requests_per_hour` | Integer | Yes | 1-100000 | Max requests per hour |
| `created_at` | DateTime | Yes | ISO8601 | When key was created |
| `expires_at` | DateTime | No | ISO8601, future date | Key expiration (optional) |
| `last_used_at` | DateTime | No | ISO8601 | Last request timestamp |
| `is_active` | Boolean | Yes | Default: true | Whether key is enabled |
| `owner_email` | String | No | Valid email | Key owner contact |

### Validation Rules
- **FR-009**: API key format: `Authorization: ApiKey <base64(id:api_key)>`
- **FR-010**: Unauthenticated requests (missing/invalid key) → 401 Unauthorized
- **FR-011**: Permission check before processing (e.g., `write` permission for document upload)
- **FR-012**: Log all authentication failures with key_id (if parseable)
- **Secret hashing**: Use bcrypt with cost factor 12 for security

### State Transitions
```
None → Created (key_id, key_secret_hash, created_at set)
     ↓
   Active (is_active=true, ready for use)
     ↓
   Used (last_used_at updated on each request)
     ↓
   Expired (expires_at < current_time) → Cannot authenticate
     ↓
   Revoked (is_active=false) → Cannot authenticate
```

### IRIS Storage
- **Table**: `RAG_API.ApiKey`
- **Indexes**: `key_id` (primary), `is_active` (for fast active key lookup)
- **Security**: key_secret_hash column encrypted at rest
- **Audit**: All key creation/revocation logged to separate audit table

### Relationships
- **One-to-Many** with API Request (one key used by many requests)
- **One-to-One** with Rate Limit Quota (each key has quota tracking)

### Example JSON (Pydantic Model)
```json
{
  "key_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "key_secret_hash": "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYk3H.olfm",
  "name": "production-app-key",
  "permissions": ["read", "write"],
  "rate_limit_tier": "premium",
  "requests_per_minute": 100,
  "requests_per_hour": 5000,
  "created_at": "2025-01-15T08:00:00.000Z",
  "expires_at": "2026-01-15T08:00:00.000Z",
  "last_used_at": "2025-10-16T12:34:56.789Z",
  "is_active": true,
  "owner_email": "developer@example.com"
}
```

---

## Entity 4: Rate Limit Quota

### Description
Represents allowed request rate for a specific API key. Tracks current usage, reset timestamp, and enforcement policy. Supports both static quotas and adaptive concurrency.

### Fields

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `quota_id` | UUID | Yes | Auto-generated | Unique quota identifier |
| `api_key_id` | UUID | Yes | Foreign key | Associated API key |
| `quota_type` | Enum | Yes | `requests_per_minute`, `requests_per_hour`, `concurrent_requests` | Type of quota |
| `limit` | Integer | Yes | > 0 | Maximum allowed value |
| `current_usage` | Integer | Yes | >= 0 | Current usage in time window |
| `window_start` | DateTime | Yes | ISO8601 | When current window started |
| `window_end` | DateTime | Yes | ISO8601 | When current window ends |
| `next_reset_at` | DateTime | Yes | ISO8601 | When quota will reset |
| `exceeded_count` | Integer | Yes | >= 0 | Number of times quota exceeded |
| `last_exceeded_at` | DateTime | No | ISO8601 | Last time quota was exceeded |

### Validation Rules
- **FR-013**: Enforce per-API-key rate limits to prevent abuse
- **FR-014**: Return rate limit headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`
- **FR-015**: Return 429 when limit exceeded with `Retry-After` header
- **FR-016**: Support configurable per-key quotas with fallback to static limits

### Rate Limit Algorithm
```python
# Sliding window counter (Redis-based)
def check_rate_limit(api_key_id: str, quota_type: str) -> tuple[bool, int]:
    """
    Returns: (is_allowed, remaining_quota)
    """
    window_size = 60 if quota_type == "requests_per_minute" else 3600
    current_time = time.time()
    window_start = current_time - window_size

    # Count requests in sliding window
    request_count = redis.zcount(
        f"rate_limit:{api_key_id}:{quota_type}",
        window_start,
        current_time
    )

    quota = get_quota_limit(api_key_id, quota_type)
    is_allowed = request_count < quota
    remaining = max(0, quota - request_count)

    if is_allowed:
        # Add current request to window
        redis.zadd(
            f"rate_limit:{api_key_id}:{quota_type}",
            {current_time: current_time}
        )
        # Clean old entries
        redis.zremrangebyscore(
            f"rate_limit:{api_key_id}:{quota_type}",
            0,
            window_start
        )

    return is_allowed, remaining
```

### State Transitions
```
None → Active (quota tracking started on first request)
     ↓
   Within Limit (current_usage < limit, allow requests)
     ↓
   At Limit (current_usage == limit, allow but warn in headers)
     ↓
   Exceeded (current_usage > limit, reject with 429)
     ↓
   Reset (window_end reached, current_usage = 0)
```

### Storage
- **Redis**: Primary storage for real-time rate limiting (fast ZSET operations)
- **IRIS Table**: `RAG_API.RateLimitQuota` for persistent quota configuration
- **Fallback**: In-memory storage if Redis unavailable (single-server deployments)

### Relationships
- **Many-to-One** with Authentication Token (one key has multiple quota types)
- **One-to-Many** with API Request (rate limit checked for each request)

### Example JSON (Pydantic Model)
```json
{
  "quota_id": "f4a3b2c1-5678-4d3e-9f0a-123456789def",
  "api_key_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "quota_type": "requests_per_minute",
  "limit": 100,
  "current_usage": 73,
  "window_start": "2025-10-16T12:34:00.000Z",
  "window_end": "2025-10-16T12:35:00.000Z",
  "next_reset_at": "2025-10-16T12:35:00.000Z",
  "exceeded_count": 5,
  "last_exceeded_at": "2025-10-16T11:23:45.000Z"
}
```

---

## Entity 5: Query Response

### Description
Structured result containing generated answer, retrieved documents with sources, execution metadata (pipeline used, latency, tokens), and optional streaming chunks for WebSocket responses.

### Fields

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `response_id` | UUID | Yes | Auto-generated | Unique response identifier |
| `request_id` | UUID | Yes | Foreign key | Associated API request |
| `answer` | String | Yes | Non-empty | LLM-generated answer text |
| `retrieved_documents` | List[Document] | Yes | Non-empty | Retrieved documents (see Document schema) |
| `sources` | List[String] | Yes | Can be empty | Source references (URLs, filenames, etc.) |
| `pipeline_name` | String | Yes | 1-100 chars | Pipeline that processed query |
| `execution_time_ms` | Integer | Yes | >= 0 | Total query execution time |
| `retrieval_time_ms` | Integer | Yes | >= 0 | Time spent retrieving documents |
| `generation_time_ms` | Integer | Yes | >= 0 | Time spent generating answer |
| `tokens_used` | Integer | Yes | >= 0 | Total LLM tokens consumed |
| `confidence_score` | Float | No | 0.0-1.0 | Answer confidence (if available) |
| `metadata` | JSON Object | No | Valid JSON | Pipeline-specific metadata |

### Nested Schema: Document
```json
{
  "doc_id": "uuid",
  "content": "string (document text)",
  "score": "float (similarity score 0.0-1.0)",
  "metadata": {
    "source": "string (filename, URL, etc.)",
    "chunk_index": "integer (for chunked documents)",
    "page_number": "integer (optional)",
    "created_at": "datetime (ISO8601)"
  }
}
```

### Validation Rules
- **FR-002**: Response MUST contain answer, retrieved_documents, sources, execution_metadata
- **FR-004**: Support multiple pipeline types with consistent response format
- **100% LangChain compatible**: Response format matches LangChain Document structure
- **RAGAS evaluation ready**: Include `contexts` field (List[str] of document content)

### Response Format (Pydantic Model)
```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from uuid import UUID

class DocumentMetadata(BaseModel):
    source: str
    chunk_index: Optional[int] = None
    page_number: Optional[int] = None
    created_at: Optional[str] = None

class Document(BaseModel):
    doc_id: UUID
    content: str = Field(..., min_length=1)
    score: float = Field(..., ge=0.0, le=1.0)
    metadata: DocumentMetadata

class QueryResponse(BaseModel):
    response_id: UUID
    request_id: UUID
    answer: str = Field(..., min_length=1)
    retrieved_documents: List[Document] = Field(..., min_items=1)
    sources: List[str] = Field(default_factory=list)
    pipeline_name: str
    execution_time_ms: int = Field(..., ge=0)
    retrieval_time_ms: int = Field(..., ge=0)
    generation_time_ms: int = Field(..., ge=0)
    tokens_used: int = Field(..., ge=0)
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    metadata: Optional[Dict] = None

    # RAGAS compatibility field
    @property
    def contexts(self) -> List[str]:
        """Extract document content for RAGAS evaluation"""
        return [doc.content for doc in self.retrieved_documents]
```

### Example JSON
```json
{
  "response_id": "9a8b7c6d-5e4f-3210-9876-543210fedcba",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "answer": "Diabetes is a chronic metabolic disorder characterized by elevated blood glucose levels...",
  "retrieved_documents": [
    {
      "doc_id": "1a2b3c4d-5678-90ab-cdef-1234567890ab",
      "content": "Diabetes mellitus is a group of metabolic diseases...",
      "score": 0.95,
      "metadata": {
        "source": "medical_textbook_ch5.pdf",
        "chunk_index": 3,
        "page_number": 127,
        "created_at": "2025-01-10T00:00:00Z"
      }
    }
  ],
  "sources": ["medical_textbook_ch5.pdf"],
  "pipeline_name": "graphrag",
  "execution_time_ms": 1456,
  "retrieval_time_ms": 345,
  "generation_time_ms": 1089,
  "tokens_used": 2345,
  "confidence_score": 0.92,
  "metadata": {
    "graph_entities_found": 5,
    "graph_relationships_traversed": 12
  }
}
```

---

## Entity 6: Document Upload Operation

### Description
Represents an asynchronous document loading task with status tracking, progress percentage, validation results, and completion/error information. Supports batch uploads and progress streaming via WebSocket.

### Fields

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `operation_id` | UUID | Yes | Auto-generated | Unique operation identifier |
| `api_key_id` | UUID | Yes | Foreign key | API key that initiated upload |
| `status` | Enum | Yes | `pending`, `validating`, `processing`, `completed`, `failed` | Current operation status |
| `created_at` | DateTime | Yes | ISO8601 | When operation was created |
| `started_at` | DateTime | No | ISO8601 | When processing started |
| `completed_at` | DateTime | No | ISO8601 | When operation completed/failed |
| `total_documents` | Integer | Yes | > 0 | Total documents to process |
| `processed_documents` | Integer | Yes | >= 0 | Documents processed so far |
| `failed_documents` | Integer | Yes | >= 0 | Documents that failed validation/indexing |
| `progress_percentage` | Float | Yes | 0.0-100.0 | Completion percentage |
| `file_size_bytes` | Integer | Yes | 1-104857600 (100MB max) | Total file size |
| `pipeline_type` | Enum | Yes | `basic`, `graphrag`, etc. | Pipeline for indexing |
| `validation_errors` | List[String] | No | Max 100 errors | Validation error messages |
| `error_message` | String | No | Max 1000 chars | Error message if operation failed |

### Validation Rules
- **FR-021**: Async document upload (no blocking query processing)
- **FR-022**: Validate document format and size before accepting (max 100MB per file)
- **FR-023**: Return operation status (pending, processing, completed, failed)
- **FR-024**: Document loading must not block query processing

### State Transitions
```
None → Pending (operation_id, created_at, status="pending")
     ↓
   Validating (status="validating", file format/size checks)
     ↓
   Processing (status="processing", started_at set, indexing documents)
     ↓ ← (progress updates via WebSocket: processed_documents, progress_percentage)
     ↓
   Completed (status="completed", completed_at set, processed_documents == total_documents)
     ↓
   Failed (status="failed", completed_at set, error_message set)
```

### IRIS Storage
- **Table**: `RAG_API.DocumentUploadOperation`
- **Indexes**: `operation_id` (primary), `api_key_id` (foreign key), `status` (for filtering)
- **Retention**: Keep completed operations for 7 days, then archive

### Relationships
- **Many-to-One** with Authentication Token (one key can initiate many uploads)
- **One-to-Many** with WebSocket events (operation progress streamed via WebSocket)

### Example JSON (Pydantic Model)
```json
{
  "operation_id": "b1c2d3e4-5678-90ab-cdef-fedcba987654",
  "api_key_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "status": "processing",
  "created_at": "2025-10-16T12:30:00.000Z",
  "started_at": "2025-10-16T12:30:05.000Z",
  "total_documents": 100,
  "processed_documents": 47,
  "failed_documents": 2,
  "progress_percentage": 47.0,
  "file_size_bytes": 52428800,
  "pipeline_type": "graphrag",
  "validation_errors": [
    "Document 23: Invalid UTF-8 encoding",
    "Document 45: Exceeds maximum chunk size"
  ]
}
```

---

## Entity 7: WebSocket Session

### Description
Represents an active WebSocket connection for streaming responses. Manages message queue, reconnection state, and client subscription preferences for progress updates.

### Fields

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `session_id` | UUID | Yes | Auto-generated | Unique session identifier |
| `api_key_id` | UUID | Yes | Foreign key | Authenticated API key |
| `connected_at` | DateTime | Yes | ISO8601 | When connection was established |
| `last_activity_at` | DateTime | Yes | ISO8601 | Last message sent/received |
| `client_ip` | IP Address | Yes | Valid IPv4/IPv6 | Client IP address |
| `subscription_type` | Enum | Yes | `query_streaming`, `document_upload`, `all` | Event subscriptions |
| `is_active` | Boolean | Yes | Default: true | Whether connection is open |
| `message_count` | Integer | Yes | >= 0 | Total messages sent |
| `reconnection_token` | String | No | UUID-like | Token for reconnecting to session |

### Validation Rules
- **FR-025**: Support WebSocket connections for real-time streaming
- **FR-026**: Stream incremental query results as processing progresses
- **FR-027**: Stream document loading progress with percentage completion
- **FR-028**: Use JSON-based event streaming protocol
- **Idle timeout**: Close connections idle for >5 minutes
- **Heartbeat**: Ping every 30 seconds to detect dead connections

### State Transitions
```
None → Connected (session_id, connected_at, is_active=true)
     ↓
   Active (sending/receiving messages, last_activity_at updated)
     ↓
   Idle (no activity for >2 minutes, send ping)
     ↓
   Disconnected (client closes or timeout, is_active=false)
     ↓
   Reconnected (client reconnects with reconnection_token, new session_id)
```

### In-Memory Storage
- **Not persisted to IRIS**: Sessions stored in FastAPI app state
- **Registry**: `Dict[UUID, WebSocket]` keyed by session_id
- **Cleanup**: Remove disconnected sessions after 5 minutes

### Relationships
- **Many-to-One** with Authentication Token (one key can have multiple sessions)
- **One-to-Many** with WebSocket events (one session receives many events)

### Example JSON (Pydantic Model)
```json
{
  "session_id": "c1d2e3f4-5678-90ab-cdef-123456789abc",
  "api_key_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
  "connected_at": "2025-10-16T12:34:00.000Z",
  "last_activity_at": "2025-10-16T12:34:56.789Z",
  "client_ip": "192.168.1.100",
  "subscription_type": "query_streaming",
  "is_active": true,
  "message_count": 23,
  "reconnection_token": "reconnect_d4e5f6a7-8901-2345-6789-abcdef012345"
}
```

---

## Entity 8: Health Status

### Description
Represents current operational state of system components (pipelines, database, cache) with status level (healthy/degraded/unavailable), last check timestamp, and diagnostic details.

### Fields

| Field | Type | Required | Validation | Description |
|-------|------|----------|------------|-------------|
| `component_name` | String | Yes | 1-100 chars | Component identifier (e.g., "iris_database", "redis_cache", "graphrag_pipeline") |
| `status` | Enum | Yes | `healthy`, `degraded`, `unavailable` | Current health status |
| `last_checked_at` | DateTime | Yes | ISO8601 | Last health check timestamp |
| `response_time_ms` | Integer | No | >= 0 | Health check response time |
| `version` | String | No | Semver format | Component version |
| `dependencies` | List[String] | No | Component names | Components this depends on |
| `error_message` | String | No | Max 500 chars | Error details if unhealthy |
| `metrics` | JSON Object | No | Valid JSON | Component-specific metrics |

### Health Check Logic
```python
async def check_health(component: str) -> HealthStatus:
    """
    Check health of system component
    """
    start_time = time.time()

    try:
        if component == "iris_database":
            # Test database connection
            await db.execute("SELECT 1")
            status = "healthy"
        elif component == "redis_cache":
            # Test Redis connection
            await redis.ping()
            status = "healthy"
        elif component.endswith("_pipeline"):
            # Check pipeline status
            pipeline = get_pipeline(component)
            status = pipeline.status
        else:
            status = "unavailable"

    except Exception as e:
        status = "unavailable"
        error_message = str(e)

    response_time_ms = (time.time() - start_time) * 1000

    return HealthStatus(
        component_name=component,
        status=status,
        last_checked_at=datetime.utcnow(),
        response_time_ms=response_time_ms,
        error_message=error_message if status != "healthy" else None
    )
```

### Validation Rules
- **FR-032**: Health check endpoint reports overall system status
- **FR-033**: Report status of all dependencies (IRIS, Redis, LLM services)
- **FR-034**: Expose metrics (request count, latency, error rates, resource usage)
- **Health check interval**: Run checks every 30 seconds (background task)

### State Transitions
```
None → Healthy (component initialized successfully)
     ↓
   Degraded (partial functionality, some features unavailable)
     ↓
   Unavailable (component completely down)
     ↓
   Recovering (health checks starting to pass)
     ↓
   Healthy (full functionality restored)
```

### In-Memory Storage
- **Not persisted to IRIS**: Health status stored in FastAPI app state
- **Cache**: Last health check results cached for 30 seconds
- **Refresh**: Background task updates health every 30 seconds

### Relationships
- **Aggregated in** GET /health endpoint (returns all component statuses)
- **Monitored by** external health checkers (Kubernetes liveness/readiness probes)

### Example JSON (Pydantic Model)
```json
{
  "component_name": "iris_database",
  "status": "healthy",
  "last_checked_at": "2025-10-16T12:34:56.789Z",
  "response_time_ms": 12,
  "version": "2025.3.0",
  "dependencies": [],
  "metrics": {
    "connection_pool_size": 20,
    "active_connections": 8,
    "query_count": 15234,
    "avg_query_time_ms": 345
  }
}
```

### Aggregated Health Endpoint Response
```json
{
  "status": "healthy",
  "timestamp": "2025-10-16T12:34:56.789Z",
  "components": {
    "iris_database": { "status": "healthy", "response_time_ms": 12 },
    "redis_cache": { "status": "healthy", "response_time_ms": 5 },
    "graphrag_pipeline": { "status": "healthy", "response_time_ms": 8 },
    "basic_pipeline": { "status": "healthy", "response_time_ms": 6 }
  },
  "overall_health": "healthy"
}
```

---

## Entity Relationships Diagram

```
┌─────────────────────┐
│ Authentication Token│
│   (API Key)         │
└──────────┬──────────┘
           │ 1
           │
           │ N
           ├───────────┐
           │           │
           │ N         │ N
  ┌────────▼───────┐  │
  │  API Request   │  │
  │   (Logged)     │  │
  └────────┬───────┘  │
           │ 1        │
           │          │
           │ N        │ N
  ┌────────▼───────┐ ┌▼────────────────────┐
  │ Query Response │ │ Rate Limit Quota    │
  │   (Result)     │ │   (Redis/IRIS)      │
  └────────────────┘ └─────────────────────┘
           │ N
           │
           │ 1
  ┌────────▼───────┐        ┌──────────────────┐
  │ Pipeline       │        │ WebSocket Session│
  │  Instance      │────────│  (Streaming)     │
  │ (In-Memory)    │   N    └──────────────────┘
  └────────┬───────┘
           │ 1
           │
           │ N
  ┌────────▼────────────────┐
  │ Document Upload         │
  │   Operation             │
  └─────────────────────────┘

Health Status (independent monitoring)
```

---

## Summary

### Persistent Entities (IRIS Storage)
1. **API Request** - Logged requests for monitoring/analytics (30-day retention)
2. **Authentication Token** - API keys with permissions and rate limits
3. **Rate Limit Quota** - Quota configuration (Redis for state, IRIS for config)
4. **Document Upload Operation** - Upload task tracking (7-day retention)

### In-Memory Entities (FastAPI State)
5. **Pipeline Instance** - Active RAG pipeline instances
6. **Query Response** - Generated responses (not persisted)
7. **WebSocket Session** - Active WebSocket connections
8. **Health Status** - Component health checks (30s cache)

### Total Fields Across All Entities
- **8 entities** with **81 total fields**
- **4 persistent** entities requiring IRIS tables
- **4 in-memory** entities for runtime state
- **100% Pydantic V2** validation for all entities
- **LangChain & RAGAS compatible** response format
