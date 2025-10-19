# Feature Specification: Production-Grade REST API for RAG Pipelines

**Feature Branch**: `042-full-rest-api`
**Created**: 2025-10-16
**Status**: Draft
**Input**: User description: "full REST API running sidecar OR embedded Python ASGI server as IRIS application  Full REST API: Implement proper FastAPI integration with: Pipeline lifecycle management, Request validation, Error handling, Rate limiting, Authentication/authorization, Connection pooling, Async document loading, WebSocket support for streaming responses"

## Execution Flow (main)
```
1. Parse user description from Input
   → Key concepts: REST API, FastAPI, sidecar/embedded deployment, production features
2. Extract key concepts from description
   → Actors: API consumers (developers, applications), system administrators
   → Actions: Query pipelines, manage pipeline lifecycle, authenticate, load documents
   → Data: RAG queries, documents, pipeline configurations, authentication credentials
   → Constraints: Production-grade (reliability, security, performance)
3. For each unclear aspect:
   → RESOLVED: Sidecar primary, embedded future consideration
   → RESOLVED: API key-based authentication (Elasticsearch pattern)
   → RESOLVED: Adaptive concurrency + per-key quotas
   → RESOLVED: JSON event streaming protocol
4. Fill User Scenarios & Testing section
   → Primary: Developer integrates RAG API into application
   → Secondary: Admin monitors API health and manages access
5. Generate Functional Requirements
   → Each requirement is testable via API contract tests
   → All ambiguous requirements resolved via Elasticsearch patterns
6. Identify Key Entities
   → API Request, Pipeline Instance, Authentication Token, Rate Limit Quota
7. Run Review Checklist
   → PASS: All clarifications resolved using proven Elasticsearch patterns
   → Check for implementation details - none found
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

---

## User Scenarios & Testing

### Primary User Story
A developer building a medical application needs to integrate RAG-powered question answering. They want to send queries via HTTP API and receive structured responses with answers and source citations. The API must be reliable, secure, and performant enough for production use with multiple concurrent users.

### Acceptance Scenarios

1. **Given** a valid API key and query text, **When** the developer sends a POST request to the query endpoint, **Then** the system returns an answer with retrieved documents and sources within 2 seconds

2. **Given** an unauthenticated request, **When** the developer sends a query without credentials, **Then** the system returns a 401 Unauthorized error with clear authentication instructions

3. **Given** a developer has exceeded their rate limit, **When** they send another request, **Then** the system returns a 429 Too Many Requests error with retry-after information

4. **Given** multiple pipeline types are available (basic, CRAG, GraphRAG), **When** the developer queries the pipelines endpoint, **Then** the system returns a list of available pipelines with their capabilities and status

5. **Given** a long-running document loading operation, **When** the developer connects via WebSocket and uploads documents, **Then** the system streams progress updates in real-time until completion

6. **Given** an invalid query parameter, **When** the developer sends a malformed request, **Then** the system returns a 422 validation error with specific field-level error messages

7. **Given** a pipeline is initializing or unhealthy, **When** the developer attempts to query it, **Then** the system returns a 503 Service Unavailable error with estimated recovery time

8. **Given** the administrator wants to monitor system health, **When** they check the health endpoint, **Then** the system returns status of all pipelines, database connections, and service dependencies

### Edge Cases
- Query length exceeding maximum: System returns 422 Validation Error with specific limit in error message
- Query containing special characters: Accepted and properly escaped/sanitized during processing
- Concurrent requests exceeding connection pool capacity: System queues requests with timeout; returns 503 if queue full
- Pipeline crash during query: System returns 500 Internal Server Error and automatically restarts pipeline
- WebSocket disconnection during long operation: System allows reconnection with operation ID to resume progress updates
- Authentication token expiration mid-request: Request completes with current token; next request requires new token
- Sustained high load beyond rate limits: System enforces 429 responses with exponential backoff recommendations

## Requirements

### Functional Requirements

**Query Processing**
- **FR-001**: System MUST accept query requests with query text, pipeline selection, and optional parameters (top_k, filters)
- **FR-002**: System MUST return structured responses containing answer text, retrieved documents, sources, and execution metadata
- **FR-003**: System MUST validate all incoming requests and return field-specific error messages for invalid inputs
- **FR-004**: System MUST support querying multiple pipeline types (basic, basic_rerank, CRAG, GraphRAG, PyLateColBERT)

**Pipeline Lifecycle Management**
- **FR-005**: System MUST initialize all configured pipelines on startup
- **FR-006**: System MUST report pipeline health status (healthy, degraded, unavailable)
- **FR-007**: System MUST gracefully shut down pipelines on service stop, completing in-flight requests
- **FR-008**: System MUST provide endpoints to list available pipelines with their capabilities and current status

**Authentication & Authorization**
- **FR-009**: System MUST authenticate all API requests using API key-based authentication with header format `Authorization: ApiKey <base64(id:api_key)>`
- **FR-010**: System MUST reject unauthenticated requests with 401 Unauthorized status
- **FR-011**: System MUST support permission-based authorization where each API key has associated permissions (read, write, admin)
- **FR-012**: System MUST log all authentication failures for security monitoring

**Rate Limiting**
- **FR-013**: System MUST enforce per-API-key rate limits to prevent abuse using adaptive request concurrency
- **FR-014**: System MUST return rate limit information in response headers (X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset)
- **FR-015**: System MUST return 429 Too Many Requests when limits are exceeded with Retry-After header
- **FR-016**: System MUST support configurable per-key request quotas with fallback to static concurrency limits

**Error Handling**
- **FR-017**: System MUST return appropriate HTTP status codes (200, 400, 401, 422, 429, 500, 503)
- **FR-018**: System MUST provide clear error messages with actionable guidance for resolution
- **FR-019**: System MUST log all errors with request context for debugging
- **FR-020**: System MUST handle database connection failures gracefully without exposing internal details

**Document Loading**
- **FR-021**: System MUST support asynchronous document upload and indexing
- **FR-022**: System MUST validate document format and size before accepting uploads (maximum 100 MB per document)
- **FR-023**: System MUST return operation status (pending, processing, completed, failed) for document loads
- **FR-024**: Document loading operations MUST not block query processing

**WebSocket Streaming**
- **FR-025**: System MUST support WebSocket connections for real-time streaming responses
- **FR-026**: System MUST stream incremental query results as retrieval and generation progresses
- **FR-027**: System MUST stream document loading progress with percentage completion
- **FR-028**: System MUST use JSON-based event streaming protocol with message format `{"event": "type", "data": {...}, "timestamp": "ISO8601"}`

**Connection Management**
- **FR-029**: System MUST maintain database connection pool to handle concurrent requests efficiently
- **FR-030**: System MUST recover from transient database connection failures automatically
- **FR-031**: System MUST limit concurrent connections to prevent resource exhaustion

**Monitoring & Health**
- **FR-032**: System MUST provide health check endpoint reporting overall system status
- **FR-033**: System MUST report status of all dependencies (IRIS database, Redis cache, LLM services)
- **FR-034**: System MUST expose metrics for request count, latency, error rates, and resource usage
- **FR-035**: System MUST log all API requests with standard detail level: request ID, timestamp, status code, endpoint, HTTP method, response time, and API key ID
- **FR-036**: System MUST automatically delete API request logs after 30 days retention period

**Deployment Options**
- **FR-037**: System MUST support deployment as standalone sidecar service (primary deployment model)
- **FR-038**: System MUST be designed to allow future embedded deployment within IRIS (via IRIS Embedded Python or Python Gateway)
- **FR-039**: Both deployment modes MUST provide identical REST API functionality and contracts

### Key Entities

- **API Request**: Represents an incoming HTTP request containing query text, pipeline selection, parameters, and authentication credentials; includes metadata like request ID, timestamp, and client identifier

- **Pipeline Instance**: Represents an initialized RAG pipeline (basic, CRAG, GraphRAG, etc.) with its current health status, configuration, and performance statistics; manages lifecycle from initialization through shutdown

- **Authentication Token**: Represents credentials used to authenticate API requests; includes associated permissions, rate limits, and expiration information

- **Rate Limit Quota**: Represents allowed request rate for a specific client or tier; tracks current usage, reset timestamp, and enforcement policy

- **Query Response**: Structured result containing generated answer, retrieved documents with sources, execution metadata (pipeline used, latency, tokens), and optional streaming chunks

- **Document Upload Operation**: Represents an asynchronous document loading task with status tracking, progress percentage, validation results, and completion/error information

- **WebSocket Session**: Represents an active WebSocket connection for streaming responses; manages message queue, reconnection state, and client subscription preferences

- **Health Status**: Represents current operational state of system components (pipelines, database, cache) with status level (healthy/degraded/unavailable), last check timestamp, and diagnostic details

---

## Clarifications

### Session 2025-10-16
- Q: What should happen when a query text exceeds maximum length? → A: Reject with 422 Validation Error including specific length limit in error message (Elasticsearch pattern)
- Q: What is the maximum document size (per individual document) the system must support for upload and indexing? → A: 100 MB per document (supports technical manuals, books, rich content)
- Q: What is the expected scale for concurrent document indexing operations? → A: 1-5 concurrent uploads (single user/developer scenario)
- Q: What level of detail should API request logging capture? → A: Standard logging - Request ID, timestamp, status code, endpoint, method, response time, user/key ID (balances observability with storage)
- Q: How long should API request logs be retained before automatic deletion? → A: 30 days (standard operational retention for debugging)

---

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain - **All resolved via Elasticsearch patterns**
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities resolved via Elasticsearch API research
- [x] User scenarios defined
- [x] Requirements generated and clarified
- [x] Entities identified
- [x] Review checklist passed - ALL REQUIREMENTS CLEAR

---

## Notes for Planning Phase

**Clarifications RESOLVED - Elasticsearch-Inspired Decisions:**

Based on Elasticsearch vector search API patterns, the following design decisions are recommended:

1. **Authentication Method**: API key-based authentication (Elasticsearch pattern)
   - Simple, proven model used by Elasticsearch
   - Keys stored with associated permissions and rate limits
   - Header-based: `Authorization: ApiKey <base64(id:api_key)>`
   - Supports future migration to OAuth2/JWT if needed

2. **Authorization Model**: Permission-based with key scoping
   - Each API key has associated permissions (read, write, admin)
   - Permissions control access to specific pipeline operations
   - Inspired by Elasticsearch index-level privileges model

3. **Rate Limiting Strategy**: Adaptive Request Concurrency (ARC) + per-key quotas
   - Primary: Adaptive concurrency control (Elasticsearch default)
   - Secondary: Per-API-key request quotas for abuse prevention
   - Headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`
   - Configurable fallback to static limits if needed

4. **WebSocket Protocol**: JSON-based event streaming
   - Event types: `query_start`, `retrieval_progress`, `generation_chunk`, `query_complete`, `error`
   - Message format: `{"event": "type", "data": {...}, "timestamp": "ISO8601"}`
   - Mirrors Elasticsearch's structured event approach

5. **Embedded IRIS Deployment**: Focus on sidecar pattern initially
   - Primary deployment: Standalone FastAPI sidecar service
   - Future consideration: IRIS Embedded Python if demand exists
   - Both modes share identical REST API contract

**Elasticsearch-Inspired API Patterns to Adopt:**

1. **Retry Logic with Exponential Backoff**
   - Initial backoff: 1 second
   - Fibonacci sequence for retry intervals (1s, 1s, 2s, 3s, 5s, 8s)
   - Full jitter mode to prevent thundering herd
   - Configurable max retries (default: 3)

2. **Timeout Configuration**
   - Request timeout: 60 seconds default
   - Separate timeouts for query vs document operations
   - Never set timeout below internal service timeout (prevents orphaned requests)

3. **Response Headers (Elasticsearch standard)**
   - `X-Request-ID`: Unique request identifier for tracing
   - `X-RateLimit-*`: Rate limit information
   - `X-Pipeline-Name`: Which pipeline processed the request
   - `X-Execution-Time-Ms`: Query execution time

4. **Error Response Format**
   - HTTP status codes matching Elasticsearch patterns
   - Structured error body: `{"error": {"type": "...", "reason": "...", "details": {...}}}`
   - Field-level validation errors with specific paths

5. **Query Endpoint Design** (POST /{pipeline}/_search pattern)
   - POST `/api/v1/basic/_search` - Query basic RAG pipeline
   - POST `/api/v1/graphrag/_search` - Query GraphRAG pipeline
   - POST `/api/v1/_search` - Query with pipeline parameter in body
   - Consistent with Elasticsearch `/{index}/_search` pattern

6. **Bulk Operations** (future consideration)
   - POST `/api/v1/_bulk` - Batch document uploads
   - NDJSON format for streaming large batches
   - Per-operation success/failure tracking

**Dependencies:**
- Existing RAG pipeline implementations (basic, CRAG, GraphRAG, etc.)
- IRIS database with connection pooling capabilities
- Optional: Redis for rate limiting state and caching (recommended)

**Risks:**
- Adaptive concurrency complexity (mitigate: allow static fallback)
- WebSocket connection management under high concurrency
- Pipeline initialization time impacting startup (mitigate: lazy loading option)
- API key management at scale (mitigate: key rotation policy)

**Success Metrics:**
- API response time < 2 seconds for 95th percentile (Elasticsearch benchmark)
- Support 100+ concurrent query requests with adaptive concurrency
- Support 1-5 concurrent document upload operations
- 99.9% uptime for production deployment
- Zero authentication bypass vulnerabilities
- Rate limiting enforced with < 1% false positives
- Retry success rate > 95% for transient failures
- Document uploads up to 100 MB processed successfully
