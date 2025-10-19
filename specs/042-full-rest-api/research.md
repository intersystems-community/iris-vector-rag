# Research: Production-Grade REST API for RAG Pipelines

**Date**: 2025-10-16
**Feature**: 042-full-rest-api
**Purpose**: Document technical research and design decisions for REST API implementation

---

## Research Task 1: FastAPI Production Patterns

### Decision
Use FastAPI with the following production patterns:
- **Application Factory**: Create FastAPI app instance via factory function for testability
- **Dependency Injection**: Use FastAPI's dependency injection for auth, DB connections, pipeline instances
- **Lifespan Events**: Use `@asynccontextmanager` lifespan pattern for startup/shutdown (replaces deprecated on_event)
- **Middleware Stack**: Custom middleware for auth, rate limiting, request logging, error handling
- **Async Handlers**: All route handlers use `async def` for non-blocking I/O with IRIS database
- **Pydantic V2**: Use Pydantic V2 models for request/response validation with strict mode

### Rationale
- **Factory pattern** enables multiple app instances for testing (unit tests vs integration tests)
- **Dependency injection** follows SOLID principles and simplifies testing with mocks
- **Lifespan events** are the modern FastAPI pattern (on_event deprecated in 0.109.0+)
- **Async handlers** prevent thread blocking during IRIS queries, improving concurrency
- **Pydantic V2** provides 5-10x faster validation and better error messages than V1

### Alternatives Considered
- **Flask + async extensions**: Rejected - FastAPI native async support is superior
- **Django REST Framework**: Rejected - too heavyweight for API-only service
- **Starlette directly**: Rejected - FastAPI provides better developer experience with OpenAPI auto-generation

### Implementation Notes
```python
# Application factory pattern
from contextlib import asynccontextmanager
from fastapi import FastAPI

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize connection pool, load pipelines
    app.state.pipeline_manager = PipelineManager()
    await app.state.pipeline_manager.initialize()
    yield
    # Shutdown: Close connections, cleanup
    await app.state.pipeline_manager.shutdown()

def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    # Register routes, middleware
    return app
```

### References
- FastAPI Async Best Practices: https://fastapi.tiangolo.com/async/
- Lifespan Events: https://fastapi.tiangolo.com/advanced/events/
- Pydantic V2 Migration: https://docs.pydantic.dev/latest/migration/

---

## Research Task 2: Elasticsearch API Design Patterns

### Decision
Adopt Elasticsearch-inspired patterns for the REST API:

**Authentication**:
- Header format: `Authorization: ApiKey <base64(id:api_key)>`
- Base64 encoding of `{id}:{api_key}` tuple for easy parsing
- API keys stored in IRIS with hashed secrets (bcrypt)
- Permissions model: read, write, admin scopes per key

**Rate Limiting**:
- Primary: Adaptive Request Concurrency (ARC) - dynamically adjust based on latency
- Secondary: Per-key quotas (requests per minute/hour) for abuse prevention
- Response headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`
- 429 status with `Retry-After` header when limits exceeded

**Error Format**:
```json
{
  "error": {
    "type": "validation_exception",
    "reason": "Invalid query parameter",
    "details": {
      "field": "top_k",
      "rejected_value": -5,
      "message": "Must be positive integer"
    }
  }
}
```

**Endpoint Design**:
- Pattern: `POST /api/v1/{pipeline}/_search` (Elasticsearch `/{index}/_search` inspiration)
- Examples:
  - `POST /api/v1/basic/_search` - Query basic RAG pipeline
  - `POST /api/v1/graphrag/_search` - Query GraphRAG pipeline
  - `POST /api/v1/_search` - Query with pipeline in request body

**Retry Logic**:
- Fibonacci backoff sequence: 1s, 1s, 2s, 3s, 5s, 8s
- Full jitter mode: `sleep = random(0, backoff)`
- Max retries: 3 (configurable)
- Retryable errors: 429 (rate limit), 503 (service unavailable), connection timeouts

**Response Headers**:
- `X-Request-ID`: UUID for request tracing
- `X-Pipeline-Name`: Which pipeline processed the request
- `X-Execution-Time-Ms`: Query execution time in milliseconds
- `X-RateLimit-*`: Rate limiting information

### Rationale
- **Proven at scale**: Elasticsearch handles billions of requests with these patterns
- **Developer familiarity**: Many developers already know Elasticsearch API conventions
- **Base64 API key encoding**: Simplifies parsing while remaining secure over HTTPS
- **Adaptive concurrency**: More intelligent than static rate limits; adjusts to system load
- **Structured errors**: Field-level validation errors help developers debug quickly
- **Retry with jitter**: Prevents thundering herd when services recover from outages

### Alternatives Considered
- **OAuth2/JWT**: Rejected for v1 - adds complexity; API keys sufficient for initial use case
- **Token bucket rate limiting**: Rejected - less adaptive to system load than ARC
- **Generic error format**: Rejected - Elasticsearch format provides better developer experience
- **Random endpoint paths**: Rejected - `/{pipeline}/_search` pattern is intuitive and consistent

### Implementation Notes
```python
# Adaptive Request Concurrency (simplified)
class AdaptiveRateLimiter:
    def __init__(self, min_concurrency=10, max_concurrency=100):
        self.current_concurrency = min_concurrency
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.latency_p95 = deque(maxlen=100)

    def adjust(self, latency_ms):
        self.latency_p95.append(latency_ms)
        p95 = np.percentile(self.latency_p95, 95)

        # Increase concurrency if latency is low
        if p95 < 1000 and self.current_concurrency < self.max_concurrency:
            self.current_concurrency = min(
                self.current_concurrency * 1.1,
                self.max_concurrency
            )
        # Decrease concurrency if latency is high
        elif p95 > 2000 and self.current_concurrency > self.min_concurrency:
            self.current_concurrency = max(
                self.current_concurrency * 0.9,
                self.min_concurrency
            )
```

### References
- Elasticsearch Security: https://www.elastic.co/guide/en/elasticsearch/reference/current/security-api.html
- Elasticsearch API Conventions: https://www.elastic.co/guide/en/elasticsearch/reference/current/api-conventions.html
- Adaptive Concurrency: https://aws.amazon.com/builders-library/timeouts-retries-and-backoff-with-jitter/

---

## Research Task 3: IRIS Database Connection Pooling

### Decision
Implement IRIS connection pooling with the following strategy:

**Connection Pool Configuration**:
- Library: `intersystems-irispython` native connection pooling
- Pool size: 20 connections (default), configurable via environment
- Max overflow: 10 additional connections during peak load
- Pool timeout: 30 seconds (wait for available connection)
- Pool recycle: 3600 seconds (1 hour) - refresh connections periodically
- Pre-ping: True - validate connection health before use

**Pool Management**:
- **Startup**: Initialize pool in FastAPI lifespan startup
- **Request handling**: Acquire connection via dependency injection
- **Error recovery**: Auto-retry on connection failures (max 3 attempts)
- **Shutdown**: Graceful pool disposal in lifespan shutdown

**Concurrency Model**:
- FastAPI async handlers + thread pool executor for IRIS queries
- IRIS Python connector is synchronous - wrap in `asyncio.to_thread()`
- Prevents blocking event loop during database operations

### Rationale
- **Native pooling**: intersystems-irispython has built-in pooling; no need for external libraries
- **20 base connections**: Supports 100+ concurrent requests (each request holds connection briefly)
- **Pre-ping validation**: Prevents failures from stale connections (network timeouts, IRIS restarts)
- **1 hour recycle**: Balance between connection freshness and overhead of re-establishing
- **Thread pool approach**: Cleanly integrates synchronous IRIS driver with async FastAPI

### Alternatives Considered
- **SQLAlchemy pooling**: Rejected - adds unnecessary ORM layer; native IRIS pooling sufficient
- **Connection per request**: Rejected - high overhead; connection establishment takes 50-100ms
- **Global connection object**: Rejected - not thread-safe; causes race conditions under load
- **Run IRIS queries in asyncio**: Rejected - IRIS Python driver is synchronous; would require rewrite

### Implementation Notes
```python
from intersystems_irispython import iris
import asyncio
from functools import lru_cache

class IRISConnectionPool:
    def __init__(self, host, port, namespace, username, password, pool_size=20):
        self.config = {
            "hostname": host,
            "port": port,
            "namespace": namespace,
            "username": username,
            "password": password
        }
        self.pool_size = pool_size
        self.pool = []
        self._initialize_pool()

    def _initialize_pool(self):
        for _ in range(self.pool_size):
            conn = iris.connect(**self.config)
            self.pool.append(conn)

    @contextmanager
    def acquire(self):
        """Acquire connection from pool with timeout"""
        conn = self.pool.pop(0) if self.pool else self._create_connection()
        try:
            # Pre-ping to validate health
            conn.ping()
            yield conn
        finally:
            self.pool.append(conn)

    async def execute_async(self, query, params=None):
        """Execute IRIS query in thread pool to avoid blocking"""
        def _execute():
            with self.acquire() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params or [])
                return cursor.fetchall()

        return await asyncio.to_thread(_execute)

# FastAPI dependency injection
async def get_iris_connection():
    pool = request.app.state.iris_pool
    return pool
```

### Performance Characteristics
- **Connection acquisition**: <5ms (from pool)
- **New connection overhead**: 50-100ms (only during overflow)
- **Query execution**: 10-500ms (depends on query complexity)
- **Total request latency**: 100-2000ms (within p95 target of <2s)

### References
- IRIS Python API: https://docs.intersystems.com/irislatest/csp/docbook/DocBook.UI.Page.cls
- Python Connection Pooling Patterns: https://www.compose.com/articles/connection-pooling-with-python/
- FastAPI Background Tasks: https://fastapi.tiangolo.com/tutorial/background-tasks/

---

## Research Task 4: WebSocket Streaming Protocols

### Decision
Implement JSON-based event streaming protocol for WebSocket connections:

**Event Types**:
- `query_start`: Query processing initiated
- `retrieval_progress`: Document retrieval in progress (n/total)
- `generation_chunk`: LLM generating response (streaming tokens)
- `query_complete`: Query finished with final result
- `error`: Error occurred during processing
- `document_upload_progress`: Document indexing progress (percentage)

**Message Format**:
```json
{
  "event": "retrieval_progress",
  "data": {
    "documents_retrieved": 5,
    "total_documents": 10,
    "current_pipeline": "graphrag"
  },
  "timestamp": "2025-10-16T12:34:56.789Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Connection Management**:
- **Authentication**: API key validation on WebSocket upgrade request
- **Heartbeat**: Ping/pong every 30 seconds to detect dead connections
- **Reconnection**: Support reconnection with `request_id` to resume progress
- **Timeout**: Close idle connections after 5 minutes
- **Concurrency**: Limit to 100 concurrent WebSocket connections per server

**Protocol Flow**:
1. Client connects: `ws://api.example.com/api/v1/ws/query`
2. Client sends auth + query: `{"api_key": "...", "query": "...", "pipeline": "graphrag"}`
3. Server validates auth, initiates query, sends `query_start` event
4. Server streams progress events as query executes
5. Server sends `query_complete` with final result
6. Client can disconnect or send another query on same connection

### Rationale
- **JSON format**: Human-readable, easy to debug, widely supported by clients
- **Event-based**: Allows clients to handle different event types (progress vs final result)
- **Request ID tracking**: Enables reconnection and distributed tracing
- **Heartbeat protocol**: Detects network failures quickly; prevents resource leaks
- **Stateless events**: Each event is self-contained; client can process out-of-order

### Alternatives Considered
- **Server-Sent Events (SSE)**: Rejected - unidirectional; doesn't support client→server messages
- **gRPC streaming**: Rejected - requires protobuf; more complex for web clients
- **Raw binary protocol**: Rejected - harder to debug; JSON overhead is negligible
- **Polling**: Rejected - inefficient for long-running operations; high latency

### Implementation Notes
```python
from fastapi import WebSocket, WebSocketDisconnect
import json
from datetime import datetime
from uuid import uuid4

class WebSocketManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, request_id: str):
        await websocket.accept()
        self.active_connections[request_id] = websocket

    async def send_event(self, request_id: str, event: str, data: dict):
        websocket = self.active_connections.get(request_id)
        if websocket:
            message = {
                "event": event,
                "data": data,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "request_id": request_id
            }
            await websocket.send_json(message)

    async def disconnect(self, request_id: str):
        if request_id in self.active_connections:
            del self.active_connections[request_id]

# FastAPI WebSocket route
@app.websocket("/api/v1/ws/query")
async def websocket_query(websocket: WebSocket):
    request_id = str(uuid4())
    await ws_manager.connect(websocket, request_id)

    try:
        # Receive initial query message
        message = await websocket.receive_json()
        api_key = message.get("api_key")

        # Validate authentication
        if not validate_api_key(api_key):
            await ws_manager.send_event(request_id, "error", {
                "type": "authentication_error",
                "message": "Invalid API key"
            })
            await websocket.close()
            return

        # Send query_start event
        await ws_manager.send_event(request_id, "query_start", {
            "query": message["query"],
            "pipeline": message["pipeline"]
        })

        # Stream query execution (simplified)
        async for progress in execute_query_streaming(message):
            await ws_manager.send_event(request_id, progress["event"], progress["data"])

    except WebSocketDisconnect:
        await ws_manager.disconnect(request_id)
```

### Error Handling
- **Connection failures**: Client receives `error` event with retry guidance
- **Authentication failures**: WebSocket closed with 403 status code
- **Timeout**: Server sends `error` event, then closes connection gracefully
- **Rate limiting**: Reject new connections with 429 status when limit reached

### Performance Considerations
- **Memory per connection**: ~50KB (buffers + state)
- **Max connections**: 100 concurrent (configurable) = ~5MB overhead
- **Message throughput**: 1000+ messages/second per connection
- **Latency**: <10ms for event delivery (local network)

### References
- FastAPI WebSockets: https://fastapi.tiangolo.com/advanced/websockets/
- WebSocket Protocol RFC: https://datatracker.ietf.org/doc/html/rfc6455
- JSON Event Stream Format: https://www.w3.org/TR/eventsource/

---

## Summary of Design Decisions

### Technology Stack
- **Framework**: FastAPI 0.104+ with async/await
- **ASGI Server**: Uvicorn with uvloop for production
- **Validation**: Pydantic V2 for request/response schemas
- **Database**: IRIS with native Python connection pooling
- **Rate Limiting**: Redis (optional) or in-memory adaptive concurrency
- **WebSockets**: FastAPI native WebSocket support
- **Authentication**: API key with bcrypt hashing

### Architecture Patterns
- **Application Factory**: Testable app creation with lifespan management
- **Dependency Injection**: FastAPI DI for auth, connections, pipelines
- **Middleware Stack**: Auth → Rate Limit → Logging → Error Handling
- **Async I/O**: All handlers async; IRIS queries wrapped in thread pool
- **Event Streaming**: JSON-based WebSocket protocol for progress updates

### API Design
- **Endpoint Pattern**: `POST /api/v1/{pipeline}/_search` (Elasticsearch-inspired)
- **Authentication**: `Authorization: ApiKey <base64(id:api_key)>`
- **Error Format**: Structured JSON with type, reason, details
- **Rate Limiting**: Adaptive concurrency + per-key quotas
- **Retry Logic**: Fibonacci backoff with full jitter

### Performance Targets
- **Query Latency**: <2s p95 for basic queries
- **Concurrent Queries**: 100+ simultaneous requests
- **Document Uploads**: 1-5 concurrent (100MB max per document)
- **WebSocket Connections**: 100 concurrent per server instance
- **Connection Pool**: 20 base connections + 10 overflow

### Production Readiness
- **Logging**: Structured JSON logs with request tracing
- **Health Checks**: `/health` endpoint with dependency status
- **Monitoring**: Prometheus metrics for latency, error rates, pool usage
- **Docker**: Multi-stage build with non-root user
- **Configuration**: Environment variables + YAML config files

---

## Next Steps (Phase 1)

1. **Create data-model.md**: Document 8 entities with field definitions and validation rules
2. **Create OpenAPI contracts**: 6 YAML files covering all 39 functional requirements
3. **Generate contract tests**: Failing tests for each endpoint and requirement
4. **Create quickstart.md**: Step-by-step guide based on acceptance scenarios
5. **Update CLAUDE.md**: Add REST API context and deployment commands

All design decisions are now documented and ready for Phase 1 implementation planning.
