# RAG API - Production-Grade REST API

Production-ready REST API for Retrieval-Augmented Generation (RAG) pipelines built with FastAPI and InterSystems IRIS.

## Features

- **Multiple RAG Pipelines**: BasicRAG, CRAG, GraphRAG, HybridGraphRAG, PyLate ColBERT
- **API Key Authentication**: Secure bcrypt-hashed credentials with permission-based access
- **Three-Tier Rate Limiting**: 60/100/1000 requests per minute (Basic/Premium/Enterprise)
- **Request/Response Logging**: Complete audit trail with execution metrics
- **WebSocket Streaming**: Real-time query execution and document upload progress
- **Async Document Upload**: Background processing with validation and progress tracking
- **Health Monitoring**: Comprehensive health checks for all system components
- **Elasticsearch-Inspired Design**: Structured error responses with actionable guidance
- **100% RAGAS Compatible**: Standardized response format for evaluation frameworks

## Quick Start

### 1. Setup Database

```bash
make api-setup-db
```

This creates all required tables in your IRIS database.

### 2. Create API Key

```bash
make api-create-key NAME="My First Key" EMAIL=user@example.com
```

**Save the key secret** - it won't be shown again!

Output:
```
================================================================================
API Key Created Successfully!
================================================================================
Key ID:      7c9e6679-7425-40de-944b-e07fc1f90ae7
Key Secret:  a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6
Name:        My First Key
Permissions: read
Tier:        basic
Created:     2025-01-16 12:34:56
================================================================================
IMPORTANT: Save the Key Secret - it will not be shown again!
================================================================================

Base64-encoded credentials for Authorization header:
Authorization: ApiKey N2M5ZTY2NzktNzQyNS00MGRlLTk0NGItZTA3ZmMxZjkwYWU3OmExYjJjM2Q0ZTVmNmc3aDhpOWowazFsMm0zbjRvNXA2
================================================================================
```

### 3. Start API Server

Development mode (auto-reload):
```bash
make api-run
```

Production mode (4 workers):
```bash
make api-run-prod
```

### 4. Test the API

Check health:
```bash
make api-health
```

Open documentation:
```bash
make api-docs
# Opens http://localhost:8000/docs in browser
```

Query a pipeline:
```bash
curl -X POST http://localhost:8000/api/v1/basic/_search \
  -H "Authorization: ApiKey <your-base64-encoded-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the symptoms of diabetes?",
    "top_k": 5
  }'
```

## API Endpoints

### Authentication

All endpoints (except `/health` and `/pipelines`) require API key authentication:

```
Authorization: ApiKey <base64(key_id:key_secret)>
```

### Query Endpoints

**POST /{pipeline}/_search** - Execute semantic search query

Supported pipelines:
- `basic` - Standard vector similarity search
- `basic_rerank` - Vector search with cross-encoder reranking
- `crag` - Corrective RAG with self-evaluation
- `graphrag` - Hybrid search (vector + text + graph + RRF)
- `pylate_colbert` - Late interaction retrieval

Request:
```json
{
  "query": "What is diabetes?",
  "top_k": 5,
  "filters": {},
  "include_sources": true,
  "include_metadata": true
}
```

Response:
```json
{
  "response_id": "9a8b7c6d-5e4f-3210-9876-543210fedcba",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "answer": "Diabetes is a chronic metabolic disorder...",
  "retrieved_documents": [
    {
      "doc_id": "1a2b3c4d-5678-90ab-cdef-1234567890ab",
      "content": "Diabetes mellitus is...",
      "score": 0.95,
      "metadata": {
        "source": "medical_textbook_ch5.pdf",
        "page_number": 127
      }
    }
  ],
  "sources": ["medical_textbook_ch5.pdf"],
  "contexts": ["Diabetes mellitus is..."],
  "pipeline_name": "basic",
  "execution_time_ms": 1456,
  "retrieval_time_ms": 345,
  "generation_time_ms": 1089,
  "tokens_used": 2345
}
```

### Pipeline Discovery

**GET /api/v1/pipelines** - List all available pipelines

**GET /api/v1/pipelines/{name}** - Get pipeline details

### Document Upload

**POST /api/v1/documents/upload** - Upload documents for indexing

Requires `write` permission.

Request (multipart/form-data):
```
file: <document-file>
pipeline_type: graphrag
chunk_size: 1000
chunk_overlap: 200
```

Response:
```json
{
  "operation_id": "b1c2d3e4-5678-90ab-cdef-fedcba987654",
  "status": "pending",
  "message": "Document upload initiated. Use operation_id to track progress."
}
```

**GET /api/v1/documents/operations/{operation_id}** - Track upload progress

Response:
```json
{
  "operation_id": "b1c2d3e4-5678-90ab-cdef-fedcba987654",
  "status": "processing",
  "total_documents": 100,
  "processed_documents": 47,
  "progress_percentage": 47.0,
  "pipeline_type": "graphrag"
}
```

### Health Monitoring

**GET /api/v1/health** - System health check

No authentication required.

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-01-16T12:34:56.789Z",
  "components": {
    "iris_database": {
      "status": "healthy",
      "response_time_ms": 12
    },
    "redis_cache": {
      "status": "healthy",
      "response_time_ms": 5
    },
    "graphrag_pipeline": {
      "status": "healthy",
      "response_time_ms": 8
    }
  },
  "overall_health": "healthy"
}
```

### WebSocket Streaming

**WS /ws** - Real-time event streaming

Connect and authenticate:
```python
import websockets
import json
import base64

async with websockets.connect("ws://localhost:8000/ws") as ws:
    # Authenticate
    auth = {
        "api_key": base64.b64encode(b"key-id:secret").decode(),
        "subscription_type": "all"
    }
    await ws.send(json.dumps(auth))

    # Request query streaming
    request = {
        "type": "query",
        "query": "What is diabetes?",
        "pipeline": "basic",
        "request_id": "550e8400-e29b-41d4-a716-446655440000"
    }
    await ws.send(json.dumps(request))

    # Receive events
    async for message in ws:
        event = json.loads(message)
        print(f"Event: {event['event']}, Data: {event['data']}")
```

Event types:
- `query_start` - Query execution started
- `retrieval_progress` - Documents being retrieved
- `generation_chunk` - Partial answer text
- `query_complete` - Query finished
- `document_upload_progress` - Upload progress
- `ping/pong` - Heartbeat

## API Key Management

### Create API Key

```bash
# Basic key (60 requests/min)
make api-create-key NAME="My Key" EMAIL=user@example.com

# Premium key (100 requests/min)
make api-create-key NAME="Premium Key" EMAIL=user@example.com TIER=premium

# Enterprise key with all permissions (1000 requests/min)
make api-create-key \
  NAME="Enterprise Key" \
  EMAIL=admin@example.com \
  TIER=enterprise \
  PERMISSIONS="read write admin"
```

### List API Keys

```bash
# List all keys
make api-list-keys

# List keys for specific email
make api-list-keys EMAIL=user@example.com
```

### Revoke API Key

```bash
make api-revoke-key KEY_ID=7c9e6679-7425-40de-944b-e07fc1f90ae7
```

## Rate Limiting

Three tiers with different limits:

| Tier | Requests/Minute | Requests/Hour | Max Concurrent |
|------|----------------|---------------|----------------|
| Basic | 60 | 1,000 | 5 |
| Premium | 100 | 5,000 | 10 |
| Enterprise | 1,000 | 50,000 | 20 |

Rate limit headers in every response:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1705411200
```

When exceeded (HTTP 429):
```json
{
  "error": {
    "type": "rate_limit_exceeded",
    "reason": "Too many requests",
    "details": {
      "limit": 60,
      "window": "requests per minute",
      "retry_after_seconds": 60
    }
  }
}
```

## Error Handling

All errors follow Elasticsearch-inspired structure:

```json
{
  "error": {
    "type": "validation_exception",
    "reason": "Invalid parameter value",
    "details": {
      "field": "top_k",
      "rejected_value": -5,
      "message": "Must be positive integer between 1 and 100",
      "min_value": 1,
      "max_value": 100
    }
  }
}
```

Error types:
- `authentication_error` - Missing or invalid credentials (401)
- `authorization_error` - Insufficient permissions (403)
- `validation_exception` - Invalid request parameters (422)
- `rate_limit_exceeded` - Too many requests (429)
- `service_unavailable` - Pipeline unavailable (503)
- `internal_server_error` - Unexpected error (500)

## Configuration

Configuration file: `config/api_config.yaml`

```yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 4

database:
  host: localhost
  port: 1972
  namespace: USER
  username: demo
  password: demo
  pool_size: 20
  max_overflow: 10

redis:
  enabled: true
  host: localhost
  port: 6379

pipelines:
  enabled:
    - basic
    - basic_rerank
    - crag
    - graphrag
    - pylate_colbert

auth:
  bcrypt_rounds: 12

rate_limiting:
  max_concurrent_per_key: 10

websocket:
  max_connections_per_key: 10
  heartbeat_interval: 30
  idle_timeout: 300

logging:
  level: INFO
  retention_days: 30
```

## Database Maintenance

### Run Cleanup Job

Automatically clean up old logs and expired data:

```bash
python -m iris_rag.api.cleanup_job
```

Schedule with cron (runs daily at 2 AM):
```cron
0 2 * * * cd /path/to/rag-templates && .venv/bin/python -m iris_rag.api.cleanup_job >> logs/cleanup.log 2>&1
```

Cleanup policies:
- Request logs: 30 days retention
- Rate limit history: 7 days retention
- WebSocket sessions: 24 hours inactive
- Upload operations: 30 days for completed/failed
- Expired API keys: Auto-deactivated

## Testing

### Run All Tests

```bash
make api-test
```

### Contract Tests

```bash
make api-test-contracts
```

### Integration Tests

```bash
make api-test-integration
```

## Production Deployment

### Using Docker

See `docker-compose.api.yml` for complete configuration.

### Using Kubernetes

Health check endpoints for probes:

```yaml
livenessProbe:
  httpGet:
    path: /api/v1/health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /api/v1/health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
```

### Environment Variables

```bash
# Database
IRIS_HOST=localhost
IRIS_PORT=1972
IRIS_NAMESPACE=USER
IRIS_USERNAME=demo
IRIS_PASSWORD=demo

# Redis (optional)
REDIS_HOST=localhost
REDIS_PORT=6379

# API Keys
API_KEY_SECRET=your-secret-key-for-signing

# Logging
LOG_LEVEL=INFO
LOG_RETENTION_DAYS=30
```

## Monitoring

### Metrics Export

Access metrics at `/api/v1/health` for Prometheus/Grafana integration.

Key metrics:
- Request rate (requests/second)
- Response time (p50, p95, p99)
- Error rate (% of failed requests)
- Pipeline latency (retrieval + generation)
- Token usage (LLM consumption)

### Logging

Structured JSON logs for easy parsing:

```json
{
  "timestamp": "2025-01-16T12:34:56.789Z",
  "level": "INFO",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "POST",
  "endpoint": "/api/v1/basic/_search",
  "status_code": 200,
  "execution_time_ms": 1456,
  "api_key_id": "7c9e6679-7425-40de-944b-e07fc1f90ae7"
}
```

## Security

- **API Key Storage**: bcrypt hashed with cost factor 12
- **Permission-Based Access**: Read, Write, Admin roles
- **Rate Limiting**: Prevents abuse and DoS attacks
- **Request Logging**: Complete audit trail
- **CORS**: Configurable allowed origins
- **Input Validation**: Pydantic models with strict validation

## Performance

Target latencies:
- Query execution: <2s (p95)
- Health check: <100ms
- API key validation: <50ms
- Document upload: Async, non-blocking

Capacity:
- 100+ concurrent query requests
- 1000+ requests/minute (Enterprise tier)
- 100 MB max document size
- 10 WebSocket connections per API key

## Troubleshooting

### API Key Not Working

1. Check key is active: `make api-list-keys`
2. Verify base64 encoding is correct
3. Ensure key hasn't expired
4. Check permissions for endpoint

### Rate Limit Errors

1. Check current tier limits
2. Upgrade to higher tier if needed
3. Implement exponential backoff
4. Check `X-RateLimit-*` headers

### Health Check Failing

1. Check IRIS database connection
2. Verify Redis is running (if enabled)
3. Check pipeline initialization
4. Review component logs

### Slow Query Performance

1. Check pipeline health: `GET /api/v1/pipelines/{name}`
2. Review execution time breakdown
3. Optimize document chunking
4. Consider reranking for better relevance

## CLI Reference

```bash
# Server operations
python -m iris_rag.api.cli run [--host HOST] [--port PORT] [--workers N] [--reload]

# API key management
python -m iris_rag.api.cli create-key --name NAME --owner-email EMAIL [--tier TIER] [--permissions PERMS]
python -m iris_rag.api.cli list-keys [--owner-email EMAIL]
python -m iris_rag.api.cli revoke-key --key-id KEY_ID

# Database operations
python -m iris_rag.api.cli setup-db

# Health check
python -m iris_rag.api.cli health [--host HOST] [--port PORT]
```

## Support

For issues, feature requests, or questions:
- GitHub Issues: https://github.com/your-org/rag-templates/issues
- Documentation: http://localhost:8000/docs (when server is running)

## License

See LICENSE file in repository root.
