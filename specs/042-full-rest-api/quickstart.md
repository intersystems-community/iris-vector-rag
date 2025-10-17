# Quickstart: Production-Grade REST API for RAG Pipelines

**Feature**: 042-full-rest-api
**Date**: 2025-10-16
**Purpose**: Step-by-step guide to using the RAG REST API based on acceptance scenarios

---

## Prerequisites

- IRIS database running (port 1972)
- API server running on http://localhost:8000
- Valid API key credentials
- Python 3.11+ with `requests` library (for examples)

## Setup

### 1. Start the API Server

```bash
# Development mode
make api-dev

# Or production mode
make api-prod

# Or using Docker
docker-compose up -d api
```

### 2. Verify Server is Running

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-16T12:34:56.789Z",
  "components": {
    "iris_database": {"status": "healthy", "response_time_ms": 12},
    "redis_cache": {"status": "healthy", "response_time_ms": 5},
    "graphrag_pipeline": {"status": "healthy", "response_time_ms": 8}
  },
  "overall_health": "healthy"
}
```

### 3. Obtain API Key

Contact your administrator or use the API key management CLI:

```bash
# Create a new API key
python -m iris_rag.api.cli create-key \
  --name "my-app-key" \
  --permissions read write \
  --tier premium

# Output:
# API Key ID: 7c9e6679-7425-40de-944b-e07fc1f90ae7
# API Key Secret: my_secret_key_12345
# Save this secret securely - it will not be shown again!
```

### 4. Encode API Key for Authentication

```python
import base64

key_id = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
key_secret = "my_secret_key_12345"

# Combine and encode
credentials = f"{key_id}:{key_secret}"
encoded = base64.b64encode(credentials.encode()).decode()

# Use in Authorization header
auth_header = f"ApiKey {encoded}"
print(auth_header)
# Output: ApiKey N2M5ZTY2NzktNzQyNS00MGRlLTk0NGItZTA3ZmMxZjkwYWU3Om15X3NlY3JldF9rZXlfMTIzNDU=
```

---

## Acceptance Scenario 1: Query with Valid API Key

**Given**: A valid API key and query text
**When**: Developer sends POST request to query endpoint
**Then**: System returns answer with documents and sources within 2 seconds

### cURL Example

```bash
curl -X POST http://localhost:8000/api/v1/graphrag/_search \
  -H "Authorization: ApiKey N2M5ZTY2NzktNzQyNS00MGRlLTk0NGItZTA3ZmMxZjkwYWU3Om15X3NlY3JldF9rZXlfMTIzNDU=" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is diabetes?",
    "top_k": 5
  }'
```

### Python Example

```python
import requests
import base64

# Setup authentication
key_id = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
key_secret = "my_secret_key_12345"
credentials = f"{key_id}:{key_secret}"
auth_token = base64.b64encode(credentials.encode()).decode()

# Send query
response = requests.post(
    "http://localhost:8000/api/v1/graphrag/_search",
    headers={
        "Authorization": f"ApiKey {auth_token}",
        "Content-Type": "application/json"
    },
    json={
        "query": "What is diabetes?",
        "top_k": 5
    }
)

# Check response
assert response.status_code == 200
result = response.json()

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
print(f"Execution time: {result['execution_time_ms']}ms")
```

### Expected Response

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
        "page_number": 127
      }
    }
  ],
  "sources": ["medical_textbook_ch5.pdf"],
  "pipeline_name": "graphrag",
  "execution_time_ms": 1456
}
```

---

## Acceptance Scenario 2: Unauthenticated Request

**Given**: Request without API key
**When**: Developer sends query without credentials
**Then**: System returns 401 Unauthorized with authentication instructions

### Example

```bash
curl -X POST http://localhost:8000/api/v1/basic/_search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is diabetes?"
  }'
```

### Expected Response (401)

```json
{
  "error": {
    "type": "authentication_error",
    "reason": "Missing Authorization header",
    "details": {
      "message": "API requests must include 'Authorization: ApiKey <base64(id:secret)>' header"
    }
  }
}
```

---

## Acceptance Scenario 3: Rate Limit Enforcement

**Given**: Developer exceeds rate limit
**When**: Another request is sent
**Then**: System returns 429 with retry-after information

### Example (Send 101 requests in 60 seconds with 100/min limit)

```python
import requests
import base64
import time

auth_token = "..."  # Your base64-encoded credentials

# Send requests until rate limited
for i in range(105):
    response = requests.post(
        "http://localhost:8000/api/v1/basic/_search",
        headers={"Authorization": f"ApiKey {auth_token}"},
        json={"query": f"Query {i}"}
    )

    if response.status_code == 429:
        print(f"Rate limited after {i} requests")
        print(f"Retry after: {response.headers.get('Retry-After')} seconds")
        print(f"Response: {response.json()}")
        break
```

### Expected Response (429)

```json
{
  "error": {
    "type": "rate_limit_exceeded",
    "reason": "Too many requests",
    "details": {
      "limit": 100,
      "window": "requests per minute",
      "retry_after_seconds": 60
    }
  }
}
```

Response Headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1697461200
Retry-After: 60
```

---

## Acceptance Scenario 4: List Available Pipelines

**Given**: Multiple pipeline types available
**When**: Developer queries pipelines endpoint
**Then**: System returns list with capabilities and status

### Example

```bash
curl http://localhost:8000/api/v1/pipelines \
  -H "Authorization: ApiKey N2M5ZTY2NzktNzQyNS00MGRlLTk0NGItZTA3ZmMxZjkwYWU3Om15X3NlY3JldF9rZXlfMTIzNDU="
```

### Expected Response

```json
{
  "pipelines": [
    {
      "pipeline_type": "basic",
      "name": "basic-production",
      "status": "healthy",
      "capabilities": ["vector_search"],
      "avg_latency_ms": 234.5
    },
    {
      "pipeline_type": "graphrag",
      "name": "graphrag-production",
      "status": "healthy",
      "capabilities": ["vector_search", "graph_traversal", "entity_extraction"],
      "avg_latency_ms": 1234.5
    }
  ]
}
```

---

## Acceptance Scenario 5: WebSocket Streaming for Document Upload

**Given**: Long-running document loading operation
**When**: Developer connects via WebSocket and uploads documents
**Then**: System streams progress updates in real-time

### Python WebSocket Example

```python
import asyncio
import websockets
import json
import base64

async def upload_with_progress():
    # Encode API key
    credentials = f"{key_id}:{key_secret}"
    auth_token = base64.b64encode(credentials.encode()).decode()

    # Connect to WebSocket
    uri = "ws://localhost:8000/api/v1/ws/query"
    async with websockets.connect(uri) as websocket:
        # Send upload request
        await websocket.send(json.dumps({
            "api_key": auth_token,
            "operation_id": "b1c2d3e4-5678-90ab-cdef-fedcba987654"
        }))

        # Receive progress events
        async for message in websocket:
            event = json.loads(message)
            print(f"Event: {event['event']}")

            if event['event'] == 'document_upload_progress':
                progress = event['data']['progress_percentage']
                print(f"Progress: {progress}%")

            elif event['event'] == 'query_complete':
                print("Upload complete!")
                break

            elif event['event'] == 'error':
                print(f"Error: {event['data']['error']['reason']}")
                break

asyncio.run(upload_with_progress())
```

### Expected WebSocket Events

```json
// Event 1: Upload started
{
  "event": "document_upload_progress",
  "data": {
    "operation_id": "b1c2d3e4-5678-90ab-cdef-fedcba987654",
    "processed_documents": 0,
    "total_documents": 100,
    "progress_percentage": 0.0
  },
  "timestamp": "2025-10-16T12:30:00.000Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}

// Event 2: Progress update
{
  "event": "document_upload_progress",
  "data": {
    "operation_id": "b1c2d3e4-5678-90ab-cdef-fedcba987654",
    "processed_documents": 47,
    "total_documents": 100,
    "progress_percentage": 47.0
  },
  "timestamp": "2025-10-16T12:30:15.000Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}

// Event 3: Upload complete
{
  "event": "query_complete",
  "data": {
    "operation_id": "b1c2d3e4-5678-90ab-cdef-fedcba987654",
    "status": "completed",
    "processed_documents": 100,
    "total_documents": 100
  },
  "timestamp": "2025-10-16T12:30:45.000Z",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## Acceptance Scenario 6: Validation Error Handling

**Given**: Invalid query parameter
**When**: Developer sends malformed request
**Then**: System returns 422 with field-level errors

### Example

```bash
curl -X POST http://localhost:8000/api/v1/basic/_search \
  -H "Authorization: ApiKey ..." \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is diabetes?",
    "top_k": -5
  }'
```

### Expected Response (422)

```json
{
  "error": {
    "type": "validation_exception",
    "reason": "Invalid parameter value",
    "details": {
      "field": "top_k",
      "rejected_value": -5,
      "message": "Must be positive integer between 1 and 100"
    }
  }
}
```

---

## Acceptance Scenario 7: Unhealthy Pipeline Returns 503

**Given**: Pipeline is initializing or unhealthy
**When**: Developer attempts to query
**Then**: System returns 503 with estimated recovery time

### Example

```bash
curl -X POST http://localhost:8000/api/v1/graphrag/_search \
  -H "Authorization: ApiKey ..." \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is diabetes?"
  }'
```

### Expected Response (503)

```json
{
  "error": {
    "type": "service_unavailable",
    "reason": "Pipeline is currently unavailable",
    "details": {
      "pipeline": "graphrag",
      "status": "degraded",
      "estimated_recovery_time": 120,
      "message": "Pipeline is initializing. Please try again in 2 minutes."
    }
  }
}
```

---

## Acceptance Scenario 8: Health Endpoint with Dependency Status

**Given**: Administrator wants to monitor system health
**When**: Health endpoint is checked
**Then**: System returns status of all components

### Example

```bash
curl http://localhost:8000/api/v1/health
```

### Expected Response

```json
{
  "status": "healthy",
  "timestamp": "2025-10-16T12:34:56.789Z",
  "components": {
    "iris_database": {
      "status": "healthy",
      "response_time_ms": 12,
      "version": "2025.3.0",
      "metrics": {
        "connection_pool_size": 20,
        "active_connections": 8
      }
    },
    "redis_cache": {
      "status": "healthy",
      "response_time_ms": 5
    },
    "graphrag_pipeline": {
      "status": "healthy",
      "response_time_ms": 8,
      "metrics": {
        "total_queries": 15234,
        "avg_latency_ms": 1234.5
      }
    },
    "basic_pipeline": {
      "status": "healthy",
      "response_time_ms": 6
    }
  },
  "overall_health": "healthy"
}
```

---

## Common API Patterns

### 1. Check Response Headers for Debugging

```python
response = requests.post(...)

# Request tracing
request_id = response.headers.get('X-Request-ID')
print(f"Request ID: {request_id}")

# Performance monitoring
execution_time = response.headers.get('X-Execution-Time-Ms')
print(f"Execution time: {execution_time}ms")

# Rate limiting
remaining = response.headers.get('X-RateLimit-Remaining')
print(f"Rate limit remaining: {remaining}")
```

### 2. Retry Logic with Exponential Backoff

```python
import time

def query_with_retry(query, max_retries=3):
    backoff_sequence = [1, 1, 2, 3, 5, 8]  # Fibonacci

    for attempt in range(max_retries):
        response = requests.post(
            "http://localhost:8000/api/v1/basic/_search",
            headers={"Authorization": f"ApiKey {auth_token}"},
            json={"query": query}
        )

        if response.status_code == 200:
            return response.json()

        elif response.status_code == 429:
            # Rate limited - use Retry-After header
            retry_after = int(response.headers.get('Retry-After', backoff_sequence[attempt]))
            print(f"Rate limited. Retrying in {retry_after}s...")
            time.sleep(retry_after)

        elif response.status_code == 503:
            # Service unavailable - exponential backoff
            backoff = backoff_sequence[min(attempt, len(backoff_sequence) - 1)]
            print(f"Service unavailable. Retrying in {backoff}s...")
            time.sleep(backoff)

        else:
            # Other errors - don't retry
            raise Exception(f"Query failed: {response.json()}")

    raise Exception("Max retries exceeded")
```

### 3. Query Multiple Pipelines and Compare Results

```python
pipelines = ['basic', 'basic_rerank', 'crag', 'graphrag']
query = "What is diabetes?"

results = {}
for pipeline in pipelines:
    response = requests.post(
        f"http://localhost:8000/api/v1/{pipeline}/_search",
        headers={"Authorization": f"ApiKey {auth_token}"},
        json={"query": query, "top_k": 5}
    )

    if response.status_code == 200:
        result = response.json()
        results[pipeline] = {
            "answer": result['answer'],
            "execution_time_ms": result['execution_time_ms'],
            "confidence": result.get('confidence_score', 0.0)
        }

# Compare results
for pipeline, data in results.items():
    print(f"{pipeline}: {data['execution_time_ms']}ms, confidence: {data['confidence']}")
```

---

## Next Steps

1. **Explore API Documentation**: Review OpenAPI spec at `/specs/042-full-rest-api/contracts/openapi.yaml`
2. **Run Integration Tests**: Execute test suite to validate all acceptance scenarios
3. **Monitor API Performance**: Use health endpoint and response headers for monitoring
4. **Review Rate Limits**: Check your API key tier and request quotas
5. **Implement Client SDK**: Build language-specific SDKs using OpenAPI contracts

---

## Troubleshooting

### Authentication Errors
- Verify API key format: `Authorization: ApiKey <base64(id:secret)>`
- Check key is active: `is_active=true`
- Verify key hasn't expired: `expires_at > current_time`

### Rate Limiting
- Check response headers: `X-RateLimit-Remaining`
- Implement retry logic with `Retry-After` header
- Consider upgrading to higher tier (premium, enterprise)

### Slow Queries
- Check `X-Execution-Time-Ms` header to identify bottlenecks
- Review pipeline health: `GET /api/v1/pipelines/{pipeline_type}`
- Consider using faster pipeline (basic vs graphrag)

### WebSocket Connection Issues
- Verify WebSocket support in client library
- Check authentication in first message
- Implement heartbeat handling (ping/pong)
- Handle reconnection with `request_id`

---

## Success Criteria

All acceptance scenarios should complete successfully:

1. ✓ Query with valid API key returns answer within 2 seconds
2. ✓ Unauthenticated request returns 401 with clear instructions
3. ✓ Rate limit enforcement returns 429 with retry guidance
4. ✓ Pipeline listing returns capabilities and status
5. ✓ WebSocket streaming provides real-time progress updates
6. ✓ Validation errors return 422 with field-level details
7. ✓ Unhealthy pipeline returns 503 with recovery estimate
8. ✓ Health endpoint returns status of all dependencies

**Ready for production deployment when all scenarios pass!**
