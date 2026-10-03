# Route table

All HTTP routes live under `/api/v1`. Authentication is `Authorization: ApiKey
<base64(key_id:secret)>` (see [auth-and-rate-limits.md](auth-and-rate-limits.md)).

| Method | Path                                | Permission                                         | Request                                                           | Response                                    |
| ------ | ----------------------------------- | -------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------- |
| POST   | `/api/v1/{pipeline}/_search`        | read                                               | `QueryRequest` (JSON)                                             | `QueryResponse`                             |
| GET    | `/api/v1/pipelines`                 | none intended (AP-03)                              | none                                                              | `PipelineListResponse`                      |
| GET    | `/api/v1/pipelines/{pipeline_name}` | none intended (AP-03)                              | none                                                              | `PipelineInstance`, 404 if unknown          |
| POST   | `/api/v1/documents/upload`          | write                                              | multipart: `file`, `pipeline_type`, `chunk_size`, `chunk_overlap` | `DocumentUploadResponse`                    |
| GET    | `/api/v1/documents/operations/{id}` | write; owner or admin                              | none                                                              | `DocumentUploadOperation`, 403 if not yours |
| GET    | `/api/v1/documents/operations`      | write; admin sees all                              | query: `limit` (default 50, unbounded)                            | `List[DocumentUploadOperation]`             |
| GET    | `/api/v1/health`                    | public                                             | none                                                              | `HealthCheckResponse`, 200 or 503           |
| WS     | `/ws`                               | per [websocket-protocol.md](websocket-protocol.md) | first message authenticates                                       | events                                      |
| GET    | `/docs`, `/redoc`, `/openapi.json`  | public                                             | none                                                              | FastAPI defaults                            |

## Models

`QueryRequest`: `query` (1 to 10,000 characters), `top_k` (1 to 100, default 5),
`filters` (object, optional), `include_sources` (bool), `include_metadata` (bool),
`pipeline` (optional `PipelineType`).

`QueryResponse`: `response_id`, `request_id`, `answer`, `retrieved_documents[]`
(`Document`: `doc_id`, `content`, `score`, `metadata` with `source`, `chunk_index`,
`page_number`, `created_at`), `contexts[]`, `sources[]`, `pipeline_name`,
`execution_time_ms`, and optional `retrieval_time_ms`, `generation_time_ms`,
`tokens_used`, `confidence_score`, `metadata`.

`PipelineInstance`: `pipeline_id`, `pipeline_type`, `name`, `status`
(`healthy|degraded|unavailable`), `initialized_at`, `last_health_check`, `total_queries`,
`avg_latency_ms`, `error_count`, `error_rate`, `config`, `capabilities[]`.
`PipelineListResponse`: `pipelines[]` (plus total). `PipelineManager` built this model
with field names it does not have (AP-23).

`DocumentUploadOperation`: `operation_id`, `api_key_id`, `status`
(`pending|validating|processing|completed|failed`), `created_at`, `started_at`,
`completed_at`, `total_documents`, `processed_documents`, `failed_documents`,
`progress_percentage`, `file_size_bytes`, `pipeline_type`, `validation_errors[]`,
`error_message`. `DocumentUploadResponse`: `operation_id`, `status`, `message`.

`HealthCheckResponse`: `status`, `timestamp`, `components{name: HealthStatus}`,
`overall_health`. `HealthStatus`: `component_name`, `status`, `last_checked_at`,
`response_time_ms`, `version`, `dependencies[]`, `error_message`, `metrics`.

## Behaviour per route

- **Query.** Validate length and `top_k` (duplicating the model bounds, AP-22), look up
  the pipeline (404 as `bad_request`), refuse if its status is `unavailable` (503), call
  `pipeline.query(...)` synchronously (AP-13), map the result into `QueryResponse`, then
  update the pipeline's counters and moving-average latency.
- **Upload.** Reject files over 100 MB (413, skipped when the size is unknown, AP-11), map a `ValueError` from the service to 422, create
  an operation, return immediately, and process in a background task. The processing
  step never read the file; it slept 0.1 s per fake document and reported `completed`
  (AP-06). The config lists allowed types (`text/plain`, `application/pdf`,
  `application/json`, `text/markdown`), but no code enforced them.
- **Operations.** Owner or admin may read a single operation; the list is filtered to the
  caller unless the caller is admin.
- **Health.** Checks IRIS (a trivial query), Redis (`ping`) and each pipeline; returns
  503 when unhealthy. The 503 path built `JSONResponse(content=model_dump())` from
  objects that are not JSON-serialisable (AP-07).

## Error model (Elasticsearch-inspired)

```json
{
  "error": {
    "type": "validation_exception",
    "reason": "human-readable summary",
    "details": {
      "message": "...",
      "field": "top_k",
      "rejected_value": 500,
      "min_value": 1,
      "max_value": 100
    }
  }
}
```

`ErrorType`: `authentication_error`, `authorization_error`, `invalid_api_key`,
`expired_api_key`, `validation_exception`, `bad_request`, `rate_limit_exceeded`,
`service_unavailable`, `internal_server_error`, `processing_error`, `timeout_error`.
`ErrorDetails` carries optional context such as `required_permissions`,
`current_permissions`, `pipeline`, `key_id`, `window`.

Status codes used: 200, 400, 401, 403, 404, 413, 422, 429 (with `Retry-After`,
`X-RateLimit-*`), 500, 503. Every response was meant to carry `X-Request-ID` and
`X-Execution-Time-Ms`.
