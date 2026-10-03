# Architecture and module map

## Modules (`iris_vector_rag/api/`, about 9,200 lines including docs and SQL)

| Path                            | Role                                                                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `main.py`                       | `AppState`, `load_config()`, `get_default_config()`, `lifespan()`, `create_app()`, module-level `app = create_app()`, `uvicorn.run()` |
| `middleware/auth.py`            | `ApiKeyAuth` (parse, verify, permission check) and `AuthenticationMiddleware`                                                         |
| `middleware/rate_limit.py`      | `RateLimiter` (Redis sorted sets) and `RateLimitMiddleware`                                                                           |
| `middleware/logging.py`         | `RequestLoggingMiddleware`: request id, timing, writes `api_request_logs`                                                             |
| `routes/query.py`               | `POST /api/v1/{pipeline}/_search`                                                                                                     |
| `routes/pipeline.py`            | `GET /api/v1/pipelines`, `GET /api/v1/pipelines/{pipeline_name}`                                                                      |
| `routes/document.py`            | Upload, operation status, operation list                                                                                              |
| `routes/health.py`              | `GET /api/v1/health`                                                                                                                  |
| `services/pipeline_manager.py`  | Builds one pipeline per configured type with `create_pipeline()`, tracks health and latency                                           |
| `services/document_service.py`  | Upload operations: create, process in the background, persist, list                                                                   |
| `services/auth_service.py`      | API-key CRUD (never exposed by any route)                                                                                             |
| `websocket/`                    | `ConnectionManager`, `QueryStreamingHandler`, `DocumentUploadProgressHandler`, `/ws` route                                            |
| `models/`                       | Pydantic models: auth, errors, health, pipeline, quota, request, response, upload, websocket                                          |
| `schema.sql`, `migrations/`     | IRIS tables and views                                                                                                                 |
| `scripts/check_code_quality.sh` | Lint helper                                                                                                                           |

Dependencies: `fastapi`, `uvicorn[standard]`, `python-multipart`, `websockets`, `redis`,
`bcrypt`, `python-jose[cryptography]`, `email-validator` (the `api` extra), plus the
package's own `create_pipeline()` and connection pool.

## Start-up sequence (`lifespan`)

1. `load_config("config/api_config.yaml")`; on `FileNotFoundError`, use
   `get_default_config()` (which carries `demo`/`demo` database credentials).
2. Build an IRIS connection pool from `database.*` (`host`, `port`, `namespace`,
   `username`, `password`, `pool_size`, `max_overflow`, `pool_recycle`).
3. If `redis.enabled`, create a Redis client and `ping()` it. Otherwise log that rate
   limiting is in-memory. (No in-memory limiter exists.)
4. Build `PipelineManager(config)`, `AuthService(pool, bcrypt_rounds)` and
   `DocumentService(pool, pipeline_manager)`.
5. Build `ConnectionManager`, `QueryStreamingHandler`, `DocumentUploadProgressHandler`.
6. On shutdown, close the pool and the Redis client.

`create_app()` builds the FastAPI app, CORS, three middleware, the global exception
handler and the five routers from `app_state`. The first version called `create_app()`
at import time, before `lifespan` had filled `app_state`, so every router and middleware
was built with `None` (AP-01).

## Middleware stack

Added in this order: CORS, `RequestLoggingMiddleware`, `RateLimitMiddleware` (only if a
Redis client exists at build time), `AuthenticationMiddleware`. Starlette makes the
_last_ one added the _outermost_, so the real order is Auth, then rate limit, then
logging, then CORS; the code comments say the opposite (AP-17). Public paths skipped by
`AuthenticationMiddleware`: `/api/v1/health`, `/api/v1/health/`, `/docs`, `/docs/`, and
the OpenAPI and ReDoc paths.

Routes additionally call `ApiKeyAuth.__call__(request, required_permission)` through a
`Depends`, so authentication runs twice per request.

## Pipelines

`PipelineManager` creates one pipeline for each name in `pipelines.enabled`
(`basic`, `basic_rerank`, `crag`, `graphrag`, `pylate_colbert`) with
`create_pipeline(pipeline_type=...)`, and records a `PipelineInstance` with status
`healthy`, `degraded` or `unavailable`, counters, a moving-average latency and a
capability list:

| Pipeline         | Capabilities                                                              |
| ---------------- | ------------------------------------------------------------------------- |
| `basic`          | vector_search, semantic_retrieval                                         |
| `basic_rerank`   | vector_search, semantic_retrieval, reranking                              |
| `crag`           | vector_search, self_evaluation, corrective_retrieval                      |
| `graphrag`       | vector_search, text_search, knowledge_graph, hybrid_retrieval, rrf_fusion |
| `pylate_colbert` | late_interaction, token_level_similarity                                  |

A pipeline that fails to initialise is recorded as `unavailable` with its error text.

## Response-shape goal

Query responses were designed to be RAGAS- and LangChain-compatible: `answer`,
`retrieved_documents[]` (with `score` and `metadata`), `contexts` (plain strings) and
`sources`. Keep that goal; it is the one part of the design with a clear consumer.
