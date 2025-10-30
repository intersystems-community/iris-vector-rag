# Tasks: Production-Grade REST API for RAG Pipelines

**Branch**: `042-full-rest-api`
**Feature**: Full REST API with FastAPI integration for RAG pipelines
**Prerequisites**: plan.md, research.md, data-model.md, contracts/, quickstart.md

---

## Phase 3.1: Setup & Infrastructure

- [ ] **T001** Create project structure for REST API at `iris_rag/api/` with subdirectories: routes/, middleware/, models/, services/, websocket/
- [ ] **T002** Add FastAPI dependencies to `pyproject.toml`: fastapi>=0.104.0, uvicorn[standard]>=0.24.0, python-multipart>=0.0.6, websockets>=12.0, redis>=5.0.0 (optional)
- [ ] **T003** [P] Create API configuration file at `config/api_config.yaml` with connection pool settings, rate limit defaults, CORS config
- [ ] **T004** [P] Implement IRIS connection pool manager in `common/connection_pool.py` with 20 base connections, 10 overflow, 1-hour recycle, pre-ping validation

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests (All tests must fail initially)

- [ ] **T005** [P] Contract test for authentication endpoints in `tests/contract/test_auth_contracts.py` - validate API key creation, validation, revocation endpoints against auth.yaml OpenAPI spec
- [ ] **T006** [P] Contract test for query endpoints in `tests/contract/test_query_contracts.py` - validate POST /{pipeline}/_search endpoints against query.yaml OpenAPI spec for all 5 pipeline types (basic, basic_rerank, crag, graphrag, pylate_colbert)
- [ ] **T007** [P] Contract test for pipeline management in `tests/contract/test_pipeline_contracts.py` - validate GET /pipelines and GET /pipelines/{type} endpoints against pipeline.yaml OpenAPI spec
- [ ] **T008** [P] Contract test for document upload in `tests/contract/test_document_contracts.py` - validate POST /documents/upload and GET /documents/operations/{id} endpoints against document.yaml OpenAPI spec
- [ ] **T009** [P] Contract test for WebSocket events in `tests/contract/test_websocket_contracts.py` - validate WebSocket event message schemas (query_start, retrieval_progress, generation_chunk, query_complete, error, document_upload_progress) against websocket.yaml OpenAPI spec
- [ ] **T010** [P] Contract test for health endpoint in `tests/contract/test_health_contracts.py` - validate GET /health endpoint returns component status structure per openapi.yaml spec

### Integration Tests (User Stories → Tests)

- [ ] **T011** [P] Integration test for Scenario 1 (query with valid key) in `tests/integration/test_query_e2e.py` - verify authenticated query returns answer with documents within 2s
- [ ] **T012** [P] Integration test for Scenario 2 (unauthenticated request) in `tests/integration/test_auth_e2e.py` - verify missing/invalid API key returns 401 with clear error message
- [ ] **T013** [P] Integration test for Scenario 3 (rate limit enforcement) in `tests/integration/test_rate_limit_e2e.py` - verify exceeding rate limit returns 429 with Retry-After header
- [ ] **T014** [P] Integration test for Scenario 4 (list pipelines) in `tests/integration/test_pipeline_listing_e2e.py` - verify GET /pipelines returns all configured pipelines with status and capabilities
- [ ] **T015** [P] Integration test for Scenario 5 (WebSocket streaming) in `tests/integration/test_websocket_streaming.py` - verify document upload progress streams events with percentage completion
- [ ] **T016** [P] Integration test for Scenario 6 (validation errors) in `tests/integration/test_validation_e2e.py` - verify invalid parameters return 422 with field-level error details
- [ ] **T017** [P] Integration test for Scenario 7 (unhealthy pipeline) in `tests/integration/test_pipeline_health_e2e.py` - verify degraded/unavailable pipeline returns 503 with estimated recovery time
- [ ] **T018** [P] Integration test for Scenario 8 (health endpoint) in `tests/integration/test_health_e2e.py` - verify GET /health returns status of all dependencies (IRIS, Redis, pipelines)

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Pydantic Models (from data-model.md entities)

- [ ] **T019** [P] Create API Request model in `iris_rag/api/models/request.py` - Pydantic model for QueryRequest with query_text (1-10000 chars), pipeline_type enum, parameters (top_k, filters), validation rules per Entity 1
- [ ] **T020** [P] Create Pipeline Instance model in `iris_rag/api/models/pipeline.py` - Pydantic model for PipelineInstance with status enum (healthy/degraded/unavailable), capabilities list, performance stats per Entity 2
- [ ] **T021** [P] Create Authentication Token model in `iris_rag/api/models/auth.py` - Pydantic model for ApiKey with permissions enum (read/write/admin), rate_limit_tier, bcrypt hash validation per Entity 3
- [ ] **T022** [P] Create Rate Limit Quota model in `iris_rag/api/models/quota.py` - Pydantic model for RateLimitQuota with quota_type enum, sliding window fields, exceeded count tracking per Entity 4
- [ ] **T023** [P] Create Query Response model in `iris_rag/api/models/response.py` - Pydantic model for QueryResponse with Document nested schema, execution timing fields, RAGAS-compatible contexts property per Entity 5
- [ ] **T024** [P] Create Document Upload Operation model in `iris_rag/api/models/upload.py` - Pydantic model for DocumentUploadOperation with status enum, progress tracking, validation errors list per Entity 6
- [ ] **T025** [P] Create WebSocket Session model in `iris_rag/api/models/websocket.py` - Pydantic model for WebSocketSession with subscription_type enum, reconnection token, activity tracking per Entity 7
- [ ] **T026** [P] Create Health Status model in `iris_rag/api/models/health.py` - Pydantic model for HealthStatus with component_name, status enum, dependencies list, metrics object per Entity 8
- [ ] **T027** [P] Create Error Response models in `iris_rag/api/models/errors.py` - Elasticsearch-inspired error structure with type, reason, details fields for all HTTP status codes (400, 401, 422, 429, 500, 503)

### Middleware Components

- [ ] **T028** [P] Implement API key authentication middleware in `iris_rag/api/middleware/auth.py` - base64 decode Authorization header, validate against IRIS ApiKey table, set request.state.api_key_id, return 401 on failure (FR-009, FR-010, FR-012)
- [ ] **T029** [P] Implement rate limiting middleware in `iris_rag/api/middleware/rate_limit.py` - Redis-based sliding window counter, check per-key quotas, set X-RateLimit-* headers, return 429 when exceeded (FR-013, FR-014, FR-015)
- [ ] **T030** [P] Implement request/response logging middleware in `iris_rag/api/middleware/logging.py` - log request ID, timestamp, status code, endpoint, method, response time, API key ID to IRIS ApiRequestLog table (FR-035, FR-036)

### Service Layer

- [ ] **T031** Implement PipelineManager service in `iris_rag/api/services/pipeline_manager.py` - initialize all configured pipelines on startup using create_pipeline(), track instance registry, run health checks every 30s, handle graceful shutdown (FR-005, FR-006, FR-007, FR-008)
- [ ] **T032** [P] Implement AuthService in `iris_rag/api/services/auth_service.py` - API key CRUD operations (create with bcrypt, validate, revoke), permission checking, expiration validation, store in IRIS ApiKey table (FR-009, FR-011)
- [ ] **T033** [P] Implement RateLimiter service in `iris_rag/api/services/rate_limiter.py` - adaptive concurrency control, per-key quota enforcement, sliding window implementation with Redis ZSET, fallback to in-memory if Redis unavailable (FR-013, FR-016)

### API Routes

- [ ] **T034** Implement query routes in `iris_rag/api/routes/query.py` - POST /{pipeline}/_search endpoints for all 5 pipeline types, call pipeline.query(), return standardized QueryResponse, include X-Request-ID and X-Execution-Time-Ms headers (FR-001, FR-002, FR-004)
- [ ] **T035** [P] Implement pipeline routes in `iris_rag/api/routes/pipeline.py` - GET /pipelines (list all with status), GET /pipelines/{type} (specific pipeline health), aggregate from PipelineManager registry (FR-008)
- [ ] **T036** [P] Implement document upload routes in `iris_rag/api/routes/document.py` - POST /documents/upload (async multipart), validate file size (<100MB) and format, create DocumentUploadOperation, return operation_id, ensure non-blocking (FR-021, FR-022, FR-023, FR-024)
- [ ] **T037** [P] Implement health routes in `iris_rag/api/routes/health.py` - GET /health endpoint, check IRIS connection (SELECT 1), Redis ping, pipeline statuses, return aggregated HealthStatus for all components (FR-032, FR-033, FR-034)

### WebSocket Handlers

- [ ] **T038** Implement WebSocket connection handler in `iris_rag/api/websocket/handlers.py` - accept connection, authenticate via first message, register in session registry, handle heartbeat pings, reconnection with token (FR-025, FR-028)
- [ ] **T039** Implement query streaming in `iris_rag/api/websocket/events.py` - stream query_start, retrieval_progress, generation_chunk, query_complete events with ISO8601 timestamps and request_id (FR-026, FR-028)
- [ ] **T040** Implement document upload progress streaming in `iris_rag/api/websocket/events.py` - stream document_upload_progress events with percentage, processed_documents count, total_documents (FR-027, FR-028)

### FastAPI Application

- [ ] **T041** Implement FastAPI application factory in `iris_rag/api/main.py` - create_app() with lifespan events, register all routes, add middleware (auth, rate limit, logging, CORS), initialize PipelineManager on startup, shutdown pipelines gracefully
- [ ] **T042** Create CLI entrypoint in `iris_rag/api/cli.py` - commands for create-key (API key generation), list-keys, revoke-key, run-server (uvicorn wrapper)
- [ ] **T043** Add Makefile targets - make api-dev (uvicorn with reload), make api-prod (gunicorn with 4 workers), make api-test (run integration tests against live server)

## Phase 3.4: Database Setup

- [ ] **T044** Create IRIS schema migration in `iris_rag/storage/migrations/042_api_tables.sql` - create RAG_API schema, ApiKey table (key_id, key_secret_hash, permissions, rate_limit_tier, etc. per Entity 3), ApiRequestLog table with 30-day retention, RateLimitQuota table, DocumentUploadOperation table
- [ ] **T045** Implement IRIS table initialization in `iris_rag/api/services/storage.py` - run migration on first startup, create indexes (key_id, api_key_id, timestamp), setup monthly partitioning for ApiRequestLog
- [ ] **T046** [P] Implement log retention cleanup job in `iris_rag/api/services/cleanup.py` - background task to delete ApiRequestLog records older than 30 days, run daily at midnight (FR-036)

## Phase 3.5: Deployment & Documentation

- [ ] **T047** [P] Create Dockerfile for API server in `docker/api.Dockerfile` - multi-stage build, copy only api/ package, expose port 8000, set entrypoint to uvicorn
- [ ] **T048** [P] Update docker-compose.yml - add api service, link to iris and redis, set environment variables (IRIS_HOST, IRIS_PORT, REDIS_URL), health check on GET /health
- [ ] **T049** [P] Create API server README at `iris_rag/api/README.md` - quickstart guide, authentication setup, deployment options (sidecar, Docker), configuration reference
- [ ] **T050** [P] Update CLAUDE.md agent context - add REST API section with make targets (make api-dev, make api-prod), authentication flow, WebSocket examples, health monitoring

## Phase 3.6: Unit Tests & Polish

- [ ] **T051** [P] Unit test for authentication middleware in `tests/unit/test_auth_middleware.py` - test base64 decoding, bcrypt validation, permission checking, 401 error cases
- [ ] **T052** [P] Unit test for rate limiter in `tests/unit/test_rate_limiter.py` - test sliding window logic, quota enforcement, Redis fallback, header generation
- [ ] **T053** [P] Unit test for pipeline manager in `tests/unit/test_pipeline_manager.py` - test initialization, health checks, graceful shutdown, error handling
- [ ] **T054** [P] Performance test for concurrent requests in `tests/integration/test_concurrent_requests.py` - send 100+ concurrent queries, verify <2s p95 latency, check connection pool doesn't exhaust
- [ ] **T055** Run full acceptance scenario validation - execute all 8 quickstart.md scenarios against live API, verify success criteria, document any issues
- [ ] **T056** Code quality pass - run black, isort, flake8, mypy on iris_rag/api/, fix all linting errors, ensure 100% type hints on public functions
- [ ] **T057** Remove duplication - extract common error handling to helpers, consolidate validation logic, DRY up middleware patterns
- [ ] **T058** Final documentation review - ensure all endpoints documented in contracts/, quickstart.md examples tested, README complete

---

## Dependencies

**Setup Phase**:
- T001 (project structure) blocks all other tasks
- T002 (dependencies) required for any implementation
- T003-T004 can run in parallel (different files)

**Tests Phase**:
- Contract tests (T005-T010) can all run in parallel (different files)
- Integration tests (T011-T018) can all run in parallel (different files)
- All tests (T005-T018) must complete and FAIL before starting Phase 3.3

**Implementation Phase**:
- Pydantic models (T019-T027) can all run in parallel (different files)
- Middleware (T028-T030) can all run in parallel (different files)
- Services (T031-T033): T031 blocks T034, T032-T033 independent
- Routes (T034-T037): T034 depends on T031, others can run in parallel
- WebSocket (T038-T040): T038 blocks T039-T040
- Application (T041-T043): T041 depends on all routes/middleware, T042-T043 independent

**Database Phase**:
- T044 (SQL migration) blocks T045 (initialization)
- T046 can run in parallel with T045

**Deployment Phase**:
- All tasks (T047-T050) can run in parallel (different files)

**Polish Phase**:
- Unit tests (T051-T053) can all run in parallel (different files)
- T054-T058 should run sequentially after all implementation complete

---

## Parallel Execution Examples

### Phase 3.2: Launch All Contract Tests Together
```bash
# All contract tests can run in parallel (different files, no dependencies)
pytest tests/contract/test_auth_contracts.py &
pytest tests/contract/test_query_contracts.py &
pytest tests/contract/test_pipeline_contracts.py &
pytest tests/contract/test_document_contracts.py &
pytest tests/contract/test_websocket_contracts.py &
pytest tests/contract/test_health_contracts.py &
wait
```

### Phase 3.2: Launch All Integration Tests Together
```bash
# All integration tests can run in parallel
pytest tests/integration/test_query_e2e.py &
pytest tests/integration/test_auth_e2e.py &
pytest tests/integration/test_rate_limit_e2e.py &
pytest tests/integration/test_pipeline_listing_e2e.py &
pytest tests/integration/test_websocket_streaming.py &
pytest tests/integration/test_validation_e2e.py &
pytest tests/integration/test_pipeline_health_e2e.py &
pytest tests/integration/test_health_e2e.py &
wait
```

### Phase 3.3: Launch All Pydantic Model Tasks Together
```bash
# All model creation can run in parallel (8 different files)
# Task T019-T027: Create all Pydantic models simultaneously
```

### Phase 3.3: Launch All Middleware Tasks Together
```bash
# All middleware can run in parallel (3 different files)
# Task T028-T030: Implement all middleware simultaneously
```

---

## Validation Checklist

**Contract Coverage**:
- [x] All 6 contract files (openapi.yaml, auth.yaml, query.yaml, pipeline.yaml, document.yaml, websocket.yaml) have corresponding contract tests
- [x] All 8 entities from data-model.md have Pydantic model tasks
- [x] All 8 acceptance scenarios from quickstart.md have integration test tasks

**TDD Compliance**:
- [x] All tests written before implementation (Phase 3.2 before 3.3)
- [x] Tests must fail initially (no implementation exists yet)
- [x] Each contract → test task exists

**Parallel Execution**:
- [x] All [P] tasks are truly independent (different files, no shared state)
- [x] Tasks without [P] have documented sequential dependencies

**File Path Specificity**:
- [x] Every task specifies exact file path
- [x] No tasks modify same file (except T039-T040 both in events.py - sequential)

**Constitutional Compliance**:
- [x] Framework-First: All routes call create_pipeline() from existing framework
- [x] TDD: Contract tests (T005-T010) before implementation
- [x] Production Readiness: Logging (T030), health checks (T037, T045), Docker (T047-T048)
- [x] Explicit Errors: Error models (T027), structured error responses in all routes
- [x] Standardized DB: Connection pool (T004), VectorStore abstraction maintained

---

## Estimated Effort

- **Phase 3.1 (Setup)**: 4 tasks, ~2 hours
- **Phase 3.2 (Tests)**: 14 tasks, ~8 hours (all parallel if multiple developers)
- **Phase 3.3 (Implementation)**: 25 tasks, ~20 hours (many parallelizable)
- **Phase 3.4 (Database)**: 3 tasks, ~2 hours
- **Phase 3.5 (Deployment)**: 4 tasks, ~2 hours (all parallel)
- **Phase 3.6 (Polish)**: 8 tasks, ~4 hours

**Total**: 58 tasks, ~38 hours sequential, ~20 hours with full parallelization

---

## Success Criteria

All tasks complete when:

1. ✓ All 6 contract test files passing (OpenAPI validation)
2. ✓ All 8 integration test scenarios passing (quickstart.md workflows)
3. ✓ All 8 Pydantic models implement data-model.md entities with validation
4. ✓ API server starts successfully with `make api-dev`
5. ✓ All 5 pipeline types queryable via POST /{pipeline}/_search
6. ✓ Authentication enforced (401 on missing/invalid key)
7. ✓ Rate limiting functional (429 on quota exceeded)
8. ✓ WebSocket streaming working (progress updates received)
9. ✓ Health endpoint returns component status
10. ✓ Docker deployment working (`docker-compose up api`)
11. ✓ <2s p95 latency on 100+ concurrent queries
12. ✓ All linting passes (black, isort, flake8, mypy)

**Ready for production deployment when all 12 criteria met!**
