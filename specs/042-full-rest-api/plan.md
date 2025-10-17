
# Implementation Plan: Production-Grade REST API for RAG Pipelines

**Branch**: `042-full-rest-api` | **Date**: 2025-10-16 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/Users/tdyar/ws/rag-templates/specs/042-full-rest-api/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from file system structure or context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 8. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary

Implement a production-grade REST API for the RAG framework that enables HTTP-based query processing, pipeline lifecycle management, and document loading. The API will support multiple RAG pipeline types (basic, CRAG, GraphRAG, etc.) with enterprise features including API key authentication, adaptive rate limiting, WebSocket streaming, and comprehensive error handling. Primary deployment mode is standalone FastAPI sidecar service with architecture supporting future IRIS-embedded deployment.

Technical approach leverages proven Elasticsearch API patterns for authentication (API key with base64 encoding), rate limiting (adaptive request concurrency + per-key quotas), error formatting (structured JSON with actionable messages), and endpoint design (POST /{pipeline}/_search). The API will maintain IRIS database connection pooling for efficient concurrent request handling and support both synchronous REST and WebSocket streaming for long-running operations.

## Technical Context
**Language/Version**: Python 3.11+
**Primary Dependencies**: FastAPI 0.104+, uvicorn[standard], python-multipart, websockets, redis (optional for rate limiting)
**Storage**: InterSystems IRIS database (vector store + metadata), Redis (optional, for rate limiting state)
**Testing**: pytest with contract tests (OpenAPI validation), integration tests (full pipeline workflows), unit tests (auth, rate limiting, validation)
**Target Platform**: Linux server (Docker container), macOS/Windows for development
**Project Type**: Single project (REST API server extending existing iris_rag framework)
**Performance Goals**: <2s p95 query latency, 100+ concurrent queries, 1-5 concurrent document uploads, <100ms authentication overhead
**Constraints**: 100MB max document upload size, 30-day log retention, adaptive concurrency with fallback to static limits, must not exhaust IRIS connection pool
**Scale/Scope**: 5 pipeline types (basic, basic_rerank, CRAG, GraphRAG, PyLateColBERT), 8 core API endpoints, 4 WebSocket event types, support for 100+ concurrent API clients

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**:
- ✓ Component extends existing RAGPipeline framework via create_pipeline()
- ✓ No application-specific logic (pure API layer over framework)
- ✓ CLI interface exposed via Make targets (make api-dev, make api-prod)

**II. Pipeline Validation & Requirements**:
- ✓ Automated requirement validation via existing pipeline validation framework
- ✓ Setup procedures idempotent (health checks, connection pool initialization)

**III. Test-Driven Development**:
- ✓ Contract tests written before implementation (OpenAPI schema validation)
- ✓ Performance tests for concurrent load scenarios (100+ queries, connection pool limits)

**IV. Performance & Enterprise Scale**:
- ✓ Incremental indexing supported via async document upload endpoints
- ✓ IRIS vector operations optimized via connection pooling and async handlers

**V. Production Readiness**:
- ✓ Structured logging included (request tracing, latency metrics, error tracking)
- ✓ Health checks implemented (GET /health endpoint with dependency status)
- ✓ Docker deployment ready (Dockerfile, docker-compose integration)

**VI. Explicit Error Handling**:
- ✓ No silent failures (all errors return structured JSON responses)
- ✓ Clear exception messages (Elasticsearch-inspired error format)
- ✓ Actionable error context (field-level validation errors, retry guidance)

**VII. Standardized Database Interfaces**:
- ✓ Uses proven SQL/vector utilities (iris_rag.storage.vector_store_iris)
- ✓ No ad-hoc IRIS queries (all DB access via VectorStore abstraction)
- ✓ New patterns contributed back (connection pooling utilities added to common/)

**GATE STATUS**: ✅ PASS - All constitutional principles satisfied

## Project Structure

### Documentation (this feature)
```
specs/042-full-rest-api/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
│   ├── openapi.yaml    # Main OpenAPI 3.1 specification
│   ├── auth.yaml       # Authentication schemas
│   ├── query.yaml      # Query endpoint contracts
│   ├── pipeline.yaml   # Pipeline management contracts
│   ├── document.yaml   # Document upload contracts
│   └── websocket.yaml  # WebSocket event schemas
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
iris_rag/
├── api/                 # NEW: REST API implementation
│   ├── __init__.py
│   ├── main.py         # FastAPI application factory
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── query.py    # Query endpoints
│   │   ├── pipeline.py # Pipeline management
│   │   ├── document.py # Document upload
│   │   └── health.py   # Health checks
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth.py     # API key authentication
│   │   ├── rate_limit.py # Adaptive rate limiting
│   │   └── logging.py  # Request/response logging
│   ├── models/
│   │   ├── __init__.py
│   │   ├── request.py  # Request schemas (Pydantic)
│   │   ├── response.py # Response schemas (Pydantic)
│   │   └── errors.py   # Error response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pipeline_manager.py  # Pipeline lifecycle
│   │   ├── auth_service.py      # API key management
│   │   └── rate_limiter.py      # Rate limit logic
│   └── websocket/
│       ├── __init__.py
│       ├── handlers.py # WebSocket connection handlers
│       └── events.py   # Event streaming protocol
├── core/               # EXISTING: RAG framework base classes
├── pipelines/          # EXISTING: Pipeline implementations
└── storage/            # EXISTING: Vector store implementations

common/                 # EXISTING: Shared utilities
├── connection_pool.py  # NEW: IRIS connection pool manager
└── api_utils.py       # NEW: API helper functions

tests/
├── contract/          # NEW: OpenAPI contract validation
│   ├── test_query_contracts.py
│   ├── test_auth_contracts.py
│   ├── test_pipeline_contracts.py
│   └── test_document_contracts.py
├── integration/       # NEW: Full API workflow tests
│   ├── test_query_e2e.py
│   ├── test_document_upload_e2e.py
│   ├── test_websocket_streaming.py
│   └── test_concurrent_requests.py
└── unit/              # NEW: Component-level tests
    ├── test_auth_middleware.py
    ├── test_rate_limiter.py
    └── test_pipeline_manager.py

config/
└── api_config.yaml    # NEW: API server configuration

docker/
└── api.Dockerfile     # NEW: API server Dockerfile
```

**Structure Decision**: Single project structure extending existing `iris_rag` framework. The REST API is implemented as a new `iris_rag/api/` package that wraps existing pipeline implementations. This follows Framework-First Architecture by keeping the API layer thin and reusable while leveraging the established RAG framework components.

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - All technical decisions resolved via Elasticsearch patterns (no NEEDS CLARIFICATION)
   - Research best practices for FastAPI production deployment
   - Research IRIS connection pooling strategies for concurrent requests
   - Research WebSocket connection management patterns

2. **Generate and dispatch research agents**:
   ```
   Task 1: "Research FastAPI production patterns for authentication, rate limiting, WebSocket support"
   Task 2: "Research Elasticsearch API design patterns for authentication, error formatting, retry logic"
   Task 3: "Research IRIS database connection pooling best practices for concurrent RAG queries"
   Task 4: "Research WebSocket streaming protocols for long-running operations with progress updates"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with FastAPI patterns, Elasticsearch design decisions, IRIS pooling strategies, WebSocket protocols

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - 8 entities from spec: API Request, Pipeline Instance, Authentication Token, Rate Limit Quota, Query Response, Document Upload Operation, WebSocket Session, Health Status
   - Field definitions, validation rules, state transitions
   - IRIS table mappings for persistent entities (API keys, rate limit state)

2. **Generate API contracts** from functional requirements:
   - OpenAPI 3.1 specification covering all 39 functional requirements
   - 6 contract files: openapi.yaml (main), auth.yaml, query.yaml, pipeline.yaml, document.yaml, websocket.yaml
   - Request/response schemas with validation rules
   - Error response formats (Elasticsearch-inspired)

3. **Generate contract tests** from contracts:
   - tests/contract/test_query_contracts.py - FR-001 to FR-004
   - tests/contract/test_auth_contracts.py - FR-009 to FR-012
   - tests/contract/test_pipeline_contracts.py - FR-005 to FR-008
   - tests/contract/test_document_contracts.py - FR-021 to FR-024
   - All tests must fail initially (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Acceptance scenario 1 → test_query_with_valid_key()
   - Acceptance scenario 2 → test_unauthenticated_request_rejected()
   - Acceptance scenario 3 → test_rate_limit_enforcement()
   - Acceptance scenario 4 → test_list_available_pipelines()
   - Acceptance scenario 5 → test_websocket_document_upload_streaming()
   - Acceptance scenario 6 → test_validation_error_handling()
   - Acceptance scenario 7 → test_unhealthy_pipeline_503_response()
   - Acceptance scenario 8 → test_health_endpoint_dependency_status()

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/bash/update-agent-context.sh claude`
   - Add REST API context: FastAPI patterns, OpenAPI contracts, deployment commands
   - Preserve existing RAG pipeline context
   - Update recent changes section

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, CLAUDE.md update

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Contract-first TDD approach:
  1. Each contract file → contract test task [P]
  2. Each entity → Pydantic model creation task [P]
  3. Each endpoint → route implementation task (after contract test)
  4. Each middleware → unit test + implementation tasks
  5. Each acceptance scenario → integration test task
  6. Infrastructure tasks: Docker, connection pool, logging

**Ordering Strategy**:
- TDD order: Contract tests → Pydantic models → Route implementation → Integration tests
- Dependency order: Models → Middleware → Routes → WebSocket handlers
- Infrastructure first: Connection pool, logging setup before route implementation
- Mark [P] for parallel execution: All contract tests, all Pydantic models, independent middleware components

**Task Grouping**:
- Group 1: Foundation (contracts, data models, connection pool) - 8 tasks
- Group 2: Authentication & Rate Limiting (middleware + tests) - 6 tasks
- Group 3: Core API Routes (query, pipeline, document endpoints) - 10 tasks
- Group 4: WebSocket Streaming (handlers, events, tests) - 5 tasks
- Group 5: Integration & Deployment (E2E tests, Docker, docs) - 6 tasks

**Estimated Output**: 35-40 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

No complexity violations identified. All constitutional principles are satisfied:
- Framework-First: API layer wraps existing RAG framework
- TDD: Contract tests drive implementation
- Production Readiness: Logging, health checks, Docker deployment included
- Explicit Errors: Structured JSON errors with actionable messages
- Standardized DB: Uses existing VectorStore abstraction + connection pooling

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [x] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none)

---
*Based on Constitution v1.6.0 - See `/.specify/memory/constitution.md`*
