# Tasks: Complete MCP Tools Implementation

**Input**: Design documents from `/Users/intersystems-community/ws/rag-templates/specs/043-complete-mcp-tools/`
**Prerequisites**: plan.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓, quickstart.md ✓
**Feature Branch**: `043-complete-mcp-tools`

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → ✓ Found: Python 3.11+ bridge, Node.js 18+ server, TypeScript 5.0+
   → ✓ Structure: iris_rag/mcp/, nodejs/src/mcp/, tests/
2. Load optional design documents:
   → ✓ data-model.md: 5 entities (MCP Tool, Tool Request, Tool Response, Pipeline Config, Health Status)
   → ✓ contracts/: 3 files (mcp_tool_schema.json, python_bridge_interface.py, nodejs_server_interface.ts)
   → ✓ research.md: 8 decisions (MCP SDK, transports, architecture, auth, deployment, etc.)
   → ✓ quickstart.md: 7 integration scenarios
3. Generate tasks by category:
   → Setup: Python/Node.js deps, directory structure
   → Tests: 3 contract tests, 7 integration tests (from quickstart scenarios)
   → Core: 6 Python files, 7 Node.js/TypeScript files
   → Integration: REST API integration, health checks
   → Polish: Documentation, Makefile targets
4. Apply task rules:
   → Contract tests [P] (different files)
   → Python implementations [P] (different files)
   → Node.js implementations [P] where independent
   → Sequential for interdependent files
5. Number tasks sequentially (T001-T058)
6. Dependency graph generated below
7. Parallel execution examples included
8. Validation: All contracts tested, all entities modeled, TDD order enforced
9. ✓ SUCCESS (58 tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Exact file paths included in descriptions

---

## Phase 3.1: Setup (T001-T005)

### Project Structure & Dependencies

- [ ] **T001** [P] Create Python MCP bridge directory structure
  - Create `/Users/intersystems-community/ws/rag-templates/iris_rag/mcp/__init__.py`
  - Verify directory accessible from REST API code
  - Add to Python package structure

- [ ] **T002** [P] Create Node.js MCP server directory structure
  - Create `/Users/intersystems-community/ws/rag-templates/nodejs/src/mcp/` directories
  - Create subdirectories: `transport/`, `config/`
  - Initialize TypeScript project if needed

- [ ] **T003** [P] Install Python dependencies for MCP bridge
  - Add to `pyproject.toml` or `requirements.txt`: `fastapi`, `uvicorn`, `pydantic>=2.0`
  - Install dependencies: `pip install -e .` or `uv sync`
  - Verify imports work

- [ ] **T004** [P] Install Node.js dependencies for MCP server
  - Add to `/Users/intersystems-community/ws/rag-templates/nodejs/package.json`: `@modelcontextprotocol/sdk`, `express`, `websockets`
  - Run `npm install` in nodejs/ directory
  - Verify TypeScript compilation works

- [ ] **T005** [P] Configure linting and TypeScript for MCP server
  - Add `tsconfig.json` to nodejs/ with strict mode
  - Configure ESLint for TypeScript
  - Add npm script for type checking

---

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests (T006-T008) - Can run in parallel [P]

- [ ] **T006** [P] Contract test for Python MCP Bridge interface
  - File: `/Users/intersystems-community/ws/rag-templates/tests/contract/test_mcp_bridge_contract.py`
  - Test `IMCPBridge` interface implementation
  - Test methods: `invoke_technique()`, `get_available_techniques()`, `health_check()`, `get_metrics()`
  - Assert interface compliance (TDD - should FAIL initially)
  - Based on: `contracts/python_bridge_interface.py`

- [ ] **T007** [P] Contract test for Python Technique Handlers
  - File: `/Users/intersystems-community/ws/rag-templates/tests/contract/test_technique_handlers_contract.py`
  - Test `ITechniqueHandler` interface for each of 6 pipelines
  - Test methods: `execute()`, `validate_params()`, `health_check()`
  - Assert handler registry works correctly
  - Based on: `contracts/python_bridge_interface.py`

- [ ] **T008** [P] Contract test for MCP tool schemas
  - File: `/Users/intersystems-community/ws/rag-templates/tests/contract/test_mcp_tool_schemas.py`
  - Validate all 6 RAG tool schemas (basic, basic_rerank, crag, graphrag, pylate_colbert, iris_global_graphrag)
  - Validate 2 utility tool schemas (health_check, metrics)
  - Test parameter validation against schemas
  - Based on: `contracts/mcp_tool_schema.json`

### Integration Tests (T009-T015) - From quickstart.md scenarios [P]

- [ ] **T009** [P] Integration test: List available tools (Scenario 1)
  - File: `/Users/intersystems-community/ws/rag-templates/tests/integration/test_mcp_list_tools.py`
  - Test MCP server `tools/list` endpoint
  - Assert 8 tools returned (6 RAG + 2 utility)
  - Assert schemas include parameter descriptions
  - Based on: `quickstart.md` Scenario 1

- [ ] **T010** [P] Integration test: Execute BasicRAG query (Scenario 2)
  - File: `/Users/intersystems-community/ws/rag-templates/tests/integration/test_mcp_basic_rag.py`
  - Test `rag_basic` tool execution
  - Query: "What are the symptoms of diabetes?"
  - Assert response has answer, documents, sources, performance metrics
  - Assert execution time < 2000ms (p95 requirement)
  - Based on: `quickstart.md` Scenario 2

- [ ] **T011** [P] Integration test: CRAG with corrective measures (Scenario 3)
  - File: `/Users/intersystems-community/ws/rag-templates/tests/integration/test_mcp_crag_correction.py`
  - Test `rag_crag` tool with low-confidence query
  - Assert `correction_applied` in metadata
  - Assert `confidence_score` and `rewritten_query` present
  - Based on: `quickstart.md` Scenario 3

- [ ] **T012** [P] Integration test: Health check tool (Scenario 4)
  - File: `/Users/intersystems-community/ws/rag-templates/tests/integration/test_mcp_health_check.py`
  - Test `rag_health_check` tool
  - Assert status for all 6 pipelines
  - Assert database connectivity reported
  - Assert performance metrics included
  - Based on: `quickstart.md` Scenario 4

- [ ] **T013** [P] Integration test: Concurrent queries (Scenario 5)
  - File: `/Users/intersystems-community/ws/rag-templates/tests/integration/test_mcp_concurrent_queries.py`
  - Test 3 concurrent requests (BasicRAG, CRAG, HybridGraphRAG)
  - Assert all complete successfully
  - Assert no resource conflicts
  - Test connection limit (max 5, reject 6th)
  - Based on: `quickstart.md` Scenario 5

- [ ] **T014** [P] Integration test: HybridGraphRAG with graph traversal (Scenario 6)
  - File: `/Users/intersystems-community/ws/rag-templates/tests/integration/test_mcp_hybrid_graphrag.py`
  - Test `rag_graphrag` tool with `search_method="hybrid"`
  - Assert response includes vector, text, and graph results
  - Assert `graph_traversal_depth` and `rrf_score` in metadata
  - Based on: `quickstart.md` Scenario 6

- [ ] **T015** [P] Integration test: MCP-REST API consistency (Scenario 7)
  - File: `/Users/intersystems-community/ws/rag-templates/tests/integration/test_mcp_rest_consistency.py`
  - Execute same query via MCP and REST API
  - Assert responses are identical (excluding timestamps/IDs)
  - Validates FR-006 (pipeline instance reuse) and FR-025 (response format consistency)
  - Based on: `quickstart.md` Scenario 7

---

## Phase 3.3: Core Implementation - Python MCP Bridge (T016-T027)

**ONLY after contract tests (T006-T008) are failing**

### Python Bridge Core (T016-T021) - Can run in parallel [P]

- [ ] **T016** [P] Implement MCP tool schema definitions
  - File: `/Users/intersystems-community/ws/rag-templates/iris_rag/mcp/tool_schemas.py`
  - Load schemas from `contracts/mcp_tool_schema.json`
  - Provide schema access for all 8 tools
  - Implement schema validation helper functions
  - Based on: `data-model.md` (MCP Tool entity), `contracts/mcp_tool_schema.json`

- [ ] **T017** [P] Implement parameter validation module
  - File: `/Users/intersystems-community/ws/rag-templates/iris_rag/mcp/validation.py`
  - Implement `ValidationError` exception class
  - Implement parameter validators for all 6 RAG tools
  - Implement common validators (query length, top_k range, etc.)
  - Based on: `contracts/python_bridge_interface.py`, `research.md` (Error Handling)

- [ ] **T018** [P] Implement MCP bridge configuration
  - File: `/Users/intersystems-community/ws/rag-templates/iris_rag/mcp/config.py`
  - Load configuration from YAML/env vars
  - Support both auth modes (api_key, none)
  - Support both deployment modes (standalone, integrated)
  - Based on: `research.md` (Authentication Strategy, Deployment Modes)

- [ ] **T019** [P] Implement Technique Handler Registry
  - File: `/Users/intersystems-community/ws/rag-templates/iris_rag/mcp/technique_handlers.py`
  - Implement `ITechniqueHandler` interface for each of 6 RAG pipelines
  - Implement `TechniqueHandlerRegistry` to manage handlers
  - Implement `register_handlers()`, `get_handler()`, `list_techniques()`
  - Based on: `data-model.md` (MCP Tool entity), `contracts/python_bridge_interface.py`

- [ ] **T020** [P] Implement MCP Server Manager
  - File: `/Users/intersystems-community/ws/rag-templates/iris_rag/mcp/server_manager.py`
  - Implement `IMCPServerManager` interface
  - Implement `start_server()`, `stop_server()`, `handle_tool_call()`
  - Implement `list_tools()`, `health_check()`
  - Manage Node.js subprocess lifecycle (if standalone mode)
  - Based on: `contracts/python_bridge_interface.py`, `research.md` (Deployment Modes)

- [ ] **T021** [P] Implement main MCPBridge class
  - File: `/Users/intersystems-community/ws/rag-templates/iris_rag/mcp/bridge.py`
  - Implement `IMCPBridge` interface
  - Implement `invoke_technique()` with authentication
  - Implement `get_available_techniques()`, `health_check()`, `get_metrics()`
  - Integrate with REST API's `PipelineManager` (pipeline instance reuse)
  - Based on: `contracts/python_bridge_interface.py`, `research.md` (Pipeline Instance Reuse)

### REST API Integration (T022-T024) - Sequential (same files)

- [ ] **T022** Integrate MCP bridge with REST API PipelineManager
  - File: `/Users/intersystems-community/ws/rag-templates/iris_rag/mcp/bridge.py` (modify T021)
  - Import `iris_rag.api.services.PipelineManager`
  - Use `PipelineManager.get_instance()` to access shared pipelines
  - Ensure same configuration and pipeline instances used by both MCP and REST API
  - Based on: `research.md` (Pipeline Instance Reuse)

- [ ] **T023** Add MCP health endpoints to REST API
  - File: `/Users/intersystems-community/ws/rag-templates/iris_rag/api/mcp_routes.py` (new)
  - Create FastAPI router with MCP health endpoints
  - Endpoints: `GET /api/v1/mcp/health`, `GET /api/v1/mcp/status`
  - Include in REST API's `main.py`
  - Based on: `plan.md` Project Structure

- [ ] **T024** Update REST API main.py for integrated MCP mode
  - File: `/Users/intersystems-community/ws/rag-templates/iris_rag/api/main.py` (modify)
  - Add conditional MCP bridge startup based on environment variable
  - Start MCP bridge as background asyncio task
  - Ensure graceful shutdown when REST API stops
  - Based on: `research.md` (Mode 2: Integrated Deployment)

### Python Bridge FastAPI Endpoints (T025-T027) - Can run in parallel [P]

- [ ] **T025** [P] Implement FastAPI endpoint: POST /mcp/invoke_technique
  - File: `/Users/intersystems-community/ws/rag-templates/iris_rag/mcp/bridge.py` (extend T021)
  - FastAPI route for technique invocation
  - Body: `{"technique": str, "query": str, "params": dict, "api_key": Optional[str]}`
  - Response: `{"success": bool, "result": dict, "error": Optional[str]}`
  - Based on: `research.md` (Python Bridge API)

- [ ] **T026** [P] Implement FastAPI endpoint: GET /mcp/list_techniques
  - File: `/Users/intersystems-community/ws/rag-templates/iris_rag/mcp/bridge.py` (extend T021)
  - FastAPI route to list available techniques
  - Response: `["basic", "basic_rerank", "crag", "graphrag", "pylate_colbert", "iris_global_graphrag"]`
  - Based on: `research.md` (Python Bridge API)

- [ ] **T027** [P] Implement FastAPI endpoint: GET /mcp/health_check
  - File: `/Users/intersystems-community/ws/rag-templates/iris_rag/mcp/bridge.py` (extend T021)
  - FastAPI route for health check
  - Response: `{"status": str, "pipelines": dict, "database": dict, "performance_metrics": dict}`
  - Based on: `data-model.md` (Health Status entity), `research.md` (Python Bridge API)

---

## Phase 3.4: Core Implementation - Node.js MCP Server (T028-T038)

**After Python bridge (T016-T027) is complete**

### Node.js Transport Layer (T028-T029) - Can run in parallel [P]

- [ ] **T028** [P] Implement stdio transport
  - File: `/Users/intersystems-community/ws/rag-templates/nodejs/src/mcp/transport/stdio_transport.ts`
  - Implement `ITransport` interface for stdio (JSON-RPC over stdin/stdout)
  - Use `StdioServerTransport` from @modelcontextprotocol/sdk
  - Handle message parsing and sending
  - Based on: `contracts/nodejs_server_interface.ts`, `research.md` (stdio Transport)

- [ ] **T029** [P] Implement HTTP/SSE transport
  - File: `/Users/intersystems-community/ws/rag-templates/nodejs/src/mcp/transport/http_sse_transport.ts`
  - Implement `ITransport` interface for HTTP/SSE
  - Use Express.js with Server-Sent Events
  - Handle WebSocket upgrade if needed
  - Based on: `contracts/nodejs_server_interface.ts`, `research.md` (HTTP/SSE Transport)

### Node.js Core Components (T030-T034) - Mixed parallelism

- [ ] **T030** [P] Implement Python bridge client
  - File: `/Users/intersystems-community/ws/rag-templates/nodejs/src/mcp/python_bridge_client.ts`
  - Implement `IPythonBridgeClient` interface
  - HTTP client to call Python bridge endpoints
  - Methods: `invokeTechnique()`, `listTechniques()`, `healthCheck()`, `getMetrics()`
  - Based on: `contracts/nodejs_server_interface.ts`, `research.md` (Node.js Client Code)

- [ ] **T031** [P] Implement RAG tools manager
  - File: `/Users/intersystems-community/ws/rag-templates/nodejs/src/mcp/rag_tools.ts`
  - Implement `IRAGToolsManager` interface
  - Load tool definitions from Python bridge
  - Methods: `createTools()`, `getTool()`, `validateParams()`, `getToolSchemas()`
  - Based on: `contracts/nodejs_server_interface.ts`, `contracts/mcp_tool_schema.json`

- [ ] **T032** [P] Implement connection manager
  - File: `/Users/intersystems-community/ws/rag-templates/nodejs/src/mcp/connection_manager.ts`
  - Implement `IConnectionManager` interface
  - Enforce max 5 concurrent connections (FR-032)
  - Methods: `registerConnection()`, `unregisterConnection()`, `getActiveConnectionCount()`, `isAtMaxConnections()`
  - Reject 6th connection with clear error
  - Based on: `contracts/nodejs_server_interface.ts`, `research.md` (Connection Limit)

- [ ] **T033** [P] Implement MCP server configuration
  - File: `/Users/intersystems-community/ws/rag-templates/nodejs/src/mcp/config/mcp_config.ts`
  - Load configuration from environment variables or config file
  - Support `MCPServerConfig` interface
  - Handle both transport modes, both auth modes, both deployment modes
  - Based on: `contracts/nodejs_server_interface.ts`, `research.md` (Configuration)

- [ ] **T034** Implement main MCP server
  - File: `/Users/intersystems-community/ws/rag-templates/nodejs/src/mcp/server.ts`
  - Implement `IMCPServer` interface
  - Orchestrate all components (transports, tools manager, connection manager, Python bridge client)
  - Methods: `start()`, `stop()`, `handleToolCall()`, `listTools()`, `healthCheck()`
  - Handle MCP protocol messages (tools/list, tools/call, etc.)
  - Based on: `contracts/nodejs_server_interface.ts`, `research.md` (Architecture Flow)

### MCP Protocol Integration (T035-T036) - Sequential

- [ ] **T035** Implement MCP tool execution via Python bridge
  - File: `/Users/intersystems-community/ws/rag-templates/nodejs/src/mcp/server.ts` (extend T034)
  - Handle `tools/call` MCP messages
  - Validate parameters using tool schemas
  - Call Python bridge via `PythonBridgeClient.invokeTechnique()`
  - Map Python responses to MCP protocol format
  - Based on: `research.md` (Architecture Flow)

- [ ] **T036** Implement MCP error handling and responses
  - File: `/Users/intersystems-community/ws/rag-templates/nodejs/src/mcp/server.ts` (extend T034)
  - Map Python exceptions to MCP error codes
  - Implement structured error responses (invalid_params, internal_error, etc.)
  - Return actionable error messages
  - Based on: `research.md` (Error Handling & MCP Protocol Compliance)

### Server Entry Points (T037-T038) - Can run in parallel [P]

- [ ] **T037** [P] Implement standalone server CLI
  - File: `/Users/intersystems-community/ws/rag-templates/nodejs/src/mcp/cli.ts` (new)
  - Parse command-line arguments (--transport, --port, --config)
  - Start MCP server in standalone mode
  - Handle graceful shutdown (SIGINT, SIGTERM)
  - Based on: `research.md` (Mode 1: Standalone Deployment)

- [ ] **T038** [P] Implement Python entry point for standalone mode
  - File: `/Users/intersystems-community/ws/rag-templates/iris_rag/mcp/__main__.py` (new)
  - Start Python FastAPI bridge on port 8001
  - Spawn Node.js MCP server as subprocess
  - Handle graceful shutdown
  - Based on: `research.md` (Mode 1: Standalone Deployment)

---

## Phase 3.5: Integration & Validation (T039-T048)

### Run Existing TDD Tests (T039-T041) - Sequential

- [ ] **T039** Run existing MCP TDD tests (30+ tests)
  - File: `/Users/intersystems-community/ws/rag-templates/tests/test_mcp/test_mcp_server_integration.py`
  - Run pytest on existing test suite
  - Assert all 30+ tests now PASS (were failing before implementation)
  - Fix any failures
  - Based on: spec.md FR-021

- [ ] **T040** Run MCP integration tests
  - File: `/Users/intersystems-community/ws/rag-templates/tests/test_mcp_integration.py`
  - Run pytest on existing integration tests
  - Assert tests pass with real IRIS database
  - Based on: spec.md FR-022

- [ ] **T041** Run MCP real data integration tests
  - File: `/Users/intersystems-community/ws/rag-templates/tests/test_mcp/test_mcp_real_data_integration.py`
  - Run pytest with production-like data
  - Assert realistic queries work correctly
  - Based on: spec.md FR-022

### Validate Functional Requirements (T042-T044) - Can run in parallel [P]

- [ ] **T042** [P] Validate all 6 RAG pipelines exposed as tools (FR-001)
  - Test: Call `tools/list` via MCP
  - Assert 6 RAG tools returned (basic, basic_rerank, crag, graphrag, pylate_colbert, iris_global_graphrag)
  - Assert 2 utility tools (health_check, metrics)
  - Based on: spec.md FR-001, FR-002

- [ ] **T043** [P] Validate MCP-REST API response format consistency (FR-025)
  - Test: Execute same query via MCP and REST API
  - Assert responses match byte-for-byte (excluding timestamps/IDs)
  - Test with all 6 pipelines
  - Based on: spec.md FR-025, quickstart.md Scenario 7

- [ ] **T044** [P] Validate connection limit enforcement (FR-032)
  - Test: Create 5 concurrent MCP connections (should succeed)
  - Test: Create 6th connection (should fail with "connection_limit_exceeded" error)
  - Assert error message includes max_connections=5
  - Based on: spec.md FR-032, research.md (Connection Limit)

### Performance & Load Testing (T045-T046) - Can run in parallel [P]

- [ ] **T045** [P] Performance test: Query latency < 2s (FR-010)
  - Test: Execute 100 queries via MCP (mix of all 6 pipelines)
  - Measure p95 latency
  - Assert p95 < 2000ms
  - Based on: spec.md FR-010

- [ ] **T046** [P] Load test: Concurrent query handling (FR-008)
  - Test: Submit 5 concurrent queries (at connection limit)
  - Assert all complete successfully without resource conflicts
  - Measure throughput (queries/minute)
  - Based on: spec.md FR-008, quickstart.md Scenario 5

### Error Handling Validation (T047-T048) - Can run in parallel [P]

- [ ] **T047** [P] Validate MCP error responses (FR-011)
  - Test: Invalid parameters (missing query, top_k out of range)
  - Test: Invalid technique name
  - Test: Database connection failure (simulate)
  - Assert structured MCP error responses with actionable guidance
  - Based on: spec.md FR-011, FR-014, FR-015

- [ ] **T048** [P] Validate logging of MCP tool invocations (FR-012)
  - Test: Execute queries and check logs
  - Assert logs include: query text, pipeline used, response time, success/failure
  - Assert request_id present for tracing
  - Based on: spec.md FR-012

---

## Phase 3.6: Deployment & Operations (T049-T053)

### Docker & Deployment (T049-T051) - Can run in parallel [P]

- [ ] **T049** [P] Create Dockerfile for MCP server
  - File: `/Users/intersystems-community/ws/rag-templates/Dockerfile.mcp` (new, or extend existing Dockerfile)
  - Multi-stage build: Python dependencies + Node.js dependencies
  - Support both deployment modes via MODE environment variable
  - Entrypoint: `if MODE=standalone then python -m iris_rag.mcp else uvicorn iris_rag.api.main:app`
  - Based on: research.md (Docker Configuration)

- [ ] **T050** [P] Update docker-compose.yml for MCP server
  - File: `/Users/intersystems-community/ws/rag-templates/docker-compose.yml` (modify)
  - Add `mcp-standalone` service (MODE=standalone, port 3000 for HTTP/SSE)
  - Add `api-with-mcp` service (MODE=integrated, port 8000 for REST API + MCP)
  - Include IRIS database, Redis cache
  - Based on: research.md (Docker Configuration)

- [ ] **T051** [P] Create Makefile targets for MCP server
  - File: `/Users/intersystems-community/ws/rag-templates/Makefile` (extend)
  - Targets: `mcp-run` (standalone), `mcp-run-integrated`, `mcp-stop`, `mcp-health`, `mcp-logs`
  - Target: `mcp-test` (run all MCP tests)
  - Based on: plan.md Phase 2 Task Generation Strategy

### CLI Commands (T052) - Sequential

- [ ] **T052** Implement CLI commands for MCP server management
  - File: `/Users/intersystems-community/ws/rag-templates/iris_rag/mcp/cli.py` (new)
  - Commands: `python -m iris_rag.mcp.cli start`, `stop`, `status`, `health`
  - Support both deployment modes
  - Based on: plan.md Phase 2 Task Generation Strategy

### Configuration Files (T053) - Sequential

- [ ] **T053** Create MCP server configuration file
  - File: `/Users/intersystems-community/ws/rag-templates/config/mcp_config.yaml` (new)
  - Configuration for both deployment modes, both transports, both auth modes
  - Document all options
  - Based on: research.md (Configuration)

---

## Phase 3.7: Documentation & Polish (T054-T058)

### Documentation Updates (T054-T058) - Can run in parallel [P]

- [ ] **T054** [P] Update README.md with MCP server capability (FR-016)
  - File: `/Users/intersystems-community/ws/rag-templates/README.md` (modify)
  - Add MCP server to features list
  - Add quickstart: connecting Claude Code to MCP server
  - Link to MCP quickstart guide
  - Based on: spec.md FR-016

- [ ] **T055** [P] Update CLAUDE.md with MCP development workflow (FR-017)
  - File: `/Users/intersystems-community/ws/rag-templates/CLAUDE.md` (modify - already partially updated)
  - Add MCP testing instructions
  - Add MCP debugging guide
  - Document MCP-specific make targets
  - Based on: spec.md FR-017

- [ ] **T056** [P] Create MCP quickstart guide (FR-018)
  - File: `/Users/intersystems-community/ws/rag-templates/docs/MCP_QUICKSTART.md` (new)
  - Setup instructions for both deployment modes
  - Example: connecting Claude Code via stdio
  - Example: connecting remote client via HTTP/SSE
  - Troubleshooting guide
  - Based on: spec.md FR-018

- [ ] **T057** [P] Document all 6 RAG tool schemas with examples (FR-019)
  - File: `/Users/intersystems-community/ws/rag-templates/docs/MCP_TOOL_REFERENCE.md` (new)
  - Document each tool: name, description, parameters, examples
  - Include example queries for each pipeline
  - Include example responses with all metadata fields
  - Based on: spec.md FR-019, contracts/mcp_tool_schema.json

- [ ] **T058** [P] Update architecture docs with MCP-REST API relationship (FR-020)
  - File: `/Users/intersystems-community/ws/rag-templates/docs/architecture/COMPREHENSIVE_ARCHITECTURE_OVERVIEW.md` (modify)
  - Clarify MCP server as protocol adapter over RAG framework
  - Document pipeline instance sharing between MCP and REST API
  - Include architecture diagrams
  - Based on: spec.md FR-020, research.md (Architecture Flow)

---

## Task Dependencies

### Critical Path
```
T001-T005 (Setup)
  → T006-T015 (Tests - MUST FAIL before implementation)
    → T016-T027 (Python Bridge)
      → T028-T038 (Node.js Server)
        → T039-T041 (Run existing TDD tests - MUST PASS)
          → T042-T048 (Validation & Performance)
            → T049-T053 (Deployment)
              → T054-T058 (Documentation)
```

### Detailed Dependencies
- **Setup (T001-T005)**: All [P], no dependencies
- **Contract Tests (T006-T008)**: All [P], depend on T001-T005
- **Integration Tests (T009-T015)**: All [P], depend on T001-T005
- **Python Bridge (T016-T021)**: All [P] within group, depend on T006-T008 failing
- **REST Integration (T022-T024)**: Sequential, depend on T021
- **Python Endpoints (T025-T027)**: All [P], depend on T021
- **Node.js Transports (T028-T029)**: All [P], depend on T002, T004
- **Node.js Core (T030-T034)**: Mostly [P], T034 depends on T030-T033
- **MCP Protocol (T035-T036)**: Sequential, depend on T034
- **Entry Points (T037-T038)**: All [P], depend on T034-T036
- **TDD Validation (T039-T041)**: Sequential, depend on T016-T038 complete
- **FR Validation (T042-T044)**: All [P], depend on T039-T041 passing
- **Performance (T045-T046)**: All [P], depend on T039-T041 passing
- **Error Handling (T047-T048)**: All [P], depend on T039-T041 passing
- **Deployment (T049-T053)**: T049-T051 [P], T052-T053 sequential, depend on T039-T048
- **Documentation (T054-T058)**: All [P], depend on T039-T053

---

## Parallel Execution Examples

### Example 1: Run all contract tests in parallel (T006-T008)
```python
# After setup (T001-T005) is complete
from claude_code import Task

tasks = [
    Task("Contract test for Python MCP Bridge interface in tests/contract/test_mcp_bridge_contract.py"),
    Task("Contract test for Python Technique Handlers in tests/contract/test_technique_handlers_contract.py"),
    Task("Contract test for MCP tool schemas in tests/contract/test_mcp_tool_schemas.py"),
]

# Execute all in parallel
await execute_parallel(tasks)
```

### Example 2: Run all integration tests in parallel (T009-T015)
```python
# After contract tests (T006-T008) are written and failing
tasks = [
    Task("Integration test: List available tools in tests/integration/test_mcp_list_tools.py"),
    Task("Integration test: Execute BasicRAG query in tests/integration/test_mcp_basic_rag.py"),
    Task("Integration test: CRAG with corrective measures in tests/integration/test_mcp_crag_correction.py"),
    Task("Integration test: Health check tool in tests/integration/test_mcp_health_check.py"),
    Task("Integration test: Concurrent queries in tests/integration/test_mcp_concurrent_queries.py"),
    Task("Integration test: HybridGraphRAG in tests/integration/test_mcp_hybrid_graphrag.py"),
    Task("Integration test: MCP-REST API consistency in tests/integration/test_mcp_rest_consistency.py"),
]

await execute_parallel(tasks)
```

### Example 3: Implement Python bridge core in parallel (T016-T021)
```python
# After tests (T006-T015) are written and failing
tasks = [
    Task("Implement MCP tool schema definitions in iris_rag/mcp/tool_schemas.py"),
    Task("Implement parameter validation module in iris_rag/mcp/validation.py"),
    Task("Implement MCP bridge configuration in iris_rag/mcp/config.py"),
    Task("Implement Technique Handler Registry in iris_rag/mcp/technique_handlers.py"),
    Task("Implement MCP Server Manager in iris_rag/mcp/server_manager.py"),
    Task("Implement main MCPBridge class in iris_rag/mcp/bridge.py"),
]

await execute_parallel(tasks)
```

### Example 4: Implement Node.js core components in parallel (T028-T033)
```python
# After Python bridge (T016-T027) is complete
tasks = [
    Task("Implement stdio transport in nodejs/src/mcp/transport/stdio_transport.ts"),
    Task("Implement HTTP/SSE transport in nodejs/src/mcp/transport/http_sse_transport.ts"),
    Task("Implement Python bridge client in nodejs/src/mcp/python_bridge_client.ts"),
    Task("Implement RAG tools manager in nodejs/src/mcp/rag_tools.ts"),
    Task("Implement connection manager in nodejs/src/mcp/connection_manager.ts"),
    Task("Implement MCP server configuration in nodejs/src/mcp/config/mcp_config.ts"),
]

await execute_parallel(tasks)
```

### Example 5: Run all documentation tasks in parallel (T054-T058)
```python
# After implementation and validation (T039-T053) is complete
tasks = [
    Task("Update README.md with MCP server capability"),
    Task("Update CLAUDE.md with MCP development workflow"),
    Task("Create MCP quickstart guide in docs/MCP_QUICKSTART.md"),
    Task("Document all 6 RAG tool schemas in docs/MCP_TOOL_REFERENCE.md"),
    Task("Update architecture docs with MCP-REST API relationship"),
]

await execute_parallel(tasks)
```

---

## Validation Checklist

*GATE: Checked before marking tasks.md as complete*

- [x] All contracts have corresponding tests
  - ✓ T006: Python bridge interface contract test
  - ✓ T007: Technique handlers contract test
  - ✓ T008: MCP tool schemas contract test

- [x] All entities have model/implementation tasks
  - ✓ T016: MCP Tool entity (tool_schemas.py)
  - ✓ T019: Technique Handler Registry
  - ✓ T021: MCPBridge class
  - ✓ T034: Main MCP Server
  - ✓ T027: Health Status implementation

- [x] All tests come before implementation
  - ✓ Tests (T006-T015) → Implementation (T016-T038)
  - ✓ Tests MUST FAIL before implementation starts

- [x] Parallel tasks truly independent
  - ✓ All [P] tasks operate on different files
  - ✓ No [P] task modifies same file as another [P] task

- [x] Each task specifies exact file path
  - ✓ All tasks include absolute paths or relative from repo root
  - ✓ File paths match project structure from plan.md

- [x] TDD order enforced
  - ✓ Phase 3.2 (Tests) before Phase 3.3 (Implementation)
  - ✓ T039-T041 (Run TDD tests) after implementation complete

---

## Notes

- **TDD Critical**: Tests (T006-T015) MUST be written and MUST FAIL before any implementation (T016-T038)
- **Constitution Compliance**: Minor deviation on 10K+ performance tests (documented in plan.md Complexity Tracking)
- **Pipeline Reuse**: T022 ensures MCP uses same pipeline instances as REST API (FR-006)
- **Response Format Consistency**: T043 validates MCP responses match REST API byte-for-byte (FR-025)
- **Connection Limit**: T044 enforces max 5 concurrent connections (FR-032)
- **Deployment Flexibility**: Support both standalone (T037-T038) and integrated (T024) modes
- **Transport Flexibility**: Support both stdio (T028) and HTTP/SSE (T029) transports
- **Authentication**: Configurable API key (T018, T021) or unauthenticated mode

---

## Summary

**Total Tasks**: 58
- **Setup**: 5 tasks (T001-T005)
- **Tests First (TDD)**: 10 tasks (T006-T015)
- **Python Implementation**: 12 tasks (T016-T027)
- **Node.js Implementation**: 11 tasks (T028-T038)
- **Integration & Validation**: 10 tasks (T039-T048)
- **Deployment**: 5 tasks (T049-T053)
- **Documentation**: 5 tasks (T054-T058)

**Parallel Tasks**: 42 tasks marked [P] (can run concurrently)
**Sequential Tasks**: 16 tasks (interdependent)

**Estimated Completion Time**:
- With sequential execution: ~58 hours
- With parallel execution (5 concurrent tasks): ~25-30 hours

**Ready for execution**: All tasks have clear file paths, dependencies documented, and TDD order enforced.
