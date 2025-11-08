# Implementation Plan: Complete MCP Tools Implementation

**Branch**: `043-complete-mcp-tools` | **Date**: 2025-10-18 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/Users/intersystems-community/ws/rag-templates/specs/043-complete-mcp-tools/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → ✅ Spec loaded successfully, clarifications resolved
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → ✅ All technical unknowns resolved via clarification session
   → ✅ Project Type: Single project (RAG framework with REST API)
3. Fill the Constitution Check section
   → Constitution v1.6.0 principles applied
4. Evaluate Constitution Check section
   → See Constitution Check section below
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → MCP protocol best practices, transport mechanisms, authentication
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md
   → MCP tool schemas, Python bridge interfaces, Node.js handlers
7. Re-evaluate Constitution Check section
   → Verify TDD compliance, framework-first approach
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 8. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary

Complete the Model Context Protocol (MCP) server implementation to expose all 6 RAG pipelines (BasicRAG, BasicRAGReranking, CRAG, HybridGraphRAG, PyLateColBERT, IRIS-Global-GraphRAG) as MCP tools for AI agent integration. The implementation includes both stdio and HTTP/SSE transports, configurable authentication, and two deployment modes (standalone + integrated with REST API).

**Technical Approach**:
- Implement Python bridge layer (`iris_rag/mcp/`) to interface with existing RAG pipelines
- Complete Node.js MCP server (`nodejs/src/mcp/`) for protocol handling
- Reuse REST API pipeline instances for consistency
- Support both authenticated (API keys) and unauthenticated modes
- Docker deployment matching REST API approach

## Technical Context
**Language/Version**: Python 3.11+ (backend bridge), Node.js 18+ (MCP server), TypeScript 5.0+
**Primary Dependencies**:
- Python: FastAPI (REST API integration), existing RAG pipelines, InterSystems IRIS DB API
- Node.js: @modelcontextprotocol/sdk, express (HTTP/SSE transport), stdio-jsonrpc (stdio transport)
**Storage**: InterSystems IRIS (vector database, shared with REST API pipelines)
**Testing**: pytest (Python bridge), jest (Node.js server), existing 30+ TDD tests must pass
**Target Platform**: Linux/macOS servers, Docker containers, local development (stdio)
**Project Type**: Single project (RAG framework with dual deployment modes)
**Performance Goals**: <2s query latency (p95), 5 concurrent MCP connections max, match REST API performance
**Constraints**: Reuse REST API pipeline instances, Docker deployment, stdio + HTTP/SSE transports, max 5 concurrent connections
**Scale/Scope**: 6 RAG pipelines, 2 deployment modes, 2 transports, 32 functional requirements, 30+ existing TDD tests

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**:
- ✓ Component extends established patterns (MCP protocol layer over RAG framework)
- ✓ No application-specific logic (generic MCP server for any RAG pipeline)
- ✓ CLI interface exposed (via Makefile targets and MCP CLI commands)

**II. Pipeline Validation & Requirements**:
- ✓ Automated requirement validation included (30+ TDD tests already written)
- ✓ Setup procedures idempotent (MCP server can start/stop/restart)

**III. Test-Driven Development**:
- ✓ Contract tests written before implementation (30+ tests in `tests/test_mcp/` expect failure)
- ⚠️ Performance tests for 10K+ scenarios (defer to future - focus on 6 pipeline integration first)

**IV. Performance & Enterprise Scale**:
- ✓ Incremental indexing supported (reuses existing RAG pipelines)
- ✓ IRIS vector operations optimized (leverages REST API's proven pipeline instances)

**V. Production Readiness**:
- ✓ Structured logging included (requirement FR-012)
- ✓ Health checks implemented (requirement FR-005, FR-024)
- ✓ Docker deployment ready (requirement FR-030, matches REST API)

**VI. Explicit Error Handling**:
- ✓ No silent failures (requirement FR-011, FR-014)
- ✓ Clear exception messages (MCP protocol error responses)
- ✓ Actionable error context (parameter validation, retry guidance)

**VII. Standardized Database Interfaces**:
- ✓ Uses proven SQL/vector utilities (reuses REST API pipeline instances)
- ✓ No ad-hoc IRIS queries (all DB access via existing RAG pipelines)
- ✓ New patterns contributed back (MCP bridge pattern documented)

**Constitution Compliance**: PASS ✅
- Minor deviation on performance tests (10K+ scenarios deferred)
- Justification: Focus on completing 6-pipeline MCP integration first, scale testing in follow-up

## Project Structure

### Documentation (this feature)
```
specs/043-complete-mcp-tools/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
│   ├── mcp_tool_schema.json          # MCP tool definitions for 6 RAG pipelines
│   ├── python_bridge_interface.py    # Python bridge API contract
│   ├── nodejs_server_interface.ts    # Node.js server API contract
│   └── health_metrics_schema.json    # Health check and metrics schemas
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# MCP Server Implementation (Node.js/TypeScript)
nodejs/
├── src/
│   └── mcp/
│       ├── server.ts                 # Main MCP server (stdio + HTTP/SSE transports)
│       ├── rag_tools.ts              # RAG tool manager (6 pipeline tools)
│       ├── python_bridge_client.ts   # Client to communicate with Python bridge
│       ├── transport/
│       │   ├── stdio_transport.ts    # stdio JSON-RPC transport
│       │   └── http_sse_transport.ts # HTTP/SSE transport
│       └── config/
│           └── mcp_config.ts         # MCP server configuration
└── package.json                      # Node.js dependencies (@modelcontextprotocol/sdk)

# Python MCP Bridge (Python)
iris_rag/mcp/
├── __init__.py
├── bridge.py                         # MCPBridge class - main Python bridge
├── server_manager.py                 # MCPServerManager - lifecycle management
├── technique_handlers.py             # TechniqueHandlerRegistry - pipeline adapters
├── tool_schemas.py                   # MCP tool schema definitions
├── validation.py                     # Parameter validation for MCP tools
└── config.py                         # MCP bridge configuration

# REST API Integration
iris_rag/api/
├── main.py                           # Add MCP server integration hooks
└── mcp_routes.py                     # (NEW) MCP health/status endpoints

# Test Files (TDD - already exist, expecting failures)
tests/
├── contract/
│   └── test_mcp_contracts.py         # (NEW) Contract tests for MCP interfaces
├── integration/
│   └── test_mcp_integration.py       # Existing integration tests
└── test_mcp/
    ├── test_mcp_server_integration.py # Existing 30+ TDD tests
    └── test_mcp_real_data_integration.py

# Documentation Updates
├── README.md                         # Add MCP server capability
├── CLAUDE.md                         # Add MCP development workflow
└── docs/
    └── MCP_QUICKSTART.md             # (NEW) MCP setup and usage guide
```

**Structure Decision**: Single project structure (Option 1) with Node.js MCP server layer and Python bridge integration. The MCP server acts as a protocol adapter over the existing RAG framework, maintaining the framework-first architecture. Both standalone and integrated deployment modes are supported through configuration.

## Phase 0: Outline & Research

### Research Tasks

1. **MCP Protocol Best Practices**
   - **Decision**: Use official @modelcontextprotocol/sdk from Anthropic
   - **Rationale**: Official SDK ensures protocol compliance, handles versioning, provides TypeScript types
   - **Alternatives considered**: Custom MCP implementation (rejected - reinvents wheel, harder to maintain)

2. **Transport Mechanisms (stdio vs HTTP/SSE)**
   - **Decision**: Support both stdio (for local Claude Code) and HTTP/SSE (for remote clients)
   - **Rationale**: Clarification Q4 specified "Both", stdio enables local dev, HTTP/SSE enables remote deployment
   - **Implementation**: Abstracted transport layer with unified interface

3. **Python-Node.js Bridge Architecture**
   - **Decision**: Node.js MCP server invokes Python bridge via subprocess/HTTP
   - **Rationale**: Reuses existing Python RAG pipelines without porting to Node.js
   - **Alternatives considered**:
     - Port RAG pipelines to Node.js (rejected - massive duplication, breaks REST API integration)
     - Pure Python MCP server (rejected - MCP SDK is TypeScript-first, less mature Python support)
   - **Implementation**: Python bridge exposes FastAPI endpoints, Node.js calls via HTTP

4. **Authentication Strategy**
   - **Decision**: Configurable - reuse REST API's bcrypt API keys OR unauthenticated mode
   - **Rationale**: Clarification Q2 specified "Configurable", enables local dev (no auth) + production (API keys)
   - **Implementation**: Python bridge validates API keys using REST API's AuthService

5. **Deployment Modes (Standalone vs Integrated)**
   - **Decision**: Both standalone (separate process) and integrated (within REST API server)
   - **Rationale**: Clarification Q1 specified "Both modes"
   - **Implementation**:
     - Standalone: `python -m iris_rag.mcp.server` (starts Python bridge + Node.js server)
     - Integrated: REST API's `main.py` starts MCP bridge as background service
   - **Docker**: Single Dockerfile with conditional entrypoint based on mode

6. **Pipeline Instance Reuse**
   - **Decision**: MCP bridge uses REST API's PipelineManager singleton
   - **Rationale**: FR-006 requires reusing pipeline instances for consistency
   - **Implementation**: Python bridge imports `iris_rag.api.services.PipelineManager`

7. **Connection Limit (5 max concurrent)**
   - **Decision**: Implement connection pooling with semaphore (max 5)
   - **Rationale**: Clarification Q5 specified "5" max connections
   - **Implementation**: Node.js server tracks active connections, rejects 6th with clear error

8. **Error Handling & MCP Protocol Compliance**
   - **Decision**: Use MCP protocol's structured error responses
   - **Rationale**: FR-011 requires MCP spec compliance, enables actionable errors
   - **Implementation**: Python exceptions mapped to MCP error codes (invalid_params, internal_error, etc.)

**Output**: [research.md](./research.md) - Complete research findings with all decisions documented

## Phase 1: Design & Contracts

### 1. Data Model → `data-model.md`

Extracted from spec Key Entities (lines 134-171):

**MCP Tool** (represents a RAG pipeline as MCP tool)
- `name`: string (e.g., "rag_basic", "rag_crag")
- `description`: string (human-readable pipeline capabilities)
- `schema`: MCPToolSchema (parameter definitions, validation rules)
- `pipeline_reference`: Pipeline (link to underlying RAG implementation)
- `status`: HealthStatus (healthy, degraded, unavailable)

**Tool Request** (incoming query from AI agent)
- `tool_name`: string (which RAG pipeline tool to invoke)
- `query_text`: string (user's question or search query)
- `parameters`: dict (pipeline-specific options: top_k, thresholds, etc.)
- `request_id`: UUID (unique identifier for tracing)
- `timestamp`: datetime (when request was received)

**Tool Response** (result returned to AI agent)
- `answer`: string (generated text response)
- `retrieved_documents`: List[Document] (relevant documents with scores)
- `sources`: List[string] (file names or references for citations)
- `metadata`: dict (pipeline-specific info: correction_applied, token_interactions, etc.)
- `performance_metrics`: PerformanceMetrics (execution time, token usage)
- `response_id`: UUID (unique identifier matching request)

**Pipeline Configuration** (settings for each RAG pipeline)
- `pipeline_name`: string (basic, crag, graphrag, etc.)
- `default_parameters`: dict (standard settings for queries)
- `resource_limits`: ResourceLimits (max concurrent queries, timeout settings)
- `database_connection`: IRISConnectionPool (IRIS connection pool reference)
- `health_status`: HealthStatus (current operational status)

**Health Status** (system-wide operational state)
- `overall_status`: HealthStatusEnum (healthy, degraded, unavailable)
- `pipeline_statuses`: Dict[str, HealthStatus] (individual status for each of 6 pipelines)
- `database_status`: DatabaseHealth (IRIS connection health)
- `performance_metrics`: PerformanceMetrics (recent response times, error rates)
- `last_updated`: datetime (timestamp of health check)

### 2. API Contracts → `/contracts/`

**MCP Tool Schemas** (`contracts/mcp_tool_schema.json`):
```json
{
  "tools": [
    {
      "name": "rag_basic",
      "description": "Basic RAG with vector similarity search",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "User's question"},
          "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50}
        },
        "required": ["query"]
      }
    },
    {
      "name": "rag_basic_rerank",
      "description": "Vector search with cross-encoder reranking",
      "inputSchema": { /* similar to rag_basic with reranking params */ }
    },
    {
      "name": "rag_crag",
      "description": "Corrective RAG with self-evaluation",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": {"type": "string"},
          "top_k": {"type": "integer", "default": 5},
          "confidence_threshold": {"type": "number", "default": 0.8},
          "correction_strategy": {"type": "string", "enum": ["rewrite", "web_search"]}
        },
        "required": ["query"]
      }
    },
    {
      "name": "rag_graphrag",
      "description": "Hybrid search (vector + text + graph + RRF)",
      "inputSchema": { /* vector + graph params */ }
    },
    {
      "name": "rag_pylate_colbert",
      "description": "ColBERT late interaction retrieval",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": {"type": "string"},
          "top_k": {"type": "integer", "default": 5},
          "interaction_threshold": {"type": "number", "default": 0.5}
        },
        "required": ["query"]
      }
    },
    {
      "name": "rag_health_check",
      "description": "Check health of all RAG pipelines",
      "inputSchema": {
        "type": "object",
        "properties": {
          "include_details": {"type": "boolean", "default": false}
        }
      }
    }
  ]
}
```

**Python Bridge Interface** (`contracts/python_bridge_interface.py`):
```python
from typing import Dict, Any, List
from abc import ABC, abstractmethod

class IMCPBridge(ABC):
    """Interface for Python MCP bridge."""

    @abstractmethod
    async def invoke_technique(
        self,
        technique: str,
        query: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Invoke a RAG technique.

        Args:
            technique: Pipeline name (basic, crag, graphrag, etc.)
            query: User's question
            params: Pipeline-specific parameters

        Returns:
            {
                "success": bool,
                "result": {
                    "answer": str,
                    "retrieved_documents": List[Dict],
                    "sources": List[str],
                    "metadata": Dict,
                    "performance": Dict
                },
                "error": Optional[str]
            }
        """
        pass

    @abstractmethod
    async def get_available_techniques(self) -> List[str]:
        """Get list of available RAG techniques."""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all pipelines and database."""
        pass
```

**Node.js Server Interface** (`contracts/nodejs_server_interface.ts`):
```typescript
interface MCPServerConfig {
  name: string;
  description: string;
  version: string;
  enabled_techniques: string[];
  transport: 'stdio' | 'http' | 'both';
  http_port?: number;
  auth_mode: 'api_key' | 'none';
  max_connections: number;
}

interface IMCPServer {
  start(): Promise<void>;
  stop(): Promise<void>;
  handleToolCall(toolName: string, params: any): Promise<MCPResponse>;
  listTools(): Promise<MCPTool[]>;
  healthCheck(): Promise<HealthStatus>;
}
```

### 3. Contract Tests → `tests/contract/test_mcp_contracts.py`

```python
import pytest
from iris_rag.mcp.bridge import MCPBridge
from iris_rag.mcp.server_manager import MCPServerManager

class TestMCPBridgeContract:
    """Contract tests for Python MCP bridge (TDD - should fail initially)."""

    def test_bridge_implements_interface(self):
        """Verify MCPBridge implements IMCPBridge interface."""
        bridge = MCPBridge()
        assert hasattr(bridge, 'invoke_technique')
        assert hasattr(bridge, 'get_available_techniques')
        assert hasattr(bridge, 'health_check')

    @pytest.mark.asyncio
    async def test_invoke_technique_returns_standard_response(self):
        """Verify invoke_technique returns standardized response."""
        bridge = MCPBridge()
        result = await bridge.invoke_technique('basic', 'test query', {})

        assert 'success' in result
        assert 'result' in result or 'error' in result
        if result['success']:
            assert 'answer' in result['result']
            assert 'retrieved_documents' in result['result']
            assert 'sources' in result['result']
            assert 'metadata' in result['result']
            assert 'performance' in result['result']
```

### 4. Integration Test Scenarios → `quickstart.md`

Based on acceptance scenarios (spec lines 56-72):

**Scenario 1**: List available tools
- Connect to MCP server
- Call `list_tools()`
- Verify 6 RAG pipeline tools returned
- Verify schemas include parameter descriptions

**Scenario 2**: Execute BasicRAG query
- Call `rag_basic` tool with query "What are the symptoms of diabetes?" and top_k=5
- Verify response has answer, retrieved_documents, sources
- Verify performance metrics included

**Scenario 3**: CRAG with corrective measures
- Call `rag_crag` tool with low-confidence query
- Verify correction_applied in metadata
- Verify confidence_score returned

**Scenario 4**: Health check tool
- Call `rag_health_check` tool
- Verify status of all 6 pipelines
- Verify database connectivity reported
- Verify performance metrics included

**Scenario 5**: Concurrent queries
- Submit concurrent requests to BasicRAG, CRAG, HybridGraphRAG
- Verify all complete successfully
- Verify no resource conflicts (5 max concurrent)

### 5. Update CLAUDE.md

Will run `.specify/scripts/bash/update-agent-context.sh claude` to add MCP context:
- MCP server capability (stdio + HTTP/SSE transports)
- Python bridge architecture (`iris_rag/mcp/`)
- Node.js server (`nodejs/src/mcp/`)
- MCP tool schemas for 6 RAG pipelines
- Deployment modes (standalone vs integrated)
- Testing approach (30+ TDD tests)

**Output**:
- `data-model.md` - Entity definitions and relationships
- `contracts/mcp_tool_schema.json` - MCP tool definitions
- `contracts/python_bridge_interface.py` - Python bridge contract
- `contracts/nodejs_server_interface.ts` - Node.js server contract
- `contracts/health_metrics_schema.json` - Health check schemas
- `tests/contract/test_mcp_contracts.py` - Contract tests (failing)
- `quickstart.md` - Integration test scenarios
- `CLAUDE.md` - Updated with MCP context

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:

1. **Load base template**: `.specify/templates/tasks-template.md`

2. **Generate contract test tasks** (from Phase 1 contracts):
   - T001: [P] Create contract test for MCPBridge interface
   - T002: [P] Create contract test for MCPServerManager lifecycle
   - T003: [P] Create contract test for TechniqueHandlerRegistry
   - T004: [P] Create contract test for each MCP tool schema (6 tools)

3. **Generate Python implementation tasks** (from data-model.md):
   - T010: [P] Implement MCPBridge class (iris_rag/mcp/bridge.py)
   - T011: [P] Implement MCPServerManager class (iris_rag/mcp/server_manager.py)
   - T012: [P] Implement TechniqueHandlerRegistry (iris_rag/mcp/technique_handlers.py)
   - T013: [P] Implement tool schema definitions (iris_rag/mcp/tool_schemas.py)
   - T014: [P] Implement parameter validation (iris_rag/mcp/validation.py)
   - T015: Integrate with REST API PipelineManager (reuse pipeline instances)

4. **Generate Node.js implementation tasks**:
   - T020: [P] Implement stdio transport (nodejs/src/mcp/transport/stdio_transport.ts)
   - T021: [P] Implement HTTP/SSE transport (nodejs/src/mcp/transport/http_sse_transport.ts)
   - T022: [P] Implement Python bridge client (nodejs/src/mcp/python_bridge_client.ts)
   - T023: [P] Implement RAG tools manager (nodejs/src/mcp/rag_tools.ts)
   - T024: Implement main MCP server (nodejs/src/mcp/server.ts) - orchestrates all above
   - T025: Implement connection pooling (max 5 concurrent)

5. **Generate integration test tasks** (from quickstart.md scenarios):
   - T030: Integration test - list available tools
   - T031: Integration test - BasicRAG query execution
   - T032: Integration test - CRAG corrective measures
   - T033: Integration test - health check tool
   - T034: Integration test - concurrent queries (5 max)
   - T035: Integration test - HybridGraphRAG with graph traversal

6. **Generate deployment tasks**:
   - T040: [P] Create Dockerfile with dual deployment modes
   - T041: [P] Update docker-compose.yml with MCP service
   - T042: [P] Create Makefile targets for MCP server (mcp-run, mcp-stop, mcp-health)
   - T043: Implement CLI commands (python -m iris_rag.mcp.server)

7. **Generate documentation tasks**:
   - T050: [P] Update README.md with MCP server capability (FR-016)
   - T051: [P] Update CLAUDE.md with MCP development workflow (FR-017)
   - T052: [P] Create MCP quickstart guide (docs/MCP_QUICKSTART.md) (FR-018)
   - T053: [P] Document all 6 RAG tool schemas with examples (FR-019)
   - T054: [P] Update architecture docs with MCP-REST API relationship (FR-020)

8. **Verification tasks** (run existing 30+ TDD tests):
   - T060: Run tests/test_mcp/test_mcp_server_integration.py (30+ tests)
   - T061: Run tests/test_mcp_integration.py
   - T062: Run tests/test_mcp/test_mcp_real_data_integration.py
   - T063: Validate all 32 functional requirements met

**Ordering Strategy**:
- **TDD order**: Contract tests (T001-T004) → Implementation (T010-T025) → Integration tests (T030-T035)
- **Dependency order**:
  1. Python bridge first (T010-T015) - provides backend for Node.js to call
  2. Node.js server second (T020-T025) - consumes Python bridge
  3. Integration tests third (T030-T035) - validates end-to-end
  4. Deployment (T040-T043) - packages everything
  5. Documentation (T050-T054) - final polish
- **Parallel execution**: Mark [P] for tasks in same layer (e.g., all contract tests can run in parallel)

**Estimated Output**: 35-40 numbered, dependency-ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Performance tests for 10K+ scenarios deferred | Focus on 6-pipeline integration first | 10K testing requires significant test data setup; deferring to follow-up ensures core MCP functionality complete first |

**Justification**: The constitutional principle of TDD performance testing for 10K+ scenarios is being partially deferred. The MCP implementation will:
1. ✅ Reuse existing REST API pipelines (already tested at scale)
2. ✅ Pass 30+ existing TDD tests for MCP integration
3. ✅ Validate concurrency (5 max connections)
4. ⏭️ Defer dedicated MCP 10K performance tests to future work

This is acceptable because the MCP server is a protocol adapter over already-validated RAG pipelines, not a new RAG implementation.

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS (minor deviation documented)
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved (5 questions answered in clarification session)
- [x] Complexity deviations documented (10K performance tests deferred)

---
*Based on Constitution v1.6.0 - See `/.specify/memory/constitution.md`*
