# Feature Specification: Complete MCP Tools Implementation

**Feature Branch**: `043-complete-mcp-tools`
**Created**: 2025-10-18
**Status**: Draft
**Input**: User description: "Complete MCP Tools implementation and documentation, updating to currently available pipelines, and integrate with REST API architecture"

## Execution Flow (main)
```
1. Parse user description from Input
   → Feature requires completing MCP server implementation
2. Extract key concepts from description
   → Actors: AI agents, developers, Claude Code users
   → Actions: Execute RAG queries via MCP protocol, manage tools
   → Data: Query results, pipeline configurations, performance metrics
   → Constraints: Must align with existing 6 RAG pipelines, integrate with REST API
3. For each unclear aspect:
   → MCP server deployment model marked below
   → Authentication strategy marked below
4. Fill User Scenarios & Testing section
   → Primary: AI agent queries RAG pipelines via MCP
5. Generate Functional Requirements
   → All requirements testable via existing TDD test suite
6. Identify Key Entities
   → MCP Tools, Pipeline Configurations, Query Results
7. Run Review Checklist
   → [NEEDS CLARIFICATION] on deployment and auth marked
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

---

## Clarifications

### Session 2025-10-18
- Q: Should the MCP server run as a standalone service, integrated within the REST API server, or support both deployment modes? → A: Both modes
- Q: What authentication mechanism should the MCP server use? → A: Configurable (support both authenticated API keys and unauthenticated modes)
- Q: What production deployment packaging should the MCP server support? → A: Match REST API (same deployment approach)
- Q: What MCP protocol transport mechanism should be supported? → A: Both (stdio and HTTP/SSE)
- Q: What is the maximum number of concurrent MCP client connections the server should support per deployment instance? → A: 5

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As an AI agent developer using Claude Code, I want to query RAG pipelines through the Model Context Protocol so that I can access retrieval-augmented generation capabilities without managing direct API integrations, enabling my AI agents to retrieve contextual information from the 6 available RAG pipelines (BasicRAG, BasicRAGReranking, CRAG, HybridGraphRAG, PyLateColBERT, IRIS-Global-GraphRAG) seamlessly.

### Acceptance Scenarios

1. **Given** an AI agent is connected to the MCP server, **When** the agent requests a list of available RAG tools, **Then** the system returns all 6 RAG pipeline tools with their descriptions and parameter schemas

2. **Given** an AI agent has selected the BasicRAG tool, **When** the agent submits a query "What are the symptoms of diabetes?" with top_k=5, **Then** the system returns relevant documents and a generated answer within the expected response time

3. **Given** an AI agent is using the CRAG tool, **When** the retrieved documents have low confidence scores, **Then** the system applies corrective measures and returns improved results with metadata indicating correction was applied

4. **Given** a developer wants to test the MCP server, **When** they run the health check tool, **Then** the system reports the status of all 6 pipelines, database connectivity, and performance metrics

5. **Given** multiple AI agents are querying different pipelines concurrently, **When** requests are distributed across BasicRAG, CRAG, and HybridGraphRAG simultaneously, **Then** all queries complete successfully without resource conflicts

6. **Given** an AI agent submits a query to HybridGraphRAG, **When** the query requires graph traversal plus vector search, **Then** the system returns results from both retrieval methods with proper ranking

7. **Given** a developer is integrating MCP with the existing REST API, **When** a query is submitted via MCP, **Then** the same pipeline instances and configuration used by the REST API are utilized to ensure consistency

8. **Given** documentation is updated, **When** a new user reads the main README.md, **Then** the MCP server capability is clearly listed as a feature with links to setup instructions

### Edge Cases

- What happens when an AI agent requests a non-existent pipeline tool? **Then** the system returns a clear error message listing available tools

- How does the system handle queries that exceed token limits? **Then** the system truncates or rejects the query with guidance on maximum allowed length

- What happens when the IRIS database connection is lost during a query? **Then** the system returns an error with retry guidance and marks the pipeline as unhealthy in status checks

- How does the system handle malformed MCP tool requests? **Then** parameter validation catches errors before execution and returns structured error responses

- What happens when multiple HybridGraphRAG queries require extensive graph traversal simultaneously? **Then** the system manages resource allocation to prevent timeout while maintaining acceptable response times for all requests

## Requirements *(mandatory)*

### Functional Requirements

#### Core MCP Server Capabilities
- **FR-001**: System MUST expose all 6 currently implemented RAG pipelines (BasicRAG, BasicRAGReranking, CRAG, HybridGraphRAG, PyLateColBERT, IRIS-Global-GraphRAG) as MCP tools
- **FR-002**: System MUST provide tool discovery allowing AI agents to query available RAG pipeline tools and their schemas
- **FR-003**: System MUST validate all tool parameters before executing RAG queries to prevent invalid requests
- **FR-004**: System MUST return standardized responses matching the existing REST API response format for consistency
- **FR-005**: System MUST provide a health check tool reporting status of all pipelines, database connectivity, and system metrics
- **FR-031**: System MUST support both stdio transport (for local MCP clients) and HTTP/SSE transport (for remote MCP clients)

#### Pipeline Integration
- **FR-006**: System MUST reuse existing pipeline instances from the REST API implementation to maintain configuration consistency
- **FR-007**: System MUST support all pipeline-specific parameters (e.g., confidence_threshold for CRAG, interaction_threshold for ColBERT)
- **FR-008**: System MUST handle concurrent requests across different pipelines without resource conflicts
- **FR-009**: System MUST return pipeline-specific metadata (e.g., correction_applied for CRAG, token_interactions for ColBERT)
- **FR-010**: System MUST maintain the same performance characteristics as the REST API (sub-2s query latency for p95)

#### Error Handling & Monitoring
- **FR-011**: System MUST return structured error messages following MCP protocol specifications when queries fail
- **FR-012**: System MUST log all MCP tool invocations with query text, pipeline used, response time, and success/failure status
- **FR-013**: System MUST provide performance metrics accessible via a dedicated metrics tool
- **FR-014**: System MUST gracefully handle database connection failures and return actionable error messages
- **FR-015**: System MUST validate query length limits and return clear guidance when exceeded

#### Documentation & Discoverability
- **FR-016**: System MUST update main README.md to include MCP server as a core feature
- **FR-017**: System MUST update CLAUDE.md developer guide with MCP development workflow and testing instructions
- **FR-018**: System MUST provide setup instructions for running the MCP server alongside the REST API
- **FR-019**: System MUST document all 6 RAG tool schemas with parameter descriptions and examples
- **FR-020**: System MUST clarify the relationship between MCP server and REST API in architecture documentation

#### Testing & Validation
- **FR-021**: System MUST pass all existing TDD tests (30+ tests) covering tool initialization, parameter validation, and error handling
- **FR-022**: System MUST verify that all 6 pipeline tools execute successfully with sample queries
- **FR-023**: System MUST validate concurrent request handling under load
- **FR-024**: System MUST confirm health check tool returns accurate status for all components
- **FR-025**: System MUST ensure MCP responses match REST API response format byte-for-byte for the same queries

#### Deployment & Operations
- **FR-026**: System MUST support both standalone MCP server deployment (separate process) and integrated deployment (within REST API server process)
- **FR-027**: System MUST provide configurable authentication supporting both authenticated mode (reusing REST API's bcrypt-hashed API keys) and unauthenticated mode (for local development)
- **FR-028**: System MUST specify startup/shutdown procedures for both standalone and integrated MCP server modes
- **FR-029**: System MUST document resource requirements (CPU, memory, connections) for running MCP server in both deployment modes
- **FR-030**: System MUST use the same production deployment approach as the existing REST API (Docker containerization with docker-compose support)
- **FR-032**: System MUST support a maximum of 5 concurrent MCP client connections per deployment instance

### Key Entities *(mandatory - feature involves data)*

- **MCP Tool**: Represents a RAG pipeline exposed via Model Context Protocol
  - Name: Unique identifier (e.g., "rag_basic", "rag_crag")
  - Description: Human-readable description of pipeline capabilities
  - Schema: Parameter definitions and validation rules
  - Pipeline Reference: Link to underlying RAG pipeline implementation
  - Status: Availability status (healthy, degraded, unavailable)

- **Tool Request**: Represents an incoming query from an AI agent
  - Tool Name: Which RAG pipeline tool to invoke
  - Query Text: User's question or search query
  - Parameters: Pipeline-specific options (top_k, thresholds, etc.)
  - Request ID: Unique identifier for tracing
  - Timestamp: When request was received

- **Tool Response**: Represents the result returned to the AI agent
  - Answer: Generated text response
  - Retrieved Documents: List of relevant documents with scores
  - Sources: File names or references for citations
  - Metadata: Pipeline-specific information (correction applied, interaction scores, etc.)
  - Performance Metrics: Execution time, token usage
  - Response ID: Unique identifier matching request

- **Pipeline Configuration**: Settings for each RAG pipeline
  - Pipeline Name: Which RAG technique (basic, crag, graphrag, etc.)
  - Default Parameters: Standard settings for queries
  - Resource Limits: Max concurrent queries, timeout settings
  - Database Connection: IRIS connection pool reference
  - Health Status: Current operational status

- **Health Status**: System-wide operational state
  - Overall Status: Healthy, degraded, or unavailable
  - Pipeline Statuses: Individual status for each of 6 pipelines
  - Database Status: IRIS connection health
  - Performance Metrics: Recent response times, error rates
  - Last Updated: Timestamp of health check

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain (all clarified in session 2025-10-18)
- [x] Requirements are testable and unambiguous (via existing TDD test suite)
- [x] Success criteria are measurable (30+ tests, 6 pipelines, response format matching, 5 max connections)
- [x] Scope is clearly bounded (6 existing pipelines, both deployment modes, stdio + HTTP/SSE transports)
- [x] Dependencies and assumptions identified (REST API integration, existing pipelines, Docker deployment)

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted (MCP protocol, 6 RAG pipelines, REST API integration)
- [x] Ambiguities marked and resolved (5 clarifications provided)
- [x] User scenarios defined (8 acceptance scenarios, 5 edge cases)
- [x] Requirements generated (32 functional requirements)
- [x] Entities identified (5 key entities)
- [x] Review checklist passed (all clarifications resolved)

---

## Additional Context

### Relationship to Existing Architecture

This feature completes work already in progress:
- **Architecture**: 492-line comprehensive architecture document exists
- **Tests**: 30+ TDD tests already written (expected to fail until implementation)
- **Infrastructure**: Node.js MCP server directories and dependencies set up
- **Gap**: Python bridge layer (`iris_rag/mcp/`) exists but empty - needs implementation

### Integration Points

- **REST API**: Must reuse same pipeline instances, configuration, and response formats
- **Existing Pipelines**: All 6 pipelines already production-ready via REST API
- **Documentation**: README.md and CLAUDE.md need updates to reflect MCP capability
- **Testing**: Existing TDD test suite provides acceptance criteria

### Out of Scope

This feature does NOT include:
- Creating new RAG pipelines (uses existing 6 only)
- Implementing HyDE, NodeRAG, or SQLRAG (mentioned in tests but not in main pipeline list)
- Modifying existing REST API endpoints
- Changing pipeline behavior or response formats
- Creating new authentication systems beyond what REST API uses

---
