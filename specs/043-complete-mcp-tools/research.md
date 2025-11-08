# Research: MCP Tools Implementation

**Feature**: Complete MCP Tools Implementation
**Branch**: 043-complete-mcp-tools
**Date**: 2025-10-18

## Research Objectives

Resolve all technical unknowns to enable confident implementation of MCP server for 6 RAG pipelines with dual deployment modes and dual transport mechanisms.

## 1. MCP Protocol Best Practices

### Decision
Use official **@modelcontextprotocol/sdk** from Anthropic for Node.js MCP server implementation.

### Rationale
- **Protocol compliance**: Official SDK ensures adherence to MCP specification
- **Version management**: Handles protocol versioning automatically
- **TypeScript support**: Provides type definitions for MCP messages, tools, resources
- **Community adoption**: Used by Anthropic's own tools (Claude Desktop, etc.)
- **Maintenance**: Actively maintained by protocol authors

### Alternatives Considered
1. **Custom MCP implementation**
   - Rejected: Reinvents wheel, harder to maintain protocol updates
   - Risks: Protocol drift, incompatibility with MCP clients

2. **Python MCP SDK** (if mature enough)
   - Rejected: Python support less mature than TypeScript
   - Node.js SDK is reference implementation

### Implementation Details
```typescript
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { SSEServerTransport } from '@modelcontextprotocol/sdk/server/sse.js';
```

### References
- MCP SDK: https://github.com/modelcontextprotocol/sdk
- MCP Specification: https://spec.modelcontextprotocol.io/

---

## 2. Transport Mechanisms (stdio vs HTTP/SSE)

### Decision
Support **both stdio (JSON-RPC over stdin/stdout) and HTTP/SSE** transports with abstracted transport layer.

### Rationale
- **Clarification requirement**: User specified "Both" in Q4
- **stdio use case**: Local development with Claude Code (primary use case)
- **HTTP/SSE use case**: Remote deployment, multi-client scenarios, web integrations
- **Abstraction benefit**: Unified server logic regardless of transport

### stdio Transport Details
- **Protocol**: JSON-RPC 2.0 over stdin/stdout
- **Use case**: `claude-code` connects to MCP server via subprocess
- **Implementation**: `StdioServerTransport` from @modelcontextprotocol/sdk
- **Example**:
  ```bash
  # Claude Code invokes MCP server
  node nodejs/src/mcp/server.js --transport stdio
  ```

### HTTP/SSE Transport Details
- **Protocol**: Server-Sent Events over HTTP
- **Use case**: Remote clients, web applications, multi-user scenarios
- **Implementation**: Express.js server with SSE endpoint
- **Example**:
  ```bash
  # Start HTTP/SSE server on port 3000
  node nodejs/src/mcp/server.js --transport http --port 3000
  ```

### Transport Abstraction
```typescript
interface ITransport {
  start(): Promise<void>;
  stop(): Promise<void>;
  onMessage(handler: (message: MCPMessage) => void): void;
  sendMessage(message: MCPMessage): Promise<void>;
}

class StdioTransport implements ITransport { /* ... */ }
class HttpSseTransport implements ITransport { /* ... */ }
```

### Configuration
```yaml
mcp_server:
  transport: both  # or 'stdio' or 'http'
  http_port: 3000  # only used if transport includes http
```

---

## 3. Python-Node.js Bridge Architecture

### Decision
**Node.js MCP server invokes Python bridge via HTTP API** (FastAPI service).

### Rationale
- **Reuse existing pipelines**: All 6 RAG pipelines are Python-based
- **REST API integration**: Python bridge can import and reuse REST API's `PipelineManager`
- **No code duplication**: Avoids porting RAG pipelines to Node.js
- **Clear separation**: Node.js handles MCP protocol, Python handles RAG execution
- **Performance**: HTTP overhead negligible compared to LLM query latency (1-2 seconds)

### Alternatives Considered
1. **Port RAG pipelines to Node.js**
   - Rejected: Massive duplication (6 pipelines × hundreds of lines each)
   - Breaks REST API integration requirement (FR-006)
   - Maintenance nightmare (two implementations drift)

2. **Pure Python MCP server**
   - Rejected: MCP SDK is TypeScript-first
   - Python MCP support less mature
   - Would miss out on official SDK benefits

3. **Python subprocess invocation**
   - Considered: Node.js spawns Python scripts
   - Rejected: Complex state management, slow process startup
   - HTTP API is cleaner, supports connection pooling

### Architecture Flow
```
MCP Client (Claude Code)
    ↓ (stdio JSON-RPC)
Node.js MCP Server (nodejs/src/mcp/server.ts)
    ↓ (HTTP POST)
Python MCP Bridge (iris_rag/mcp/bridge.py - FastAPI)
    ↓ (function call)
REST API PipelineManager (iris_rag/api/services.py)
    ↓ (function call)
RAG Pipelines (iris_rag/pipelines/)
    ↓ (SQL/vector ops)
InterSystems IRIS Database
```

### Python Bridge API
```python
# FastAPI endpoints exposed by Python bridge
POST /mcp/invoke_technique
  Body: {"technique": "basic", "query": "...", "params": {...}}
  Response: {"success": true, "result": {...}}

GET /mcp/list_techniques
  Response: ["basic", "basic_rerank", "crag", "graphrag", "pylate_colbert", "iris_global_graphrag"]

GET /mcp/health_check
  Response: {"status": "healthy", "pipelines": {...}, "database": {...}}
```

### Node.js Client Code
```typescript
class PythonBridgeClient {
  private baseUrl: string = 'http://localhost:8001';  // Python bridge port

  async invokeTechnique(technique: string, query: string, params: any) {
    const response = await fetch(`${this.baseUrl}/mcp/invoke_technique`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ technique, query, params })
    });
    return response.json();
  }
}
```

---

## 4. Authentication Strategy

### Decision
**Configurable authentication** - support both API key mode (production) and unauthenticated mode (development).

### Rationale
- **Clarification requirement**: User specified "Configurable" in Q2
- **Development convenience**: Local dev with Claude Code doesn't need auth
- **Production security**: Remote deployments require API keys
- **REST API integration**: Reuse existing bcrypt-hashed API key system (FR-027)

### Authentication Modes

#### Mode 1: Unauthenticated (Development)
```yaml
mcp_server:
  auth_mode: none
```
- No API key validation
- Suitable for local stdio transport with Claude Code
- Not recommended for HTTP/SSE transport

#### Mode 2: API Key Authentication (Production)
```yaml
mcp_server:
  auth_mode: api_key
```
- Reuses REST API's `AuthService` (bcrypt-hashed keys)
- Same API keys work for both REST API and MCP server
- Validation happens in Python bridge before pipeline execution

### Implementation
```python
# Python bridge (iris_rag/mcp/bridge.py)
class MCPBridge:
    def __init__(self, auth_mode: str = 'none'):
        self.auth_mode = auth_mode
        if auth_mode == 'api_key':
            from iris_rag.api.services import AuthService
            self.auth_service = AuthService(...)

    async def invoke_technique(self, technique, query, params, api_key=None):
        if self.auth_mode == 'api_key':
            if not api_key or not await self.auth_service.validate_key(api_key):
                return {"success": False, "error": "Invalid API key"}
        # ... proceed with technique invocation
```

### MCP Protocol Integration
```typescript
// Node.js MCP server passes API key to Python bridge
const result = await pythonBridge.invokeTechnique(technique, query, params, {
  headers: { 'Authorization': `Bearer ${apiKey}` }
});
```

---

## 5. Deployment Modes (Standalone vs Integrated)

### Decision
Support **both standalone deployment (separate process) and integrated deployment (within REST API server)**.

### Rationale
- **Clarification requirement**: User specified "Both modes" in Q1
- **Standalone benefits**: Independent scaling, simpler dev workflow
- **Integrated benefits**: Shared resources, single deployment, consistent configuration
- **Docker packaging**: Single Dockerfile with conditional entrypoint

### Mode 1: Standalone Deployment
```bash
# Start Python bridge + Node.js server as separate process
python -m iris_rag.mcp.server
```

**Process architecture**:
- Python script starts FastAPI bridge on port 8001
- Python script spawns Node.js MCP server as subprocess
- Node.js server connects to Python bridge via HTTP
- MCP server listens on stdio or HTTP/SSE

**Use cases**:
- Development with Claude Code
- Dedicated MCP server deployment
- Testing MCP in isolation

### Mode 2: Integrated Deployment (within REST API)
```bash
# Start REST API with embedded MCP server
uvicorn iris_rag.api.main:app --host 0.0.0.0 --port 8000
```

**Process architecture**:
- REST API's `main.py` starts MCP bridge as background task (asyncio)
- Node.js MCP server started as subprocess by REST API
- MCP server shares REST API's `PipelineManager` singleton
- Single health check endpoint reports both REST API and MCP status

**Use cases**:
- Production deployment (single container)
- Unified monitoring and logging
- Consistent configuration across API and MCP

### Docker Configuration
```dockerfile
# Dockerfile supports both modes via ENTRYPOINT
FROM python:3.11-slim

# Install Python dependencies
RUN pip install -e .

# Install Node.js for MCP server
RUN apt-get update && apt-get install -y nodejs npm
RUN cd nodejs && npm install

# Conditional entrypoint based on MODE env var
ENTRYPOINT ["sh", "-c", "if [ \"$MODE\" = 'standalone' ]; then python -m iris_rag.mcp.server; else uvicorn iris_rag.api.main:app --host 0.0.0.0 --port 8000; fi"]
```

```yaml
# docker-compose.yml supports both modes
services:
  mcp-standalone:
    image: rag-templates:latest
    environment:
      - MODE=standalone
    ports:
      - "3000:3000"  # HTTP/SSE transport

  api-with-mcp:
    image: rag-templates:latest
    environment:
      - MODE=integrated
    ports:
      - "8000:8000"  # REST API + MCP
```

---

## 6. Pipeline Instance Reuse

### Decision
**MCP bridge imports and reuses REST API's `PipelineManager` singleton** to maintain configuration consistency.

### Rationale
- **Functional requirement**: FR-006 requires reusing pipeline instances
- **Configuration consistency**: Same YAML configs, same LLM settings, same IRIS connection
- **Performance**: Avoid duplicate pipeline initialization (saves memory)
- **Response format consistency**: FR-004 requires matching REST API responses

### Implementation
```python
# iris_rag/mcp/bridge.py
from iris_rag.api.services import PipelineManager

class MCPBridge:
    def __init__(self):
        # Reuse REST API's PipelineManager
        self.pipeline_manager = PipelineManager.get_instance()

    async def invoke_technique(self, technique: str, query: str, params: dict):
        # Get pipeline from shared manager
        pipeline = self.pipeline_manager.get_pipeline(technique)

        # Execute query (same as REST API)
        result = pipeline.query(query=query, **params)

        # Return in MCP-compatible format (matches REST API response)
        return {
            "success": True,
            "result": {
                "answer": result["answer"],
                "retrieved_documents": result["retrieved_documents"],
                "sources": result["sources"],
                "metadata": result.get("metadata", {}),
                "performance": {
                    "execution_time_ms": result.get("execution_time_ms"),
                    "tokens_used": result.get("tokens_used")
                }
            }
        }
```

### Shared Configuration
```yaml
# config/pipelines.yaml (used by both REST API and MCP)
pipelines:
  basic:
    class: BasicRAGPipeline
    config:
      embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
      llm: "gpt-4"
      top_k: 5

  crag:
    class: CRAGPipeline
    config:
      confidence_threshold: 0.8
      correction_strategy: "rewrite"
```

---

## 7. Connection Limit (5 max concurrent)

### Decision
Implement **connection pooling with semaphore** limiting max 5 concurrent MCP client connections.

### Rationale
- **Clarification requirement**: User specified "5" in Q5
- **Resource management**: Prevents overload on shared pipeline instances
- **Fair access**: FIFO queue for connections waiting for slot
- **Clear error**: 6th connection receives immediate rejection with guidance

### Implementation
```typescript
// nodejs/src/mcp/server.ts
class MCPServer {
  private maxConnections = 5;
  private activeConnections = 0;
  private connectionQueue: Array<() => void> = [];

  async handleNewConnection(transport: ITransport) {
    if (this.activeConnections >= this.maxConnections) {
      // Reject 6th connection
      transport.sendMessage({
        error: {
          code: 'connection_limit_exceeded',
          message: `Maximum ${this.maxConnections} concurrent connections. Please try again later.`,
          data: {
            max_connections: this.maxConnections,
            current_connections: this.activeConnections,
            retry_after_seconds: 30
          }
        }
      });
      transport.close();
      return;
    }

    this.activeConnections++;
    try {
      await this.handleConnection(transport);
    } finally {
      this.activeConnections--;
      // Process next in queue if any
      const next = this.connectionQueue.shift();
      if (next) next();
    }
  }
}
```

### Configuration
```yaml
mcp_server:
  max_connections: 5  # Constitutional limit from clarification
```

---

## 8. Error Handling & MCP Protocol Compliance

### Decision
Map Python exceptions to **MCP protocol's structured error responses** with actionable guidance.

### Rationale
- **Functional requirement**: FR-011 requires MCP spec compliance
- **User experience**: Clear error messages enable self-service debugging
- **Error codes**: Standard MCP error codes (invalid_params, internal_error, etc.)
- **Constitutional principle**: Explicit error handling (no silent failures)

### MCP Error Response Format
```json
{
  "error": {
    "code": "invalid_params",
    "message": "Parameter 'top_k' out of range",
    "data": {
      "field": "top_k",
      "rejected_value": 100,
      "valid_range": [1, 50],
      "suggestion": "Use top_k between 1 and 50"
    }
  }
}
```

### Python Exception Mapping
```python
# iris_rag/mcp/bridge.py
class MCPBridge:
    def map_exception_to_mcp_error(self, exc: Exception) -> dict:
        if isinstance(exc, ValidationError):
            return {
                "code": "invalid_params",
                "message": str(exc),
                "data": {
                    "field": exc.field,
                    "rejected_value": exc.value,
                    "valid_range": exc.valid_range
                }
            }
        elif isinstance(exc, DatabaseConnectionError):
            return {
                "code": "service_unavailable",
                "message": "Database connection failed",
                "data": {
                    "retry_after_seconds": 30,
                    "guidance": "Check IRIS database health"
                }
            }
        else:
            return {
                "code": "internal_error",
                "message": "An unexpected error occurred",
                "data": {"error_id": generate_error_id()}
            }
```

### Error Categories
- `invalid_params`: Query missing, top_k out of range, invalid technique name
- `authentication_error`: Invalid/missing API key (when auth_mode=api_key)
- `service_unavailable`: Database down, pipeline unavailable
- `timeout`: Query exceeded timeout (FR-010 requires <2s p95)
- `internal_error`: Unexpected exceptions (logged with trace ID)

---

## Summary

All 8 research tasks completed with concrete decisions:

1. ✅ **MCP Protocol**: Use @modelcontextprotocol/sdk (official TypeScript SDK)
2. ✅ **Transports**: Support both stdio (local) and HTTP/SSE (remote) with abstraction
3. ✅ **Bridge Architecture**: Node.js MCP server → HTTP → Python FastAPI bridge → RAG pipelines
4. ✅ **Authentication**: Configurable (none for dev, API keys for production)
5. ✅ **Deployment**: Both standalone (separate process) and integrated (within REST API)
6. ✅ **Pipeline Reuse**: Import REST API's PipelineManager singleton
7. ✅ **Connection Limit**: Semaphore-based pooling (max 5 concurrent)
8. ✅ **Error Handling**: MCP-compliant structured errors with actionable guidance

No unknowns remain. Ready to proceed to Phase 1 (Design & Contracts).
