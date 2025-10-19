# Data Model: MCP Tools Implementation

**Feature**: Complete MCP Tools Implementation
**Branch**: 043-complete-mcp-tools
**Date**: 2025-10-18

## Overview

This data model defines the core entities for the MCP server implementation, extracted from the feature specification's Key Entities section (spec.md lines 134-171).

## Entity Definitions

### 1. MCP Tool

Represents a RAG pipeline exposed via Model Context Protocol.

**Attributes**:
- `name` (string, required): Unique identifier for the tool
  - Examples: "rag_basic", "rag_crag", "rag_graphrag"
  - Naming convention: "rag_" prefix + pipeline type
  - Must match MCP tool name in client requests

- `description` (string, required): Human-readable description of pipeline capabilities
  - Displayed in Claude Code's tool picker
  - Should explain when to use this technique
  - Example: "Basic RAG with vector similarity search - best for general Q&A"

- `schema` (MCPToolSchema, required): Parameter definitions and validation rules
  - JSON Schema format (type, properties, required, defaults)
  - Includes all pipeline-specific parameters
  - Validated before tool execution

- `pipeline_reference` (Pipeline, required): Link to underlying RAG pipeline implementation
  - Reference to `iris_rag.pipelines.BasicRAGPipeline` instance
  - Shared with REST API (from `PipelineManager`)

- `status` (HealthStatus, computed): Availability status
  - Values: "healthy", "degraded", "unavailable"
  - Determined by health check tool
  - Reflects pipeline and database health

**Relationships**:
- One MCP Tool → One RAG Pipeline (1:1)
- One MCP Tool → Many Tool Requests (1:N)
- One MCP Tool → One MCPToolSchema (1:1)

**Validation Rules**:
- `name` must be unique across all tools
- `name` must start with "rag_" prefix
- `schema` must be valid JSON Schema
- `pipeline_reference` must point to initialized pipeline

**Example**:
```python
{
  "name": "rag_basic",
  "description": "Basic RAG with vector similarity search",
  "schema": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50}
    },
    "required": ["query"]
  },
  "pipeline_reference": <BasicRAGPipeline instance>,
  "status": "healthy"
}
```

---

### 2. Tool Request

Represents an incoming query from an AI agent (Claude Code).

**Attributes**:
- `tool_name` (string, required): Which RAG pipeline tool to invoke
  - Must match one of the 6 available tools
  - Example: "rag_basic", "rag_crag"

- `query_text` (string, required): User's question or search query
  - Main input to RAG pipeline
  - Max length: 8000 characters (validated)
  - Example: "What are the symptoms of diabetes?"

- `parameters` (dict, optional): Pipeline-specific options
  - Common: `top_k`, `include_sources`, `include_metadata`
  - CRAG-specific: `confidence_threshold`, `correction_strategy`
  - ColBERT-specific: `interaction_threshold`
  - Validated against tool's schema

- `request_id` (UUID, required): Unique identifier for tracing
  - Auto-generated if not provided
  - Used for logging and performance tracking
  - Returned in response for correlation

- `timestamp` (datetime, required): When request was received
  - ISO 8601 format
  - Used for performance metrics
  - Used for request log retention

**Relationships**:
- Many Tool Requests → One MCP Tool (N:1)
- One Tool Request → One Tool Response (1:1)

**Validation Rules**:
- `tool_name` must exist in available tools
- `query_text` must be non-empty string (1-8000 chars)
- `parameters` must conform to tool's schema
- `request_id` must be valid UUID v4

**State Transitions**:
```
[Received] → [Validating] → [Executing] → [Completed]
                  ↓
              [Failed]
```

**Example**:
```python
{
  "tool_name": "rag_crag",
  "query_text": "What are the symptoms of diabetes?",
  "parameters": {
    "top_k": 5,
    "confidence_threshold": 0.8,
    "correction_strategy": "rewrite"
  },
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-10-18T14:23:45.123Z"
}
```

---

### 3. Tool Response

Represents the result returned to the AI agent.

**Attributes**:
- `answer` (string, required): LLM-generated text response
  - Main output from RAG pipeline
  - Synthesized from retrieved documents
  - May be empty if no relevant docs found

- `retrieved_documents` (List[Document], required): Relevant documents with scores
  - Each document has: `doc_id`, `content`, `score`, `metadata`
  - Ordered by relevance score (descending)
  - Length determined by `top_k` parameter

- `sources` (List[string], required): File names or references for citations
  - Extracted from document metadata
  - Deduplicated list
  - Used by AI agent for attribution

- `metadata` (dict, required): Pipeline-specific information
  - Common: `pipeline_name`, `execution_time_ms`, `tokens_used`
  - CRAG: `correction_applied`, `confidence_score`
  - ColBERT: `token_interactions`, `compression_ratio`
  - HybridGraphRAG: `graph_traversal_depth`, `rrf_score`

- `performance_metrics` (PerformanceMetrics, required): Execution details
  - `execution_time_ms`: Total query time
  - `retrieval_time_ms`: Document retrieval time
  - `generation_time_ms`: LLM generation time
  - `tokens_used`: LLM token count

- `response_id` (UUID, required): Unique identifier matching request
  - Same as `request_id` for correlation
  - Used for debugging and tracing

**Relationships**:
- One Tool Response → One Tool Request (1:1)
- One Tool Response → One MCP Tool (N:1)

**Validation Rules**:
- `answer` can be empty but not null
- `retrieved_documents` must be list (can be empty)
- `sources` must be list (can be empty)
- `metadata` must include `pipeline_name`
- `performance_metrics` times must be non-negative

**Example**:
```python
{
  "answer": "Diabetes symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections.",
  "retrieved_documents": [
    {
      "doc_id": "1a2b3c4d-5678-90ab-cdef-1234567890ab",
      "content": "Common symptoms of diabetes mellitus include...",
      "score": 0.95,
      "metadata": {
        "source": "medical_textbook_ch5.pdf",
        "page_number": 127
      }
    }
  ],
  "sources": ["medical_textbook_ch5.pdf"],
  "metadata": {
    "pipeline_name": "basic",
    "execution_time_ms": 1456,
    "tokens_used": 2345
  },
  "performance_metrics": {
    "execution_time_ms": 1456,
    "retrieval_time_ms": 345,
    "generation_time_ms": 1089,
    "tokens_used": 2345
  },
  "response_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

### 4. Pipeline Configuration

Settings for each RAG pipeline.

**Attributes**:
- `pipeline_name` (string, required): Which RAG technique
  - Values: "basic", "basic_rerank", "crag", "graphrag", "pylate_colbert", "iris_global_graphrag"
  - Matches pipeline registry keys

- `default_parameters` (dict, required): Standard settings for queries
  - Common: `top_k`, `embedding_model`, `llm_model`
  - Technique-specific defaults
  - Can be overridden per request

- `resource_limits` (ResourceLimits, required): Operational constraints
  - `max_concurrent_queries`: Per-pipeline limit (default: 5)
  - `timeout_seconds`: Query timeout (default: 30)
  - `max_tokens`: LLM token limit (default: 4096)

- `database_connection` (IRISConnectionPool, required): IRIS connection pool reference
  - Shared with REST API
  - Managed by `PipelineManager`

- `health_status` (HealthStatus, computed): Current operational status
  - Values: "healthy", "degraded", "unavailable"
  - Updated by health check tool
  - Based on recent query success rate

**Relationships**:
- One Pipeline Configuration → One RAG Pipeline (1:1)
- One Pipeline Configuration → One Database Connection Pool (N:1, shared)

**Validation Rules**:
- `pipeline_name` must match factory registry
- `default_parameters` must be valid for pipeline type
- `resource_limits.max_concurrent_queries` <= 5 (MCP connection limit)
- `resource_limits.timeout_seconds` <= 60 (reasonable limit)

**Example**:
```python
{
  "pipeline_name": "crag",
  "default_parameters": {
    "top_k": 5,
    "confidence_threshold": 0.8,
    "correction_strategy": "rewrite",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "gpt-4"
  },
  "resource_limits": {
    "max_concurrent_queries": 5,
    "timeout_seconds": 30,
    "max_tokens": 4096
  },
  "database_connection": <IRISConnectionPool instance>,
  "health_status": "healthy"
}
```

---

### 5. Health Status

System-wide operational state.

**Attributes**:
- `overall_status` (HealthStatusEnum, computed): Aggregate health
  - Values: "healthy", "degraded", "unavailable"
  - "healthy": All pipelines healthy, DB connected
  - "degraded": Some pipelines unhealthy, DB connected
  - "unavailable": DB disconnected or all pipelines failed

- `pipeline_statuses` (Dict[str, HealthStatus], required): Individual pipeline health
  - Keys: Pipeline names ("basic", "crag", etc.)
  - Values: {"status": "healthy|degraded|unavailable", "last_success": datetime, "error_rate": float}
  - Updated every health check

- `database_status` (DatabaseHealth, required): IRIS connection health
  - `connected`: boolean
  - `response_time_ms`: Last ping latency
  - `connection_pool_usage`: Active connections / max connections

- `performance_metrics` (PerformanceMetrics, required): Recent performance
  - `average_response_time_ms`: p50 latency across all pipelines
  - `p95_response_time_ms`: p95 latency
  - `error_rate`: Failed queries / total queries (last hour)
  - `queries_per_minute`: Throughput

- `last_updated` (datetime, required): Timestamp of health check
  - ISO 8601 format
  - Used to determine staleness

**Relationships**:
- One Health Status → Many Pipeline Configurations (1:N)
- One Health Status → One Database Connection (1:1)

**Validation Rules**:
- `overall_status` must reflect pipeline + DB health
- `pipeline_statuses` must include all 6 pipelines
- `database_status.response_time_ms` should be < 100ms (healthy)
- `last_updated` should be within last 60 seconds

**Status Computation Logic**:
```python
if not database_status.connected:
    overall_status = "unavailable"
elif all(p.status == "healthy" for p in pipeline_statuses.values()):
    overall_status = "healthy"
elif all(p.status == "unavailable" for p in pipeline_statuses.values()):
    overall_status = "unavailable"
else:
    overall_status = "degraded"
```

**Example**:
```python
{
  "overall_status": "healthy",
  "pipeline_statuses": {
    "basic": {"status": "healthy", "last_success": "2025-10-18T14:23:45Z", "error_rate": 0.01},
    "crag": {"status": "healthy", "last_success": "2025-10-18T14:23:40Z", "error_rate": 0.02},
    "graphrag": {"status": "degraded", "last_success": "2025-10-18T14:20:00Z", "error_rate": 0.15},
    "pylate_colbert": {"status": "healthy", "last_success": "2025-10-18T14:23:30Z", "error_rate": 0.005},
    "basic_rerank": {"status": "healthy", "last_success": "2025-10-18T14:23:35Z", "error_rate": 0.01},
    "iris_global_graphrag": {"status": "healthy", "last_success": "2025-10-18T14:23:25Z", "error_rate": 0.03}
  },
  "database_status": {
    "connected": true,
    "response_time_ms": 12,
    "connection_pool_usage": "5/20"
  },
  "performance_metrics": {
    "average_response_time_ms": 1234,
    "p95_response_time_ms": 1890,
    "error_rate": 0.02,
    "queries_per_minute": 15
  },
  "last_updated": "2025-10-18T14:23:45.678Z"
}
```

---

## Entity Relationship Diagram

```
                      ┌─────────────────┐
                      │  MCP Tool       │
                      ├─────────────────┤
                      │ name            │
                      │ description     │
                      │ schema          │
                      │ pipeline_ref    │
                      │ status          │
                      └────────┬────────┘
                               │ 1
                               │
                               │ N
                      ┌────────▼────────┐
                      │  Tool Request   │
                      ├─────────────────┤
                      │ tool_name       │
                      │ query_text      │
                      │ parameters      │
                      │ request_id      │
                      │ timestamp       │
                      └────────┬────────┘
                               │ 1
                               │
                               │ 1
                      ┌────────▼────────┐
                      │  Tool Response  │
                      ├─────────────────┤
                      │ answer          │
                      │ documents       │
                      │ sources         │
                      │ metadata        │
                      │ perf_metrics    │
                      │ response_id     │
                      └─────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Pipeline Configuration                                 │
├─────────────────────────────────────────────────────────┤
│ pipeline_name  │ default_params  │ resource_limits     │
│ db_connection  │ health_status                         │
└───────────────┬─────────────────────────────────────────┘
                │ N
                │
                │ 1
       ┌────────▼────────┐
       │  Health Status  │
       ├─────────────────┤
       │ overall_status  │
       │ pipeline_statuses│
       │ database_status │
       │ perf_metrics    │
       │ last_updated    │
       └─────────────────┘
```

---

## Type Definitions

### HealthStatusEnum
```python
from enum import Enum

class HealthStatusEnum(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
```

### ResourceLimits
```python
from pydantic import BaseModel

class ResourceLimits(BaseModel):
    max_concurrent_queries: int = 5
    timeout_seconds: int = 30
    max_tokens: int = 4096
```

### PerformanceMetrics
```python
from pydantic import BaseModel

class PerformanceMetrics(BaseModel):
    execution_time_ms: int
    retrieval_time_ms: int
    generation_time_ms: int
    tokens_used: int
```

### DatabaseHealth
```python
from pydantic import BaseModel

class DatabaseHealth(BaseModel):
    connected: bool
    response_time_ms: int
    connection_pool_usage: str  # e.g., "5/20"
```

---

## Summary

This data model defines 5 core entities for MCP Tools implementation:
1. **MCP Tool**: RAG pipeline exposed as MCP tool
2. **Tool Request**: Incoming query from AI agent
3. **Tool Response**: Result returned to AI agent
4. **Pipeline Configuration**: Settings for each pipeline
5. **Health Status**: System-wide operational state

All entities include validation rules, relationships, and state transitions where applicable. The model supports both standalone and integrated deployment modes, with shared pipeline instances between MCP and REST API.
