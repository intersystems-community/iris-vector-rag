# MCP Tools Quickstart Guide

**Feature**: Complete MCP Tools Implementation
**Branch**: 043-complete-mcp-tools
**Date**: 2025-10-18

## Overview

This quickstart guide provides integration test scenarios based on the feature specification's acceptance scenarios (spec.md lines 56-85). Each scenario validates a functional requirement through end-to-end testing.

## Prerequisites

Before running these scenarios:
1. IRIS database running and accessible
2. REST API operational (MCP reuses pipeline instances)
3. Python MCP bridge started (port 8001)
4. Node.js MCP server started (stdio or HTTP/SSE)

## Scenario 1: List Available Tools (FR-002)

**Objective**: Verify AI agent can discover all 6 RAG pipeline tools with their schemas.

**User Story**: As an AI agent, I want to list available tools so I can choose the right technique for my query.

**Steps**:
1. Connect to MCP server (stdio or HTTP/SSE)
2. Send `tools/list` request
3. Receive tool definitions

**Expected Result**:
```json
{
  "tools": [
    {
      "name": "rag_basic",
      "description": "Basic RAG with vector similarity search...",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": {"type": "string"},
          "top_k": {"type": "integer", "default": 5}
        },
        "required": ["query"]
      }
    },
    {
      "name": "rag_basic_rerank",
      "description": "Vector search with cross-encoder reranking...",
      "inputSchema": { /* ... */ }
    },
    {
      "name": "rag_crag",
      "description": "Corrective RAG with self-evaluation...",
      "inputSchema": { /* ... */ }
    },
    {
      "name": "rag_graphrag",
      "description": "Hybrid search (vector + text + graph + RRF)...",
      "inputSchema": { /* ... */ }
    },
    {
      "name": "rag_pylate_colbert",
      "description": "ColBERT late interaction retrieval...",
      "inputSchema": { /* ... */ }
    },
    {
      "name": "rag_iris_global_graphrag",
      "description": "IRIS Global GraphRAG for academic papers...",
      "inputSchema": { /* ... */ }
    },
    {
      "name": "rag_health_check",
      "description": "Check health of all RAG pipelines...",
      "inputSchema": { /* ... */ }
    },
    {
      "name": "rag_metrics",
      "description": "Retrieve performance metrics...",
      "inputSchema": { /* ... */ }
    }
  ]
}
```

**Validation**:
- ✓ Exactly 8 tools returned (6 RAG pipelines + 2 utility tools)
- ✓ Each tool has `name`, `description`, `inputSchema`
- ✓ All 6 RAG pipelines present
- ✓ Schemas include parameter descriptions and validation rules

**Test Command**:
```bash
# Via stdio (Claude Code)
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' | node nodejs/src/mcp/server.js --transport stdio

# Via HTTP/SSE
curl -X POST http://localhost:3000/mcp/tools/list
```

---

## Scenario 2: Execute BasicRAG Query (FR-001, FR-004, FR-010)

**Objective**: Verify BasicRAG tool executes successfully and returns standardized response format.

**User Story**: As an AI agent, I want to query the BasicRAG tool so I can get fast answers from the knowledge base.

**Steps**:
1. Call `rag_basic` tool with sample query
2. Provide parameters: `query="What are the symptoms of diabetes?"`, `top_k=5`
3. Receive answer and retrieved documents

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "rag_basic",
    "arguments": {
      "query": "What are the symptoms of diabetes?",
      "top_k": 5,
      "include_sources": true,
      "include_metadata": true
    }
  }
}
```

**Expected Result**:
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "answer": "Diabetes symptoms include increased thirst, frequent urination, extreme hunger, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections. These symptoms occur when blood glucose levels are too high.",
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
      // ... 4 more documents
    ],
    "sources": ["medical_textbook_ch5.pdf", "diabetes_guide.pdf"],
    "metadata": {
      "pipeline_name": "basic",
      "execution_time_ms": 1456,
      "tokens_used": 2345
    },
    "performance": {
      "execution_time_ms": 1456,
      "retrieval_time_ms": 345,
      "generation_time_ms": 1089,
      "tokens_used": 2345
    }
  }
}
```

**Validation**:
- ✓ Response has `answer` field (non-empty string)
- ✓ Response has `retrieved_documents` (array of 5 docs)
- ✓ Response has `sources` (array of filenames)
- ✓ Response has `metadata.pipeline_name = "basic"`
- ✓ Response has `performance` with execution times
- ✓ Performance: `execution_time_ms < 2000` (p95 latency requirement)
- ✓ Response format matches REST API response (FR-004)

**Test Command**:
```python
# Python test
async def test_basic_rag_query():
    result = await mcp_client.call_tool("rag_basic", {
        "query": "What are the symptoms of diabetes?",
        "top_k": 5
    })
    assert result["answer"]
    assert len(result["retrieved_documents"]) == 5
    assert result["metadata"]["pipeline_name"] == "basic"
    assert result["performance"]["execution_time_ms"] < 2000
```

---

## Scenario 3: CRAG with Corrective Measures (FR-009)

**Objective**: Verify CRAG tool applies corrective measures when document confidence is low.

**User Story**: As an AI agent, I want CRAG to automatically improve low-quality retrievals so I get better answers.

**Steps**:
1. Call `rag_crag` tool with query likely to have low confidence
2. Set `confidence_threshold=0.8`, `correction_strategy="rewrite"`
3. Verify correction metadata in response

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "rag_crag",
    "arguments": {
      "query": "What is the molecular structure of compound XYZ-123?",
      "top_k": 5,
      "confidence_threshold": 0.8,
      "correction_strategy": "rewrite"
    }
  }
}
```

**Expected Result**:
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "answer": "...",
    "retrieved_documents": [ /* ... */ ],
    "sources": [ /* ... */ ],
    "metadata": {
      "pipeline_name": "crag",
      "correction_applied": true,
      "confidence_score": 0.65,
      "original_query": "What is the molecular structure of compound XYZ-123?",
      "rewritten_query": "What is the chemical composition and molecular formula of compound XYZ-123?",
      "correction_strategy_used": "rewrite"
    },
    "performance": { /* ... */ }
  }
}
```

**Validation**:
- ✓ Response has `metadata.correction_applied = true`
- ✓ Response has `metadata.confidence_score` (float 0.0-1.0)
- ✓ Response has `metadata.original_query` (original query text)
- ✓ Response has `metadata.rewritten_query` (corrected query)
- ✓ Response has `metadata.correction_strategy_used = "rewrite"`

**Test Command**:
```python
async def test_crag_correction():
    result = await mcp_client.call_tool("rag_crag", {
        "query": "What is the molecular structure of compound XYZ-123?",
        "confidence_threshold": 0.8,
        "correction_strategy": "rewrite"
    })
    assert result["metadata"]["correction_applied"] == True
    assert "confidence_score" in result["metadata"]
    assert "rewritten_query" in result["metadata"]
```

---

## Scenario 4: Health Check Tool (FR-005, FR-024)

**Objective**: Verify health check tool reports accurate status for all components.

**User Story**: As a developer, I want to check system health so I can monitor RAG pipeline availability.

**Steps**:
1. Call `rag_health_check` tool
2. Set `include_details=true` for full report
3. Receive health status for all 6 pipelines + database

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tools/call",
  "params": {
    "name": "rag_health_check",
    "arguments": {
      "include_details": true,
      "include_performance_metrics": true
    }
  }
}
```

**Expected Result**:
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "status": "healthy",
    "timestamp": "2025-10-18T14:23:45.678Z",
    "pipelines": {
      "basic": {"status": "healthy", "last_success": "2025-10-18T14:23:45Z", "error_rate": 0.01},
      "basic_rerank": {"status": "healthy", "last_success": "2025-10-18T14:23:35Z", "error_rate": 0.01},
      "crag": {"status": "healthy", "last_success": "2025-10-18T14:23:40Z", "error_rate": 0.02},
      "graphrag": {"status": "degraded", "last_success": "2025-10-18T14:20:00Z", "error_rate": 0.15},
      "pylate_colbert": {"status": "healthy", "last_success": "2025-10-18T14:23:30Z", "error_rate": 0.005},
      "iris_global_graphrag": {"status": "healthy", "last_success": "2025-10-18T14:23:25Z", "error_rate": 0.03}
    },
    "database": {
      "connected": true,
      "response_time_ms": 12,
      "connection_pool_usage": "5/20"
    },
    "performance_metrics": {
      "average_response_time_ms": 1234,
      "p95_response_time_ms": 1890,
      "error_rate": 0.02,
      "queries_per_minute": 15
    }
  }
}
```

**Validation**:
- ✓ Response has `status` (healthy/degraded/unavailable)
- ✓ Response has `pipelines` with all 6 pipeline statuses
- ✓ Response has `database.connected = true`
- ✓ Response has `database.response_time_ms < 100` (healthy DB)
- ✓ Response has `performance_metrics` with p95 latency
- ✓ All pipeline statuses are valid (healthy/degraded/unavailable)

**Test Command**:
```python
async def test_health_check():
    result = await mcp_client.call_tool("rag_health_check", {
        "include_details": True
    })
    assert result["status"] in ["healthy", "degraded", "unavailable"]
    assert len(result["pipelines"]) == 6
    assert result["database"]["connected"] == True
```

---

## Scenario 5: Concurrent Queries (FR-008, FR-032)

**Objective**: Verify system handles concurrent requests without resource conflicts (max 5 concurrent).

**User Story**: As an AI agent, I want to submit concurrent queries so I can get multiple answers in parallel.

**Steps**:
1. Submit 3 concurrent queries to different pipelines
2. Verify all complete successfully
3. Verify no resource conflicts or errors

**Request** (3 parallel requests):
```python
import asyncio

async def test_concurrent_queries():
    tasks = [
        mcp_client.call_tool("rag_basic", {"query": "What is diabetes?", "top_k": 3}),
        mcp_client.call_tool("rag_crag", {"query": "What causes diabetes?", "top_k": 3}),
        mcp_client.call_tool("rag_graphrag", {"query": "How is diabetes treated?", "top_k": 3})
    ]

    results = await asyncio.gather(*tasks)

    # All should succeed
    for result in results:
        assert "answer" in result
        assert "retrieved_documents" in result
```

**Expected Result**:
- All 3 queries complete successfully
- No timeout errors
- No resource conflict errors
- Response times acceptable (<2s per query)

**Validation**:
- ✓ All 3 queries return successful responses
- ✓ No errors related to resource conflicts
- ✓ Each response has correct `pipeline_name` in metadata
- ✓ Total concurrent connections ≤ 5 (FR-032)

**Connection Limit Test**:
```python
async def test_connection_limit():
    # Try to create 6 concurrent connections (exceeds limit)
    connections = []
    for i in range(6):
        try:
            conn = await mcp_client.connect()
            connections.append(conn)
        except MCPError as e:
            # 6th connection should fail
            if i == 5:
                assert e.code == "connection_limit_exceeded"
                assert "5" in e.message  # mentions max connections
```

---

## Scenario 6: HybridGraphRAG with Graph Traversal (FR-001, FR-009)

**Objective**: Verify HybridGraphRAG returns results from both vector search and graph traversal.

**User Story**: As an AI agent, I want to use GraphRAG for complex queries requiring relationship understanding.

**Steps**:
1. Call `rag_graphrag` tool with query requiring graph relationships
2. Set `search_method="hybrid"`, `graph_traversal_depth=2`
3. Verify response includes both vector and graph results

**Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "method": "tools/call",
  "params": {
    "name": "rag_graphrag",
    "arguments": {
      "query": "What treatments for diabetes also help with hypertension?",
      "top_k": 5,
      "search_method": "hybrid",
      "graph_traversal_depth": 2,
      "rrf_k": 60
    }
  }
}
```

**Expected Result**:
```json
{
  "jsonrpc": "2.0",
  "id": 6,
  "result": {
    "answer": "Several treatments are effective for both diabetes and hypertension, including...",
    "retrieved_documents": [ /* hybrid results */ ],
    "sources": [ /* ... */ ],
    "metadata": {
      "pipeline_name": "graphrag",
      "search_method_used": "hybrid",
      "graph_traversal_depth": 2,
      "vector_results_count": 5,
      "text_results_count": 5,
      "graph_results_count": 3,
      "rrf_combined_results": 5,
      "rrf_score": 0.87
    },
    "performance": { /* ... */ }
  }
}
```

**Validation**:
- ✓ Response has `metadata.search_method_used = "hybrid"`
- ✓ Response has `metadata.graph_traversal_depth = 2`
- ✓ Response has counts for vector, text, and graph results
- ✓ Response has `metadata.rrf_score` (reciprocal rank fusion)
- ✓ Documents include both vector-similar and graph-related results

**Test Command**:
```python
async def test_hybrid_graphrag():
    result = await mcp_client.call_tool("rag_graphrag", {
        "query": "What treatments for diabetes also help with hypertension?",
        "search_method": "hybrid",
        "graph_traversal_depth": 2
    })
    assert result["metadata"]["search_method_used"] == "hybrid"
    assert "vector_results_count" in result["metadata"]
    assert "graph_results_count" in result["metadata"]
    assert result["metadata"]["graph_traversal_depth"] == 2
```

---

## Scenario 7: MCP-REST API Integration (FR-006, FR-025)

**Objective**: Verify MCP responses match REST API responses byte-for-byte.

**User Story**: As a developer, I want MCP and REST API to return identical results for the same query.

**Steps**:
1. Execute same query via MCP tool
2. Execute same query via REST API endpoint
3. Compare responses (should be identical)

**MCP Request**:
```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/call",
  "params": {
    "name": "rag_basic",
    "arguments": {
      "query": "What is diabetes?",
      "top_k": 5
    }
  }
}
```

**REST API Request**:
```bash
curl -X POST http://localhost:8000/api/v1/basic/_search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is diabetes?", "top_k": 5}'
```

**Validation**:
```python
async def test_mcp_rest_consistency():
    # Query via MCP
    mcp_result = await mcp_client.call_tool("rag_basic", {
        "query": "What is diabetes?",
        "top_k": 5
    })

    # Query via REST API
    rest_result = await rest_client.post("/api/v1/basic/_search", {
        "query": "What is diabetes?",
        "top_k": 5
    })

    # Compare (excluding timestamps and request IDs)
    assert mcp_result["answer"] == rest_result["answer"]
    assert len(mcp_result["retrieved_documents"]) == len(rest_result["retrieved_documents"])
    assert mcp_result["sources"] == rest_result["sources"]
    assert mcp_result["metadata"]["pipeline_name"] == rest_result["pipeline_name"]
```

---

## Summary

These 7 scenarios validate all key functional requirements:

1. **Tool Discovery** (FR-002): List all 6 RAG tools with schemas
2. **BasicRAG Execution** (FR-001, FR-004, FR-010): Execute query, standardized response, <2s latency
3. **CRAG Correction** (FR-009): Pipeline-specific metadata (correction_applied)
4. **Health Check** (FR-005, FR-024): Status of all 6 pipelines + database
5. **Concurrency** (FR-008, FR-032): 3 concurrent queries, 5 connection limit
6. **HybridGraphRAG** (FR-001, FR-009): Graph traversal + vector search
7. **MCP-REST Consistency** (FR-006, FR-025): Identical responses

All scenarios are automatable and align with existing TDD test suite in `tests/test_mcp/`.
