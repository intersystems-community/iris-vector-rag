# Feature Specification: OpenTelemetry Integration for iris-vector-rag

**Feature ID**: F-OTEL-001
**Version**: 0.6.0
**Status**: Draft
**Date**: 2025-11-22
**Author**: System (based on aigw_mockup reference implementation)

---

## Executive Summary

Add comprehensive OpenTelemetry (OTel) observability to iris-vector-rag with **native IRIS integration**. Implement dual telemetry approach: OTLP export for external platforms (Grafana, Datadog) AND direct IRIS storage for fast queries and audit trails.

**Key Benefits**:
- **Distributed Tracing**: Track RAG queries across LLM, vector search, and IRIS database
- **Cost Tracking**: Monitor token usage and estimate costs per user/model/pipeline
- **Performance Monitoring**: Identify bottlenecks with span-level metrics
- **IRIS Native Compatibility**: Mesh seamlessly with IRIS 2024.1+ native OTel support
- **Audit Trail**: Store all spans in IRIS TelemetrySpan table for compliance

**Integration Architecture**: Aligns with `aigw_mockup` reference implementation to ensure compatibility with IRIS native telemetry infrastructure.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Goals and Non-Goals](#goals-and-non-goals)
3. [Architecture](#architecture)
4. [Implementation Plan](#implementation-plan)
5. [Span Definitions](#span-definitions)
6. [API Design](#api-design)
7. [Configuration](#configuration)
8. [Migration Strategy](#migration-strategy)
9. [Testing Strategy](#testing-strategy)
10. [Dependencies](#dependencies)
11. [Risks and Mitigations](#risks-and-mitigations)
12. [Success Metrics](#success-metrics)

---

## Problem Statement

### Current State

iris-vector-rag **lacks observability infrastructure**:
- **No tracing**: Cannot track query execution across components
- **No cost tracking**: Unknown token usage and LLM costs
- **No performance metrics**: Difficult to identify bottlenecks
- **No audit trail**: No record of who executed what queries
- **No integration**: Cannot leverage IRIS 2024.1+ native OTel features

### Pain Points

1. **Debugging RAG Failures**: When a query fails, no visibility into which component failed (embedding, retrieval, LLM)
2. **Cost Overruns**: Users report unexpectedly high LLM bills with no breakdown by user/pipeline
3. **Performance Tuning**: Cannot identify whether slowness is from vector search, graph traversal, or LLM generation
4. **Compliance**: No audit trail for queries containing sensitive data
5. **IRIS Integration**: Cannot use IRIS's native OTel collector and monitoring tools

### User Stories

**Story 1: DevOps Engineer**
> "As a DevOps engineer, I need to identify why GraphRAG queries are taking 10+ seconds so I can optimize the slowest component."

**Story 2: Finance Manager**
> "As a finance manager, I need to track LLM costs by user and department to allocate expenses accurately."

**Story 3: Data Scientist**
> "As a data scientist, I need to see which retrieved documents the LLM used to generate each answer for quality assessment."

**Story 4: Compliance Officer**
> "As a compliance officer, I need an audit trail showing who ran queries on patient data and what was retrieved."

---

## Goals and Non-Goals

### Goals

**Primary Goals**:
1. ✅ Implement OpenTelemetry tracing for all RAG pipelines
2. ✅ Store spans in IRIS TelemetrySpan table (dual storage with OTLP)
3. ✅ Track token usage and calculate costs per user/model
4. ✅ Mesh with IRIS 2024.1+ native OTel infrastructure
5. ✅ Follow GenAI semantic conventions for LLM spans
6. ✅ Support distributed tracing (W3C Trace Context propagation)

**Secondary Goals**:
- Export metrics in Prometheus format
- Provide CLI for querying telemetry data
- Support sampling (e.g., 10% in production)
- Automatic FastAPI instrumentation for REST API

### Non-Goals

- ❌ Real-time alerting (use external tools like Grafana)
- ❌ Custom visualization dashboard (use Grafana/Datadog)
- ❌ Log aggregation (focus on traces/spans only)
- ❌ Profiling (CPU/memory profiling out of scope)

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│  iris-vector-rag Pipeline (Python)                      │
│  - BasicRAG, CRAG, GraphRAG, ColBERT                    │
│  - LLM calls (Anthropic, OpenAI)                        │
│  - Vector search (IRISVectorStore)                      │
│  - Knowledge graph traversal                            │
└─────────────────┬───────────────────────────────────────┘
                  │ OpenTelemetry SDK
                  │ (spans, metrics, logs)
                  │
    ┌─────────────▼─────────────────────────────┐
    │  Dual Telemetry Storage                   │
    ├───────────────────┬───────────────────────┤
    │ IRIS TelemetrySpan│ OTLP Exporter         │
    │ (SQL queries)     │ (gRPC/HTTP)           │
    └───────────────────┴───────────────────────┘
            │                       │
            │                       ▼
            │           ┌─────────────────────────┐
            │           │ OpenTelemetry Collector │
            │           │ (IRIS native 2024.1+)   │
            │           └─────────────────────────┘
            │                       │
            │           ┌───────────┴────────────┐
            │           │                        │
            │           ▼                        ▼
            │    IRIS Metrics            External Platform
            │    (/api/monitor/metrics)  (Grafana, Datadog)
            │
            ▼
    ┌─────────────────────────┐
    │ Query APIs              │
    │ - TelemetryQueryAPI     │
    │ - CostTracker           │
    │ - CLI (iris-rag-telemetry) │
    └─────────────────────────┘
```

### Component Design

#### 1. Span Instrumentation (`iris_rag/telemetry/spans.py`)

**Responsibilities**:
- Create OpenTelemetry spans for RAG operations
- Follow GenAI semantic conventions for LLM spans
- Store spans to IRIS TelemetrySpan table
- Support parent-child span relationships

**Key Classes**:
- `RAGSpan`: Base class for RAG operation spans
- `LLMSpan`: LLM completion spans with token usage
- `RetrievalSpan`: Vector/graph retrieval spans
- `SpanInstrumentor`: Context manager for span creation

**Example Usage**:
```python
from iris_rag.telemetry.spans import LLMSpanInstrumentor

instrumentor = LLMSpanInstrumentor(iris_conn, username="alice")

with instrumentor.start_llm_span(
    model="claude-3-5-sonnet",
    provider="anthropic",
    operation="chat.completion"
) as span:
    response = llm.complete(prompt)

    span.set_usage(
        prompt_tokens=response.usage.prompt_tokens,
        completion_tokens=response.usage.completion_tokens
    )
    span.set_finish_reason("stop")
```

#### 2. OTLP Exporter Configuration (`iris_rag/telemetry/exporters.py`)

**Responsibilities**:
- Configure OpenTelemetry SDK with OTLP exporter
- Set up trace context propagation (W3C standard)
- Configure sampling (10% in production, 100% in dev)
- Enable automatic FastAPI instrumentation

**Key Functions**:
- `configure_telemetry(service_name, otlp_endpoint, sampling_ratio)`
- `get_tracer(name)` - Get tracer for creating spans
- `get_current_trace_id()` - For IRIS storage correlation
- `shutdown_telemetry()` - Flush spans on shutdown

**Configuration Variables**:
- `OTEL_EXPORTER_OTLP_ENDPOINT` (default: `http://localhost:4318`)
- `OTEL_SERVICE_NAME` (default: `iris-rag-api`)
- `OTEL_TRACES_SAMPLER_ARG` (default: `0.1` for 10% sampling)

#### 3. Retrieval Instrumentation (`iris_rag/telemetry/retrieval.py`)

**Responsibilities**:
- Instrument vector search operations
- Track similarity scores (top, avg, min)
- Record cache hit/miss
- Support hybrid search (vector + text + graph)

**Span Attributes** (following aigw_mockup patterns):
- `retrieval.query`: Text query or embedding description
- `retrieval.collection`: Collection name/ID
- `retrieval.top_k`: Number of results requested
- `retrieval.result_count`: Actual results returned
- `retrieval.top_score`: Highest similarity score
- `retrieval.avg_score`: Average similarity score
- `retrieval.search_type`: "vector", "hybrid", "keyword", "graph"
- `retrieval.cache_hit`: Whether results were cached

**Example Usage**:
```python
from iris_rag.telemetry.retrieval import VectorSearchInstrumentor

instrumentor = VectorSearchInstrumentor(
    iris_conn, username="alice", trace_id=trace_id
)

with instrumentor.start_search_span(
    query="diabetes treatment guidelines",
    collection_id="pmc-documents",
    top_k=10,
    search_type="hybrid",
    parent_span_id=tool_span_id
) as span:
    results = vector_store.search(query, top_k=10)

    span.calculate_and_set_scores_from_results(results)
    span.set_embedding_model("text-embedding-3-large")
```

#### 4. Cost Tracking (`iris_rag/telemetry/cost.py`)

**Responsibilities**:
- Calculate LLM costs based on token usage
- Track costs per user, provider, model
- Generate cost reports for time periods
- Store cost data in span attributes

**Pricing Tiers** (configurable):
```python
DEFAULT_PRICING = {
    "claude-3-5-sonnet-20241022": PricingTier(
        input_price_per_1k=0.003,  # $3 per 1M tokens
        output_price_per_1k=0.015  # $15 per 1M tokens
    ),
    "gpt-4-turbo": PricingTier(
        input_price_per_1k=0.01,
        output_price_per_1k=0.03
    ),
    # ... more models
}
```

**Example Usage**:
```python
from iris_rag.telemetry.cost import CostTracker

tracker = CostTracker(iris_conn)

# Track request cost
cost = tracker.track_request_cost(
    username="alice",
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    input_tokens=1000,
    output_tokens=500,
    trace_id=trace_id
)
print(f"Request cost: ${cost:.4f}")

# Generate cost report
report = tracker.generate_cost_report(
    start_time="2025-01-01T00:00:00",
    end_time="2025-01-31T23:59:59"
)
print(f"Total cost: ${report.total_cost:.2f}")
print(f"By user: {report.by_user}")
```

#### 5. Query API (`iris_rag/telemetry/query.py`)

**Responsibilities**:
- Query spans from IRIS TelemetrySpan table
- Calculate aggregated metrics (avg duration, error rate)
- Export metrics in Prometheus format
- Support filtering by operation type, user, time range

**Example Usage**:
```python
from iris_rag.telemetry.query import TelemetryQueryAPI

api = TelemetryQueryAPI(iris_conn)

# Query LLM completion spans
spans = api.query_spans(
    operation_type="llm.completion",
    start_time="2025-01-01T00:00:00",
    limit=100
)

# Get aggregated metrics
metrics = api.get_aggregated_metrics(
    operation_type="llm.completion",
    start_time="2025-01-01T00:00:00"
)
print(f"Average duration: {metrics[0].avg_duration_ms:.2f}ms")
print(f"Error rate: {metrics[0].error_count / metrics[0].total_count * 100:.1f}%")

# Export Prometheus metrics
prometheus_data = api.export_prometheus_metrics()
```

#### 6. IRIS TelemetrySpan Table (ObjectScript)

**Schema** (aligns with aigw_mockup):
```objectscript
Class AIGateway.Storage.TelemetrySpan Extends %Persistent
{
    Property SpanId As %String(MAXLEN = 64) [Required];
    Property TraceId As %String(MAXLEN = 64) [Required];
    Property ParentSpanId As %String(MAXLEN = 64);
    Property OperationType As %String(MAXLEN = 100) [Required];
    Property OperationName As %String(MAXLEN = 200);
    Property StartTime As %BigInt [Required];  // Microseconds since epoch
    Property EndTime As %BigInt;
    Property DurationMicros As %BigInt [Calculated];
    Property Status As %String(VALUELIST = ",OK,ERROR") [Required];
    Property StatusMessage As %String(MAXLEN = 500);
    Property Attributes As %String(MAXLEN = 10000);  // JSON
    Property Username As %String(MAXLEN = 100);
    Property ServiceName As %String(MAXLEN = 100);

    Index TraceIdIdx On TraceId;
    Index OperationTypeIdx On OperationType;
    Index StartTimeIdx On StartTime;
    Index UsernameIdx On Username;
}
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

**Goal**: Set up OpenTelemetry SDK and IRIS storage

**Tasks**:
1. Create `iris_rag/telemetry/` module structure
2. Implement `exporters.py` - OTLP exporter configuration
3. Create IRIS TelemetrySpan table (ObjectScript class)
4. Implement `spans.py` - Base span classes
5. Add configuration support in `config/default_config.yaml`

**Deliverables**:
- ✅ OpenTelemetry SDK configured with OTLP exporter
- ✅ IRIS TelemetrySpan table created
- ✅ Basic span creation and storage working
- ✅ Unit tests for span creation

**Test Verification**:
```python
# Verify span is stored in IRIS
from iris_rag.telemetry.exporters import configure_telemetry, get_tracer

configure_telemetry(service_name="test-service")
tracer = get_tracer("test")

with tracer.start_as_current_span("test.operation") as span:
    span.set_attribute("test.key", "value")

# Query IRIS to verify span exists
query = "SELECT COUNT(*) FROM AIGateway_Storage.TelemetrySpan"
# Assert: count > 0
```

### Phase 2: LLM Instrumentation (Week 2)

**Goal**: Instrument LLM calls with GenAI semantic conventions

**Tasks**:
1. Implement `LLMSpan` class with GenAI attributes
2. Instrument `common/utils.py` `get_llm_func()` wrapper
3. Add automatic token usage tracking
4. Implement `CostTracker` for cost calculation
5. Add unit tests for LLM span creation

**Deliverables**:
- ✅ All LLM calls create spans with GenAI attributes
- ✅ Token usage tracked automatically
- ✅ Cost calculated and stored in span attributes
- ✅ Integration tests with real LLM calls

**GenAI Semantic Conventions**:
```python
span.set_attribute("gen_ai.system", "anthropic")  # or "openai"
span.set_attribute("gen_ai.request.model", "claude-3-5-sonnet")
span.set_attribute("gen_ai.request.temperature", 0.7)
span.set_attribute("gen_ai.request.max_tokens", 1000)
span.set_attribute("gen_ai.usage.prompt_tokens", 100)
span.set_attribute("gen_ai.usage.completion_tokens", 50)
span.set_attribute("gen_ai.response.finish_reason", "stop")
```

### Phase 3: Retrieval Instrumentation (Week 3)

**Goal**: Instrument vector search and knowledge graph operations

**Tasks**:
1. Implement `VectorSearchSpan` class
2. Instrument `IRISVectorStore.search()` method
3. Add graph traversal instrumentation for GraphRAG
4. Implement score metric calculation
5. Add integration tests for retrieval spans

**Deliverables**:
- ✅ Vector search operations create spans
- ✅ Similarity scores tracked (top, avg, min)
- ✅ Graph traversal spans for GraphRAG
- ✅ Hybrid search spans with search type

**Retrieval Span Example**:
```python
# Automatic instrumentation in IRISVectorStore.search()
def search(self, query: str, top_k: int = 10) -> List[Document]:
    with self.instrumentor.start_search_span(
        query=query,
        collection_id=self.collection_name,
        top_k=top_k,
        search_type="vector"
    ) as span:
        results = self._execute_search(query, top_k)
        span.calculate_and_set_scores_from_results(results)
        return results
```

### Phase 4: Pipeline Integration (Week 4)

**Goal**: Integrate tracing into all RAG pipelines

**Tasks**:
1. Instrument `BasicRAGPipeline.query()`
2. Instrument `CRAGPipeline.query()` with evaluator spans
3. Instrument `HybridGraphRAGPipeline.query()` with RRF spans
4. Add parent-child span relationships
5. E2E tests for full pipeline traces

**Deliverables**:
- ✅ All 5 pipelines create hierarchical spans
- ✅ Distributed trace shows complete RAG flow
- ✅ Parent-child relationships correct
- ✅ E2E tests validate trace structure

**Span Hierarchy Example**:
```
RAG Query Span (pipeline.query)
├─ Embedding Span (embedding.encode)
├─ Retrieval Span (retrieval.vector_search)
│  ├─ IRIS SQL Span (iris.sql.query)
│  └─ Score Calculation Span (retrieval.score_calculation)
├─ Graph Traversal Span (graph.traversal)  [GraphRAG only]
│  └─ IRIS SQL Span (iris.sql.query)
└─ LLM Generation Span (llm.completion)
   ├─ Token Counting Span (llm.token_count)
   └─ Cost Calculation Span (cost.calculate)
```

### Phase 5: Query and Reporting (Week 5)

**Goal**: Enable querying and analysis of telemetry data

**Tasks**:
1. Implement `TelemetryQueryAPI` for span queries
2. Add aggregation methods (metrics, error rates)
3. Implement Prometheus exporter
4. Create CLI tool `iris-rag-telemetry`
5. Add cost reporting functionality

**Deliverables**:
- ✅ Query API for spans with filters
- ✅ Aggregated metrics (avg duration, error rate)
- ✅ Prometheus metrics export
- ✅ CLI tool for telemetry queries
- ✅ Cost reports by user/model/time period

**CLI Examples**:
```bash
# Query recent LLM spans
iris-rag-telemetry query --operation-type llm.completion --limit 10

# Get aggregated metrics
iris-rag-telemetry metrics --operation-type llm.completion --start-time 2025-01-01

# Generate cost report
iris-rag-telemetry cost-report --start-time 2025-01-01 --end-time 2025-01-31

# Export Prometheus metrics
iris-rag-telemetry export-prometheus > /tmp/metrics.txt
```

### Phase 6: FastAPI Integration (Week 6)

**Goal**: Automatic instrumentation for REST API

**Tasks**:
1. Enable FastAPI automatic instrumentation
2. Add trace context propagation (W3C headers)
3. Instrument document upload operations
4. Add API-specific span attributes
5. Integration tests for API tracing

**Deliverables**:
- ✅ FastAPI requests create spans automatically
- ✅ Trace context propagated via HTTP headers
- ✅ API operations instrumented
- ✅ Integration tests for API traces

**FastAPI Setup**:
```python
from iris_rag.telemetry.exporters import (
    configure_telemetry,
    configure_fastapi_instrumentation
)

app = FastAPI()

@app.on_event("startup")
async def startup():
    configure_telemetry(service_name="iris-rag-api")
    configure_fastapi_instrumentation(app)

@app.on_event("shutdown")
async def shutdown():
    shutdown_telemetry()
```

---

## Span Definitions

### 1. RAG Pipeline Span

**Operation Type**: `pipeline.query`
**Parent**: None (root span for RAG query)

**Attributes**:
- `pipeline.type`: Pipeline name ("basic", "crag", "graphrag", etc.)
- `pipeline.query`: User query text
- `pipeline.top_k`: Number of documents requested
- `pipeline.method`: Query method for GraphRAG ("hybrid", "vector", "text", "kg", "rrf")
- `pipeline.result_count`: Number of documents retrieved
- `pipeline.answer_length`: Length of generated answer in characters

### 2. LLM Completion Span

**Operation Type**: `llm.completion`
**Parent**: Pipeline span or retrieval span

**GenAI Attributes**:
- `gen_ai.system`: "anthropic", "openai", "azure_openai"
- `gen_ai.request.model`: "claude-3-5-sonnet", "gpt-4-turbo"
- `gen_ai.request.temperature`: 0.0-2.0
- `gen_ai.request.max_tokens`: Integer
- `gen_ai.request.top_p`: 0.0-1.0
- `gen_ai.usage.prompt_tokens`: Integer
- `gen_ai.usage.completion_tokens`: Integer
- `gen_ai.usage.total_tokens`: Integer
- `gen_ai.response.finish_reason`: "stop", "length", "tool_calls", "error"
- `gen_ai.response.id`: Response ID from provider

**Cost Attributes**:
- `cost.total_usd`: Calculated cost in USD
- `cost.input_tokens`: Prompt tokens
- `cost.output_tokens`: Completion tokens
- `cost.model`: Model identifier
- `cost.provider`: Provider name

### 3. Vector Search Span

**Operation Type**: `retrieval.vector_search`
**Parent**: Pipeline span

**Attributes**:
- `retrieval.query`: Text query or embedding description
- `retrieval.collection`: Collection name/ID
- `retrieval.top_k`: Number of results requested
- `retrieval.result_count`: Actual results returned
- `retrieval.top_score`: Highest similarity score
- `retrieval.avg_score`: Average similarity score
- `retrieval.min_score_result`: Lowest score in results
- `retrieval.search_type`: "vector", "hybrid", "keyword"
- `retrieval.embedding_model`: Model used for query embedding
- `retrieval.cache_hit`: Boolean

### 4. Graph Traversal Span

**Operation Type**: `graph.traversal`
**Parent**: Pipeline span

**Attributes**:
- `graph.start_entity`: Starting entity for traversal
- `graph.max_depth`: Maximum traversal depth
- `graph.path_count`: Number of paths found
- `graph.entity_count`: Number of entities retrieved
- `graph.relationship_count`: Number of relationships traversed

### 5. Embedding Span

**Operation Type**: `embedding.encode`
**Parent**: Vector search span or pipeline span

**Attributes**:
- `embedding.model`: Model used ("text-embedding-3-large", "all-MiniLM-L6-v2")
- `embedding.text_length`: Length of input text
- `embedding.vector_dimension`: Dimension of output vector
- `embedding.batch_size`: Number of texts encoded in batch

### 6. CRAG Evaluator Span

**Operation Type**: `crag.evaluation`
**Parent**: Pipeline span

**Attributes**:
- `crag.relevance_score`: Relevance score (0.0-1.0)
- `crag.action`: "correct", "incorrect", "ambiguous"
- `crag.correction_applied`: Boolean
- `crag.external_search_triggered`: Boolean

---

## API Design

### 1. Instrumentor API

```python
from iris_rag.telemetry import (
    configure_telemetry,
    LLMSpanInstrumentor,
    VectorSearchInstrumentor,
    get_current_trace_id
)

# Initialize telemetry (once at startup)
configure_telemetry(
    service_name="iris-rag-api",
    otlp_endpoint="http://localhost:4318",
    sampling_ratio=0.1  # 10% sampling
)

# Create LLM span
llm_instrumentor = LLMSpanInstrumentor(iris_conn, username="alice")

with llm_instrumentor.start_llm_span(
    model="claude-3-5-sonnet",
    provider="anthropic"
) as span:
    response = llm.complete(prompt)
    span.set_usage(response.usage.prompt_tokens, response.usage.completion_tokens)

# Create vector search span
search_instrumentor = VectorSearchInstrumentor(
    iris_conn,
    username="alice",
    trace_id=get_current_trace_id()
)

with search_instrumentor.start_search_span(
    query="diabetes treatment",
    collection_id="pmc-docs",
    top_k=10
) as span:
    results = vector_store.search(query, top_k=10)
    span.calculate_and_set_scores_from_results(results)
```

### 2. Query API

```python
from iris_rag.telemetry.query import TelemetryQueryAPI

api = TelemetryQueryAPI(iris_conn)

# Query spans with filters
spans = api.query_spans(
    operation_type="llm.completion",
    username="alice",
    start_time="2025-01-01T00:00:00",
    end_time="2025-01-31T23:59:59",
    limit=100
)

# Get aggregated metrics
metrics = api.get_aggregated_metrics(
    operation_type="llm.completion",
    start_time="2025-01-01T00:00:00"
)

# Get error rate
error_rate = api.get_error_rate(
    operation_type="llm.completion",
    start_time="2025-01-01T00:00:00"
)
```

### 3. Cost Tracking API

```python
from iris_rag.telemetry.cost import CostTracker

tracker = CostTracker(iris_conn)

# Calculate cost for a request
cost = tracker.calculate_cost(
    model="claude-3-5-sonnet-20241022",
    input_tokens=1000,
    output_tokens=500
)

# Track and store cost
tracker.track_request_cost(
    username="alice",
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    input_tokens=1000,
    output_tokens=500,
    trace_id=trace_id
)

# Generate cost report
report = tracker.generate_cost_report(
    start_time="2025-01-01T00:00:00",
    end_time="2025-01-31T23:59:59"
)

print(f"Total cost: ${report.total_cost:.2f}")
print(f"By user: {report.by_user}")
print(f"By model: {report.by_model}")
```

### 4. CLI Commands

```bash
# Query spans
iris-rag-telemetry query \
  --operation-type llm.completion \
  --username alice \
  --start-time 2025-01-01T00:00:00 \
  --limit 10

# Get metrics
iris-rag-telemetry metrics \
  --operation-type llm.completion \
  --start-time 2025-01-01T00:00:00

# Cost report
iris-rag-telemetry cost-report \
  --start-time 2025-01-01T00:00:00 \
  --end-time 2025-01-31T23:59:59 \
  --by-user

# Export Prometheus metrics
iris-rag-telemetry export-prometheus > /tmp/metrics.txt
```

---

## Configuration

### Environment Variables

```bash
# OTLP Exporter Configuration
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318  # OpenTelemetry Collector
OTEL_SERVICE_NAME=iris-rag-api  # Service identifier
OTEL_TRACES_SAMPLER=traceidratio  # Sampling strategy
OTEL_TRACES_SAMPLER_ARG=0.1  # 10% sampling in production
OTEL_EXPORTER_OTLP_PROTOCOL=grpc  # Protocol (grpc or http/protobuf)

# IRIS Native OTel Integration (IRIS 2024.1+)
# These are set in IRIS configuration, not Python environment
# OTELMetrics=1  # Enable IRIS native metrics export
# OTELInterval=10  # Push interval in seconds
```

### Configuration File (`config/default_config.yaml`)

```yaml
telemetry:
  # Enable/disable telemetry
  enabled: true

  # Service identification
  service_name: "iris-rag-api"
  service_version: "0.6.0"

  # OTLP exporter configuration
  otlp:
    endpoint: "http://localhost:4318"
    protocol: "grpc"  # or "http/protobuf"

  # Sampling configuration
  sampling:
    type: "traceidratio"  # Sample X% of traces
    ratio: 0.1  # 10% in production

  # Storage configuration
  storage:
    iris_enabled: true  # Store spans in IRIS TelemetrySpan table
    otlp_enabled: true  # Export to OTLP collector

  # Cost tracking configuration
  cost_tracking:
    enabled: true
    pricing_config: "config/llm_pricing.yaml"

  # Console export (for debugging)
  console_export: false  # Enable in development only
```

### Pricing Configuration (`config/llm_pricing.yaml`)

```yaml
pricing:
  - model: "claude-3-5-sonnet-20241022"
    provider: "anthropic"
    input_price_per_1k: 0.003
    output_price_per_1k: 0.015
    currency: "USD"

  - model: "claude-3-5-haiku-20241022"
    provider: "anthropic"
    input_price_per_1k: 0.0008
    output_price_per_1k: 0.004
    currency: "USD"

  - model: "gpt-4-turbo"
    provider: "openai"
    input_price_per_1k: 0.01
    output_price_per_1k: 0.03
    currency: "USD"

  - model: "gpt-3.5-turbo"
    provider: "openai"
    input_price_per_1k: 0.0005
    output_price_per_1k: 0.0015
    currency: "USD"

  - model: "text-embedding-3-large"
    provider: "openai"
    input_price_per_1k: 0.00013
    output_price_per_1k: 0.0
    currency: "USD"

  # Default pricing for unknown models
  - model: "default"
    provider: "unknown"
    input_price_per_1k: 0.01
    output_price_per_1k: 0.03
    currency: "USD"
```

---

## Migration Strategy

### For Existing Users

**Zero Breaking Changes**: Telemetry is opt-in and disabled by default.

**Migration Path**:

1. **Update iris-vector-rag** to v0.6.0
   ```bash
   pip install --upgrade iris-vector-rag
   ```

2. **Enable telemetry in configuration**
   ```yaml
   # config/default_config.yaml
   telemetry:
     enabled: true
   ```

3. **Set environment variables**
   ```bash
   export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
   export OTEL_SERVICE_NAME=iris-rag-api
   ```

4. **Create IRIS TelemetrySpan table** (one-time setup)
   ```bash
   python -m iris_rag.telemetry.setup_db
   ```

5. **Start using telemetry**
   ```python
   from iris_rag.telemetry import configure_telemetry

   # At application startup
   configure_telemetry(service_name="my-rag-app")

   # Telemetry now automatic for all pipeline queries
   ```

### For New Users

Telemetry disabled by default. Enable with:
```bash
export IRIS_RAG_TELEMETRY_ENABLED=true
```

---

## Testing Strategy

### Unit Tests

**Coverage**: 90%+ for telemetry module

**Test Files**:
- `tests/unit/telemetry/test_spans.py` - Span creation and attributes
- `tests/unit/telemetry/test_exporters.py` - OTLP configuration
- `tests/unit/telemetry/test_retrieval.py` - Retrieval span instrumentation
- `tests/unit/telemetry/test_cost.py` - Cost calculation
- `tests/unit/telemetry/test_query.py` - Query API

**Mock Strategy**:
- Mock IRIS connection for span storage
- Mock OTLP exporter for export verification
- Mock LLM responses for token usage tracking

### Integration Tests

**Coverage**: All RAG pipelines instrumented

**Test Files**:
- `tests/integration/test_telemetry_basic_pipeline.py` - BasicRAG tracing
- `tests/integration/test_telemetry_crag_pipeline.py` - CRAG tracing
- `tests/integration/test_telemetry_graphrag_pipeline.py` - GraphRAG tracing
- `tests/integration/test_telemetry_iris_storage.py` - IRIS span storage
- `tests/integration/test_telemetry_cost_tracking.py` - Cost tracking

**Verification**:
- Verify spans stored in IRIS TelemetrySpan table
- Verify parent-child relationships correct
- Verify GenAI semantic conventions followed
- Verify cost calculation accuracy

### E2E Tests

**Scenario 1: Full RAG Query with Tracing**
```python
def test_full_rag_query_with_tracing():
    # Configure telemetry
    configure_telemetry(service_name="test-rag")

    # Execute RAG query
    pipeline = create_pipeline("basic")
    result = pipeline.query("What is diabetes?", top_k=5)

    # Verify trace structure
    api = TelemetryQueryAPI(iris_conn)
    spans = api.query_spans(limit=100)

    # Assert: Pipeline span exists
    pipeline_span = [s for s in spans if s.operation_type == "pipeline.query"][0]
    assert pipeline_span is not None

    # Assert: Child spans exist (embedding, retrieval, llm)
    child_spans = [s for s in spans if s.parent_span_id == pipeline_span.span_id]
    assert len(child_spans) >= 3

    # Assert: LLM span has GenAI attributes
    llm_span = [s for s in child_spans if s.operation_type == "llm.completion"][0]
    assert "gen_ai.system" in llm_span.attributes
    assert "gen_ai.usage.prompt_tokens" in llm_span.attributes
```

**Scenario 2: Cost Tracking**
```python
def test_cost_tracking_end_to_end():
    # Configure telemetry with cost tracking
    configure_telemetry(service_name="test-rag")
    tracker = CostTracker(iris_conn)

    # Execute multiple queries
    pipeline = create_pipeline("basic")
    for i in range(10):
        pipeline.query(f"Query {i}", top_k=5)

    # Generate cost report
    report = tracker.generate_cost_report(
        start_time=start_time,
        end_time=end_time
    )

    # Assert: Cost calculated
    assert report.total_cost > 0
    assert len(report.by_model) > 0
```

---

## Dependencies

### Python Dependencies

**New Dependencies**:
```toml
[project.dependencies]
opentelemetry-api = ">=1.20.0"
opentelemetry-sdk = ">=1.20.0"
opentelemetry-exporter-otlp-proto-grpc = ">=1.20.0"
opentelemetry-instrumentation-fastapi = ">=0.41b0"
opentelemetry-instrumentation-requests = ">=0.41b0"
prometheus-client = ">=0.18.0"  # For Prometheus export
```

**Optional Dependencies** (for development):
```toml
[project.optional-dependencies]
telemetry = [
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
    "opentelemetry-exporter-otlp-proto-grpc>=1.20.0",
]
```

### IRIS Dependencies

**IRIS Version**: 2024.1+ (for native OTel support)

**Required IRIS Features**:
- Native OpenTelemetry support (OTELMetrics parameter)
- TelemetrySpan table schema support
- SQL query support for span queries

**Optional IRIS Features**:
- `/api/monitor/metrics` endpoint (Prometheus metrics)
- OTLP Collector integration

---

## Risks and Mitigations

### Risk 1: Performance Overhead

**Risk**: Telemetry adds latency to RAG queries

**Mitigation**:
- **Sampling**: 10% sampling in production reduces overhead by 90%
- **Async Storage**: IRIS span storage is asynchronous (doesn't block query)
- **Batch Export**: OTLP spans exported in batches (every 5 seconds)
- **Benchmarks**: Run performance tests before/after telemetry to measure impact

**Expected Overhead**: <5ms per query with 10% sampling

### Risk 2: IRIS Storage Growth

**Risk**: TelemetrySpan table grows too large (millions of spans)

**Mitigation**:
- **Retention Policy**: Delete spans older than 30 days (configurable)
- **Archival**: Move old spans to IRIS Archive before deletion
- **Indexes**: Create indexes on TraceId, OperationType, StartTime for fast queries
- **Partitioning**: Use IRIS date partitioning for TelemetrySpan table

**Expected Growth**: ~1 GB per 1M spans (with 30-day retention)

### Risk 3: OTLP Collector Unavailable

**Risk**: OTLP collector down causes span export failures

**Mitigation**:
- **Dual Storage**: IRIS storage still works if OTLP fails
- **Retry Logic**: OpenTelemetry SDK retries failed exports
- **Graceful Degradation**: Telemetry failures don't break RAG queries
- **Monitoring**: Alert if OTLP export error rate > 10%

**Fallback**: Spans always stored in IRIS even if OTLP export fails

### Risk 4: Cost Calculation Accuracy

**Risk**: Pricing tiers become outdated, causing inaccurate cost reports

**Mitigation**:
- **Configurable Pricing**: Store pricing in YAML file (easy to update)
- **Version Tracking**: Log pricing config version in cost reports
- **Pricing API**: Consider integrating with provider pricing APIs
- **Manual Review**: Finance team reviews cost reports monthly

**Update Frequency**: Review pricing quarterly or when providers announce changes

---

## Success Metrics

### Adoption Metrics

- **Week 1**: 10% of iris-vector-rag users enable telemetry
- **Month 1**: 50% of iris-vector-rag users enable telemetry
- **Month 3**: 80% of iris-vector-rag users enable telemetry

### Performance Metrics

- **Latency Overhead**: <5ms per query (with 10% sampling)
- **Storage Overhead**: <1 GB per 1M spans
- **Export Success Rate**: >99% of spans exported to OTLP collector

### Business Metrics

- **Cost Visibility**: 100% of LLM costs tracked per user
- **Issue Resolution Time**: 50% reduction (with distributed tracing)
- **Performance Optimization**: 20% improvement (with span-level metrics)

### Technical Metrics

- **Span Completeness**: 100% of RAG queries create spans
- **Attribute Accuracy**: 100% of LLM spans have GenAI attributes
- **Parent-Child Correctness**: 100% of spans have correct parent IDs

---

## Appendix A: IRIS Native OTel Integration

### IRIS 2024.1+ Native OpenTelemetry

**Configuration Parameters**:
```
OTELMetrics = 1  # Enable metrics export
OTELInterval = 10  # Push interval in seconds
OTEL_EXPORTER_OTLP_ENDPOINT = http://localhost:4318  # Collector endpoint
```

**Prometheus Metrics Endpoint**:
```
GET /api/monitor/metrics
```

**IRIS Native Telemetry Integration**:
- IRIS automatically exports system metrics (CPU, memory, queries)
- Python spans stored in IRIS TelemetrySpan table
- IRIS and Python spans share same OTLP collector
- Grafana dashboards can visualize both IRIS and Python metrics

### Shared OTLP Collector

**Architecture**:
```
┌─────────────────────┐
│ IRIS Native Metrics │ ──┐
└─────────────────────┘   │
                          ▼
┌─────────────────────┐ ┌────────────────────┐
│ Python OTel Spans   │ │ OTLP Collector     │
│ (iris-vector-rag)   │ │ (port 4318)        │
└─────────────────────┘ └────────────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │ Grafana     │
                    │ Prometheus  │
                    │ Datadog     │
                    └─────────────┘
```

**Benefits**:
- Unified observability for IRIS + Python
- Correlate IRIS queries with RAG operations
- Single monitoring dashboard for entire stack

---

## Appendix B: Example Traces

### Example 1: BasicRAG Query

```
Trace ID: 7c3e6679-7425-40de-944b-e07fc1f90ae7

Pipeline Span (pipeline.query) - 2.5s
├─ Embedding Span (embedding.encode) - 150ms
│  └─ attributes:
│     - embedding.model: "text-embedding-3-large"
│     - embedding.text_length: 45
│     - embedding.vector_dimension: 1536
│
├─ Vector Search Span (retrieval.vector_search) - 250ms
│  ├─ attributes:
│  │  - retrieval.query: "What are the symptoms of diabetes?"
│  │  - retrieval.collection: "pmc-documents"
│  │  - retrieval.top_k: 5
│  │  - retrieval.result_count: 5
│  │  - retrieval.top_score: 0.95
│  │  - retrieval.avg_score: 0.82
│  │  - retrieval.search_type: "vector"
│  └─ IRIS SQL Span (iris.sql.query) - 200ms
│
└─ LLM Completion Span (llm.completion) - 2.0s
   ├─ attributes:
   │  - gen_ai.system: "anthropic"
   │  - gen_ai.request.model: "claude-3-5-sonnet-20241022"
   │  - gen_ai.request.temperature: 0.7
   │  - gen_ai.usage.prompt_tokens: 1000
   │  - gen_ai.usage.completion_tokens: 500
   │  - cost.total_usd: 0.0105
   │  - cost.model: "claude-3-5-sonnet-20241022"
   └─ Token Counting Span (llm.token_count) - 10ms
```

### Example 2: HybridGraphRAG Query

```
Trace ID: a1b2c3d4-e5f6-4789-a0b1-c2d3e4f56789

Pipeline Span (pipeline.query) - 5.2s
├─ Embedding Span (embedding.encode) - 150ms
│
├─ Hybrid Search Span (retrieval.hybrid_search) - 800ms
│  ├─ attributes:
│  │  - retrieval.search_type: "hybrid"
│  │  - retrieval.top_k: 10
│  │  - retrieval.result_count: 30  # Before RRF fusion
│  │
│  ├─ Vector Search Span (retrieval.vector_search) - 250ms
│  ├─ Text Search Span (retrieval.text_search) - 200ms
│  └─ RRF Fusion Span (retrieval.rrf_fusion) - 300ms
│
├─ Graph Traversal Span (graph.traversal) - 1.5s
│  ├─ attributes:
│  │  - graph.start_entity: "Diabetes"
│  │  - graph.max_depth: 2
│  │  - graph.entity_count: 15
│  │  - graph.relationship_count: 20
│  │
│  └─ IRIS SQL Span (iris.sql.query) - 1.2s
│
└─ LLM Completion Span (llm.completion) - 2.5s
   └─ attributes:
      - gen_ai.usage.prompt_tokens: 2500  # Larger due to graph context
      - gen_ai.usage.completion_tokens: 800
      - cost.total_usd: 0.0195
```

---

## Appendix C: Comparison with aigw_mockup

### Alignment Points

This specification **aligns with aigw_mockup** in:

1. **Dual Storage Approach**:
   - ✅ IRIS TelemetrySpan table for internal queries
   - ✅ OTLP export for external platforms
   - ✅ Same TelemetrySpan schema

2. **GenAI Semantic Conventions**:
   - ✅ `gen_ai.*` attributes for LLM spans
   - ✅ `retrieval.*` attributes for search spans
   - ✅ Cost tracking in span attributes

3. **Architecture Patterns**:
   - ✅ Context manager API (`with instrumentor.start_span()`)
   - ✅ Span hierarchy (parent-child relationships)
   - ✅ W3C Trace Context propagation

4. **Configuration**:
   - ✅ `OTEL_EXPORTER_OTLP_ENDPOINT` environment variable
   - ✅ Sampling configuration (10% in production)
   - ✅ Console export for debugging

### Differences from aigw_mockup

1. **RAG-Specific Spans**:
   - iris-vector-rag adds RAG pipeline spans
   - aigw_mockup focused on MCP tool spans

2. **Pipeline Instrumentation**:
   - iris-vector-rag instruments 5 RAG pipelines
   - aigw_mockup instruments FastMCP server

3. **Domain**:
   - iris-vector-rag: Retrieval-Augmented Generation
   - aigw_mockup: AI Gateway with MCP tools

---

## Appendix D: CLI Reference

### iris-rag-telemetry CLI

**Installation**:
```bash
# CLI installed automatically with iris-vector-rag[telemetry]
pip install iris-vector-rag[telemetry]
```

**Commands**:

#### Query Spans
```bash
iris-rag-telemetry query \
  --operation-type llm.completion \
  --username alice \
  --start-time 2025-01-01T00:00:00 \
  --end-time 2025-01-31T23:59:59 \
  --limit 100 \
  --format json
```

**Options**:
- `--operation-type`: Filter by operation type
- `--username`: Filter by user
- `--trace-id`: Filter by trace ID
- `--start-time`: Start time (ISO format)
- `--end-time`: End time (ISO format)
- `--limit`: Maximum results (default: 100)
- `--format`: Output format (json, table, csv)

#### Get Metrics
```bash
iris-rag-telemetry metrics \
  --operation-type llm.completion \
  --start-time 2025-01-01T00:00:00 \
  --group-by model
```

**Options**:
- `--operation-type`: Operation type
- `--start-time`: Start time
- `--end-time`: End time
- `--group-by`: Group by field (operation_type, model, username)

#### Cost Report
```bash
iris-rag-telemetry cost-report \
  --start-time 2025-01-01T00:00:00 \
  --end-time 2025-01-31T23:59:59 \
  --by-user \
  --by-model \
  --format table
```

**Options**:
- `--start-time`: Start time (required)
- `--end-time`: End time (required)
- `--by-user`: Show cost by user
- `--by-model`: Show cost by model
- `--by-provider`: Show cost by provider
- `--format`: Output format (json, table, csv)

#### Export Prometheus Metrics
```bash
iris-rag-telemetry export-prometheus \
  --operation-types llm.completion,retrieval.vector_search \
  --output /tmp/metrics.txt
```

**Options**:
- `--operation-types`: Comma-separated list of operation types
- `--output`: Output file (default: stdout)

---

## Appendix E: Next Steps

### After Feature Approval

1. **Create GitHub Issue**: `Feature: OpenTelemetry Integration #XXX`
2. **Create Feature Branch**: `feature/opentelemetry-integration`
3. **Implementation Phases**: Follow 6-week implementation plan
4. **Documentation**: Update README.md, CLAUDE.md, and user guides
5. **Release**: Package as v0.6.0 with comprehensive CHANGELOG

### Documentation Updates Needed

- [ ] README.md - Add telemetry section
- [ ] CLAUDE.md - Add telemetry commands
- [ ] Architecture diagram - Add telemetry components
- [ ] User guide - Add telemetry configuration instructions
- [ ] API documentation - Add telemetry API reference

---

**End of Feature Specification**
