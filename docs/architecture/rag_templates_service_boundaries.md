# RAG Templates Service Boundaries & Integration Patterns

> Status source of truth: Implementation progress and E2E readiness live in [UNIFIED_PROJECT_ROADMAP.md](../UNIFIED_PROJECT_ROADMAP.md). This guide documents boundaries/patterns; defer status to the roadmap.

## 1. Executive Summary

This document defines the service boundaries and integration patterns for the RAG-Templates Adapter Layer (M1), establishing clean separation between the rag-templates RAG ecosystem and the kg-ticket-resolver knowledge graph memory system while ensuring extensible, resilient, and performant integration.

## 2. Service Boundary Architecture

### 2.1 Core Service Boundaries

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SERVICE BOUNDARY OVERVIEW                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    KG-TICKET-RESOLVER DOMAIN                           │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐│ │
│  │  │            Knowledge Graph Memory System                           ││ │
│  │  │  • ProjectMemory, TicketMemory, Insight, Trend nodes             ││ │
│  │  │  • LightRAG incremental indexing patterns                        ││ │
│  │  │  • Memory & Insight APIs with SLOs                               ││ │
│  │  └─────────────────────────────────────────────────────────────────────┘│ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    ▲                                       │
│                                    │                                       │
│                        ┌───────────▼───────────┐                          │
│                        │   ADAPTER BOUNDARY    │                          │
│                        │  rag_templates_bridge │                          │
│                        │  • Circuit Breakers   │                          │
│                        │  • Performance Monitoring                        │
│                        │  • Error Recovery     │                          │
│                        └───────────┬───────────┘                          │
│                                    │                                       │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                     RAG-TEMPLATES DOMAIN                               │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐│ │
│  │  │               RAG Pipeline Ecosystem                               ││ │
│  │  │  • BasicRAG, CRAG, GraphRAG, BasicRAGReranking                    ││ │
│  │  │  • IRIS vector store and schema management                        ││ │
│  │  │  • Environment-independent configuration                          ││ │
│  │  └─────────────────────────────────────────────────────────────────────┘│ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Service Responsibility Matrix

| Service Layer | Responsibility | Interface Contract | Error Handling | Performance Target |
|---------------|---------------|-------------------|----------------|-------------------|
| **kg-ticket-resolver** | Knowledge graph memory management, insight generation | Memory APIs, GraphQL/REST | Graceful degradation to basic RAG | <200ms p95 |
| **Adapter Bridge** | Interface abstraction, circuit breaking, monitoring | [`RAGResponse`](../iris_vector_rag/adapters/rag_templates_bridge.py:42), async/await | Circuit breaker + fallback | <50ms overhead |
| **RAG Pipelines** | Document processing, vector search, answer generation | [`RAGPipeline`](../iris_rag/core/base.py:12) interface | Pipeline-specific error handling | <500ms p95 |
| **IRIS Database** | Vector storage, graph data persistence | SQL + vector functions | Connection pooling + retry | <100ms query time |

## 3. Integration Patterns

### 3.1 Query Processing Flow

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  kg-ticket-      │    │  rag_templates_  │    │  RAG Pipeline    │
│  resolver        │    │  bridge          │    │  (Basic/CRAG/    │
│                  │    │                  │    │   Graph/Rerank) │
└────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘
         │                       │                       │
         │ 1. query(text,        │                       │
         │    technique?)        │                       │
         ├──────────────────────►│                       │
         │                       │ 2. circuit_breaker_   │
         │                       │    check()            │
         │                       ├──────────┐            │
         │                       │          ▼            │
         │                       │ 3. technique_select() │
         │                       ├──────────┬────────────►│
         │                       │          │ 4. execute_│
         │                       │          │    query() │
         │                       │          │            │
         │                       │ 5. standardize_       │
         │                       │    response()         │
         │                       │◄─────────┴────────────┤
         │ 6. RAGResponse        │                       │
         │    (standardized)     │                       │
         │◄──────────────────────┤                       │
         │                       │                       │
```

### 3.2 Incremental Indexing Pattern (LightRAG Style)

```python
# Integration with kg-ticket-resolver incremental indexing
async def handle_document_changes(change_events: List[ChangeEvent]) -> IndexingResult:
    """
    Process incremental document changes following LightRAG patterns.
    
    Pattern:
    1. Batch changes by technique requirements
    2. Apply changes incrementally to avoid full reindex
    3. Update knowledge graph memory nodes
    4. Maintain consistency across RAG and KG systems
    """
    
    results = []
    for technique in [RAGTechnique.BASIC, RAGTechnique.GRAPH]:
        if change_events_for_technique(change_events, technique):
            result = await bridge.index_documents(
                documents=extract_documents(change_events),
                technique=technique,
                incremental=True
            )
            results.append(result)
    
    # Sync with knowledge graph memory
    await sync_kg_memory_nodes(results)
    return IndexingResult.aggregate(results)
```

### 3.3 Circuit Breaker Integration Pattern

```python
# Circuit breaker states and transitions
class CircuitBreakerPattern:
    """
    Implements fault tolerance for RAG pipeline access.
    
    States:
    - CLOSED: Normal operation, failures tracked
    - OPEN: Failing fast, requests rejected
    - HALF_OPEN: Testing recovery, limited requests
    
    Fallback Strategy:
    - GraphRAG failure → BasicRAG
    - CRAG failure → BasicRAG  
    - BasicRAG failure → Error response with cached context
    """
    
    async def execute_with_fallback(self, query: str, technique: RAGTechnique) -> RAGResponse:
        try:
            return await self._execute_primary(query, technique)
        except CircuitBreakerOpenError:
            logger.warning(f"Circuit breaker open for {technique}, falling back")
            return await self._execute_fallback(query, self.fallback_technique)
        except Exception as e:
            logger.error(f"Primary and fallback failed: {e}")
            return self._create_error_response(query, e)
```

## 4. Error Handling & Graceful Degradation

### 4.1 Error Handling Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        ERROR HANDLING LAYERS                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Layer 1: Circuit Breaker (Adapter Bridge)                     │
│  ├─ Technique-specific circuit breakers                        │
│  ├─ Automatic fallback to simpler techniques                   │
│  └─ Fast failure when system is overloaded                     │
│                                                                 │
│  Layer 2: Pipeline Error Handling (RAG Core)                   │
│  ├─ Timeout handling (30s default)                             │
│  ├─ Resource exhaustion recovery                               │
│  └─ Database connection retry logic                            │
│                                                                 │
│  Layer 3: Graceful Degradation (kg-ticket-resolver)            │
│  ├─ Cached response serving                                     │
│  ├─ Simplified query processing                                │
│  └─ User notification of reduced functionality                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Graceful Degradation Strategy

| Failure Scenario | Primary Response | Fallback Response | User Experience |
|------------------|------------------|-------------------|-----------------|
| GraphRAG pipeline failure | Switch to BasicRAG | Serve from cache | Slightly reduced context quality |
| Database connectivity loss | Use connection pool retry | Return cached results | "System recovering" message |
| Memory API timeout | Skip memory integration | Direct RAG response | Missing contextual insights |
| Complete system failure | Error response | Cached emergency response | "Please try again later" |

## 5. Performance & Monitoring Integration

### 5.1 Performance SLO Enforcement

```yaml
# Service Level Objectives by boundary
slo_targets:
  adapter_bridge:
    latency_p95_ms: 50
    error_rate_percent: 0.1
    availability_percent: 99.9
    
  rag_pipelines:
    latency_p95_ms: 500
    throughput_qps: 100
    memory_usage_gb: 8
    
  kg_memory_integration:
    sync_latency_ms: 200
    consistency_lag_ms: 1000
    update_success_rate: 99.5
```

### 5.2 Cross-System Monitoring Hooks

```python
# Observability integration points
class MonitoringIntegration:
    """
    Provides monitoring hooks across service boundaries.
    """
    
    @measure_performance("rag_query_processing")
    async def process_query(self, query: str) -> RAGResponse:
        """Query processing with full observability."""
        with self.tracer.span("kg_ticket_resolver.query") as span:
            span.set_attribute("query.technique", self.technique)
            span.set_attribute("query.length", len(query))
            
            # Bridge monitoring
            response = await self.bridge.query(query, technique=self.technique)
            
            # Record metrics
            self.metrics.record_query_latency(response.processing_time_ms)
            self.metrics.record_technique_usage(response.technique_used)
            
            return response
    
    def setup_health_endpoints(self):
        """Expose health check endpoints across all boundaries."""
        return {
            "/health/bridge": self.bridge.get_health_status,
            "/health/pipelines": self.get_pipeline_health,
            "/health/integration": self.get_integration_health,
            "/metrics": self.get_all_metrics
        }
```

## 6. Security & Data Isolation

### 6.1 Service Isolation Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│                      SECURITY BOUNDARIES                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Network Boundary: kg-ticket-resolver ↔ rag-templates         │
│  ├─ No direct database access from kg-ticket-resolver          │
│  ├─ All communication through adapter bridge                   │
│  └─ API key authentication for sensitive operations            │
│                                                                 │
│  Data Boundary: RAG vs Knowledge Graph                         │
│  ├─ RAG data isolated in IRIS RAG schema                       │
│  ├─ KG memory data in separate mem0/Supabase                   │
│  └─ Cross-system sync via controlled interfaces only           │
│                                                                 │
│  Process Boundary: Memory isolation                            │
│  ├─ RAG pipelines run in isolated Python processes             │
│  ├─ Knowledge graph in separate service containers             │
│  └─ Bridge adapter manages resource limits                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Configuration Security

- **Environment Variables Only**: No hardcoded secrets in any configuration files
- **Credential Rotation**: Support for runtime credential updates via bridge
- **Audit Logging**: All cross-system calls logged with request/response metadata
- **Rate Limiting**: Per-client rate limiting to prevent system abuse

## 7. Integration Points Summary

### 7.1 Primary Integration Interfaces

| Interface | Location | Purpose | Protocol |
|-----------|----------|---------|----------|
| [`RAGTemplatesBridge`](../iris_vector_rag/adapters/rag_templates_bridge.py:85) | iris_vector_rag/adapters/rag_templates_bridge.py | Main integration adapter | Python async/await |
| [`RAGResponse`](../iris_vector_rag/adapters/rag_templates_bridge.py:42) | Standard response format | Cross-system data contract | JSON/dataclass |
| [rag_integration.yaml](../config/rag_integration.yaml:1) | Configuration management | Environment-independent config | YAML |
| Health endpoints | /health/* routes | System monitoring | HTTP REST |

### 7.2 Future Integration Points

- **PRefLexOR Bridge**: Ready for integration via adapter pattern
- **Additional RAG Techniques**: Extensible through pipeline registry
- **Enhanced Memory Types**: Support for new kg-ticket-resolver memory nodes
- **Multi-tenant Support**: User context isolation for enterprise deployment

## 8. Implementation Guidelines

### 8.1 Development Best Practices

1. **Interface First**: Always define interfaces before implementation
2. **Error Handling**: Every cross-boundary call must have timeout and error handling
3. **Observability**: All operations must be measurable and traceable
4. **Testing**: Integration tests must cover all boundary scenarios
5. **Documentation**: Service contracts must be documented and versioned

### 8.2 Deployment Considerations

- **Health Checks**: Each service boundary must expose health endpoints
- **Rolling Updates**: Support zero-downtime updates via circuit breaker
- **Resource Limits**: Clearly defined CPU/memory limits per service
- **Monitoring**: Cross-boundary call tracing and metrics collection

This architecture ensures clean separation of concerns while enabling powerful integration between the RAG-templates ecosystem and the kg-ticket-resolver knowledge graph memory system.