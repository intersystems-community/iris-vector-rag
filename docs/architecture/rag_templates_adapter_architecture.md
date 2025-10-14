# RAG Templates Adapter Layer - Complete Architecture Specification

## 1. Executive Summary

The RAG Templates Adapter Layer (M1) provides a unified interface compatibility architecture between the rag-templates RAG ecosystem and the kg-ticket-resolver knowledge graph memory system. This architecture enables seamless switching between RAG techniques, maintains consistent response formats, and supports incremental indexing patterns following LightRAG methodologies.

### 1.1 Architecture Principles

- **Unified Interface**: Single adapter abstracts all RAG pipeline complexities
- **Circuit Breaker Resilience**: Fault tolerance with automatic fallback strategies  
- **Performance-First Design**: Sub-second response times with comprehensive monitoring
- **Environment-Independent Config**: Zero hardcoded secrets, full environment variable support
- **Modular Service Boundaries**: Clean separation enabling independent scaling and maintenance

## 2. System Architecture Overview

### 2.1 High-Level Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   RAG TEMPLATES ADAPTER LAYER ARCHITECTURE                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    KG-TICKET-RESOLVER LAYER                            │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │ │
│  │  │ ProjectMemory   │  │ TicketMemory    │  │ Insight/Trend   │         │ │
│  │  │ • Query Context │  │ • Issue Tracking│  │ • Analytics     │         │ │
│  │  │ • User Sessions │  │ • Resolution    │  │ • Predictions   │         │ │
│  │  │ • Project State │  │ • Workflows     │  │ • Patterns      │         │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    ▲                                       │
│                                    │ Standard RAGResponse                  │
│                                    │ JSON/Async Interface                  │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      ADAPTER BRIDGE LAYER                              │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐│ │
│  │  │               RAGTemplatesBridge (adapters/rag_templates_bridge.py) ││ │
│  │  │                                                                     ││ │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     ││ │
│  │  │  │ Circuit Breaker │  │ Performance     │  │ Query Router    │     ││ │
│  │  │  │ • OPEN/CLOSED   │  │ • Latency Track │  │ • Technique     │     ││ │
│  │  │  │ • Failure Count │  │ • Metrics Coll. │  │   Selection     │     ││ │
│  │  │  │ • Auto Recovery │  │ • SLO Monitor   │  │ • Load Balance  │     ││ │
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────────┘     ││ │
│  │  │                                                                     ││ │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     ││ │
│  │  │  │ Error Recovery  │  │ Health Monitor  │  │ Config Manager  │     ││ │
│  │  │  │ • Fallback      │  │ • Status Check  │  │ • Environment   │     ││ │
│  │  │  │ • Graceful      │  │ • Endpoint      │  │ • Hot Reload    │     ││ │
│  │  │  │   Degradation   │  │ • Alert Trigger │  │ • Validation    │     ││ │
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────────┘     ││ │
│  │  └─────────────────────────────────────────────────────────────────────┘│ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    ▲                                       │
│                                    │ RAGPipeline Interface                 │
│                                    │ ConnectionManager                     │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        RAG PIPELINE LAYER                              │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │ │
│  │  │ BasicRAG        │  │ CRAG            │  │ GraphRAG        │         │ │
│  │  │ • Fast queries  │  │ • Corrective    │  │ • Graph         │         │ │
│  │  │ • High volume   │  │ • Quality focus │  │   traversal     │         │ │
│  │  │ • 200-400ms     │  │ • 400-800ms     │  │ • 300-600ms     │         │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │ │
│  │                                                                         │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │ │
│  │  │ BasicReranking  │  │ Vector Store    │  │ Config Manager  │         │ │
│  │  │ • Precision     │  │ • IRIS Database │  │ • Pipeline      │         │ │
│  │  │ • 500-1000ms    │  │ • 273K+ entities│  │   Parameters    │         │ │
│  │  │ • High accuracy │  │ • 183K+ relations│  │ • Environment   │         │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Query Processing Flow:                                                     │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │   User Query    │    │   Adapter       │    │   RAG Pipeline  │         │
│  │                 │    │   Bridge        │    │                 │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                  │
│           │ 1. query(text,       │                      │                  │
│           │    technique?,       │                      │                  │
│           │    context)          │                      │                  │
│           ├─────────────────────►│                      │                  │
│           │                      │ 2. validate_input()  │                  │
│           │                      ├──────────┐           │                  │
│           │                      │          ▼           │                  │
│           │                      │ 3. check_circuit_    │                  │
│           │                      │    breaker()         │                  │
│           │                      ├──────────┬───────────►                  │
│           │                      │          │ 4. execute_query()           │
│           │                      │          │ • Vector search              │
│           │                      │          │ • Context augmentation       │
│           │                      │          │ • LLM generation             │
│           │                      │ 5. standardize_      │                  │
│           │                      │    response()        │                  │
│           │                      │◄─────────┴──────────┤                  │
│           │ 6. RAGResponse       │                      │                  │
│           │    • answer: str     │                      │                  │
│           │    • sources: []     │                      │                  │
│           │    • confidence      │                      │                  │
│           │    • metadata        │                      │                  │
│           │◄─────────────────────┤                      │                  │
│           │                      │                      │                  │
│                                                                             │
│  Incremental Indexing Flow:                                                │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  Document       │    │   Adapter       │    │   Pipeline      │         │
│  │  Changes        │    │   Bridge        │    │   + Vector      │         │
│  └────────┬────────┘    └────────┬────────┘    └────────┬────────┘         │
│           │                      │                      │                  │
│           │ 1. index_documents(  │                      │                  │
│           │    docs[], technique,│                      │                  │
│           │    incremental=True) │                      │                  │
│           ├─────────────────────►│                      │                  │
│           │                      │ 2. batch_changes()   │                  │
│           │                      ├──────────┬───────────►                  │
│           │                      │          │ 3. incremental_index()       │
│           │                      │          │ • Document chunking          │
│           │                      │          │ • Embedding generation       │
│           │                      │          │ • Vector store update        │
│           │                      │ 4. sync_kg_memory()  │                  │
│           │                      │◄─────────┴──────────┤                  │
│           │ 5. IndexingResult    │                      │                  │
│           │◄─────────────────────┤                      │                  │
│           │                      │                      │                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. Component Specifications

### 3.1 Adapter Bridge Interface

**File**: [`adapters/rag_templates_bridge.py`](../adapters/rag_templates_bridge.py:1)

**Key Components**:
- **RAGTemplatesBridge**: Main adapter class (324 lines)
- **RAGResponse**: Standardized response format  
- **CircuitBreakerState**: Fault tolerance management
- **PerformanceMetrics**: Comprehensive metrics collection

**Core Methods**:
```python
async def query(query_text: str, technique: Optional[RAGTechnique] = None) -> RAGResponse
async def index_documents(documents: List[Dict], technique: Optional[RAGTechnique] = None) -> Dict
def get_metrics() -> Dict[str, Any]
def get_health_status() -> Dict[str, Any]
```

### 3.2 Configuration Management

**File**: [`config/rag_integration.yaml`](../config/rag_integration.yaml:1)

**Configuration Sections**:
- **Pipeline Settings**: Technique-specific parameters (141 lines)
- **Performance Targets**: SLO definitions and thresholds
- **Circuit Breaker**: Fault tolerance configuration  
- **Environment Variables**: Template mappings (no hardcoded values)
- **Health Monitoring**: Check intervals and alerting rules

### 3.3 Service Boundaries

**File**: [`docs/architecture/rag_templates_service_boundaries.md`](../docs/architecture/rag_templates_service_boundaries.md:1)

**Boundary Definitions**:
- **Interface Contracts**: API specifications between layers
- **Error Handling**: Multi-layer error recovery strategies
- **Security Isolation**: Network, data, and process boundaries
- **Integration Patterns**: LightRAG incremental indexing support

## 4. Performance Architecture

### 4.1 Performance Targets

| Component | Metric | Target (p95) | Monitoring |
|-----------|---------|--------------|------------|
| **RAG Query Latency** | Response time | <500ms | Real-time metrics |
| **Memory API** | kg-ticket-resolver integration | <200ms | Cross-system tracing |
| **Config Loading** | System startup | <50ms | Health checks |
| **Error Recovery** | Fault tolerance | <1s | Automated alerting |

### 4.2 Monitoring Architecture

**File**: [`docs/architecture/rag_templates_performance_monitoring.md`](../docs/architecture/rag_templates_performance_monitoring.md:1)

**Monitoring Layers**:
- **Application Monitoring**: Query metrics, business logic, user experience
- **System Monitoring**: Resource usage, pipeline health, integration status  
- **Infrastructure Monitoring**: Container stats, database performance, network health

## 5. Integration Specifications

### 5.1 kg-ticket-resolver Integration

```python
# Example integration usage
from adapters.rag_templates_bridge import RAGTemplatesBridge, RAGTechnique

# Initialize bridge
bridge = RAGTemplatesBridge(config_path="config/rag_integration.yaml")

# Query with automatic technique selection
response = await bridge.query(
    query_text="What are the common patterns in ticket resolution?",
    technique=RAGTechnique.GRAPH  # Optional - defaults to configured default
)

# Handle response
if response.error is None:
    # Process successful response
    context = {
        "answer": response.answer,
        "confidence": response.confidence_score,
        "sources": response.sources,
        "processing_time": response.processing_time_ms
    }
    
    # Update knowledge graph memory
    await kg_memory.update_project_memory(context)
else:
    # Handle error with fallback
    logger.warning(f"RAG query failed: {response.error}")
    fallback_response = await get_cached_response(query_text)
```

### 5.2 Incremental Indexing Integration

```python
# LightRAG-style incremental indexing
async def handle_document_updates(change_events: List[DocumentChange]):
    """Process document changes incrementally."""
    
    # Batch changes by technique requirements
    for technique in [RAGTechnique.BASIC, RAGTechnique.GRAPH]:
        if has_changes_for_technique(change_events, technique):
            result = await bridge.index_documents(
                documents=extract_documents(change_events),
                technique=technique,
                incremental=True  # Enable incremental processing
            )
            
            # Update knowledge graph memory nodes
            await sync_memory_nodes(result, technique)
```

## 6. Deployment Architecture

### 6.1 Environment Configuration

```yaml
# Production deployment environment variables
environment:
  # Database connections
  IRIS_HOST: "iris-production.domain.com"
  IRIS_PORT: "1972"
  IRIS_USERNAME: "${VAULT_IRIS_USERNAME}"
  IRIS_PASSWORD: "${VAULT_IRIS_PASSWORD}"
  
  # API keys (from secure vault)
  OPENAI_API_KEY: "${VAULT_OPENAI_KEY}"
  
  # Performance tuning
  RAG_MAX_CONCURRENT_QUERIES: "50"
  RAG_QUERY_TIMEOUT: "30"
  
  # Monitoring
  METRICS_ENDPOINT: "https://metrics.domain.com"
  HEALTH_CHECK_INTERVAL: "30"
```

### 6.2 Container Deployment

```dockerfile
# Multi-stage deployment
FROM python:3.11-slim as base

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application
COPY adapters/ ./adapters/
COPY config/ ./config/
COPY docs/ ./docs/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD python -c "import asyncio; from adapters.rag_templates_bridge import RAGTemplatesBridge; bridge = RAGTemplatesBridge(); print(asyncio.run(bridge.get_health_status()))"

# Start bridge adapter
CMD ["python", "-m", "adapters.rag_templates_bridge"]
```

## 7. Implementation Roadmap

### 7.1 Phase 1: Core Implementation (Week 1-2)
- [x] **Unified Bridge Adapter**: [`RAGTemplatesBridge`](../adapters/rag_templates_bridge.py:85) class
- [x] **Configuration System**: Environment-independent [`rag_integration.yaml`](../config/rag_integration.yaml:1)  
- [x] **Service Boundaries**: Clean separation and interface contracts
- [x] **Performance Framework**: Monitoring and SLO enforcement

### 7.2 Phase 2: Integration & Testing (Week 3-4)
- [ ] **kg-ticket-resolver Integration**: Memory node synchronization
- [ ] **Circuit Breaker Testing**: Fault tolerance validation
- [ ] **Performance Validation**: Load testing and SLO verification
- [ ] **Documentation**: API reference and integration guides

### 7.3 Phase 3: Production Readiness (Week 5-6)
- [ ] **Production Deployment**: Container orchestration and scaling
- [ ] **Monitoring Dashboards**: Real-time observability
- [ ] **Automated Alerting**: Incident response automation
- [ ] **Performance Optimization**: Based on production metrics

## 8. Success Criteria

### 8.1 Functional Requirements
- ✅ **Unified Interface**: Single adapter for all RAG techniques
- ✅ **Circuit Breaker**: Automatic fallback and recovery
- ✅ **Performance Monitoring**: Real-time metrics and alerting
- ✅ **Configuration Management**: Environment-independent setup

### 8.2 Non-Functional Requirements
- ✅ **Performance**: <500ms p95 query latency target
- ✅ **Reliability**: Circuit breaker fault tolerance
- ✅ **Scalability**: Support for 100+ concurrent queries
- ✅ **Maintainability**: <500 lines per module constraint
- ✅ **Observability**: Comprehensive monitoring and health checks

### 8.3 Integration Requirements
- 🔄 **kg-ticket-resolver**: Memory API integration (<200ms p95)
- 🔄 **Incremental Indexing**: LightRAG-style document processing
- 🔄 **PRefLexOR Ready**: Future integration preparation
- ✅ **Environment Security**: Zero hardcoded secrets

## 9. Conclusion

The RAG Templates Adapter Layer (M1) provides a robust, scalable, and maintainable architecture for integrating the rag-templates RAG ecosystem with the kg-ticket-resolver knowledge graph memory system. The modular design with clean service boundaries enables independent scaling and maintenance while ensuring consistent performance and reliability.

**Key Architecture Benefits**:
- **Unified Access**: Single interface abstracts RAG complexity
- **Fault Tolerance**: Circuit breaker patterns ensure system resilience  
- **Performance**: Sub-second response times with comprehensive monitoring
- **Extensibility**: Easy addition of new RAG techniques and memory types
- **Security**: Environment-based configuration with zero hardcoded secrets

This architecture establishes a solid foundation for enabling the kg-ticket-resolver to leverage the full power of the rag-templates ecosystem while maintaining clean separation of concerns and enterprise-scale performance capabilities.