# IRIS RAG MCP Server - Comprehensive Architecture Overview

> Status source of truth: Implementation scope and E2E readiness are governed by [UNIFIED_PROJECT_ROADMAP.md](../UNIFIED_PROJECT_ROADMAP.md). Architectural guidance here should not be treated as an implementation claim; consult the roadmap for current status.

## 1. Executive Summary

This document provides a comprehensive overview of the IRIS RAG MCP (Model Context Protocol) server architecture, integrating all architectural components into a cohesive, enterprise-scale system design. The architecture supports 8 RAG techniques with modular design, clear service boundaries, and enterprise scalability from 1000+ to 92K+ documents.

### 1.1 Architecture Deliverables

This comprehensive architecture includes:

1. **[Core System Architecture](./IRIS_RAG_MCP_SERVER_ARCHITECTURE.md)** - Overall system design and service boundaries
2. **[Tool Interface Architecture](./TOOL_INTERFACE_ARCHITECTURE.md)** - Modular tool system for 8 RAG techniques  
3. **[Python-Node.js Bridge Architecture](./PYTHON_BRIDGE_ARCHITECTURE.md)** - Seamless integration layer
4. **[Configuration Management Architecture](./CONFIGURATION_ARCHITECTURE.md)** - Environment-based configuration system

### 1.2 Key Architectural Achievements

- **Modular Design**: Clear separation of concerns with <500 line file constraints
- **Enterprise Scale**: Support for 1000+ to 92K+ documents with IRIS Enterprise
- **Zero Hard-Coded Secrets**: Complete environment-based configuration
- **Performance Monitoring**: Built-in metrics collection and health monitoring
- **Extensible Framework**: Easy addition of new RAG techniques

## 2. Complete System Architecture

### 2.1 Full System Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        IRIS RAG MCP SERVER                                 │
│                     Complete System Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      MCP PROTOCOL LAYER                                │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ │
│  │  │ Tool Router │  │ Schema Mgr  │  │ Request     │  │ Response    │   │ │
│  │  │ & Validator │  │ & Validator │  │ Handler     │  │ Formatter   │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                       │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    NODE.JS SERVICE LAYER                               │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ │
│  │  │ Config Mgr  │  │ Health      │  │ Performance │  │ Security &  │   │ │
│  │  │ & Secrets   │  │ Monitor     │  │ Monitor     │  │ Validation  │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                       │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                  PYTHON BRIDGE INTERFACE                               │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ │
│  │  │ Process     │  │ Load        │  │ Message     │  │ Error       │   │ │
│  │  │ Pool Mgr    │  │ Balancer    │  │ Router      │  │ Handler     │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                       │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                     PYTHON RAG CORE                                    │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ │
│  │  │ Pipeline    │  │ Vector      │  │ Schema      │  │ Monitoring  │   │ │
│  │  │ Factory &   │  │ Store       │  │ Manager &   │  │ & Metrics   │   │ │
│  │  │ Registry    │  │ Interface   │  │ Migration   │  │ Core        │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────────┐ │ │
│  │  │                    8 RAG TECHNIQUE TOOLS                           │ │ │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │ │ │
│  │  │  │ Basic   │ │  CRAG   │ │  HyDE   │ │GraphRAG │ │Hybrid   │     │ │ │
│  │  │  │  RAG    │ │         │ │         │ │         │ │ IFind   │     │ │ │
│  │  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘     │ │ │
│  │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                             │ │ │
│  │  │  │ColBERT  │ │NodeRAG  │ │ SQLRAG  │                             │ │ │
│  │  │  │         │ │         │ │         │                             │ │ │
│  │  │  └─────────┘ └─────────┘ └─────────┘                             │ │ │
│  │  └─────────────────────────────────────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                       │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      IRIS DATABASE                                     │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │ │
│  │  │ Vector      │  │ Document    │  │ Graph &     │  │ Performance │   │ │
│  │  │ Storage     │  │ Store       │  │ Metadata    │  │ & Audit     │   │ │
│  │  │(HNSW Index) │  │(Source Docs)│  │(Relations)  │  │ Logs        │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Service Boundary Matrix

| Layer | Responsibility | Interface | Dependencies | File Limit |
|-------|---------------|-----------|--------------|------------|
| **MCP Protocol** | MCP compliance, tool routing | Standard MCP over stdio/HTTP | Node.js Service Layer | <500 lines |
| **Node.js Service** | Configuration, health, bridge coordination | Internal APIs, env config | Python Bridge Interface | <500 lines |
| **Python Bridge** | Process management, load balancing | Child process, JSON-RPC | Python RAG Core | <500 lines |
| **Python RAG Core** | Pipeline execution, vector ops | Pipeline registry, vector store | IRIS Database | <500 lines |
| **IRIS Database** | Data persistence, vector search | SQL, vector functions | None | N/A |

## 3. Data Flow Architecture

### 3.1 Complete Request Processing Flow

```
Client Request (MCP Protocol)
         │
         ▼
┌─────────────────┐
│ MCP Tool Router │ ──── Schema Validation
│ & Validator     │ ──── Parameter Sanitization
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Request Handler │ ──── Authentication (if enabled)
│ & Formatter     │ ──── Rate Limiting
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Configuration   │ ──── Environment Variables
│ Manager         │ ──── Secret Resolution
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Bridge Manager  │ ──── Process Pool Selection
│ & Load Balancer │ ──── Worker Health Check
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Python Worker   │ ──── Pipeline Registry
│ Process         │ ──── Technique Selection
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ RAG Pipeline    │ ──── Vector Store Interface
│ Execution       │ ──── Schema Manager
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ IRIS Database   │ ──── Vector Search (HNSW)
│ Operations      │ ──── Document Retrieval
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Response        │ ──── Performance Metrics
│ Assembly &      │ ──── Error Handling
│ Formatting      │ ──── Monitoring
└─────────────────┘
         │
         ▼
Client Response (Standardized Format)
```

### 3.2 Performance Monitoring Flow

```
Request Start
     │
     ▼ ┌─────────────────┐
       │ Start Timers    │ ──── Request ID Generation
       │ & Metrics       │ ──── Performance Context
       └─────────────────┘
     │
     ▼ ┌─────────────────┐
       │ System Metrics  │ ──── CPU, Memory, Disk
       │ Collection      │ ──── Database Connections
       └─────────────────┘
     │
     ▼ ┌─────────────────┐
       │ Pipeline        │ ──── Retrieval Time
       │ Execution       │ ──── Generation Time
       │ Tracking        │ ──── Document Count
       └─────────────────┘
     │
     ▼ ┌─────────────────┐
       │ Metrics         │ ──── Time Series Storage
       │ Aggregation     │ ──── Performance Analysis
       │ & Storage       │ ──── Threshold Checking
       └─────────────────┘
     │
     ▼
Performance Dashboard & Alerts
```

## 4. Configuration Architecture Integration

### 4.1 Complete Configuration Hierarchy

```
Environment Variables (Highest Priority)
├── RAG_DATABASE__IRIS__HOST=localhost
├── RAG_DATABASE__IRIS__PORT=1972
├── RAG_DATABASE__IRIS__PASSWORD_SECRET=iris-db-password
├── RAG_LLM__OPENAI__API_KEY_SECRET=openai-api-key
├── RAG_PERFORMANCE__MAX_WORKERS=5
├── RAG_MONITORING__ENABLE_METRICS=true
└── RAG_SECURITY__ENABLE_AUTH=false
         │
         ▼
Environment-Specific Files
├── config/production.yaml
├── config/development.yaml
└── config/testing.yaml
         │
         ▼
Base Configuration Files
├── config/default.yaml
├── config/base.yaml
└── config/techniques/
    ├── basic.yaml
    ├── crag.yaml
    ├── hyde.yaml
    ├── graphrag.yaml
    ├── hybrid_ifind.yaml
    ├── colbert.yaml
    ├── noderag.yaml
    └── sqlrag.yaml
         │
         ▼
Built-in Defaults (Lowest Priority)
└── Framework & technique defaults in code
```

### 4.2 Secret Management Integration

```
Secret References in Configuration
├── RAG_DATABASE__IRIS__PASSWORD_SECRET=iris-db-password
├── RAG_LLM__OPENAI__API_KEY_SECRET=openai-api-key
└── RAG_SECURITY__JWT_SECRET_SECRET=jwt-signing-key
         │
         ▼
Secret Backend Resolution
├── Vault Backend (Production)
├── AWS Secrets Manager
├── Azure Key Vault
└── File Backend (Development)
         │
         ▼
Runtime Secret Injection
└── Secure in-memory storage with audit logging
```

## 5. Scalability and Deployment Architecture

### 5.1 Horizontal Scaling Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    LOAD BALANCER                               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Request distribution across MCP server instances         │ │
│  │ • Health-aware routing                                      │ │
│  │ • Technique-specific load balancing                         │ │
│  │ • Circuit breaker pattern                                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ MCP Server  │  │ MCP Server  │  │ MCP Server  │  │ MCP Server  │
│ Instance 1  │  │ Instance 2  │  │ Instance 3  │  │ Instance N  │
│             │  │             │  │             │  │             │
│ • Node.js   │  │ • Node.js   │  │ • Node.js   │  │ • Node.js   │
│ • Python    │  │ • Python    │  │ • Python    │  │ • Python    │
│   Workers   │  │   Workers   │  │   Workers   │  │   Workers   │
│ • Local     │  │ • Local     │  │ • Local     │  │ • Local     │
│   Config    │  │   Config    │  │   Config    │  │   Config    │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 SHARED IRIS DATABASE CLUSTER                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • IRIS Enterprise Edition (no data limits)                 │ │
│  │ • Vector index sharding for 92K+ documents                 │ │
│  │ • Read replicas for query distribution                     │ │
│  │ • Connection pooling and load balancing                    │ │
│  │ • Automatic failover and backup                            │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Performance Optimization Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                   CACHING LAYERS                               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Embedding cache (LRU, configurable TTL)                  │ │
│  │ • Query result cache (technique-specific)                   │ │
│  │ • Pipeline instance cache (warm startup)                    │ │
│  │ • Configuration cache (hot reloading)                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                 CONNECTION POOLING                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Database connection pools (min/max configurable)         │ │
│  │ • Python worker process pools                              │ │
│  │ • HTTP connection reuse                                     │ │
│  │ • WebSocket connections for streaming                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│               VECTOR INDEX OPTIMIZATION                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • HNSW index tuning (M=16, efConstruction=200)             │ │
│  │ • Batch vector operations                                   │ │
│  │ • Parallel search execution                                 │ │
│  │ • Index warming on startup                                  │ │
│  │ • Automatic index maintenance                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 6. Security Architecture Integration

### 6.1 Multi-Layer Security Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECURITY LAYERS                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Validation & Sanitization                               │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • JSON schema validation (AJV)                             │ │
│  │ • Parameter sanitization & type coercion                   │ │
│  │ • Query length limits (1-2048 chars)                       │ │
│  │ • SQL injection prevention                                 │ │
│  │ • XSS protection                                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│  Authentication & Authorization                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • JWT-based authentication (optional)                      │ │
│  │ • Role-based access control                                │ │
│  │ • API key validation                                       │ │
│  │ • Rate limiting per user/IP                                │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│  Process Isolation & Resource Limits                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Separate Python worker processes                         │ │
│  │ • Memory limits per worker                                  │ │
│  │ • CPU usage monitoring                                      │ │
│  │ • Request timeout enforcement                               │ │
│  │ • Error containment                                        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│  Data Security & Audit                                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Database connection encryption                            │ │
│  │ • Secret management integration                             │ │
│  │ • Audit logging (all operations)                           │ │
│  │ • Data anonymization options                               │ │
│  │ • GDPR compliance features                                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 7. Implementation Roadmap

### 7.1 Phase-Based Implementation

```
Phase 1: Core Infrastructure (Weeks 1-2)
├── MCP Protocol Layer
│   ├── Basic tool routing
│   ├── Schema validation
│   └── Request/response handling
├── Node.js Service Layer
│   ├── Configuration manager
│   ├── Basic health monitoring
│   └── Environment variable parsing
├── Python Bridge Foundation
│   ├── Process management
│   ├── JSON-RPC communication
│   └── Basic error handling
└── Basic RAG Tool
    ├── Simple vector search
    ├── Standard response format
    └── Performance metrics

Phase 2: RAG Tool Implementation (Weeks 3-4)
├── Core RAG Techniques
│   ├── BasicRAG (complete)
│   ├── CRAG implementation
│   ├── HyDE implementation
│   └── GraphRAG foundation
├── Enhanced Bridge
│   ├── Load balancing
│   ├── Worker health checks
│   └── Connection pooling
├── Performance Monitoring
│   ├── Metrics collection
│   ├── Performance dashboard
│   └── Threshold alerting
└── Security Implementation
    ├── Input validation
    ├── Rate limiting
    └── Basic authentication

Phase 3: Advanced Techniques & Production (Weeks 5-6)
├── Advanced RAG Techniques
│   ├── ColBERT implementation
│   ├── NodeRAG implementation
│   ├── SQLRAG implementation
│   └── HybridIFind implementation
├── Production Features
│   ├── Horizontal scaling
│   ├── Advanced caching
│   ├── Secret management
│   └── Comprehensive monitoring
├── Testing & Validation
│   ├── End-to-end testing
│   ├── Performance benchmarking
│   ├── Security testing
│   └── Load testing
└── Documentation & Deployment
    ├── API documentation
    ├── Deployment guides
    ├── Monitoring setup
    └── Production checklist
```

## 8. Quality Assurance & Testing

### 8.1 Testing Strategy

```
Unit Testing (Per Component)
├── MCP Protocol Layer Tests
├── Tool Interface Tests
├── Bridge Communication Tests
├── Configuration Manager Tests
├── Performance Monitor Tests
└── Security Validation Tests

Integration Testing
├── End-to-End RAG Pipeline Tests
├── Multi-Technique Comparison Tests
├── Performance Benchmark Tests
├── Error Handling Tests
└── Configuration Integration Tests

System Testing
├── Load Testing (1000+ documents)
├── Scale Testing (92K+ documents)
├── Security Penetration Testing
├── Failover & Recovery Testing
└── Performance Regression Testing

Production Testing
├── Canary Deployments
├── A/B Testing Framework
├── Real-Time Monitoring
├── User Acceptance Testing
└── Compliance Validation
```

## 9. Conclusion

This comprehensive architecture provides a robust, scalable, and maintainable foundation for the IRIS RAG MCP server. The modular design with clear service boundaries enables:

- **Rapid Development**: 6-10 hour implementation timeline for core functionality
- **Enterprise Scale**: Support for 1000+ to 92K+ documents
- **Production Ready**: Built-in monitoring, security, and error handling
- **Extensible Design**: Easy addition of new RAG techniques
- **Operational Excellence**: Comprehensive configuration and deployment support

The architecture leverages existing infrastructure patterns while introducing modern best practices for enterprise-scale RAG systems, ensuring both immediate functionality and long-term maintainability.