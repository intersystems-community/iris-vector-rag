# IRIS RAG MCP Server Architecture

## 1. Executive Summary

This document defines the comprehensive architecture for the IRIS RAG MCP (Model Context Protocol) server, providing standardized tool interfaces for 8 RAG techniques with enterprise-scale performance, modular design, and seamless Python-Node.js integration.

### 1.1 Architecture Principles

- **Modular Service Boundaries**: Clear separation between MCP layer, bridge layer, and RAG core
- **Extensible Design**: Support for adding new RAG techniques without core changes
- **Performance-First**: Sub-second response times with built-in monitoring
- **Environment-Based Configuration**: Zero hard-coded secrets, following support-tools-mcp patterns
- **Enterprise Scale**: Support for 1000+ to 92K+ documents with IRIS Enterprise

## 2. System Architecture Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           IRIS RAG MCP SERVER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                          MCP PROTOCOL LAYER                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │   Tool Router   │  │  Schema Manager │  │ Request Handler │            │
│  │   & Validator   │  │   & Validator   │  │  & Formatter    │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
├─────────────────────────────────────────────────────────────────────────────┤
│                        NODE.JS SERVICE LAYER                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ Configuration   │  │ Health Monitor  │  │ Performance     │            │
│  │ Manager         │  │ & Metrics       │  │ Monitor         │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
├─────────────────────────────────────────────────────────────────────────────┤
│                      PYTHON BRIDGE INTERFACE                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ Process Manager │  │ Pipeline Router │  │ Error Handler   │            │
│  │ & Pool          │  │ & Load Balancer │  │ & Validator     │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
├─────────────────────────────────────────────────────────────────────────────┤
│                         PYTHON RAG CORE                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ Pipeline Factory│  │ Vector Store    │  │ Schema Manager  │            │
│  │ & Registry      │  │ Interface       │  │ & Migration     │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ 8 RAG Technique │  │ Monitoring &    │  │ Configuration   │            │
│  │ Implementations │  │ Metrics Core    │  │ Management      │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
├─────────────────────────────────────────────────────────────────────────────┤
│                          IRIS DATABASE                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │ Vector Storage  │  │ Document Store  │  │ Graph & Metadata│            │
│  │ (HNSW Indexes)  │  │ (Source Docs)   │  │ (Relationships) │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Service Boundaries

#### 2.2.1 MCP Protocol Layer
- **Responsibility**: MCP protocol compliance, tool routing, request/response formatting
- **Interface**: Standard MCP protocol over stdio/HTTP
- **Dependencies**: Node.js Service Layer
- **File Limit**: <500 lines per module

#### 2.2.2 Node.js Service Layer  
- **Responsibility**: Configuration management, health monitoring, Python bridge coordination
- **Interface**: Internal APIs, environment configuration
- **Dependencies**: Python Bridge Interface
- **File Limit**: <500 lines per module

#### 2.2.3 Python Bridge Interface
- **Responsibility**: Process management, load balancing, error handling between Node.js and Python
- **Interface**: Child process communication, JSON-RPC
- **Dependencies**: Python RAG Core
- **File Limit**: <500 lines per module

#### 2.2.4 Python RAG Core
- **Responsibility**: RAG pipeline execution, vector operations, database interactions
- **Interface**: Pipeline registry, vector store interface
- **Dependencies**: IRIS Database
- **File Limit**: <500 lines per module

## 3. Component Architecture

### 3.1 MCP Protocol Layer Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP PROTOCOL LAYER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Tool Router   │    │  Schema Manager │                    │
│  │                 │    │                 │                    │
│  │ • Route tools   │    │ • Validate JSON │                    │
│  │ • Load balance  │    │ • Schema cache  │                    │
│  │ • Error handle  │    │ • Type checking │                    │
│  └─────────────────┘    └─────────────────┘                    │
│           │                       │                            │
│           └───────────┬───────────┘                            │
│                       │                                        │
│  ┌─────────────────────────────────────┐                      │
│  │         Request Handler             │                      │
│  │                                     │                      │
│  │ • MCP protocol compliance           │                      │
│  │ • Request parsing & validation      │                      │
│  │ • Response formatting & streaming   │                      │
│  │ • Performance metrics collection    │                      │
│  └─────────────────────────────────────┘                      │
│                       │                                        │
│                       ▼                                        │
│              Node.js Service Layer                             │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Tool Interface Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   TOOL INTERFACE LAYER                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Basic RAG     │  │     CRAG        │  │     HyDE        │ │
│  │   Tool          │  │     Tool        │  │     Tool        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   GraphRAG      │  │  HybridIFind    │  │   ColBERT       │ │
│  │   Tool          │  │     Tool        │  │     Tool        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   NodeRAG       │  │    SQLRAG       │  │   Health &      │ │
│  │   Tool          │  │     Tool        │  │   Metrics       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                 │
│                    Common Tool Interface                       │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Standardized input/output schemas                         │ │
│  │ • Parameter validation & sanitization                       │ │
│  │ • Performance monitoring & metrics                          │ │
│  │ • Error handling & recovery                                 │ │
│  │ • Technique-specific parameter support                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Python Bridge Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  PYTHON BRIDGE INTERFACE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │ Process Manager │    │ Pipeline Router │                    │
│  │                 │    │                 │                    │
│  │ • Process pool  │    │ • Load balancer │                    │
│  │ • Health checks │    │ • Technique map │                    │
│  │ • Auto-restart  │    │ • Request queue │                    │
│  │ • Resource mgmt │    │ • Response cache│                    │
│  └─────────────────┘    └─────────────────┘                    │
│           │                       │                            │
│           └───────────┬───────────┘                            │
│                       │                                        │
│  ┌─────────────────────────────────────┐                      │
│  │         Communication Layer         │                      │
│  │                                     │                      │
│  │ • JSON-RPC protocol                 │                      │
│  │ • Streaming support                 │                      │
│  │ • Error serialization              │                      │
│  │ • Performance metrics              │                      │
│  │ • Timeout handling                 │                      │
│  └─────────────────────────────────────┘                      │
│                       │                                        │
│                       ▼                                        │
│              Python RAG Core                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 4. Data Flow Architecture

### 4.1 Request Processing Flow

```
Client Request
      │
      ▼
┌─────────────────┐
│ MCP Tool Router │ ──── Validate Schema
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Request Handler │ ──── Parse & Sanitize
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Bridge Manager  │ ──── Route to Python
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Pipeline Router │ ──── Select Technique
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ RAG Pipeline    │ ──── Execute Query
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Vector Store    │ ──── Retrieve Documents
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ IRIS Database   │ ──── Vector Search
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Response Format │ ──── Structure Output
└─────────────────┘
      │
      ▼
Client Response
```

### 4.2 Performance Monitoring Flow

```
Request Start
      │
      ▼ ┌─────────────────┐
        │ Start Timer     │
        └─────────────────┘
      │
      ▼ ┌─────────────────┐
        │ Track Metrics   │ ──── CPU, Memory, DB
        └─────────────────┘
      │
      ▼ ┌─────────────────┐
        │ Pipeline Exec   │ ──── Retrieval Time
        └─────────────────┘
      │
      ▼ ┌─────────────────┐
        │ Response Gen    │ ──── Generation Time
        └─────────────────┘
      │
      ▼ ┌─────────────────┐
        │ Metrics Store   │ ──── Aggregate & Store
        └─────────────────┘
      │
      ▼
Performance Dashboard
```

## 5. Configuration Management Architecture

### 5.1 Environment-Based Configuration

```
┌─────────────────────────────────────────────────────────────────┐
│                 CONFIGURATION ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Environment Variables                                          │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ RAG_DATABASE__IRIS__HOST=localhost                          │ │
│  │ RAG_DATABASE__IRIS__PORT=1972                               │ │
│  │ RAG_EMBEDDING__MODEL=all-MiniLM-L6-v2                       │ │
│  │ RAG_LLM__PROVIDER=openai                                    │ │
│  │ RAG_PERFORMANCE__MAX_CONCURRENT=5                           │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │            Configuration Manager                            │ │
│  │                                                             │ │
│  │ • Environment variable parsing                              │ │
│  │ • YAML configuration merging                                │ │
│  │ • Schema validation                                         │ │
│  │ • Hot reloading support                                     │ │
│  │ • Technique-specific configs                                │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Runtime Configuration                          │ │
│  │                                                             │ │
│  │ • Database connections                                      │ │
│  │ • Model configurations                                      │ │
│  │ • Performance thresholds                                    │ │
│  │ • Security settings                                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Configuration Hierarchy

```
Default Config (YAML)
         │
         ▼
Environment Variables
         │
         ▼
Runtime Overrides
         │
         ▼
Technique-Specific Config
         │
         ▼
Final Configuration
```

## 6. Security and Validation Architecture

### 6.1 Security Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    SECURITY ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Validation Layer                                         │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • JSON schema validation                                    │ │
│  │ • Parameter sanitization                                    │ │
│  │ • Query length limits                                       │ │
│  │ • SQL injection prevention                                  │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│  Process Isolation Layer                                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Separate Python processes                                 │ │
│  │ • Resource limits                                           │ │
│  │ • Timeout enforcement                                       │ │
│  │ • Error containment                                         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│  Database Security Layer                                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Connection pooling                                        │ │
│  │ • Query parameterization                                    │ │
│  │ • Access control                                            │ │
│  │ • Audit logging                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 7. Scalability and Performance Architecture

### 7.1 Horizontal Scaling Design

```
┌─────────────────────────────────────────────────────────────────┐
│                  SCALABILITY ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Load Balancer                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Request distribution                                      │ │
│  │ • Health check routing                                      │ │
│  │ • Technique-aware balancing                                 │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│  MCP Server Instances                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Instance 1  │  │ Instance 2  │  │ Instance N  │             │
│  │             │  │             │  │             │             │
│  │ • Node.js   │  │ • Node.js   │  │ • Node.js   │             │
│  │ • Python    │  │ • Python    │  │ • Python    │             │
│  │   Pool      │  │   Pool      │  │   Pool      │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                │                                │
│                                ▼                                │
│  Shared IRIS Database Cluster                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Vector index sharding                                     │ │
│  │ • Read replicas                                             │ │
│  │ • Connection pooling                                        │ │
│  │ • Enterprise edition scaling                                │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Performance Optimization

```
┌─────────────────────────────────────────────────────────────────┐
│                PERFORMANCE OPTIMIZATION                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Caching Layer                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Embedding cache                                           │ │
│  │ • Query result cache                                        │ │
│  │ • Pipeline instance cache                                   │ │
│  │ • Configuration cache                                       │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│  Connection Pooling                                             │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • Database connection pools                                 │ │
│  │ • Python process pools                                      │ │
│  │ • HTTP connection reuse                                     │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                │                                │
│                                ▼                                │
│  Vector Index Optimization                                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ • HNSW index tuning                                         │ │
│  │ • Batch operations                                          │ │
│  │ • Parallel search                                           │ │
│  │ • Index warming                                             │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 8. Implementation Timeline

### 8.1 Phase 1: Core Infrastructure (Weeks 1-2)
- MCP Protocol Layer implementation
- Basic tool routing and validation
- Python bridge foundation
- Configuration management
- Health monitoring basics

### 8.2 Phase 2: RAG Tool Implementation (Weeks 3-4)
- BasicRAG, CRAG, HyDE tools
- Performance monitoring integration
- Error handling and validation
- Initial testing framework

### 8.3 Phase 3: Advanced Techniques (Weeks 5-6)
- GraphRAG, ColBERT, NodeRAG, SQLRAG tools
- Advanced performance optimization
- Scalability testing
- Production deployment preparation

## 9. File Organization

```
docs/architecture/
├── IRIS_RAG_MCP_SERVER_ARCHITECTURE.md     # This document
├── TOOL_INTERFACE_ARCHITECTURE.md          # Tool interface design
├── PYTHON_BRIDGE_ARCHITECTURE.md           # Bridge specification
├── CONFIGURATION_ARCHITECTURE.md           # Config management
├── PERFORMANCE_ARCHITECTURE.md             # Performance & monitoring
├── SECURITY_ARCHITECTURE.md                # Security design
└── DEPLOYMENT_ARCHITECTURE.md              # Deployment & scaling
```

This architecture provides a solid foundation for implementing the IRIS RAG MCP server with clear service boundaries, modular design, and enterprise-scale performance capabilities.