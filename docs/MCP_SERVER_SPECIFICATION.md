# IRIS RAG MCP Server Specification

## 1. Overview

This specification defines a comprehensive Model Context Protocol (MCP) server for the IRIS RAG system, providing standardized tool interfaces for all 8 RAG techniques with real-time query capabilities, performance monitoring, and seamless Python-Node.js integration.

### 1.1 Objectives

- **Unified Interface**: Standardized MCP tool interface for all RAG techniques
- **Real-time Performance**: Sub-second response times for interactive queries
- **Enterprise Scale**: Support for 1000+ documents minimum, scalable to 92K+ documents
- **Environment-based Configuration**: Zero hard-coded secrets, following support-tools-mcp patterns
- **Modular Architecture**: Each technique as separate tool with consistent interface
- **Production Ready**: Built-in health checks, metrics, and error handling

### 1.2 Supported RAG Techniques

1. **BasicRAG** - Standard retrieval-augmented generation
2. **CRAG** - Corrective RAG with retrieval evaluation
3. **HyDE** - Hypothetical Document Embeddings
4. **GraphRAG** - Graph-based retrieval with entity relationships
5. **HybridIFind** - Hybrid search combining vector and keyword search
6. **ColBERT** - Late interaction retrieval with token-level matching
7. **NodeRAG** - Node-based retrieval with hierarchical context
8. **SQLRAG** - SQL-augmented retrieval for structured data

## 2. Architecture Overview

### 2.1 System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Server (Node.js)                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Tool Router   │  │  Health Monitor │  │ Config Mgr  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                Python Bridge Interface                     │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Pipeline Factory│  │ Vector Store    │  │ Performance │ │
│  │   & Registry    │  │   Interface     │  │  Monitor    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│                    IRIS Database                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │ Vector Storage  │  │ Document Store  │  │ Graph Data  │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Query Request → MCP Tool → Python Bridge → Pipeline → Vector Store → IRIS → Response
     ↓              ↓           ↓            ↓           ↓         ↓        ↓
Performance    Parameter   Pipeline     Retrieval   Vector    Database  Formatted
Monitoring     Validation  Selection    Logic       Search    Query     Response
```

## 3. MCP Tool Interface Specification

### 3.1 Standard Tool Schema

All RAG technique tools follow this standardized interface:

```typescript
interface RAGToolSchema {
  name: string;                    // Tool identifier (e.g., "rag_basic", "rag_crag")
  description: string;             // Human-readable description
  inputSchema: {
    type: "object";
    properties: {
      query: {
        type: "string";
        description: "User query text";
        minLength: 1;
        maxLength: 2048;
      };
      options?: {
        type: "object";
        properties: {
          top_k?: {
            type: "integer";
            minimum: 1;
            maximum: 50;
            default: 5;
          };
          temperature?: {
            type: "number";
            minimum: 0.0;
            maximum: 2.0;
            default: 0.7;
          };
          max_tokens?: {
            type: "integer";
            minimum: 50;
            maximum: 4096;
            default: 1024;
          };
          include_sources?: {
            type: "boolean";
            default: true;
          };
          min_similarity?: {
            type: "number";
            minimum: 0.0;
            maximum: 1.0;
            default: 0.1;
          };
        };
      };
      technique_params?: {
        type: "object";
        description: "Technique-specific parameters";
      };
    };
    required: ["query"];
  };
}
```

### 3.2 Standard Response Format

All tools return responses following this structure:

```typescript
interface RAGResponse {
  success: boolean;
  technique: string;              // RAG technique used
  query: string;                  // Original query
  answer: string;                 // Generated answer
  retrieved_documents: Array<{
    id: string;
    content: string;
    metadata: Record<string, any>;
    similarity_score: number;
    source: string;
  }>;
  performance: {
    total_time_ms: number;
    retrieval_time_ms: number;
    generation_time_ms: number;
    documents_searched: number;
  };
  metadata: {
    timestamp: string;            // ISO 8601 format
    model_info: {
      embedding_model: string;
      llm_model: string;
    };
    technique_specific?: Record<string, any>;
  };
  error?: {
    code: string;
    message: string;
    details?: Record<string, any>;
  };
}
```

## 4. Individual RAG Technique Tools

### 4.1 BasicRAG Tool

```typescript
{
  name: "rag_basic",
  description: "Standard retrieval-augmented generation with vector similarity search",
  inputSchema: {
    // Standard schema (see 3.1)
    // No technique-specific parameters
  }
}
```

**Implementation Pseudocode:**
```python
def execute_basic_rag(query: str, options: dict) -> dict:
    start_time = time.time()
    
    # 1. Initialize pipeline
    pipeline = pipeline_registry.get_pipeline("basic")
    
    # 2. Execute query
    retrieval_start = time.time()
    result = pipeline.execute(query, **options)
    retrieval_time = time.time() - retrieval_start
    
    # 3. Format response
    return format_standard_response(
        technique="basic",
        result=result,
        performance_metrics={
            "total_time_ms": (time.time() - start_time) * 1000,
            "retrieval_time_ms": retrieval_time * 1000
        }
    )
```

### 4.2 CRAG Tool

```typescript
{
  name: "rag_crag",
  description: "Corrective RAG with retrieval quality evaluation and correction",
  inputSchema: {
    // Standard schema plus:
    technique_params: {
      type: "object",
      properties: {
        confidence_threshold: {
          type: "number",
          minimum: 0.0,
          maximum: 1.0,
          default: 0.8,
          description: "Threshold for retrieval confidence"
        },
        enable_web_search: {
          type: "boolean",
          default: false,
          description: "Enable web search fallback for low confidence"
        },
        correction_strategy: {
          type: "string",
          enum: ["rewrite", "expand", "filter"],
          default: "rewrite",
          description: "Strategy for correcting poor retrievals"
        }
      }
    }
  }
}
```

### 4.3 HyDE Tool

```typescript
{
  name: "rag_hyde",
  description: "Hypothetical Document Embeddings for improved retrieval",
  inputSchema: {
    // Standard schema plus:
    technique_params: {
      type: "object",
      properties: {
        generate_hypothetical: {
          type: "boolean",
          default: true,
          description: "Generate hypothetical document for embedding"
        },
        hypothesis_length: {
          type: "integer",
          minimum: 50,
          maximum: 500,
          default: 200,
          description: "Target length for hypothetical document"
        },
        combine_strategy: {
          type: "string",
          enum: ["replace", "combine", "weighted"],
          default: "replace",
          description: "How to combine original and hypothetical embeddings"
        }
      }
    }
  }
}
```

### 4.4 GraphRAG Tool

```typescript
{
  name: "rag_graphrag",
  description: "Graph-based retrieval using entity relationships",
  inputSchema: {
    // Standard schema plus:
    technique_params: {
      type: "object",
      properties: {
        max_hops: {
          type: "integer",
          minimum: 1,
          maximum: 5,
          default: 2,
          description: "Maximum graph traversal hops"
        },
        entity_threshold: {
          type: "number",
          minimum: 0.0,
          maximum: 1.0,
          default: 0.7,
          description: "Entity extraction confidence threshold"
        },
        relationship_weight: {
          type: "number",
          minimum: 0.0,
          maximum: 1.0,
          default: 0.5,
          description: "Weight for relationship-based scoring"
        },
        include_entity_context: {
          type: "boolean",
          default: true,
          description: "Include entity context in results"
        }
      }
    }
  }
}
```

### 4.5 HybridIFind Tool

```typescript
{
  name: "rag_hybrid_ifind",
  description: "Hybrid search combining vector similarity and keyword matching",
  inputSchema: {
    // Standard schema plus:
    technique_params: {
      type: "object",
      properties: {
        vector_weight: {
          type: "number",
          minimum: 0.0,
          maximum: 1.0,
          default: 0.7,
          description: "Weight for vector similarity scores"
        },
        keyword_weight: {
          type: "number",
          minimum: 0.0,
          maximum: 1.0,
          default: 0.3,
          description: "Weight for keyword matching scores"
        },
        enable_fuzzy_matching: {
          type: "boolean",
          default: true,
          description: "Enable fuzzy keyword matching"
        },
        keyword_boost_fields: {
          type: "array",
          items: { type: "string" },
          default: ["title", "abstract"],
          description: "Fields to boost for keyword matching"
        }
      }
    }
  }
}
```

### 4.6 ColBERT Tool

```typescript
{
  name: "rag_colbert",
  description: "Late interaction retrieval with token-level matching",
  inputSchema: {
    // Standard schema plus:
    technique_params: {
      type: "object",
      properties: {
        max_query_length: {
          type: "integer",
          minimum: 32,
          maximum: 512,
          default: 256,
          description: "Maximum query length in tokens"
        },
        interaction_threshold: {
          type: "number",
          minimum: 0.0,
          maximum: 1.0,
          default: 0.5,
          description: "Token interaction threshold"
        },
        enable_query_expansion: {
          type: "boolean",
          default: false,
          description: "Enable automatic query expansion"
        },
        compression_ratio: {
          type: "number",
          minimum: 0.1,
          maximum: 1.0,
          default: 0.8,
          description: "Token representation compression ratio"
        }
      }
    }
  }
}
```

### 4.7 NodeRAG Tool

```typescript
{
  name: "rag_noderag",
  description: "Node-based retrieval with hierarchical document structure",
  inputSchema: {
    // Standard schema plus:
    technique_params: {
      type: "object",
      properties: {
        node_depth: {
          type: "integer",
          minimum: 1,
          maximum: 10,
          default: 3,
          description: "Maximum node hierarchy depth"
        },
        context_window: {
          type: "integer",
          minimum: 1,
          maximum: 10,
          default: 2,
          description: "Number of sibling nodes to include"
        },
        aggregation_strategy: {
          type: "string",
          enum: ["concatenate", "summarize", "weighted"],
          default: "concatenate",
          description: "Strategy for combining node content"
        },
        include_parent_context: {
          type: "boolean",
          default: true,
          description: "Include parent node context"
        }
      }
    }
  }
}
```

### 4.8 SQLRAG Tool

```typescript
{
  name: "rag_sqlrag",
  description: "SQL-augmented retrieval for structured data queries",
  inputSchema: {
    // Standard schema plus:
    technique_params: {
      type: "object",
      properties: {
        enable_sql_generation: {
          type: "boolean",
          default: true,
          description: "Enable automatic SQL query generation"
        },
        max_sql_complexity: {
          type: "string",
          enum: ["simple", "moderate", "complex"],
          default: "moderate",
          description: "Maximum SQL query complexity"
        },
        allowed_tables: {
          type: "array",
          items: { type: "string" },
          description: "Whitelist of allowed database tables"
        },
        combine_with_vector: {
          type: "boolean",
          default: true,
          description: "Combine SQL results with vector search"
        }
      }
    }
  }
}
```

## 5. Python-Node.js Bridge Architecture

### 5.1 Bridge Interface Specification

```typescript
interface PythonBridge {
  // Pipeline management
  initializePipeline(technique: string, config: any): Promise<string>;
  executePipeline(pipelineId: string, query: string, options: any): Promise<RAGResponse>;
  destroyPipeline(pipelineId: string): Promise<void>;
  
  // Health and monitoring
  healthCheck(): Promise<HealthStatus>;
  getMetrics(): Promise<PerformanceMetrics>;
  
  // Configuration
  updateConfiguration(config: any): Promise<void>;
  validateConfiguration(config: any): Promise<ValidationResult>;
}
```

### 5.2 Implementation Strategy

**Node.js Side (MCP Server):**
```javascript
// mcp-server/src/python-bridge.js
class PythonBridge {
  constructor(pythonExecutable = 'uv run python') {
    this.pythonProcess = null;
    this.messageQueue = new Map();
    this.requestId = 0;
  }
  
  async initialize() {
    // Start Python bridge process
    this.pythonProcess = spawn(pythonExecutable, [
      '-m', 'iris_rag.bridge.mcp_server'
    ], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: process.env
    });
    
    // Setup message handling
    this.setupMessageHandling();
  }
  
  async executePipeline(technique, query, options) {
    const requestId = ++this.requestId;
    const message = {
      id: requestId,
      method: 'execute_pipeline',
      params: { technique, query, options }
    };
    
    return this.sendMessage(message);
  }
}
```

**Python Side (Bridge Server):**
```python
# iris_rag/bridge/mcp_server.py
class MCPBridgeServer:
    def __init__(self):
        self.pipeline_registry = PipelineRegistry()
        self.performance_monitor = PerformanceMonitor()
        self.active_pipelines = {}
    
    async def handle_message(self, message: dict) -> dict:
        method = message.get('method')
        params = message.get('params', {})
        
        if method == 'execute_pipeline':
            return await self.execute_pipeline(**params)
        elif method == 'health_check':
            return await self.health_check()
        # ... other methods
    
    async def execute_pipeline(self, technique: str, query: str, options: dict) -> dict:
        try:
            # Get or create pipeline instance
            pipeline = self.get_pipeline(technique)
            
            # Execute with performance monitoring
            with self.performance_monitor.measure(f"{technique}_execution"):
                result = pipeline.execute(query, **options)
            
            # Format response
            return self.format_response(technique, result)
            
        except Exception as e:
            return self.format_error_response(str(e))
```

### 5.3 Process Communication Protocol

**Message Format:**
```typescript
interface BridgeMessage {
  id: number;                    // Request ID for correlation
  method: string;                // Method name
  params: Record<string, any>;   // Method parameters
  timestamp: string;             // ISO 8601 timestamp
}

interface BridgeResponse {
  id: number;                    // Correlates with request ID
  success: boolean;              // Operation success status
  result?: any;                  // Result data (if successful)
  error?: {                      // Error details (if failed)
    code: string;
    message: string;
    traceback?: string;
  };
  timestamp: string;             // ISO 8601 timestamp
}
```

## 6. Configuration Management

### 6.1 Environment-based Configuration

Following support-tools-mcp patterns, all configuration uses environment variables:

```typescript
interface MCPServerConfig {
  // Server configuration
  server: {
    name: string;                // MCP_SERVER_NAME
    description: string;         // MCP_SERVER_DESCRIPTION
    version: string;             // MCP_SERVER_VERSION
    port?: number;               // MCP_SERVER_PORT
  };
  
  // IRIS database configuration
  iris: {
    host: string;                // IRIS_HOST
    port: number;                // IRIS_PORT
    namespace: string;           // IRIS_NAMESPACE
    username: string;            // IRIS_USERNAME
    password: string;            // IRIS_PASSWORD
    edition: string;             // IRIS_DOCKER_IMAGE
  };
  
  // LLM configuration
  llm: {
    provider: string;            // LLM_PROVIDER
    model: string;               // LLM_MODEL
    api_key: string;             // LLM_API_KEY
    base_url?: string;           // LLM_BASE_URL
    temperature: number;         // LLM_TEMPERATURE
    max_tokens: number;          // LLM_MAX_TOKENS
  };
  
  // Embedding configuration
  embedding: {
    provider: string;            // EMBEDDING_PROVIDER
    model: string;               // EMBEDDING_MODEL
    api_key?: string;            // EMBEDDING_API_KEY
    dimensions: number;          // EMBEDDING_DIMENSIONS
  };
  
  // Performance configuration
  performance: {
    max_concurrent_requests: number;  // MAX_CONCURRENT_REQUESTS
    request_timeout_ms: number;       // REQUEST_TIMEOUT_MS
    cache_enabled: boolean;           // CACHE_ENABLED
    cache_ttl_seconds: number;        // CACHE_TTL_SECONDS
  };
}
```

### 6.2 Configuration Loading

```javascript
// config/manager.js
class ConfigurationManager {
  constructor() {
    this.config = this.loadConfiguration();
    this.validateConfiguration();
  }
  
  loadConfiguration() {
    return {
      server: {
        name: process.env.MCP_SERVER_NAME || 'iris-rag-server',
        description: process.env.MCP_SERVER_DESCRIPTION || 'IRIS RAG MCP Server',
        version: process.env.MCP_SERVER_VERSION || '1.0.0',
        port: parseInt(process.env.MCP_SERVER_PORT || '3000')
      },
      iris: {
        host: process.env.IRIS_HOST || 'localhost',
        port: parseInt(process.env.IRIS_PORT || '52773'),
        namespace: process.env.IRIS_NAMESPACE || 'USER',
        username: process.env.IRIS_USERNAME || 'SuperUser',
        password: process.env.IRIS_PASSWORD || 'SYS',
        edition: process.env.IRIS_DOCKER_IMAGE || 'intersystemsdc/iris-ml:latest'
      },
      // ... other configuration sections
    };
  }
  
  validateConfiguration() {
    const required = [
      'IRIS_HOST', 'IRIS_USERNAME', 'IRIS_PASSWORD',
      'LLM_PROVIDER', 'LLM_API_KEY',
      'EMBEDDING_PROVIDER'
    ];
    
    const missing = required.filter(key => !process.env[key]);
    if (missing.length > 0) {
      throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
    }
  }
}
```

## 7. Performance Monitoring & Health Checks

### 7.1 Health Check Tool

```typescript
{
  name: "rag_health_check",
  description: "Check system health and component status",
  inputSchema: {
    type: "object",
    properties: {
      include_details: {
        type: "boolean",
        default: false,
        description: "Include detailed component status"
      }
    }
  }
}
```

**Response Format:**
```typescript
interface HealthCheckResponse {
  status: "healthy" | "degraded" | "unhealthy";
  timestamp: string;
  components: {
    iris_database: ComponentStatus;
    python_bridge: ComponentStatus;
    llm_service: ComponentStatus;
    embedding_service: ComponentStatus;
    vector_store: ComponentStatus;
  };
  performance: {
    avg_response_time_ms: number;
    requests_per_minute: number;
    error_rate_percent: number;
  };
  details?: Record<string, any>;
}

interface ComponentStatus {
  status: "healthy" | "degraded" | "unhealthy";
  last_check: string;
  response_time_ms?: number;
  error_message?: string;
}
```

### 7.2 Performance Metrics Tool

```typescript
{
  name: "rag_metrics",
  description: "Get performance metrics and usage statistics",
  inputSchema: {
    type: "object",
    properties: {
      time_range: {
        type: "string",
        enum: ["1h", "24h", "7d", "30d"],
        default: "1h",
        description: "Time range for metrics"
      },
      technique_filter: {
        type: "array",
        items: { type: "string" },
        description: "Filter metrics by RAG techniques"
      }
    }
  }
}
```

## 8. Error Handling & Validation

### 8.1 Error Codes

```typescript
enum ErrorCodes {
  // Input validation errors
  INVALID_QUERY = "INVALID_QUERY",
  INVALID_PARAMETERS = "INVALID_PARAMETERS",
  MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD",
  
  // System errors
  PIPELINE_NOT_FOUND = "PIPELINE_NOT_FOUND",
  PIPELINE_INITIALIZATION_FAILED = "PIPELINE_INITIALIZATION_FAILED",
  DATABASE_CONNECTION_FAILED = "DATABASE_CONNECTION_FAILED",
  
  // Processing errors
  RETRIEVAL_FAILED = "RETRIEVAL_FAILED",
  GENERATION_FAILED = "GENERATION_FAILED",
  TIMEOUT_EXCEEDED = "TIMEOUT_EXCEEDED",
  
  // Resource errors
  RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED",
  INSUFFICIENT_RESOURCES = "INSUFFICIENT_RESOURCES",
  
  // External service errors
  LLM_SERVICE_UNAVAILABLE = "LLM_SERVICE_UNAVAILABLE",
  EMBEDDING_SERVICE_UNAVAILABLE = "EMBEDDING_SERVICE_UNAVAILABLE"
}
```

### 8.2 Parameter Validation

```javascript
// validation/parameter-validator.js
class ParameterValidator {
  validateQuery(query) {
    if (!query || typeof query !== 'string') {
      throw new ValidationError(ErrorCodes.INVALID_QUERY, 'Query must be a non-empty string');
    }
    
    if (query.length > 2048) {
      throw new ValidationError(ErrorCodes.INVALID_QUERY, 'Query exceeds maximum length of 2048 characters');
    }
    
    return query.trim();
  }
  
  validateOptions(options, technique) {
    const validated = {};
    const schema = this.getOptionsSchema(technique);
    
    for (const [key, value] of Object.entries(options)) {
      if (schema[key]) {
        validated[key] = this.validateParameter(key, value, schema[key]);
      }
    }
    
    return validated;
  }
}
```

## 9. Testing Framework Requirements

### 9.1 TDD Test Anchors

**Unit Tests:**
```python
# tests/test_mcp_tools.py
class TestMCPTools:
    def test_basic_rag_tool_interface(self):
        """TDD Anchor: Basic RAG tool follows standard interface"""
        tool = get_tool("rag_basic")
        assert tool.name == "rag_basic"
        assert "query" in tool.inputSchema.properties
        assert tool.inputSchema.required == ["query"]
    
    def test_parameter_validation(self):
        """TDD Anchor: Parameter validation works correctly"""
        validator = ParameterValidator()
        
        # Valid parameters
        result = validator.validate_query("What is machine learning?")
        assert result == "What is machine learning?"
        
        # Invalid parameters
        with pytest.raises(ValidationError):
            validator.validate_query("")
    
    def test_standard_response_format(self):
        """TDD Anchor: All tools return standard response format"""
        response = execute_tool("rag_basic", {"query": "test query"})
        
        assert "success" in response
        assert "technique" in response
        assert "query" in response
        assert "answer" in response
        assert "retrieved_documents" in response
        assert "performance" in response
```

**Integration Tests:**
```python
# tests/test_mcp_integration.py
class TestMCPIntegration:
    def test_python_bridge_communication(self):
        """TDD Anchor: Python bridge communication works"""
        bridge = PythonBridge()
        response = bridge.execute_pipeline("basic", "test query", {})
        
        assert response["success"] is True
        assert "answer" in response
    
    def test_real_data_pipeline_execution(self):
        """TDD Anchor: Pipelines work with real PMC data (1000+ docs)"""
        # Requires 1000+ documents loaded
        for technique in RAG_TECHNIQUES:
            response = execute_tool(f"rag_{technique}", {
                "query": "What are the effects of COVID-19?"
            })
            
            assert response["success"] is True
            assert len(response["retrieved_documents"]) > 0
            assert response["performance"]["total_time_ms"] < 5000  # 5 second max
```

**Performance Tests:**
```python
# tests/test_mcp_performance.py
class TestMCPPerformance:
    def test_response_time_requirements(self):
        """TDD Anchor: Response times meet requirements"""
        start_time = time.time()
        response = execute_tool("rag_basic", {"query": "test query"})
        duration = time.time() - start_time
        
        assert duration < 2.0  # 2 second max for basic queries
        assert response["performance"]["total_time_ms"] < 2000
    
    def test_concurrent_request_handling(self):
        """TDD Anchor: Server handles concurrent requests"""
        import asyncio
        
        async def make_request():
            return execute_tool("rag_basic", {"query": f"test query {random.randint(1, 1000)}"})
        
        # Test 10 concurrent requests
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks)
        
        assert all(r["success"] for r in responses)
```

### 9.2 Test Data Requirements

- **Minimum Dataset**: 1000+ PMC documents for meaningful testing
- **Test Queries**: Standardized set of 50+ queries covering different domains
- **Ground Truth**: Human-annotated relevance judgments for evaluation
- **Performance Baselines**: Established benchmarks for each technique

## 10. Deployment & Operations

### 10.1 Docker Configuration

```dockerfile
# Dockerfile
FROM node:18-alpine

# Install Python and UV
RUN apk add --no-cache python3 py3-pip
RUN pip install uv

# Copy application
COPY . /app
WORKDIR /app

# Install dependencies
RUN npm install
RUN uv sync

# Environment configuration
ENV NODE_ENV=production
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:3000/health || exit 1

# Start server
CMD ["npm", "start"]
```

### 10.2 Environment Variables Template

```bash
# .env.template
# Server Configuration
MCP_SERVER_NAME=iris-rag-server
MCP_SERVER_DESCRIPTION=IRIS RAG MCP Server
MCP_SERVER_VERSION=1.0.0
MCP_SERVER_PORT=3000

# IRIS Database
IRIS_HOST=localhost
IRIS_PORT=52773
IRIS_NAMESPACE=USER
IRIS_USERNAME=SuperUser
IRIS_PASSWORD=SYS
IRIS_DOCKER_IMAGE=intersystemsdc/iris-ml:latest

# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
LLM_API_KEY=your_openai_api_key
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024

# Embedding Configuration
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_API_KEY=your_openai_api_key
EMBEDDING_DIMENSIONS=1536

# Performance Configuration
MAX_CONCURRENT_REQUESTS=10
REQUEST_TIMEOUT_MS=30000
CACHE_ENABLED=true
CACHE_TTL_SECONDS=300
```

## 11. Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)
- [ ] MCP server framework setup
- [ ] Python bridge implementation
- [ ] Basic tool interface
- [ ] Configuration management
- [ ] Health checks

### Phase 2: RAG Tool Implementation (Week 3-4)
- [ ] BasicRAG tool
- [ ] CRAG tool
- [ ] HyDE tool
- [ ] Parameter validation
- [ ] Error handling

### Phase 3: Advanced Techniques (Week 5-6)
- [ ] GraphRAG tool
- [ ] HybridIFind tool
- [ ] ColBERT tool
- [ ] NodeRAG tool
- [ ] SQLRAG tool

### Phase 