# Library Consumption Framework Architecture

## Executive Summary

This document outlines a comprehensive architectural design for transforming the rag-templates project from a complex, setup-intensive framework into a systematic library consumption framework that enables "dead-simple" integration while maintaining all enterprise capabilities.

**Key Insight from support-tools-mcp Analysis**: The existing MCP implementation demonstrates sophisticated patterns including:
- Environment-based configuration management (no hardcoded secrets)
- Modular tool registry with JSON schema validation
- Production-ready Docker container lifecycle management
- Clean separation between protocol handling and business logic
- Comprehensive error handling and logging

## Current State Analysis

### Strengths
- **Sophisticated RAG Implementations**: 7+ advanced techniques (BasicRAG, ColBERT, CRAG, GraphRAG, HyDE, NodeRAG, Hybrid iFindRAG)
- **Advanced Configuration System**: YAML-based with environment variable support
- **Dynamic Pipeline Loading**: Flexible [`config/pipelines.yaml`](config/pipelines.yaml) configuration
- **Enterprise Features**: Caching, reconciliation, monitoring, comprehensive testing
- **TDD Foundation**: Robust testing framework with real data validation
- **Node.js Foundation**: Basic [`createVectorSearchPipeline`](nodejs/src/index.js) factory function

### Pain Points Identified
1. **Complex Setup Barrier**: Multi-step setup process deters simple use cases
2. **JavaScript/Node.js Gap**: Limited config system compared to Python sophistication
3. **MCP Integration Complexity**: Requires deep framework knowledge (as seen in support-tools-mcp)
4. **Library Consumption Friction**: No simple "npm install" or "pip install" experience
5. **Configuration Overwhelm**: Powerful but complex for basic scenarios

### Touch Points from support-tools-mcp Analysis

The [`support-tools-mcp/mcp-node-server/src/lib/irisRagClient.ts`](../../../support-tools-mcp/mcp-node-server/src/lib/irisRagClient.ts) implementation reveals critical integration patterns:

```typescript
// Key integration pattern from support-tools-mcp
const { createVectorSearchPipeline } = require('../../../../rag-templates/nodejs/src/index');

// Configuration bridging
const irisConfig = {
  host: this.configManager.get('iris.host') || 'localhost',
  port: this.configManager.get('iris.webPort') || 52773,
  namespace: this.configManager.get('iris.namespace') || 'ML_RAG',
  username: this.configManager.get('iris.username') || 'demo',
  password: this.configManager.get('iris.password') || 'demo'
};

this.pipeline = createVectorSearchPipeline({
  connection: irisConfig,
  embeddingModel: this.configManager.get('iris.embeddingModel') || 'Xenova/all-MiniLM-L6-v2'
});
```

## Architecture Overview

### Design Principles

1. **Progressive Complexity**: Simple APIs for basic use, advanced APIs for enterprise
2. **Language Parity**: JavaScript capabilities mirror Python patterns
3. **Zero-Config Defaults**: Works out-of-the-box with sensible defaults
4. **Extensible Foundation**: Easy addition of new RAG techniques
5. **MCP-First Design**: Trivial MCP server creation
6. **Environment-Based Configuration**: No hardcoded secrets (learned from support-tools-mcp)

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Library Consumption Layer                    │
├─────────────────────────────────────────────────────────────────┤
│  Simple API        │  Standard API      │  Enterprise API      │
│  (Zero Config)     │  (Basic Config)    │  (Full Config)       │
├─────────────────────────────────────────────────────────────────┤
│                    Language Bindings                            │
├─────────────────────┬───────────────────────────────────────────┤
│     Python SDK      │           JavaScript SDK                 │
│  ┌─────────────────┐│  ┌─────────────────┬─────────────────────┐│
│  │ rag-templates   ││  │ @rag-templates/ │ @rag-templates/     ││
│  │                 ││  │ core            │ mcp                 ││
│  └─────────────────┘│  └─────────────────┴─────────────────────┘│
├─────────────────────┴───────────────────────────────────────────┤
│                    Core Framework Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  Config Manager  │  Pipeline Factory  │  Technique Registry    │
├─────────────────────────────────────────────────────────────────┤
│                    RAG Techniques Layer                         │
├─────────────────────────────────────────────────────────────────┤
│ BasicRAG │ ColBERT │ CRAG │ GraphRAG │ HyDE │ NodeRAG │ Hybrid │
├─────────────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  Vector Store  │  LLM Providers  │  Embedding Models  │ Cache  │
└─────────────────────────────────────────────────────────────────┘
```

## API Design Patterns

### 1. Simple API (Zero Configuration)

#### Python
```python
from rag_templates import RAG

# Dead simple - works out of the box
rag = RAG()
result = rag.query("What is machine learning?")
print(result.answer)
```

#### JavaScript
```javascript
import { RAG } from '@rag-templates/core';

// Dead simple - works out of the box
const rag = new RAG();
const result = await rag.query("What is machine learning?");
console.log(result.answer);
```

### 2. Standard API (Basic Configuration)

#### Python
```python
from rag_templates import RAG

# Simple configuration
rag = RAG({
    'technique': 'colbert',
    'llm_provider': 'openai',
    'embedding_model': 'text-embedding-3-small'
})

result = rag.query("Explain neural networks", {
    'max_results': 5,
    'include_sources': True
})
```

#### JavaScript
```javascript
import { RAG } from '@rag-templates/core';

// Simple configuration
const rag = new RAG({
    technique: 'colbert',
    llmProvider: 'openai',
    embeddingModel: 'text-embedding-3-small'
});

const result = await rag.query("Explain neural networks", {
    maxResults: 5,
    includeSources: true
});
```

### 3. Enterprise API (Full Configuration)

#### Python
```python
from rag_templates import RAG
from rag_templates.config import ConfigManager

# Enterprise configuration with full control
config = ConfigManager.from_file('enterprise-config.yaml')
rag = RAG(config)

# Advanced pipeline with monitoring
result = rag.query("Complex query", {
    'pipeline_config': {
        'caching': True,
        'monitoring': True,
        'reconciliation': True
    }
})
```

#### JavaScript
```javascript
import { RAG, ConfigManager } from '@rag-templates/core';

// Enterprise configuration with full control
const config = await ConfigManager.fromFile('enterprise-config.yaml');
const rag = new RAG(config);

// Advanced pipeline with monitoring
const result = await rag.query("Complex query", {
    pipelineConfig: {
        caching: true,
        monitoring: true,
        reconciliation: true
    }
});
```

## Configuration Strategy

### Three-Tier Configuration System

#### Tier 1: Zero Configuration (Defaults)
```yaml
# Built-in defaults - no config file needed
defaults:
  technique: "basic_rag"
  llm_provider: "openai"
  embedding_model: "text-embedding-3-small"
  vector_store: "in_memory"
  max_results: 3
  temperature: 0.7
```

#### Tier 2: Simple Configuration
```yaml
# simple-config.yaml
technique: "colbert"
llm_provider: "anthropic"
embedding_model: "text-embedding-3-large"
data_source: "./documents"
```

#### Tier 3: Enterprise Configuration
```yaml
# enterprise-config.yaml
technique: "hybrid_ifind"
llm_provider: "azure_openai"
embedding_model: "text-embedding-3-large"

database:
  type: "iris"
  connection_string: "${IRIS_CONNECTION_STRING}"
  
caching:
  enabled: true
  ttl: 3600
  
monitoring:
  enabled: true
  metrics_endpoint: "${METRICS_ENDPOINT}"
  
reconciliation:
  enabled: true
  validation_rules: ["semantic_consistency", "factual_accuracy"]
```

## MCP Integration Patterns

### 1. Simple MCP Server Creation

#### JavaScript (Inspired by support-tools-mcp patterns)
```javascript
// create-mcp-server.js
import { createMCPServer } from '@rag-templates/mcp';

const server = createMCPServer({
    name: "my-rag-server",
    description: "RAG-powered MCP server",
    // Zero config - uses defaults
});

server.start();
```

#### Python
```python
# create_mcp_server.py
from rag_templates.mcp import create_mcp_server

server = create_mcp_server(
    name="my-rag-server",
    description="RAG-powered MCP server"
    # Zero config - uses defaults
)

server.start()
```

### 2. Advanced MCP Server with Custom RAG

#### JavaScript (Following support-tools-mcp architecture)
```javascript
import { createMCPServer, RAG } from '@rag-templates/mcp';
import { ConfigurationManager } from '@rag-templates/core';

// Environment-based configuration (no hardcoded secrets)
const configManager = new ConfigurationManager();
await configManager.load();

const rag = new RAG({
    technique: 'graphrag',
    dataSource: './knowledge-base',
    connection: {
        host: configManager.get('iris.host'),
        port: configManager.get('iris.webPort'),
        username: configManager.get('iris.username'),
        password: configManager.get('iris.password')
    }
});

const server = createMCPServer({
    name: "advanced-rag-server",
    rag: rag,
    tools: [
        {
            name: "search_knowledge",
            description: "Search the knowledge base",
            inputSchema: {
                type: 'object',
                properties: {
                    query: { type: 'string', description: 'Search query' },
                    topK: { type: 'integer', minimum: 1, maximum: 100 }
                },
                required: ['query'],
                additionalProperties: false  // MCP compliance
            },
            handler: async (args) => rag.query(args.query, { topK: args.topK })
        }
    ]
});

server.start();
```

## Package Structure

### Python Package Structure
```
rag-templates/
├── rag_templates/
│   ├── __init__.py              # Simple API exports
│   ├── core/
│   │   ├── rag.py              # Main RAG class
│   │   ├── config_manager.py   # Configuration management
│   │   └── pipeline_factory.py # Pipeline creation
│   ├── techniques/             # RAG technique implementations
│   ├── mcp/                    # MCP integration
│   │   ├── __init__.py
│   │   ├── server.py          # MCP server creation
│   │   └── tools.py           # MCP tool definitions
│   └── utils/                  # Utility functions
├── setup.py
└── pyproject.toml
```

### JavaScript Package Structure
```
@rag-templates/
├── core/                       # Main package
│   ├── package.json
│   ├── src/
│   │   ├── index.js           # Simple API exports
│   │   ├── rag.js             # Main RAG class
│   │   ├── config-manager.js  # Configuration management
│   │   └── pipeline-factory.js # Pipeline creation
│   └── dist/                  # Built files
├── mcp/                       # MCP-specific package
│   ├── package.json
│   ├── src/
│   │   ├── index.js          # MCP exports
│   │   ├── server.js         # MCP server creation
│   │   └── tools.js          # MCP tool definitions
│   └── dist/
└── techniques/                # RAG techniques package
    ├── package.json
    └── src/
```

## Implementation Details

### Configuration Management System (Inspired by support-tools-mcp)

#### Python Implementation
```python
# rag_templates/core/config_manager.py
class ConfigManager:
    def __init__(self, config=None):
        self.config = self._merge_configs(
            self._load_defaults(),
            self._load_environment(),
            config or {}
        )
    
    @classmethod
    def from_file(cls, path):
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config)
    
    def _load_defaults(self):
        return {
            'technique': 'basic_rag',
            'llm_provider': 'openai',
            'embedding_model': 'text-embedding-3-small',
            'max_results': 3,
            'temperature': 0.7
        }
    
    def _load_environment(self):
        """Load configuration from environment variables"""
        return {
            'iris': {
                'host': os.getenv('IRIS_HOST', 'localhost'),
                'port': int(os.getenv('IRIS_PORT', '52773')),
                'username': os.getenv('IRIS_USERNAME'),
                'password': os.getenv('IRIS_PASSWORD'),
                'namespace': os.getenv('IRIS_NAMESPACE', 'ML_RAG')
            },
            'llm': {
                'api_key': os.getenv('OPENAI_API_KEY'),
                'model': os.getenv('LLM_MODEL', 'gpt-4o-mini')
            }
        }
```

#### JavaScript Implementation (Following support-tools-mcp patterns)
```javascript
// @rag-templates/core/src/config-manager.js
export class ConfigManager {
    constructor(config = {}) {
        this.config = this._mergeConfigs(
            this._loadDefaults(),
            this._loadEnvironment(),
            config
        );
    }
    
    static async fromFile(path) {
        const fs = await import('fs/promises');
        const yaml = await import('yaml');
        const content = await fs.readFile(path, 'utf8');
        const config = yaml.parse(content);
        return new ConfigManager(config);
    }
    
    _loadDefaults() {
        return {
            technique: 'basic_rag',
            llmProvider: 'openai',
            embeddingModel: 'text-embedding-3-small',
            maxResults: 3,
            temperature: 0.7
        };
    }
    
    _loadEnvironment() {
        return {
            iris: {
                host: process.env.IRIS_HOST || 'localhost',
                port: parseInt(process.env.IRIS_PORT || '52773'),
                username: process.env.IRIS_USERNAME,
                password: process.env.IRIS_PASSWORD,
                namespace: process.env.IRIS_NAMESPACE || 'ML_RAG'
            },
            llm: {
                apiKey: process.env.OPENAI_API_KEY,
                model: process.env.LLM_MODEL || 'gpt-4o-mini'
            }
        };
    }
    
    // Legacy compatibility for existing code (like support-tools-mcp)
    get(path) {
        const parts = path.split('.');
        let current = this.config;
        
        for (const part of parts) {
            if (current && typeof current === 'object' && part in current) {
                current = current[part];
            } else {
                return undefined;
            }
        }
        
        return current;
    }
}
```

### RAG Class Implementation

#### Python Simple API
```python
# rag_templates/core/rag.py
class RAG:
    def __init__(self, config=None):
        self.config_manager = ConfigManager(config)
        self.pipeline = PipelineFactory.create(self.config_manager.config)
    
    def query(self, question, options=None):
        """Simple query interface with optional parameters"""
        query_config = {**self.config_manager.config}
        if options:
            query_config.update(options)
        
        return self.pipeline.execute(question, query_config)
    
    def add_documents(self, documents):
        """Simple document addition interface"""
        return self.pipeline.add_documents(documents)
```

#### JavaScript Simple API (Building on existing createVectorSearchPipeline)
```javascript
// @rag-templates/core/src/rag.js
export class RAG {
    constructor(config = {}) {
        this.configManager = new ConfigManager(config);
        
        // Use existing createVectorSearchPipeline as foundation
        this.pipeline = createVectorSearchPipeline({
            connection: this.configManager.get('iris'),
            embeddingModel: this.configManager.get('embeddingModel')
        });
    }
    
    async query(question, options = {}) {
        const queryConfig = { ...this.configManager.config, ...options };
        
        // Map simple API to existing pipeline interface
        const searchOptions = {
            topK: options.maxResults || queryConfig.maxResults || 5,
            additionalWhere: options.sourceFilter,
            minSimilarity: options.minSimilarity
        };
        
        const results = await this.pipeline.search(question, searchOptions);
        
        // Return standardized format
        return {
            answer: this._generateAnswer(results, question),
            sources: results,
            query: question
        };
    }
    
    async addDocuments(documents) {
        return await this.pipeline.indexDocuments(documents);
    }
    
    async initialize() {
        return await this.pipeline.initialize();
    }
    
    async close() {
        return await this.pipeline.close();
    }
}
```

### MCP Server Creation (Following support-tools-mcp architecture)

#### JavaScript MCP Server Factory
```javascript
// @rag-templates/mcp/src/server.js
import { ToolRegistry } from './tool-registry.js';
import { ConfigurationManager } from '@rag-templates/core';
import { RAG } from '@rag-templates/core';

export function createMCPServer(options = {}) {
    const configManager = new ConfigurationManager();
    const toolRegistry = new ToolRegistry(configManager);
    
    // Initialize RAG with configuration
    const rag = options.rag || new RAG(configManager.config);
    
    // Register default RAG tools
    toolRegistry.registerTool({
        name: 'rag_search',
        description: 'Perform semantic search using RAG',
        inputSchema: {
            type: 'object',
            properties: {
                query: { type: 'string', description: 'Search query' },
                topK: { type: 'integer', minimum: 1, maximum: 100 },
                minSimilarity: { type: 'number', minimum: 0, maximum: 1 }
            },
            required: ['query'],
            additionalProperties: false
        }
    }, async (args) => {
        const result = await rag.query(args.query, {
            maxResults: args.topK,
            minSimilarity: args.minSimilarity
        });
        
        return {
            jsonrpc: '2.0',
            result: {
                content: [{
                    type: 'text',
                    text: `Found ${result.sources.length} relevant documents:\n\n${result.answer}`
                }]
            },
            id: null
        };
    });
    
    // Register custom tools if provided
    if (options.tools) {
        options.tools.forEach(tool => {
            toolRegistry.registerTool(tool, tool.handler);
        });
    }
    
    return {
        async start() {
            await configManager.load();
            await rag.initialize();
            
            // Start MCP protocol handler (similar to support-tools-mcp)
            const { startMcpHandler } = await import('./mcp-handler.js');
            await startMcpHandler(toolRegistry, configManager);
        },
        
        async stop() {
            await rag.close();
        }
    };
}
```

## Migration Strategy

### Phase 1: Foundation (Weeks 1-2)
1. **Create Simple API Layer**
   - Implement zero-config RAG class for Python
   - Create default configuration system
   - Add basic error handling and validation

2. **JavaScript SDK Foundation**
   - Port core configuration system to JavaScript
   - Enhance existing [`createVectorSearchPipeline`](nodejs/src/index.js) with simple API wrapper
   - Create package structure for npm publishing

### Phase 2: MCP Integration (Weeks 3-4)
1. **MCP Server Templates**
   - Create simple MCP server creation functions following support-tools-mcp patterns
   - Implement tool registration system with JSON schema validation
   - Add configuration bridging between rag-templates and MCP

2. **Documentation and Examples**
   - Create quick-start guides
   - Build example MCP servers
   - Document migration paths from support-tools-mcp patterns

### Phase 3: Enterprise Features (Weeks 5-6)
1. **Advanced Configuration**
   - Implement three-tier config system
   - Add enterprise feature toggles
   - Create configuration validation

2. **Performance and Monitoring**
   - Add performance metrics
   - Implement monitoring hooks
   - Create debugging utilities

### Phase 4: Publishing and Distribution (Weeks 7-8)
1. **Package Publishing**
   - Publish Python package to PyPI
   - Publish JavaScript packages to npm
   - Create installation documentation

2. **Integration Testing**
   - Test with real MCP implementations
   - Validate enterprise deployments
   - Performance benchmarking

## Key Architectural Decisions

### 1. Environment-Based Configuration (Learned from support-tools-mcp)
- **No hardcoded secrets**: All sensitive data from environment variables
- **Validation with defaults**: Required vs optional parameters clearly defined
- **Legacy compatibility**: Support existing [`config.get()`](../../../support-tools-mcp/mcp-node-server/src/config/ConfigManager.ts:157) patterns

### 2. Modular Tool Registry (Inspired by support-tools-mcp)
- **JSON Schema Validation**: All tool inputs validated against schemas
- **MCP Compliance**: [`additionalProperties: false`](../../../support-tools-mcp/mcp-node-server/src/core/ToolRegistry.ts:423) for strict compliance
- **Extensible Design**: Easy registration of custom tools

### 3. Progressive API Complexity
- **Zero Config**: Works immediately with sensible defaults
- **Simple Config**: Basic customization for common use cases
- **Enterprise Config**: Full power of existing system

### 4. Language Parity
- **Consistent APIs**: Same patterns across Python and JavaScript
- **Shared Concepts**: Configuration, pipelines, tools work identically
- **Platform Optimization**: Language-specific optimizations where appropriate

## Success Metrics

### Developer Experience
- **Time to First Query**: < 5 minutes from npm install to working query
- **MCP Server Creation**: < 10 lines of code for basic server
- **Configuration Complexity**: 80% of use cases need ≤ 3 config parameters

### Technical Performance
- **API Response Time**: < 100ms overhead vs direct pipeline usage
- **Memory Footprint**: < 50MB additional for simple API layer
- **Startup Time**: < 2 seconds for zero-config initialization

### Adoption Metrics
- **Package Downloads**: Target 1000+ monthly downloads within 6 months
- **GitHub Stars**: Target 500+ stars within 1 year
- **Community Contributions**: Target 10+ external contributors

## Research-Informed Design Patterns

### LlamaIndex-Inspired Patterns

Based on the research, LlamaIndex's success comes from several key architectural decisions that we should adopt:

#### 1. Global Settings with Local Overrides
```python
# rag_templates/core/settings.py (< 200 lines)
class Settings:
    """Global configuration singleton with local override capability"""
    
    def __init__(self):
        self.llm = None
        self.embedding_model = "text-embedding-3-small"
        self.vector_store = "in_memory"
        self.temperature = 0.7
        self.max_results = 3
    
    def configure(self, **kwargs):
        """Configure global defaults"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

# Global instance
settings = Settings()

# Usage patterns:
# Global: settings.configure(llm="gpt-4o-mini", embedding_model="text-embedding-3-large")
# Local: rag.query("question", llm=custom_llm)  # Overrides global setting
```

#### 2. Node-Centric Data Representation
```python
# rag_templates/core/document.py (< 300 lines)
@dataclass
class Document:
    """Standardized document representation with metadata"""
    
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    source: Optional[str] = None
    
    def to_node(self) -> 'Node':
        """Convert to processing node"""
        return Node(
            id=self.id,
            text=self.content,
            metadata=self.metadata,
            embedding=self.embedding
        )

@dataclass
class Node:
    """Granular processing unit with relationships"""
    
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    relationships: Dict[str, str] = field(default_factory=dict)
    
    def chunk(self, chunk_size: int = 512) -> List['Node']:
        """Split node into smaller chunks"""
        # Implementation for chunking logic
        pass
```

#### 3. Async-First Design
```python
# rag_templates/core/async_pipeline.py (< 400 lines)
class AsyncRAGPipeline:
    """Async-first pipeline following LlamaIndex patterns"""
    
    async def aquery(self, question: str, **kwargs) -> dict:
        """Async query execution"""
        # Parallel document retrieval and processing
        retrieval_task = asyncio.create_task(self._aretrieve(question))
        embedding_task = asyncio.create_task(self._aembed(question))
        
        documents, query_embedding = await asyncio.gather(
            retrieval_task, embedding_task
        )
        
        # Async LLM generation
        answer = await self._agenerate(question, documents)
        
        return {
            'query': question,
            'answer': answer,
            'retrieved_documents': documents
        }
    
    # Sync wrapper for compatibility
    def query(self, question: str, **kwargs) -> dict:
        return asyncio.run(self.aquery(question, **kwargs))
```

### Haystack-Inspired Patterns

#### 1. Component-Based Pipeline Architecture
```python
# rag_templates/core/pipeline.py (< 500 lines)
class Pipeline:
    """Declarative pipeline following Haystack DAG patterns"""
    
    def __init__(self):
        self.components = {}
        self.connections = []
    
    def add_component(self, name: str, component: Component):
        """Add component to pipeline"""
        self.components[name] = component
    
    def connect(self, sender: str, receiver: str, input_name: str = "input"):
        """Connect component outputs to inputs"""
        self.connections.append({
            'sender': sender,
            'receiver': receiver,
            'input_name': input_name
        })
    
    async def run(self, inputs: dict) -> dict:
        """Execute pipeline as DAG"""
        execution_order = self._topological_sort()
        results = {}
        
        for component_name in execution_order:
            component = self.components[component_name]
            component_inputs = self._gather_inputs(component_name, results, inputs)
            results[component_name] = await component.run(**component_inputs)
        
        return results

# Example usage:
# pipeline = Pipeline()
# pipeline.add_component("retriever", VectorRetriever())
# pipeline.add_component("generator", LLMGenerator())
# pipeline.connect("retriever", "generator", "documents")
```

#### 2. YAML Configuration Support
```yaml
# config/pipeline-templates/basic-rag.yaml
name: "basic_rag_pipeline"
description: "Simple RAG pipeline with vector retrieval"

components:
  document_store:
    type: "VectorDocumentStore"
    params:
      embedding_model: "${EMBEDDING_MODEL:text-embedding-3-small}"
      vector_store: "${VECTOR_STORE:in_memory}"
  
  retriever:
    type: "VectorRetriever"
    params:
      document_store: "document_store"
      top_k: "${TOP_K:5}"
  
  generator:
    type: "LLMGenerator"
    params:
      model: "${LLM_MODEL:gpt-4o-mini}"
      temperature: "${TEMPERATURE:0.7}"

connections:
  - from: "retriever"
    to: "generator"
    input: "documents"

inputs:
  - name: "query"
    type: "string"
    required: true

outputs:
  - name: "answer"
    from: "generator"
```

#### 3. Interchangeable Components
```python
# rag_templates/components/base.py (< 200 lines)
from abc import ABC, abstractmethod

class Component(ABC):
    """Base component interface"""
    
    @abstractmethod
    async def run(self, **inputs) -> dict:
        """Execute component logic"""
        pass
    
    @abstractmethod
    def get_schema(self) -> dict:
        """Return input/output schema"""
        pass

class Retriever(Component):
    """Base retriever interface"""
    
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve relevant documents"""
        pass

class Generator(Component):
    """Base generator interface"""
    
    @abstractmethod
    async def generate(self, query: str, documents: List[Document]) -> str:
        """Generate answer from query and documents"""
        pass
```

### Progressive Complexity Implementation

#### 1. Three-Tier API Design (Inspired by Research)
```python
# rag_templates/__init__.py (< 100 lines)
"""
Progressive complexity exports:
- Simple: RAG class with zero config
- Standard: RAG class with basic config
- Enterprise: Full pipeline and component access
"""

# Simple API (Zero Config)
from .simple import RAG

# Standard API (Basic Config)
from .standard import ConfigurableRAG

# Enterprise API (Full Control)
from .enterprise import (
    Pipeline, Component, Settings,
    VectorRetriever, LLMGenerator,
    DocumentStore, ConfigManager
)

# MCP Integration
from .mcp import create_mcp_server, MCPTool

# Convenience imports for common use cases
from .core.document import Document, Node
from .core.settings import settings

__all__ = [
    # Simple API
    'RAG',
    # Standard API
    'ConfigurableRAG',
    # Enterprise API
    'Pipeline', 'Component', 'Settings',
    'VectorRetriever', 'LLMGenerator', 'DocumentStore',
    'ConfigManager',
    # MCP
    'create_mcp_server', 'MCPTool',
    # Core
    'Document', 'Node', 'settings'
]
```

#### 2. Simple API Implementation
```python
# rag_templates/simple.py (< 150 lines)
class RAG:
    """Dead simple RAG interface - works out of the box"""
    
    def __init__(self):
        # Use global settings with sensible defaults
        self._pipeline = self._create_default_pipeline()
        self._initialized = False
    
    def query(self, question: str) -> str:
        """Simple query that returns just the answer"""
        if not self._initialized:
            self._initialize()
        
        result = self._pipeline.query(question)
        return result['answer']
    
    def add_documents(self, documents: List[str]) -> None:
        """Simple document addition"""
        if not self._initialized:
            self._initialize()
        
        doc_objects = [
            Document(id=f"doc_{i}", content=doc)
            for i, doc in enumerate(documents)
        ]
        self._pipeline.add_documents(doc_objects)
    
    def _create_default_pipeline(self):
        """Create pipeline with zero configuration"""
        from .core.pipeline_factory import PipelineFactory
        return PipelineFactory.create_simple()
    
    def _initialize(self):
        """Lazy initialization"""
        self._pipeline.initialize()
        self._initialized = True
```

### MCP Integration Architecture (Research-Informed)

#### 1. Service Encapsulation Pattern
```python
# rag_templates/mcp/server_factory.py (< 300 lines)
class MCPServerFactory:
    """Factory for creating MCP servers with RAG capabilities"""
    
    @staticmethod
    def create_simple_server(name: str, description: str = None) -> MCPServer:
        """Create zero-config MCP server"""
        rag = RAG()  # Simple API
        
        return MCPServer(
            name=name,
            description=description or f"RAG-powered MCP server: {name}",
            tools=[
                MCPTool(
                    name="search",
                    description="Search knowledge base",
                    schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"],
                        "additionalProperties": False
                    },
                    handler=lambda args: {"answer": rag.query(args["query"])}
                )
            ]
        )
    
    @staticmethod
    def create_enterprise_server(config: dict) -> MCPServer:
        """Create fully configured MCP server"""
        # Use enterprise API for full control
        pipeline = Pipeline.from_config(config)
        
        tools = []
        for tool_config in config.get('tools', []):
            tools.append(MCPTool.from_config(tool_config, pipeline))
        
        return MCPServer(
            name=config['name'],
            description=config.get('description'),
            tools=tools,
            middleware=config.get('middleware', [])
        )
```

#### 2. Dynamic Tool Routing
```python
# rag_templates/mcp/tool_router.py (< 250 lines)
class DynamicToolRouter:
    """Route queries to appropriate RAG techniques based on content"""
    
    def __init__(self, techniques: Dict[str, Pipeline]):
        self.techniques = techniques
        self.router_llm = self._create_router_llm()
    
    async def route_query(self, query: str) -> str:
        """Intelligently route query to best RAG technique"""
        
        # Use LLM to classify query type
        classification = await self.router_llm.classify(
            query, list(self.techniques.keys())
        )
        
        # Execute with selected technique
        technique = self.techniques[classification['technique']]
        result = await technique.aquery(query)
        
        return result['answer']
    
    def _create_router_llm(self):
        """Create LLM for query classification"""
        return LLMClassifier(
            model="gpt-4o-mini",
            system_prompt="""
            Classify the query type to select the best RAG technique:
            - basic_rag: Simple factual questions
            - colbert: Complex multi-part queries
            - graphrag: Relationship and connection queries
            - hyde: Hypothetical or speculative questions
            """
        )
```

## Implementation Roadmap (Research-Informed)

### Phase 1: Foundation (Weeks 1-2) - LlamaIndex Patterns
1. **Global Settings System**
   - Implement [`Settings`](rag_templates/core/settings.py) singleton with local overrides
   - Create environment variable integration
   - Add validation and type checking

2. **Document/Node Architecture**
   - Implement [`Document`](rag_templates/core/document.py) and [`Node`](rag_templates/core/document.py) classes
   - Add chunking and relationship management
   - Create serialization support

3. **Simple API Layer**
   - Build zero-config [`RAG`](rag_templates/simple.py) class
   - Implement lazy initialization
   - Add basic error handling

### Phase 2: Component System (Weeks 3-4) - Haystack Patterns
1. **Pipeline Architecture**
   - Create [`Component`](rag_templates/components/base.py) base classes
   - Implement [`Pipeline`](rag_templates/core/pipeline.py) DAG execution
   - Add YAML configuration support

2. **Interchangeable Components**
   - Build retriever, generator, and store interfaces
   - Create default implementations
   - Add component registry system

3. **Async-First Design**
   - Implement [`AsyncRAGPipeline`](rag_templates/core/async_pipeline.py)
   - Add parallel processing capabilities
   - Create sync compatibility wrappers

### Phase 3: MCP Integration (Weeks 5-6) - Research Best Practices
1. **MCP Server Factory**
   - Implement [`MCPServerFactory`](rag_templates/mcp/server_factory.py)
   - Add tool registration system
   - Create configuration bridging

2. **Dynamic Tool Routing**
   - Build [`DynamicToolRouter`](rag_templates/mcp/tool_router.py)
   - Implement query classification
   - Add technique selection logic

3. **Enterprise Features**
   - Add monitoring and observability
   - Implement caching strategies
   - Create security validation

### Phase 4: Distribution (Weeks 7-8) - Ecosystem Patterns
1. **Package Structure**
   - Create modular package architecture
   - Implement plugin system
   - Add extension points

2. **Developer Experience**
   - Build comprehensive documentation
   - Create tutorial notebooks
   - Add example templates ("Packs")

3. **Testing and Validation**
   - Implement progressive complexity tests
   - Add performance benchmarks
   - Create integration test suite

## Success Metrics (Research-Informed)

### Developer Experience (LlamaIndex-Inspired)
- **Time to First Query**: < 3 minutes (LlamaIndex: ~5 minutes)
- **Lines of Code for Basic Use**: < 5 lines (LlamaIndex: 3-4 lines)
- **Configuration Complexity**: 90% of use cases need ≤ 2 parameters

### Technical Performance (Haystack-Inspired)
- **Component Swapping**: < 1 line of code to change retrievers/generators
- **Pipeline Execution**: < 50ms overhead vs direct component calls
- **Memory Efficiency**: < 30MB additional for simple API layer

### Adoption Metrics (Industry Standards)
- **Package Downloads**: Target 500+ monthly downloads within 3 months
- **GitHub Engagement**: Target 200+ stars within 6 months
- **Community Growth**: Target 5+ external contributors within 1 year

## Conclusion

This comprehensive architecture provides a systematic approach to transforming rag-templates into a library consumption framework that maintains enterprise capabilities while dramatically simplifying the developer experience. By incorporating proven patterns from LlamaIndex (global settings, node-centric design, async-first) and Haystack (component architecture, YAML configuration, pipeline DAGs), we create a framework that:

1. **Starts Simple**: Zero-config API that works immediately
2. **Scales Progressively**: Clear path from simple to enterprise usage
3. **Maintains Power**: Full access to existing RAG techniques and enterprise features
4. **Enables Innovation**: Extensible architecture for new techniques and integrations
5. **Follows Best Practices**: Research-informed patterns from successful frameworks

The modular design ensures clean separation of concerns with files under 500 lines, while the progressive complexity approach provides multiple entry points for developers with different needs and expertise levels.