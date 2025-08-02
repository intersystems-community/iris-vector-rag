# API Reference - Library Consumption Framework

Complete API documentation for both Python and JavaScript implementations of the rag-templates Library Consumption Framework.

## Table of Contents

1. [Python API](#python-api)
2. [JavaScript API](#javascript-api)
3. [Configuration Reference](#configuration-reference)
4. [Error Handling](#error-handling)
5. [Type Definitions](#type-definitions)
6. [Environment Variables](#environment-variables)

## Python API

### Simple API

#### `RAG` Class

Zero-configuration Simple API for immediate RAG functionality.

```python
from rag_templates import RAG

rag = RAG()
```

##### Constructor

```python
RAG(config_path: Optional[str] = None, **kwargs)
```

**Parameters:**
- `config_path` (Optional[str]): Path to configuration file
- `**kwargs`: Configuration overrides

**Example:**
```python
# Zero configuration
rag = RAG()

# With configuration file
rag = RAG("config.yaml")

# With inline configuration
rag = RAG(technique="colbert", max_results=10)
```

##### Methods

###### `add_documents(documents, **kwargs)`

Add documents to the knowledge base.

**Parameters:**
- `documents` (List[Union[str, Dict]]): Documents to add
- `**kwargs`: Additional processing options

**Returns:** `None`

**Example:**
```python
# String documents
rag.add_documents([
    "Document 1 content",
    "Document 2 content"
])

# Document objects
rag.add_documents([
    {
        "content": "Document content",
        "title": "Document Title",
        "source": "file.pdf",
        "metadata": {"author": "John Doe"}
    }
])
```

###### `query(query_text, **kwargs)`

Query the RAG system and return a simple answer.

**Parameters:**
- `query_text` (str): The question or query
- `**kwargs`: Query options

**Returns:** `str` - Answer to the query

**Example:**
```python
answer = rag.query("What is machine learning?")
print(answer)  # "Machine learning is a subset of artificial intelligence..."

# With options
answer = rag.query("Explain neural networks", 
                  max_results=10, 
                  min_similarity=0.8)
```

###### `get_document_count()`

Get the number of documents in the knowledge base.

**Returns:** `int` - Number of documents

**Example:**
```python
count = rag.get_document_count()
print(f"Knowledge base contains {count} documents")
```

###### `get_config(key, default=None)`

Get a configuration value.

**Parameters:**
- `key` (str): Configuration key in dot notation
- `default` (Any): Default value if key not found

**Returns:** Configuration value or default

**Example:**
```python
host = rag.get_config("database.iris.host", "localhost")
model = rag.get_config("embeddings.model")
```

###### `set_config(key, value)`

Set a configuration value.

**Parameters:**
- `key` (str): Configuration key in dot notation
- `value` (Any): Value to set

**Example:**
```python
rag.set_config("temperature", 0.1)
rag.set_config("database.iris.host", "production-server")
```

###### `validate_config()`

Validate the current configuration.

**Returns:** `bool` - True if valid

**Raises:** `ConfigurationError` if validation fails

**Example:**
```python
try:
    is_valid = rag.validate_config()
    print(f"Configuration valid: {is_valid}")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

### Standard API

#### `ConfigurableRAG` Class

Advanced Standard API for configurable RAG operations with technique selection and complex configuration.

```python
from rag_templates import ConfigurableRAG

rag = ConfigurableRAG({"technique": "colbert"})
```

##### Constructor

```python
ConfigurableRAG(config: Union[Dict, str, ConfigManager])
```

**Parameters:**
- `config` (Union[Dict, str, ConfigManager]): Configuration object, file path, or ConfigManager instance

**Example:**
```python
# Dictionary configuration
rag = ConfigurableRAG({
    "technique": "colbert",
    "llm_provider": "openai",
    "llm_config": {
        "model": "gpt-4o-mini",
        "temperature": 0.1
    }
})

# From configuration file
rag = ConfigurableRAG("advanced-config.yaml")

# From ConfigManager
from rag_templates.config import ConfigManager
config = ConfigManager.from_file("config.yaml")
rag = ConfigurableRAG(config)
```

##### Methods

###### `query(query_text, options=None)`

Advanced query with rich result object.

**Parameters:**
- `query_text` (str): The question or query
- `options` (Optional[Dict]): Query options

**Returns:** `QueryResult` - Rich result object

**Example:**
```python
result = rag.query("What is machine learning?", {
    "max_results": 10,
    "include_sources": True,
    "min_similarity": 0.8,
    "source_filter": "academic_papers"
})

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Sources: {len(result.sources)}")
for source in result.sources:
    print(f"  - {source.title} (similarity: {source.similarity:.2f})")
```

###### `get_available_techniques()`

List available RAG techniques.

**Returns:** `List[str]` - Available technique names

**Example:**
```python
techniques = rag.get_available_techniques()
print(f"Available techniques: {techniques}")
# Output: ['basic', 'colbert', 'crag', 'hyde', 'graphrag', 'hybrid_ifind', 'noderag', 'sql_rag']
```

###### `get_technique_info(technique_name)`

Get information about a specific technique.

**Parameters:**
- `technique_name` (str): Name of the technique

**Returns:** `Dict` - Technique information

**Example:**
```python
info = rag.get_technique_info("colbert")
print(f"Description: {info['description']}")
print(f"Best for: {info['best_for']}")
print(f"Parameters: {info['parameters']}")
```

###### `switch_technique(technique_name, config=None)`

Switch to a different RAG technique.

**Parameters:**
- `technique_name` (str): Name of the technique to switch to
- `config` (Optional[Dict]): Technique-specific configuration

**Example:**
```python
# Switch to ColBERT
rag.switch_technique("colbert", {
    "max_query_length": 512,
    "top_k": 15
})

# Switch to HyDE
rag.switch_technique("hyde")
```

### Configuration Management

#### `ConfigManager` Class

Manages configuration loading from files and environment variables.

```python
from rag_templates.config import ConfigManager

config = ConfigManager.from_file("config.yaml")
```

##### Class Methods

###### `ConfigManager.from_file(path)`

Load configuration from a YAML file.

**Parameters:**
- `path` (str): Path to YAML configuration file

**Returns:** `ConfigManager` instance

**Example:**
```python
config = ConfigManager.from_file("production-config.yaml")
rag = ConfigurableRAG(config)
```

##### Methods

###### `get(key, default=None)`

Get configuration value with dot notation support.

**Parameters:**
- `key` (str): Configuration key (e.g., "database.iris.host")
- `default` (Any): Default value if key not found

**Returns:** Configuration value or default

**Example:**
```python
host = config.get("database.iris.host", "localhost")
model = config.get("llm_config.model", "gpt-4o-mini")
```

###### `set(key, value)`

Set configuration value with dot notation support.

**Parameters:**
- `key` (str): Configuration key
- `value` (Any): Value to set

**Example:**
```python
config.set("temperature", 0.1)
config.set("database.iris.port", 52773)
```

## JavaScript API

### Simple API

#### `RAG` Class

Zero-configuration Simple API for immediate RAG functionality.

```javascript
import { RAG } from '@rag-templates/core';

const rag = new RAG();
```

##### Constructor

```javascript
new RAG(configPath = null, options = {})
```

**Parameters:**
- `configPath` (string|null): Path to configuration file
- `options` (Object): Configuration overrides

**Example:**
```javascript
// Zero configuration
const rag = new RAG();

// With configuration file
const rag = new RAG("config.yaml");

// With inline configuration
const rag = new RAG(null, {technique: "colbert", maxResults: 10});
```

##### Methods

###### `addDocuments(documents, options = {})`

Add documents to the knowledge base.

**Parameters:**
- `documents` (Array<string|Object>): Documents to add
- `options` (Object): Additional processing options

**Returns:** `Promise<void>`

**Example:**
```javascript
// String documents
await rag.addDocuments([
    "Document 1 content",
    "Document 2 content"
]);

// Document objects
await rag.addDocuments([
    {
        content: "Document content",
        title: "Document Title",
        source: "file.pdf",
        metadata: {author: "John Doe"}
    }
]);
```

###### `query(queryText, options = {})`

Query the RAG system and return a simple answer.

**Parameters:**
- `queryText` (string): The question or query
- `options` (Object): Query options

**Returns:** `Promise<string>` - Answer to the query

**Example:**
```javascript
const answer = await rag.query("What is machine learning?");
console.log(answer);  // "Machine learning is a subset of artificial intelligence..."

// With options
const answer = await rag.query("Explain neural networks", {
    maxResults: 10,
    minSimilarity: 0.8
});
```

###### `getDocumentCount()`

Get the number of documents in the knowledge base.

**Returns:** `Promise<number>` - Number of documents

**Example:**
```javascript
const count = await rag.getDocumentCount();
console.log(`Knowledge base contains ${count} documents`);
```

###### `getConfig(key, defaultValue = null)`

Get a configuration value.

**Parameters:**
- `key` (string): Configuration key in dot notation
- `defaultValue` (any): Default value if key not found

**Returns:** Configuration value or default

**Example:**
```javascript
const host = rag.getConfig("database.iris.host", "localhost");
const model = rag.getConfig("embeddings.model");
```

###### `setConfig(key, value)`

Set a configuration value.

**Parameters:**
- `key` (string): Configuration key in dot notation
- `value` (any): Value to set

**Example:**
```javascript
rag.setConfig("temperature", 0.1);
rag.setConfig("database.iris.host", "production-server");
```

###### `validateConfig()`

Validate the current configuration.

**Returns:** `Promise<boolean>` - True if valid

**Throws:** `ConfigurationError` if validation fails

**Example:**
```javascript
try {
    const isValid = await rag.validateConfig();
    console.log(`Configuration valid: ${isValid}`);
} catch (error) {
    console.error(`Configuration error: ${error.message}`);
}
```

### Standard API

#### `ConfigurableRAG` Class

Advanced Standard API for configurable RAG operations.

```javascript
import { ConfigurableRAG } from '@rag-templates/core';

const rag = new ConfigurableRAG({technique: "colbert"});
```

##### Constructor

```javascript
new ConfigurableRAG(config)
```

**Parameters:**
- `config` (Object|string|ConfigManager): Configuration object, file path, or ConfigManager instance

**Example:**
```javascript
// Object configuration
const rag = new ConfigurableRAG({
    technique: "colbert",
    llmProvider: "openai",
    llmConfig: {
        model: "gpt-4o-mini",
        temperature: 0.1
    }
});

// From configuration file
const rag = await ConfigurableRAG.fromConfigFile("advanced-config.yaml");

// From ConfigManager
import { ConfigManager } from '@rag-templates/core';
const config = await ConfigManager.fromFile("config.yaml");
const rag = new ConfigurableRAG(config);
```

##### Methods

###### `query(queryText, options = {})`

Advanced query with rich result object.

**Parameters:**
- `queryText` (string): The question or query
- `options` (Object): Query options

**Returns:** `Promise<QueryResult>` - Rich result object

**Example:**
```javascript
const result = await rag.query("What is machine learning?", {
    maxResults: 10,
    includeSources: true,
    minSimilarity: 0.8,
    sourceFilter: "academic_papers"
});

console.log(`Answer: ${result.answer}`);
console.log(`Confidence: ${result.confidence}`);
console.log(`Sources: ${result.sources.length}`);
result.sources.forEach(source => {
    console.log(`  - ${source.title} (similarity: ${source.similarity.toFixed(2)})`);
});
```

###### `getAvailableTechniques()`

List available RAG techniques.

**Returns:** `Array<string>` - Available technique names

**Example:**
```javascript
const techniques = rag.getAvailableTechniques();
console.log(`Available techniques: ${techniques}`);
// Output: ['basic', 'colbert', 'crag', 'hyde', 'graphrag', 'hybrid_ifind', 'noderag', 'sql_rag']
```

###### `getTechniqueInfo(techniqueName)`

Get information about a specific technique.

**Parameters:**
- `techniqueName` (string): Name of the technique

**Returns:** `Object` - Technique information

**Example:**
```javascript
const info = rag.getTechniqueInfo("colbert");
console.log(`Description: ${info.description}`);
console.log(`Best for: ${info.bestFor}`);
console.log(`Parameters: ${JSON.stringify(info.parameters)}`);
```

###### `switchTechnique(techniqueName, config = {})`

Switch to a different RAG technique.

**Parameters:**
- `techniqueName` (string): Name of the technique to switch to
- `config` (Object): Technique-specific configuration

**Returns:** `Promise<void>`

**Example:**
```javascript
// Switch to ColBERT
await rag.switchTechnique("colbert", {
    maxQueryLength: 512,
    topK: 15
});

// Switch to HyDE
await rag.switchTechnique("hyde");
```

### Configuration Management

#### `ConfigManager` Class

Manages configuration loading from files and environment variables.

```javascript
import { ConfigManager } from '@rag-templates/core';

const config = await ConfigManager.fromFile("config.yaml");
```

##### Static Methods

###### `ConfigManager.fromFile(path)`

Load configuration from a YAML file.

**Parameters:**
- `path` (string): Path to YAML configuration file

**Returns:** `Promise<ConfigManager>` instance

**Example:**
```javascript
const config = await ConfigManager.fromFile("production-config.yaml");
const rag = new ConfigurableRAG(config);
```

##### Methods

###### `get(key, defaultValue = null)`

Get configuration value with dot notation support.

**Parameters:**
- `key` (string): Configuration key (e.g., "database.iris.host")
- `defaultValue` (any): Default value if key not found

**Returns:** Configuration value or default

**Example:**
```javascript
const host = config.get("database.iris.host", "localhost");
const model = config.get("llmConfig.model", "gpt-4o-mini");
```

###### `set(key, value)`

Set configuration value with dot notation support.

**Parameters:**
- `key` (string): Configuration key
- `value` (any): Value to set

**Example:**
```javascript
config.set("temperature", 0.1);
config.set("database.iris.port", 52773);
```

### MCP Integration

#### `createMCPServer(config)`

Create an MCP server with RAG capabilities.

```javascript
import { createMCPServer } from '@rag-templates/mcp';

const server = createMCPServer({
    name: "my-rag-server",
    description: "RAG-powered MCP server"
});
```

**Parameters:**
- `config` (Object): Server configuration

**Configuration Options:**
- `name` (string): Server name
- `description` (string): Server description
- `version` (string): Server version (default: "1.0.0")
- `ragConfig` (Object): RAG configuration (optional)
- `enabledTools` (Array<string>): List of enabled tools (optional)
- `tools` (Array<Object>): Custom tool definitions (optional)

**Returns:** MCP server instance

**Example:**
```javascript
// Simple server
const server = createMCPServer({
    name: "knowledge-assistant",
    description: "Company knowledge base"
});

// Advanced server
const server = createMCPServer({
    name: "advanced-rag-server",
    description: "Advanced RAG with custom tools",
    ragConfig: {
        technique: 'colbert',
        llmProvider: 'openai'
    },
    tools: [
        {
            name: "custom_search",
            description: "Custom search tool",
            inputSchema: {
                type: 'object',
                properties: {
                    query: { type: 'string' }
                },
                required: ['query']
            },
            handler: async (args, rag) => {
                return await rag.query(args.query);
            }
        }
    ]
});

await server.start();
```

## Storage Layer API

The storage layer provides two classes for different use cases:

### IRISVectorStore (Standard API)

LangChain-compatible vector store for standard RAG applications.

```python
from iris_rag.storage.vector_store_iris import IRISVectorStore
from iris_rag.core.connection import ConnectionManager  
from iris_rag.config.manager import ConfigurationManager

config = ConfigurationManager()
connection = ConnectionManager(config)
vector_store = IRISVectorStore(connection, config)
```

#### Key Features:
- **LangChain compatibility**: Drop-in replacement for LangChain vector stores
- **Automatic schema management**: Creates tables and indexes automatically  
- **Security validation**: Validates table names and query parameters
- **Custom table support**: Configure custom table names via config

#### Methods:

```python
# Add documents
vector_store.add_documents(documents)

# Similarity search
results = vector_store.similarity_search("query", k=5)

# Similarity search with scores
results = vector_store.similarity_search_with_score("query", k=5)

# Use as LangChain retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
```

#### Custom Table Configuration:
```yaml
# config.yaml
storage:
  iris:
    table_name: "MyCompany.Documents"  # Custom table name
```

### IRISStorage (Enterprise API)

Enterprise-grade storage with full manual control for complex scenarios.

```python
from iris_rag.storage.enterprise_storage import IRISStorage

storage = IRISStorage(connection, config)
```

#### Key Features:
- **Manual schema control**: Full control over database schema creation
- **Legacy integration**: Works with existing database schemas
- **Schema migration**: Add missing columns to existing tables
- **Enterprise flexibility**: Complete customization of storage behavior

#### Methods:

```python
# Initialize or update schema
storage.initialize_schema()  # Adds missing columns like doc_id, metadata

# Store documents directly
storage.store_documents(documents)

# Vector search with manual control
results = storage.vector_search(query_vector, top_k=5)

# Get document by ID
document = storage.get_document(doc_id)
```

### When to Use Which Storage Class

#### Use IRISVectorStore (Standard) When:
- Building standard RAG applications
- Using LangChain ecosystem 
- Want automatic schema management
- Need LangChain compatibility

#### Use IRISStorage (Enterprise) When:
- Integrating with existing databases
- Need custom schema modifications
- Require manual control over database operations
- Migrating from legacy systems

### Custom Table Names

Both storage classes support custom table names:

```python
# Via configuration
config_data = {
    "storage": {
        "iris": {
            "table_name": "Sales.CustomerDocuments"
        }
    }
}

# Both classes will use the custom table name
vector_store = IRISVectorStore(connection, config)  # Uses Sales.CustomerDocuments
storage = IRISStorage(connection, config)           # Uses Sales.CustomerDocuments
```

### Security Considerations

- **Table name validation**: Both classes validate table names to prevent SQL injection
- **Parameterized queries**: All queries use parameterized statements
- **Field validation**: Input validation for all user-provided data
- **Schema security**: Custom tables must follow `Schema.TableName` format

## Configuration Reference

### Configuration File Format

#### YAML Configuration
```yaml
# Basic configuration
technique: "colbert"
llm_provider: "openai"
embedding_model: "text-embedding-3-small"

# Advanced configuration
llm_config:
  model: "gpt-4o-mini"
  temperature: 0.1
  max_tokens: 1000

embedding_config:
  model: "text-embedding-3-small"
  dimension: 1536
  batch_size: 100

database:
  iris:
    host: "${IRIS_HOST}"
    port: "${IRIS_PORT}"
    username: "${IRIS_USERNAME}"
    password: "${IRIS_PASSWORD}"
    namespace: "RAG_PRODUCTION"

technique_config:
  colbert:
    max_query_length: 512
    doc_maxlen: 180
    top_k: 15
  hyde:
    num_hypotheses: 3
    hypothesis_length: 100

vector_index:
  type: "HNSW"
  M: 16
  efConstruction: 200

caching:
  enabled: true
  ttl: 3600
  max_size: 1000

monitoring:
  enabled: true
  log_level: "INFO"
```

### Configuration Options

#### Core Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `technique` | string | "basic" | RAG technique to use |
| `llm_provider` | string | "openai" | LLM provider |
| `embedding_model` | string | "text-embedding-3-small" | Embedding model |
| `max_results` | integer | 5 | Default number of results |
| `temperature` | number | 0.7 | LLM temperature |

#### Database Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `database.iris.host` | string | "localhost" | IRIS database host |
| `database.iris.port` | integer | 52773 | IRIS database port |
| `database.iris.username` | string | "demo" | Database username |
| `database.iris.password` | string | "demo" | Database password |
| `database.iris.namespace` | string | "RAG" | Database namespace |

#### LLM Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `llm_config.model` | string | "gpt-4o-mini" | LLM model name |
| `llm_config.temperature` | number | 0.7 | Response randomness |
| `llm_config.max_tokens` | integer | 1000 | Maximum response length |
| `llm_config.api_key` | string | - | API key (use environment variable) |

#### Embedding Configuration

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `embedding_config.model` | string | "text-embedding-3-small" | Embedding model |
| `embedding_config.dimension` | integer | 1536 | Embedding dimension |
| `embedding_config.batch_size` | integer | 100 | Batch size for processing |

## Error Handling

### Python Exceptions

#### `RAGFrameworkError`
Base exception for all RAG framework errors.

```python
from rag_templates.core.errors import RAGFrameworkError

try:
    rag = RAG()
    answer = rag.query("test")
except RAGFrameworkError as e:
    print(f"RAG error: {e}")
```

#### `ConfigurationError`
Configuration-related errors.

```python
from rag_templates.core.errors import ConfigurationError

try:
    rag = RAG("invalid-config.yaml")
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

#### `InitializationError`
Initialization and setup errors.

```python
from rag_templates.core.errors import InitializationError

try:
    rag = RAG()
    rag.add_documents(documents)
except InitializationError as e:
    print(f"Initialization error: {e}")
```

### JavaScript Errors

#### `RAGError`
Base error for all RAG framework errors.

```javascript
import { RAGError } from '@rag-templates/core';

try {
    const rag = new RAG();
    const answer = await rag.query("test");
} catch (error) {
    if (error instanceof RAGError) {
        console.error(`RAG error: ${error.message}`);
    }
}
```

#### `ConfigurationError`
Configuration-related errors.

```javascript
import { ConfigurationError } from '@rag-templates/core';

try {
    const rag = new RAG("invalid-config.yaml");
} catch (error) {
    if (error instanceof ConfigurationError) {
        console.error(`Configuration error: ${error.message}`);
    }
}
```

#### `InitializationError`
Initialization and setup errors.

```javascript
import { InitializationError } from '@rag-templates/core';

try {
    const rag = new RAG();
    await rag.addDocuments(documents);
} catch (error) {
    if (error instanceof InitializationError) {
        console.error(`Initialization error: ${error.message}`);
    }
}
```

## Type Definitions

### Python Types

#### `QueryResult`
```python
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class QueryResult:
    answer: str
    confidence: float
    sources: Optional[List[DocumentSource]]
    metadata: Optional[Dict[str, Any]]
    processing_time_ms: Optional[int]
```

#### `DocumentSource`
```python
@dataclass
class DocumentSource:
    title: str
    content: str
    source: str
    similarity: float
    metadata: Optional[Dict[str, Any]]
```

#### `Document`
```python
@dataclass
class Document:
    content: str
    title: Optional[str] = None
    source: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

### JavaScript Types

#### `QueryResult`
```typescript
interface QueryResult {
    answer: string;
    confidence: number;
    sources?: DocumentSource[];
    metadata?: Record<string, any>;
    processingTimeMs?: number;
}
```

#### `DocumentSource`
```typescript
interface DocumentSource {
    title: string;
    content: string;
    source: string;
    similarity: number;
    metadata?: Record<string, any>;
}
```

#### `Document`
```typescript
interface Document {
    content: string;
    title?: string;
    source?: string;
    metadata?: Record<string, any>;
}
```

## Environment Variables

### Database Configuration
```bash
# IRIS Database
IRIS_HOST=localhost
IRIS_PORT=52773
IRIS_USERNAME=demo
IRIS_PASSWORD=demo
IRIS_NAMESPACE=RAG_PRODUCTION

# Connection settings
IRIS_CONNECTION_TIMEOUT=30
IRIS_POOL_SIZE=10
```

### LLM Configuration
```bash
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-sonnet

# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_VERSION=2024-02-01
```

### Framework Configuration
```bash
# RAG Configuration
RAG_TECHNIQUE=colbert
RAG_MAX_RESULTS=5
RAG_CACHE_TTL=3600

# Embedding Configuration
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_BATCH_SIZE=100

# Logging
LOG_LEVEL=INFO
DEBUG_MODE=false
```

### MCP Configuration
```bash
# MCP Server
MCP_SERVER_NAME=rag-assistant
MCP_SERVER_DESCRIPTION=RAG-powered assistant
MCP_SERVER_VERSION=1.0.0

# MCP Tools
MCP_ENABLED_TOOLS=rag_search,rag_add_documents,rag_get_stats
```

---

**Next Steps:**
- [Library Consumption Guide](LIBRARY_CONSUMPTION_GUIDE.md) - Complete usage guide
- [MCP Integration Guide](MCP_INTEGRATION_GUIDE.md) - MCP server creation
- [Migration Guide](MIGRATION_GUIDE.md) - Migrate from complex setup
- [Examples](EXAMPLES.md) - Comprehensive examples