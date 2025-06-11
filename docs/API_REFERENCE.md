# RAG Templates API Reference

## Overview

The `iris_rag` package provides a modular framework for building Retrieval Augmented Generation (RAG) systems with InterSystems IRIS. This API reference documents all public classes, methods, and functions.

## Package Structure Validation

### Naming Convention Analysis

Based on research of InterSystems Python packages and Python best practices, our current structure follows established conventions:

- **Package name**: `iris_rag` (snake_case, descriptive)
- **Module structure**: Hierarchical with clear separation of concerns
- **Class names**: PascalCase (e.g., `RAGPipeline`, `ConnectionManager`)
- **Method names**: snake_case (e.g., `execute()`, `load_documents()`)
- **Parameter names**: Consistent across modules (`iris_connector`, `embedding_func`, `llm_func`)

✅ **Validation Result**: Current naming conventions align with Python PEP 8 and InterSystems best practices.

## Core API

### Factory Function

#### [`create_pipeline()`](../rag_templates/__init__.py:10)

Creates RAG pipeline instances with automatic configuration management.

```python
from iris_rag import create_pipeline

# Create a basic RAG pipeline
pipeline = create_pipeline(
    pipeline_type="basic",
    config_path="config.yaml",
    llm_func=my_llm_function
)
```

**Parameters:**
- `pipeline_type` (str): Type of pipeline ("basic")
- `config_path` (Optional[str]): Path to configuration file
- `llm_func` (Optional[Callable]): LLM function for answer generation
- `**kwargs`: Additional configuration parameters

**Returns:** [`RAGPipeline`](../rag_templates/core/base.py:3) instance

**Raises:** `ValueError` if pipeline_type is unknown

---

## Core Classes

### [`RAGPipeline`](../rag_templates/core/base.py:3)

Abstract base class defining the RAG pipeline interface.

```python
from iris_rag.core.base import RAGPipeline

class MyRAGPipeline(RAGPipeline):
    def execute(self, query_text: str, **kwargs) -> dict:
        # Implementation
        pass
```

#### Methods

##### [`execute()`](../rag_templates/core/base.py:13)

Executes the full RAG pipeline for a given query.

**Parameters:**
- `query_text` (str): Input query string
- `**kwargs`: Pipeline-specific arguments

**Returns:** dict with keys:
- `"query"`: Original query
- `"answer"`: Generated answer  
- `"retrieved_documents"`: Retrieved documents

##### [`load_documents()`](../rag_templates/core/base.py:32)

Loads and processes documents into the knowledge base.

**Parameters:**
- `documents_path` (str): Path to documents or directory
- `**kwargs`: Loading-specific arguments

##### [`query()`](../rag_templates/core/base.py:46)

Performs document retrieval step.

**Parameters:**
- `query_text` (str): Query string
- `top_k` (int): Number of documents to retrieve (default: 5)
- `**kwargs`: Retrieval-specific arguments

**Returns:** List of retrieved documents

---

### [`ConnectionManager`](../rag_templates/core/connection.py:19)

Manages database connections with caching and configuration support.

```python
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager

config_manager = ConfigurationManager("config.yaml")
conn_manager = ConnectionManager(config_manager)
connection = conn_manager.get_connection("iris")
```

#### Methods

##### [`__init__()`](../rag_templates/core/connection.py:29)

**Parameters:**
- `config_manager` (Optional[ConfigurationManager]): Configuration manager instance

##### [`get_connection()`](../rag_templates/core/connection.py:45)

Retrieves or creates a cached database connection.

**Parameters:**
- `backend_name` (str): Database backend name ("iris")

**Returns:** Database connection object

---

### [`ConfigurationManager`](../rag_templates/config/manager.py:10)

Manages configuration loading from YAML files and environment variables.

```python
from iris_rag.config.manager import ConfigurationManager

# Load from file
config = ConfigurationManager("config.yaml")

# Access nested configuration
db_config = config.get("database:iris")
```

#### Environment Variable Support

Environment variables use `RAG_` prefix with `__` for nesting:
- `RAG_DATABASE__IRIS__HOST` → `config['database']['iris']['host']`
- `RAG_EMBEDDINGS__MODEL` → `config['embeddings']['model']`

#### Methods

##### [`__init__()`](../rag_templates/config/manager.py:22)

**Parameters:**
- `config_path` (Optional[str]): Path to YAML configuration file
- `schema` (Optional[Dict]): Configuration schema for validation

**Raises:** `FileNotFoundError` if config file doesn't exist

##### [`get()`](../rag_templates/config/manager.py:50)

Retrieves configuration value with dot notation support.

**Parameters:**
- `key` (str): Configuration key (e.g., "database:iris:host")
- `default` (Any): Default value if key not found

**Returns:** Configuration value or default

---

### [`Document`](../rag_templates/core/models.py:10)

Immutable document representation with metadata support.

```python
from iris_rag.core.models import Document

doc = Document(
    page_content="Document text content",
    metadata={"source": "file.pdf", "page": 1},
    id="custom-id"  # Optional, UUID generated if not provided
)
```

#### Attributes

- `page_content` (str): Main textual content
- `metadata` (Dict[str, Any]): Additional document information
- `id` (str): Unique identifier (auto-generated UUID if not provided)

**Note:** Document instances are frozen (immutable) and hashable.

---

## Pipeline Implementations

### [`BasicRAGPipeline`](../rag_templates/pipelines/basic.py:21)

Standard RAG implementation with vector similarity search.

```python
from iris_rag.pipelines.basic import BasicRAGPipeline

pipeline = BasicRAGPipeline(
    connection_manager=conn_manager,
    config_manager=config_manager,
    llm_func=my_llm_function
)

# Load documents
pipeline.load_documents("./documents/")

# Execute query
result = pipeline.execute("What is machine learning?")
print(result["answer"])
```

#### Methods

##### [`__init__()`](../rag_templates/pipelines/basic.py:31)

**Parameters:**
- `connection_manager` (ConnectionManager): Database connection manager
- `config_manager` (ConfigurationManager): Configuration manager
- `llm_func` (Optional[Callable]): LLM function for answer generation

---

## Storage Layer

### [`IRISStorage`](../rag_templates/storage/iris.py:18)

InterSystems IRIS-specific storage implementation with vector search capabilities.

```python
from iris_rag.storage.iris import IRISStorage

storage = IRISStorage(connection_manager, config_manager)
storage.initialize_schema()

# Store documents
documents = [Document("content1"), Document("content2")]
storage.store_documents(documents)

# Vector search
results = storage.vector_search("query text", top_k=5)
```

#### Methods

##### [`__init__()`](../rag_templates/storage/iris.py:26)

**Parameters:**
- `connection_manager` (ConnectionManager): Database connection manager
- `config_manager` (ConfigurationManager): Configuration manager

##### [`initialize_schema()`](../rag_templates/storage/iris.py:49)

Creates necessary database tables and indexes for document storage.

---

## Embedding Management

### [`EmbeddingManager`](../rag_templates/embeddings/manager.py:15)

Manages embedding generation with multiple backends and fallback support.

```python
from iris_rag.embeddings.manager import EmbeddingManager

embedding_manager = EmbeddingManager(config_manager)

# Generate embeddings
embeddings = embedding_manager.embed_texts(["text1", "text2"])
```

#### Methods

##### [`__init__()`](../rag_templates/embeddings/manager.py:23)

**Parameters:**
- `config_manager` (ConfigurationManager): Configuration manager

##### [`embed_texts()`](../rag_templates/embeddings/manager.py:60)

Generates embeddings for text inputs with automatic fallback.

**Parameters:**
- `texts` (List[str]): List of texts to embed
- `batch_size` (int): Processing batch size

**Returns:** List of embedding vectors

---

## Adapters

### [`PersonalAssistantAdapter`](../rag_templates/adapters/personal_assistant.py:18)

Adapter for integrating with Personal Assistant systems.

```python
from iris_rag.adapters.personal_assistant import PersonalAssistantAdapter

adapter = PersonalAssistantAdapter(config=pa_config)
pipeline = adapter.initialize_iris_rag_pipeline()
```

#### Methods

##### [`__init__()`](../rag_templates/adapters/personal_assistant.py:27)

**Parameters:**
- `config` (Optional[Dict]): Personal Assistant configuration

##### [`initialize_iris_rag_pipeline()`](../rag_templates/adapters/personal_assistant.py:70)

Initializes RAG pipeline compatible with Personal Assistant interface.

**Parameters:**
- `config_path` (Optional[str]): Configuration file path
- `pa_specific_config` (Optional[Dict]): PA-specific configuration
- `**kwargs`: Additional pipeline arguments

**Returns:** Initialized RAG pipeline instance

---

## Error Handling

### [`ConfigValidationError`](../rag_templates/config/manager.py:6)

Custom exception for configuration validation errors.

```python
from iris_rag.config.manager import ConfigValidationError

try:
    config = ConfigurationManager("invalid_config.yaml")
except ConfigValidationError as e:
    print(f"Configuration error: {e}")
```

---

## Configuration Examples

### Basic Configuration (YAML)

```yaml
database:
  iris:
    host: localhost
    port: 1972
    namespace: USER
    username: ${IRIS_USERNAME}
    password: ${IRIS_PASSWORD}

embeddings:
  primary_backend: sentence_transformers
  model: sentence-transformers/all-MiniLM-L6-v2
  fallback_backends:
    - openai

storage:
  iris:
    table_name: rag_documents
    vector_dimension: 384

pipelines:
  basic:
    top_k: 5
    chunk_size: 1000
    chunk_overlap: 200
```

### Environment Variables

```bash
# Database configuration
export RAG_DATABASE__IRIS__HOST=localhost
export RAG_DATABASE__IRIS__PORT=1972
export RAG_DATABASE__IRIS__USERNAME=SuperUser
export RAG_DATABASE__IRIS__PASSWORD=SYS

# Embedding configuration  
export RAG_EMBEDDINGS__MODEL=sentence-transformers/all-MiniLM-L6-v2
export RAG_EMBEDDINGS__API_KEY=your-api-key
```

---

## Type Hints and Imports

```python
from typing import Dict, List, Any, Optional, Callable
from iris_rag import create_pipeline
from iris_rag.core.base import RAGPipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.core.models import Document
from iris_rag.config.manager import ConfigurationManager, ConfigValidationError
from iris_rag.storage.iris import IRISStorage
from iris_rag.embeddings.manager import EmbeddingManager
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.adapters.personal_assistant import PersonalAssistantAdapter
```

---

## Best Practices

1. **Configuration Management**: Use YAML files for static configuration and environment variables for secrets
2. **Error Handling**: Wrap pipeline operations in try-catch blocks for robust error handling
3. **Resource Management**: Use connection managers to handle database connections efficiently
4. **Type Safety**: Leverage type hints for better IDE support and code documentation
5. **Testing**: Use the provided test fixtures for consistent testing across environments

---

## Migration Support

The package includes migration utilities for upgrading from legacy Personal Assistant configurations:

```python
from iris_rag.utils.migration import migrate_pa_config

# Migrate legacy configuration
new_config = migrate_pa_config(legacy_config, mapping_rules)
```

For detailed migration guides, see the [Migration Documentation](MIGRATION.md).