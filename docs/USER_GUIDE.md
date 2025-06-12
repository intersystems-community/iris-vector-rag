# User Guide

Complete guide for installing, configuring, and using RAG Templates with InterSystems IRIS.

## Table of Contents

- [Installation](#installation)
- [Configuration](#configuration)
- [Basic Usage](#basic-usage)
- [Pipeline Types](#pipeline-types)
- [Document Management](#document-management)
- [Querying](#querying)
- [Personal Assistant Integration](#personal-assistant-integration)
- [Examples](#examples)

## Installation

### Prerequisites

- **Python**: 3.11 or higher
- **InterSystems IRIS**: 2025.1 or higher
- **Memory**: 2GB+ available RAM
- **Storage**: 5GB+ for documents and embeddings

### Package Installation

#### From PyPI (Recommended)
```bash
pip install intersystems-iris-rag
```

#### From Source
```bash
git clone https://github.com/your-org/intersystems-iris-rag.git
cd intersystems-iris-rag
pip install -e .
```

#### Development Installation
```bash
git clone https://github.com/your-org/intersystems-iris-rag.git
cd intersystems-iris-rag
pip install -e ".[dev]"
```

### IRIS Database Setup

#### Option 1: Docker (Recommended)
```bash
# Start IRIS container
docker run -d \
  --name iris-rag \
  -p 1972:1972 \
  -p 52773:52773 \
  intersystemsdc/iris-community:latest

# Verify connection
docker exec iris-rag iris session iris -U USER
```

#### Option 2: Local Installation
Download and install IRIS from [InterSystems Developer Community](https://community.intersystems.com/).

### JDBC Driver Setup

Download the IRIS JDBC driver:
```bash
curl -L -o intersystems-jdbc-3.8.4.jar \
  https://github.com/intersystems-community/iris-driver-distribution/raw/main/JDBC/JDK18/intersystems-jdbc-3.8.4.jar
```

Install Java dependencies:
```bash
pip install jaydebeapi jpype1
```

## Configuration

### Configuration File

Create a `config.yaml` file:

```yaml
# Database Configuration
database:
  iris:
    host: localhost
    port: 1972
    namespace: USER
    username: demo
    password: demo
    driver: intersystems.jdbc

# Storage Configuration
storage:
  iris:
    table_name: rag_documents
    vector_dimension: 384

# Self-Healing Schema Management

The RAG Templates framework includes automatic schema management that ensures data integrity when configuration changes occur.

## Automatic Schema Validation

When you change embedding models or vector configurations, the system automatically:

1. **Detects Configuration Changes**: Compares current database schema with new configuration
2. **Validates Vector Dimensions**: Ensures vector columns match the embedding model dimensions  
3. **Performs Automatic Migration**: Updates database schema when needed
4. **Preserves System Integrity**: Maintains consistent data structures

## Configuration Changes That Trigger Schema Updates

### Embedding Model Changes

```yaml
# Changing from one model to another triggers automatic migration
embeddings:
  model: "all-mpnet-base-v2"  # 768 dimensions (was all-MiniLM-L6-v2: 384 dimensions)
```

### Vector Data Type Changes

```yaml
storage:
  iris:
    vector_data_type: "DOUBLE"  # Changed from "FLOAT"
```

## User-Visible Effects

### Initial Setup
- On first run, the system automatically creates required database tables
- Schema metadata is initialized to track configuration state
- No user intervention required

### Configuration Updates
- When you update embedding models, the system detects the change
- Automatic migration occurs on next pipeline execution
- Brief processing delay during migration (data may be regenerated)
- System logs migration progress and completion

### Data Integrity Assurance
- The system prevents data corruption from schema mismatches
- Automatic validation before any vector operations
- Clear error messages if manual intervention is needed

## Monitoring Schema Health

You can check schema status using the self-healing system:

```bash
# Check current schema status
make check-readiness

# Trigger manual schema validation
make heal-data
```

The system will report any schema issues and automatically resolve them when possible.
# Embedding Configuration
embeddings:
  primary_backend: sentence_transformers
  fallback_backends: [openai]
  dimension: 384
  
  sentence_transformers:
    model_name: all-MiniLM-L6-v2
    
  openai:
    api_key: ${OPENAI_API_KEY}
    model_name: text-embedding-ada-002

# Pipeline Configuration
pipelines:
  basic:
    chunk_size: 1000
    chunk_overlap: 200
    default_top_k: 5
    embedding_batch_size: 32
```

### Environment Variables

Set environment variables for sensitive data:

```bash
# Database credentials
export RAG_DATABASE__IRIS__HOST=localhost
export RAG_DATABASE__IRIS__PORT=1972
export RAG_DATABASE__IRIS__USERNAME=demo
export RAG_DATABASE__IRIS__PASSWORD=demo

# API keys
export RAG_OPENAI__API_KEY=your-openai-api-key

# Optional: Override config file location
export RAG_CONFIG_PATH=/path/to/config.yaml
```

### Configuration Validation

Validate your configuration:

```python
from rag_templates.config import ConfigurationManager

config = ConfigurationManager("config.yaml")
try:
    config.validate()
    print("✅ Configuration is valid")
except Exception as e:
    print(f"❌ Configuration error: {e}")
```

## Basic Usage

### Quick Start

```python
from rag_templates import create_pipeline

# Create a basic RAG pipeline
pipeline = create_pipeline(
    pipeline_type="basic",
    config_path="config.yaml"
)

# Load documents
pipeline.load_documents("./documents")

# Query the pipeline
result = pipeline.execute("What is machine learning?")
print(f"Answer: {result['answer']}")
```

### Manual Setup

For more control over the initialization:

```python
from rag_templates.core import ConnectionManager, ConfigurationManager
from rag_templates.pipelines import BasicRAGPipeline

# Initialize managers
config_manager = ConfigurationManager("config.yaml")
connection_manager = ConnectionManager(config_manager)

# Create pipeline
pipeline = BasicRAGPipeline(
    connection_manager=connection_manager,
    config_manager=config_manager
)

# Initialize storage schema
pipeline.storage.initialize_schema()
```

## Pipeline Types

### Basic RAG
Standard retrieval-augmented generation with vector similarity search.

```python
pipeline = create_pipeline("basic", config_path="config.yaml")
```

**Features:**
- Document chunking with overlap
- Vector embeddings
- Similarity search
- Context-aware answer generation

### ColBERT
Token-level retrieval with late interaction.

```python
# Note: ColBERT requires additional configuration
pipeline = create_pipeline("colbert", config_path="config.yaml")
```

**Features:**
- Token-level embeddings
- Late interaction scoring
- High precision retrieval

### CRAG (Corrective RAG)
Self-correcting RAG with retrieval evaluation.

```python
pipeline = create_pipeline("crag", config_path="config.yaml")
```

**Features:**
- Retrieval quality assessment
- Automatic correction
- Fallback strategies

### Additional Pipelines
- **GraphRAG**: Knowledge graph-enhanced retrieval
- **HyDE**: Hypothetical document embeddings
- **NodeRAG**: Node-based document representation

## Document Management

### Loading Documents

#### From Directory
```python
# Load all documents from a directory
pipeline.load_documents("./documents")

# Load with specific options
pipeline.load_documents(
    "./documents",
    chunk_documents=True,
    generate_embeddings=True
)
```

#### From File List
```python
from rag_templates.core.models import Document

# Create documents manually
documents = [
    Document(
        page_content="Machine learning is...",
        metadata={"source": "ml_intro.txt", "topic": "AI"}
    ),
    Document(
        page_content="Deep learning involves...",
        metadata={"source": "dl_guide.txt", "topic": "AI"}
    )
]

# Load documents directly
pipeline.load_documents(
    documents_path="",  # Not used when providing documents
    documents=documents
)
```

#### Supported File Formats
- **Text files**: `.txt`, `.md`
- **Documents**: `.pdf`, `.docx` (with appropriate parsers)
- **Structured data**: `.json`, `.csv`

### Document Chunking

Configure chunking strategies:

```yaml
pipelines:
  basic:
    chunk_size: 1000        # Characters per chunk
    chunk_overlap: 200      # Overlap between chunks
    chunking_strategy: recursive  # recursive, semantic, adaptive
```

### Managing Storage

```python
# Get document count
count = pipeline.get_document_count()
print(f"Documents in storage: {count}")

# Clear all documents (use with caution!)
pipeline.clear_knowledge_base()
```

## Querying

### Basic Queries

```python
# Simple query
result = pipeline.execute("What is photosynthesis?")
print(result["answer"])
```

### Advanced Queries

```python
# Query with custom parameters
result = pipeline.execute(
    "Explain machine learning algorithms",
    top_k=10,                    # Retrieve more documents
    include_sources=True,        # Include source information
    similarity_threshold=0.7,    # Minimum similarity score
    metadata_filter={            # Filter by metadata
        "topic": "AI"
    }
)

# Access detailed results
print(f"Answer: {result['answer']}")
print(f"Retrieved {len(result['retrieved_documents'])} documents")
print(f"Sources: {result['sources']}")
print(f"Processing time: {result['metadata']['processing_time']:.2f}s")
```

### Custom Prompts

```python
custom_prompt = """
Based on the following context, answer the question in a technical manner:

Context: {context}

Question: {query}

Provide a detailed technical explanation with examples.
"""

result = pipeline.execute(
    "How does neural network training work?",
    custom_prompt=custom_prompt
)
```

### Retrieval Only

```python
# Get relevant documents without generating an answer
documents = pipeline.query(
    "machine learning algorithms",
    top_k=5
)

for doc in documents:
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Content: {doc.page_content[:200]}...")
```

## Personal Assistant Integration

### Drop-in Replacement

Replace existing Personal Assistant RAG initialization:

```python
from rag_templates.adapters import PersonalAssistantAdapter

# Initialize adapter
adapter = PersonalAssistantAdapter()

# Use existing PA configuration format
pipeline = adapter.initialize_iris_rag_pipeline(
    config_path="pa_config.yaml",
    pa_specific_config={
        "iris_host": "localhost",
        "iris_port": 1972,
        "embedding_model": "all-MiniLM-L6-v2"
    }
)

# Query using PA interface
response = adapter.query("How does photosynthesis work?")
```

### Configuration Translation

The adapter automatically translates PA configuration:

```python
# PA configuration is automatically converted
pa_config = {
    "pa_db_host": "iris-server",
    "pa_api_key": "secret-key",
    "pa_model": "gpt-3.5-turbo"
}

# Becomes RAG templates configuration
rag_config = {
    "database": {"iris": {"host": "iris-server"}},
    "openai": {"api_key": "secret-key"},
    "llm": {"model": "gpt-3.5-turbo"}
}
```

## Examples

### Complete Workflow

```python
from rag_templates import create_pipeline
import logging

# Enable logging
logging.basicConfig(level=logging.INFO)

# Create pipeline
pipeline = create_pipeline("basic", config_path="config.yaml")

# Load documents
print("Loading documents...")
pipeline.load_documents("./medical_papers")

# Query the system
queries = [
    "What are the symptoms of diabetes?",
    "How is cancer diagnosed?",
    "What are the side effects of chemotherapy?"
]

for query in queries:
    print(f"\nQuery: {query}")
    result = pipeline.execute(query)
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['retrieved_documents'])} documents")
```

### Batch Processing

```python
import json
from pathlib import Path

# Process multiple queries
queries = [
    "What is machine learning?",
    "How does deep learning work?",
    "What are neural networks?"
]

results = []
for query in queries:
    result = pipeline.execute(query)
    results.append({
        "query": query,
        "answer": result["answer"],
        "num_sources": len(result["retrieved_documents"]),
        "processing_time": result["metadata"]["processing_time"]
    })

# Save results
with open("batch_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

### Performance Monitoring

```python
import time
from collections import defaultdict

# Monitor performance
stats = defaultdict(list)

for i in range(10):
    start_time = time.time()
    result = pipeline.execute("What is artificial intelligence?")
    end_time = time.time()
    
    stats["response_time"].append(end_time - start_time)
    stats["num_documents"].append(len(result["retrieved_documents"]))

# Calculate averages
avg_response_time = sum(stats["response_time"]) / len(stats["response_time"])
avg_documents = sum(stats["num_documents"]) / len(stats["num_documents"])

print(f"Average response time: {avg_response_time:.2f}s")
print(f"Average documents retrieved: {avg_documents:.1f}")
```

## Next Steps

- **[API Reference](API_REFERENCE.md)**: Detailed API documentation
- **[Developer Guide](DEVELOPER_GUIDE.md)**: Architecture and extension patterns
- **[Performance Guide](PERFORMANCE_GUIDE.md)**: Optimization recommendations
- **[Troubleshooting](TROUBLESHOOTING.md)**: Common issues and solutions