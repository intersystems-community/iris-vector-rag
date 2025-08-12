# User Guide

## System Requirements

### Minimum Requirements
- **Operating System**: macOS, Linux, or Windows with WSL2
- **Python**: Version 3.11 or higher
- **Memory**: 2GB available RAM
- **Storage**: 5GB free disk space
- **Docker**: For running InterSystems IRIS database

### Recommended Requirements
- **Memory**: 4GB+ available RAM for better performance
- **Storage**: 10GB+ for larger document collections
- **Internet**: For downloading models and API access

### Required Software
- **Docker and Docker Compose**: For database management
- **Git**: For cloning the repository

## Installation

### Step 1: Install Prerequisites

#### Install Docker
- **macOS**: Download Docker Desktop from [docker.com](https://www.docker.com/products/docker-desktop)
- **Linux**: Follow the [official Docker installation guide](https://docs.docker.com/engine/install/)
- **Windows**: Install Docker Desktop with WSL2 backend

### Step 2: Get the Code

```bash
# Clone the repository
git clone <repository-url>
cd rag-templates

# Make CLI tools executable (if applicable, e.g. ragctl)
# chmod +x ragctl # Assuming ragctl is a script
```

### Step 3: Set Up Python Virtual Environment

#### Option 1: Using Make (Recommended)
```bash
# Create Python virtual environment (.venv) and install dependencies
make setup-env
make install

# Activate the environment for your current session
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Option 2: Manual Setup
```bash
# Create Python virtual environment
python3 -m venv .venv

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the project in editable mode (optional, for development)
pip install -e .
```

### Step 4: Start the Database

```bash
# Start InterSystems IRIS database
docker-compose up -d

# Verify it's running
docker ps | grep iris_db_rag_standalone
```

### Step 5: Initialize the System

```bash
# Set up database schema
make setup-db

# Load sample documents for testing
make load-data

# Verify everything works
make validate-iris-rag
```

## Configuration

### Basic Configuration

The system uses a main configuration file at [`config/config.yaml`](../config/config.yaml). For most users, the default settings work well.

#### Database Settings
```yaml
database:
  db_host: "localhost"
  db_port: 1972
  db_user: "SuperUser"
  db_password: "SYS"
  db_namespace: "USER"
```

#### Embedding Model Settings
```yaml
embedding_model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
```

### Environment Variables

For sensitive information like API keys, use environment variables:

```bash
# LLM API Keys (optional, for advanced features)
export OPENAI_API_KEY=your-openai-api-key

# Database connection (only if different from defaults)
export IRIS_HOST=localhost
export IRIS_PORT=1972
export IRIS_USERNAME=SuperUser
export IRIS_PASSWORD=SYS
```

### Automatic Schema Management

The system automatically manages database schema changes when you update configurations. You don't need to manually handle database migrations - the system detects changes and updates the schema automatically.

## Basic Usage

### Using Python

```python
from iris_rag.pipelines.factory import create_pipeline
from common.utils import get_llm_func
from common.iris_connection_manager import get_iris_connection

# Create a basic RAG pipeline
pipeline = create_pipeline(
    pipeline_type="basic",
    llm_func=get_llm_func(),
    external_connection=get_iris_connection()
)

# Ask a question
result = pipeline.query("What is machine learning?", top_k=5)
print(f"Answer: {result['answer']}")
print(f"Found {len(result['retrieved_documents'])} relevant documents")
```

## Document Management

### Loading Your Documents

#### From a Directory
```bash
# Load documents from a folder
make load-data

# Load from a specific directory
python -c "
from data.loader_fixed import process_and_load_documents
result = process_and_load_documents('path/to/your/documents', limit=100)
print(f'Loaded: {result}')
"
```

#### Supported File Types
- **Text files**: `.txt`, `.md`
- **PDF documents**: `.pdf` (requires additional setup)
- **Word documents**: `.docx` (requires additional setup)
- **Structured data**: `.json`, `.csv`

#### Check What's Loaded
```bash
# See how many documents are in the system
make check-data
```

#### Clear All Data
```bash
# Remove all documents (use with caution!)
make clear-rag-data
```

## Querying Your Data

### Simple Queries

```python
from iris_rag import create_pipeline

# Create a pipeline
pipeline = create_pipeline("basic")

# Ask questions
result = pipeline.query("What is photosynthesis?")
print(result["answer"])
```

### Advanced Queries

```python
# Get more detailed results
result = pipeline.query(
    "Explain machine learning algorithms",
    top_k=10,  # Get more source documents
    include_sources=True  # Include source information
)

# See what sources were used
for doc in result['retrieved_documents']:
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Content preview: {doc.page_content[:100]}...")
```

### Using Make Commands

```bash
# Test a specific pipeline
make test-pipeline PIPELINE=basic

# Run comprehensive tests
make test-1000
```

## Available RAG Techniques

The system supports multiple RAG (Retrieval Augmented Generation) techniques:

### Basic RAG
- **Best for**: General question answering
- **Features**: Standard vector similarity search

### Basic RAG
- **Best for**: General question answering with more precision
- **Features**: Overretrieval with reranking done using a cross encoder

### CRAG (Corrective RAG)
- **Best for**: Self-correcting answers
- **Features**: Automatic quality assessment and correction

## Common Use Cases

### 1. Document Q&A System

```python
# Load your company documents
pipeline = create_pipeline("basic")

# Ask questions about your documents
answer = pipeline.query("What is our return policy?")
print(answer["answer"])
```

### 2. Research Assistant

```python
# Use CRAG for self-correcting research
pipeline = create_pipeline("crag")

# Ask complex research questions
result = pipeline.query("What are the latest developments in AI?")
print(result["answer"])
```

## Troubleshooting

### Common Issues

#### "Connection failed" Error
```bash
# Check if IRIS database is running
docker ps | grep iris_db_rag_standalone

# If not running, start it
docker-compose up -d

# Test connection
make test-dbapi
```

#### "No documents found" Error
```bash
# Check if documents are loaded
make check-data

# If no documents, load sample data
make load-data
```