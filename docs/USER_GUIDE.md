# User Guide

Complete guide for installing, configuring, and using RAG Templates with InterSystems IRIS.

## Table of Contents

- [Quick Start](#quick-start)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Basic Usage](#basic-usage)
- [Using the CLI Tool (ragctl)](#using-the-cli-tool-ragctl)
- [Document Management](#document-management)
- [Querying Your Data](#querying-your-data)
- [Available RAG Techniques](#available-rag-techniques)
- [Common Use Cases](#common-use-cases)
- [Troubleshooting](#troubleshooting)
- [Getting Help](#getting-help)

## Quick Start

**🚀 NEW: One-Command Setup!** Get a complete RAG system running in minutes:

### Option 1: Quick Start Profiles (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd rag-templates

# Choose your profile and run ONE command:
make quick-start-minimal    # Development setup (50 docs, 2GB RAM, ~5 min)
make quick-start-standard   # Production setup (500 docs, 4GB RAM, ~15 min)
make quick-start-extended   # Enterprise setup (5000 docs, 8GB RAM, ~30 min)

# Or use interactive setup:
make quick-start           # Interactive wizard with profile selection
```

**That's it!** The Quick Start system automatically:
- ✅ Sets up Python environment and dependencies
- ✅ Configures and starts database services
- ✅ Loads optimized sample data for your profile
- ✅ Validates system health and functionality
- ✅ Provides ready-to-use RAG pipelines

### Option 2: Manual Setup (Advanced Users)

```bash
# 1. Clone the repository
git clone <repository-url>
cd rag-templates

# 2. Set up the Python virtual environment and install dependencies
make setup-env  # This will create .venv and install core dependencies
make install    # This will install all dependencies from requirements.txt

# 3. Activate the virtual environment (if not already done by make setup-env/install)
#    source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 4. Start the database
docker-compose up -d

# 5. Initialize and load sample data
make setup-db
make load-data

# 6. Test your installation
make validate-iris-rag
```

### Quick Start Profile Comparison

| Profile | Documents | Memory | Setup Time | Use Case |
|---------|-----------|--------|------------|----------|
| **Minimal** | 50 | 2GB | ~5 min | Development, Testing, Learning |
| **Standard** | 500 | 4GB | ~15 min | Production, Demos, Evaluation |
| **Extended** | 5000 | 8GB | ~30 min | Enterprise, Scale Testing |

### Quick Start Management

```bash
# Check system status and health
make quick-start-status

# Clean up Quick Start environment
make quick-start-clean

# Custom profile setup
make quick-start-custom PROFILE=my-profile
```

For detailed Quick Start documentation, see [`QUICK_START_GUIDE.md`](QUICK_START_GUIDE.md).

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

### Quick Test

```bash
# Test that everything is working
make test-pipeline PIPELINE=basic
```

## Using the CLI Tool (ragctl)

The [`ragctl`](../ragctl) command-line tool provides easy access to common operations.

### Basic Commands

```bash
# Check system status
./ragctl status

# Validate configuration
./ragctl validate

# Run reconciliation (data integrity check)
./ragctl run --pipeline basic

# Get help
./ragctl --help
```

### Alternative CLI Access

```bash
# Using Python module
python -m iris_rag.cli --help

# Direct execution
python iris_rag/cli/__main__.py --help
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
- **How to use**: `create_pipeline("basic")`
- **Features**: Standard vector similarity search

### ColBERT
- **Best for**: High-precision retrieval
- **How to use**: `create_pipeline("colbert")`
- **Features**: Token-level embeddings with late interaction

### CRAG (Corrective RAG)
- **Best for**: Self-correcting answers
- **How to use**: `create_pipeline("crag")`
- **Features**: Automatic quality assessment and correction

### Other Techniques
- **GraphRAG**: Knowledge graph-enhanced retrieval
- **HyDE**: Hypothetical document embeddings
- **NodeRAG**: Node-based document representation
- **Hybrid iFindRAG**: Combines multiple search strategies

### Testing All Techniques

```bash
# Validate all available techniques
make validate-all-pipelines

# Test all techniques with sample data
make auto-setup-all
```

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

### 3. Technical Documentation Search

```python
# Use ColBERT for precise technical queries
pipeline = create_pipeline("colbert")

# Search technical documentation
result = pipeline.query("How do I configure the database connection?")
print(result["answer"])
```

### 4. Batch Processing

```bash
# Process multiple queries efficiently
make eval-all-ragas-1000
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

#### "Pipeline not found" Error
```bash
# Validate all pipelines
make validate-all-pipelines

# Auto-setup missing pipelines
make auto-setup-all
```

#### Memory Issues
- Reduce batch sizes in configuration
- Use smaller embedding models
- Process fewer documents at once

#### Slow Performance
```bash
# Check system status
make status

# Run performance optimization
make heal-data
```

### Getting Detailed Logs

```bash
# View database logs
docker-compose logs -f

# Check system readiness
make check-readiness

# Run comprehensive validation
make validate-all
```

### Self-Healing Features

The system includes automatic problem detection and fixing:

```bash
# Automatically fix common issues
make heal-data

# Check what was fixed
make check-readiness
```

## Getting Help

### Documentation
- **[Developer Guide](DEVELOPER_GUIDE.md)**: Technical details and architecture
- **[Configuration Guide](CONFIGURATION.md)**: Advanced configuration options
- **[Performance Guide](guides/PERFORMANCE_GUIDE.md)**: Optimization tips
- **[Security Guide](guides/SECURITY_GUIDE.md)**: Security best practices

### Command Reference
```bash
# See all available commands
make help

# Get CLI help
./ragctl --help

# Test specific components
make validate-iris-rag
make test-dbapi
make check-data
```

### Quick Diagnostics

```bash
# Complete system check
make status

# Validate everything is working
make validate-all

# Run a quick test
make test-pipeline PIPELINE=basic
```

### Environment Information

```bash
# Check your environment
make env-info

# Verify conda environment
conda info --envs

# Check Python packages
pip list | grep -E "(iris|torch|transformers)"
```

### Support Resources

1. **Check the logs**: Most issues are explained in the system logs
2. **Run diagnostics**: Use `make status` to identify problems
3. **Use self-healing**: Run `make heal-data` to fix common issues
4. **Validate setup**: Use `make validate-all` to ensure everything is configured correctly

### Performance Tips

- Start with the Basic RAG pipeline for testing
- Use `make load-data` to load sample documents before testing
- Run `make heal-data` if you encounter data consistency issues
- Use `make validate-all-pipelines` to ensure all techniques are properly configured

The system is designed to be self-healing and will automatically detect and fix many common issues. When in doubt, run the validation and healing commands to restore the system to a working state.