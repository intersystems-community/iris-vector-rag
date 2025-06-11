# RAG Templates for InterSystems IRIS

A comprehensive, production-ready framework for implementing Retrieval Augmented Generation (RAG) pipelines using InterSystems IRIS as the vector database backend.

## 🚀 Quick Start

```bash
# Install the package
pip install intersystems-iris-rag

# Create a basic RAG pipeline
from iris_rag import create_pipeline

pipeline = create_pipeline("basic", config_path="config.yaml")
pipeline.load_documents("./documents")
result = pipeline.execute("What is machine learning?")
print(result["answer"])
```

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [📖 User Guide](docs/USER_GUIDE.md) | Installation, configuration, and usage |
| [🔧 API Reference](docs/API_REFERENCE.md) | Complete API documentation |
| [🏗️ Developer Guide](docs/DEVELOPER_GUIDE.md) | Architecture and development |
| [⚙️ Dynamic Pipeline System](docs/DYNAMIC_PIPELINE_SYSTEM.md) | Config-driven pipeline loading and custom pipeline development |
| [🧪 Testing Guide](docs/TESTING.md) | Testing strategies and execution |
| [🎯 TDD+RAGAS Integration](docs/TDD_RAGAS_INTEGRATION.md) | Performance testing with RAGAS quality metrics |
| [💾 LLM Caching Guide](docs/LLM_CACHING_GUIDE.md) | Intelligent LLM response caching with IRIS backend |
| [🔄 Migration Guide](docs/MIGRATION_GUIDE.md) | Migrate from existing implementations |
| [🛡️ Security Guide](docs/SECURITY_GUIDE.md) | Production security best practices |
| [⚡ Performance Guide](docs/PERFORMANCE_GUIDE.md) | Optimization recommendations |
| [🔍 Troubleshooting](docs/TROUBLESHOOTING.md) | Common issues and solutions |

## ✨ Features

### 🎯 Core Capabilities
- **Multiple RAG Techniques**: Basic RAG, ColBERT, CRAG, GraphRAG, HyDE, NodeRAG, and Hybrid iFindRAG
- **Dynamic Pipeline Loading**: Config-driven pipeline system with [`config/pipelines.yaml`](config/pipelines.yaml)
- **Production Ready**: Enterprise-grade architecture with comprehensive error handling
- **IRIS Integration**: Native InterSystems IRIS vector search with HNSW indexing
- **Flexible Configuration**: YAML/JSON config files with environment variable overrides
- **Personal Assistant Adapter**: Drop-in replacement for existing PA integrations

### 🔧 Technical Excellence
- **Type Safety**: Full type hints and validation throughout
- **Modular Design**: Clean separation of concerns with dependency injection
- **Test Coverage**: Comprehensive test suite with real data validation
- **Performance**: Optimized for enterprise-scale deployments (50K+ documents)
- **Security**: Secure parameter binding and SQL injection protection

### 📊 Advanced Features
- **Multiple Embedding Backends**: Sentence Transformers, OpenAI, Hugging Face
- **Intelligent Chunking**: Recursive, semantic, adaptive, and hybrid strategies
- **🚀 LLM Response Caching**: Intelligent caching layer with IRIS backend for reduced API costs and improved performance
- **Fallback Support**: Graceful degradation when services are unavailable
- **Monitoring**: Built-in performance metrics and health checks
- **Benchmarking**: RAGAS integration for quality assessment
- **🛡️ Data Integrity & Self-Healing**: Automated system to ensure 100% data readiness for all RAG tables using `make heal-data` and related targets. See [`docs/SELF_HEALING_SYSTEM.md`](docs/SELF_HEALING_SYSTEM.md) for details.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RAG Pipeline  │    │ Embedding Mgr   │    │ Storage Layer   │
│                 │────│                 │────│                 │
│ • BasicRAG      │    │ • SentenceT     │    │ • IRIS Storage  │
│ • ColBERT       │    │ • OpenAI        │    │ • Vector Search │
│ • CRAG          │    │ • HuggingFace   │    │ • HNSW Index    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Config Manager  │
                    │                 │
                    │ • YAML/JSON     │
                    │ • Environment   │
                    │ • Validation    │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │ LLM Cache Layer │
                    │                 │
                    │ • IRIS Backend  │
                    │ • TTL Support   │
                    │ • Metrics       │
                    │ • Auto-cleanup  │
                    └─────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- InterSystems IRIS 2025.1+
- 2GB+ available memory

### Option 1: Python Package Installation
```bash
pip install intersystems-iris-rag
```

### Option 2: IPM (ZPM) Installation
```objectscript
// Install via InterSystems Package Manager
zpm "install intersystems-iris-rag"

// Verify installation
do ##class(RAG.IPMInstaller).DisplayInfo()

// Run system test
do ##class(RAG.PythonBridge).RunSystemTest()
```

### Development Installation
```bash
git clone https://github.com/your-org/intersystems-iris-rag.git
cd intersystems-iris-rag
pip install -e ".[dev]"
```

### IPM Installation Guide
For detailed IPM installation instructions, see [IPM Installation Guide](docs/IPM_INSTALLATION.md).

## ⚡ Quick Examples

### Basic RAG Pipeline
```python
from iris_rag import create_pipeline

# Create pipeline with configuration
pipeline = create_pipeline(
    pipeline_type="basic",
    config_path="config.yaml"
)

# Load documents
pipeline.load_documents("./documents")

# Query the pipeline
result = pipeline.execute("What is the main topic?")
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['retrieved_documents'])} documents")
```

### Personal Assistant Integration
```python
from iris_rag.adapters import PersonalAssistantAdapter

# Drop-in replacement for existing PA integration
adapter = PersonalAssistantAdapter()
pipeline = adapter.initialize_iris_rag_pipeline(
    config_path="pa_config.yaml"
)

# Use existing PA query interface
response = adapter.query("How does photosynthesis work?")
```

### Advanced Configuration
```python
from iris_rag.core import ConnectionManager, ConfigurationManager
from iris_rag.pipelines import BasicRAGPipeline

# Manual configuration for advanced use cases
config_manager = ConfigurationManager("config.yaml")
connection_manager = ConnectionManager(config_manager)

pipeline = BasicRAGPipeline(
    connection_manager=connection_manager,
    config_manager=config_manager,
    llm_func=my_custom_llm_function
)
```

### LLM Caching Example
```python
from common.llm_cache_manager import setup_langchain_cache, get_cache_stats
from common.utils import get_llm_func

# Enable caching for LLM calls
setup_langchain_cache()

# Create LLM function with caching enabled
llm_func = get_llm_func(
    provider="openai",
    model_name="gpt-3.5-turbo",
    enable_cache=True
)

# First call - cache miss, calls API
response1 = llm_func("What is machine learning?")

# Second call - cache hit, returns instantly
response2 = llm_func("What is machine learning?")

# Check cache performance
stats = get_cache_stats()
print(f"Cache hit rate: {stats['metrics']['hit_rate']:.2%}")
```

## 🔧 Dynamic Pipeline Configuration

The framework uses a **config-driven dynamic pipeline loading system** that allows you to define, enable/disable, and configure RAG pipelines through [`config/pipelines.yaml`](config/pipelines.yaml). This system replaces hard-coded pipeline instantiation with a flexible, extensible configuration-based approach.

### Pipeline Configuration Structure

Pipelines are defined in [`config/pipelines.yaml`](config/pipelines.yaml) with the following structure:

```yaml
pipelines:
  - name: "BasicRAG"
    module: "iris_rag.pipelines.basic"
    class: "BasicRAGPipeline"
    enabled: true
    params:
      top_k: 5
      chunk_size: 1000
      similarity_threshold: 0.7

  - name: "ColBERTRAG"
    module: "iris_rag.pipelines.colbert"
    class: "ColBERTRAGPipeline"
    enabled: true
    params:
      top_k: 10
      max_query_length: 512
      doc_maxlen: 180
```

Each pipeline definition includes:
- **`name`**: Unique identifier for the pipeline
- **`module`**: Python module path containing the pipeline class
- **`class`**: Name of the pipeline class to instantiate
- **`enabled`**: Whether the pipeline should be loaded (true/false)
- **`params`**: Pipeline-specific configuration parameters

### Adding Custom Pipelines

To add a new custom pipeline:

1. **Create your pipeline class** following the standard constructor signature:
   ```python
   class MyCustomRAGPipeline:
       def __init__(self, llm_func, embedding_func, vector_store, config_manager, **kwargs):
           # Framework dependencies (automatically injected)
           self.llm_func = llm_func
           self.embedding_func = embedding_func
           self.vector_store = vector_store
           self.config_manager = config_manager
           
           # Pipeline-specific parameters from config
           self.custom_param = kwargs.get('custom_param', 'default_value')
           self.another_param = kwargs.get('another_param', 42)
   ```

2. **Add the pipeline definition** to [`config/pipelines.yaml`](config/pipelines.yaml):
   ```yaml
   - name: "MyCustomRAG"
     module: "my_package.pipelines"
     class: "MyCustomRAGPipeline"
     enabled: true
     params:
       custom_param: "my_value"
       another_param: 42
   ```

3. **Framework dependencies are automatically injected** by the [`PipelineFactory`](iris_rag/pipelines/factory.py):
   - **`llm_func`**: LLM function for answer generation
   - **`embedding_func`**: Embedding function for vector operations
   - **`vector_store`**: IRIS vector store instance
   - **`config_manager`**: Configuration manager instance

### Dynamic Loading Architecture

The system consists of four core services:

- **[`PipelineConfigService`](iris_rag/config/pipeline_config_service.py)**: Loads and validates pipeline configurations from YAML
- **[`ModuleLoader`](iris_rag/utils/module_loader.py)**: Dynamically imports pipeline classes from modules
- **[`PipelineFactory`](iris_rag/pipelines/factory.py)**: Creates pipeline instances with dependency injection
- **[`PipelineRegistry`](iris_rag/pipelines/registry.py)**: Manages pipeline inventory and lifecycle

### Using Dynamic Pipelines

```python
from iris_rag.config.pipeline_config_service import PipelineConfigService
from iris_rag.utils.module_loader import ModuleLoader
from iris_rag.pipelines.factory import PipelineFactory
from iris_rag.pipelines.registry import PipelineRegistry

# Setup framework dependencies
framework_dependencies = {
    "llm_func": my_llm_function,
    "embedding_func": my_embedding_function,
    "vector_store": my_vector_store,
    "config_manager": my_config_manager
}

# Initialize dynamic loading services
config_service = PipelineConfigService()
module_loader = ModuleLoader()
pipeline_factory = PipelineFactory(config_service, module_loader, framework_dependencies)
pipeline_registry = PipelineRegistry(pipeline_factory)

# Register all enabled pipelines from config/pipelines.yaml
pipeline_registry.register_pipelines()

# Get a specific pipeline
pipeline = pipeline_registry.get_pipeline("BasicRAG")

# Get all registered pipelines
all_pipelines = {name: pipeline_registry.get_pipeline(name)
                for name in pipeline_registry.list_pipeline_names()}
```

### Evaluation with Dynamic Pipelines

The evaluation script [`eval/execute_comprehensive_ragas_evaluation.py`](eval/execute_comprehensive_ragas_evaluation.py) supports dynamic pipeline selection:

```bash
# Evaluate all enabled pipelines from config/pipelines.yaml
python eval/execute_comprehensive_ragas_evaluation.py --pipelines ALL

# Evaluate specific pipelines by name (defined in config/pipelines.yaml)
python eval/execute_comprehensive_ragas_evaluation.py --pipelines BasicRAG ColBERTRAG

# Evaluate with custom number of queries
python eval/execute_comprehensive_ragas_evaluation.py --pipelines BasicRAG --num-queries 10
```

The `--pipelines` argument accepts:
- **`ALL`**: Evaluate all enabled pipelines from [`config/pipelines.yaml`](config/pipelines.yaml)
- **Pipeline names**: Space-separated list of pipeline names defined in the configuration file

**Note**: The script no longer uses hard-coded pipeline configurations. All pipelines must be defined in [`config/pipelines.yaml`](config/pipelines.yaml) to be available for evaluation.

## � Configuration

### Basic Configuration (config.yaml)
```yaml
database:
  iris:
    host: localhost
    port: 1972
    namespace: USER
    username: demo
    password: demo
    driver: intersystems.jdbc

embeddings:
  primary_backend: sentence_transformers
  sentence_transformers:
    model_name: all-MiniLM-L6-v2
  dimension: 384

pipelines:
  basic:
    chunk_size: 1000
    chunk_overlap: 200
    default_top_k: 5
```

### Environment Variables
```bash
# Database connection
export RAG_DATABASE__IRIS__HOST=localhost
export RAG_DATABASE__IRIS__PORT=1972
export RAG_DATABASE__IRIS__USERNAME=demo
export RAG_DATABASE__IRIS__PASSWORD=demo

# Embedding configuration
export RAG_EMBEDDINGS__PRIMARY_BACKEND=openai
export RAG_OPENAI__API_KEY=your-api-key

# LLM Caching configuration
export LLM_CACHE_ENABLED=true
export LLM_CACHE_BACKEND=iris
export LLM_CACHE_TTL=3600
```

## 🚀 LLM Response Caching

The framework includes an intelligent LLM response caching layer that significantly reduces API costs and improves response times by caching LLM responses in the IRIS database.

### Key Features

- **IRIS Backend**: Leverages existing IRIS infrastructure for persistent cache storage
- **Intelligent Key Generation**: SHA256-based cache keys with configurable parameters
- **TTL Support**: Automatic expiration of cached responses
- **Performance Metrics**: Built-in hit/miss tracking and performance monitoring
- **Graceful Fallback**: Continues operation even if cache is unavailable
- **Langchain Integration**: Seamless integration with Langchain's caching system

### Quick Setup

```python
from common.llm_cache_manager import setup_langchain_cache
from common.llm_cache_config import load_cache_config

# Setup caching with default configuration
config = load_cache_config()
cache = setup_langchain_cache(config)

# Caching is now automatically applied to all LLM calls
```

### Configuration

The caching layer is configured via [`config/cache_config.yaml`](config/cache_config.yaml):

```yaml
llm_cache:
  enabled: true
  backend: "iris"
  ttl_seconds: 3600
  
  iris:
    table_name: "llm_cache"
    schema: "USER"
    auto_cleanup: true
    cleanup_interval: 86400
  
  key_generation:
    include_temperature: true
    include_max_tokens: true
    include_model_name: true
    hash_algorithm: "sha256"
```

### Environment Variables

Override configuration with environment variables:

```bash
export LLM_CACHE_ENABLED=true
export LLM_CACHE_BACKEND=iris
export LLM_CACHE_TTL=3600
export LLM_CACHE_TABLE=llm_cache
export LLM_CACHE_IRIS_SCHEMA=USER
```

### Cache Management

```python
from common.llm_cache_manager import get_global_cache_manager

# Get cache statistics
manager = get_global_cache_manager()
stats = manager.get_cache_stats()
print(f"Hit rate: {stats['metrics']['hit_rate']:.2%}")

# Clear cache
manager.clear_cache()
```

### Performance Benefits

- **Cost Reduction**: Up to 90% reduction in LLM API costs for repeated queries
- **Response Time**: 10-100x faster responses for cached queries
- **Scalability**: Persistent cache survives application restarts
- **Analytics**: Track usage patterns and optimize cache configuration

For detailed configuration and usage examples, see the [LLM Caching Guide](docs/LLM_CACHING_GUIDE.md).

## 🧪 Testing

### Run Basic Tests
```bash
pytest tests/
```

### Run with Real Data (1000+ documents)
```bash
pytest tests/test_all_with_1000_docs.py
```

### TDD+RAGAS Performance Testing
```bash
# Run TDD performance benchmarks with RAGAS quality metrics
make test-performance-ragas-tdd

# Run scalability tests with RAGAS
make test-scalability-ragas-tdd

# Run comprehensive TDD+RAGAS integration tests
make test-tdd-comprehensive-ragas

# Run with 1000+ documents
make test-1000-enhanced

# Quick TDD+RAGAS test
make test-tdd-ragas-quick

# Generate comprehensive report
make ragas-with-tdd
```

### Performance Benchmarks
```bash
python -m iris_rag.benchmarks --techniques basic colbert --queries 50
```

### Key Makefile Targets
```bash
# Setup environment and database
make setup-env
make install
make setup-db

# Load data and ensure 100% readiness
make load-data       # Load sample data
make load-1000       # Load 1000+ documents
make heal-data       # Run self-healing for all tables

# Run tests
make test            # Run unit and integration tests
make test-1000       # Run comprehensive E2E test with 1000 docs

# TDD+RAGAS Testing
make test-performance-ragas-tdd    # Performance benchmarks with RAGAS
make test-scalability-ragas-tdd    # Scalability tests with RAGAS
make test-tdd-comprehensive-ragas  # All TDD+RAGAS tests
make test-1000-enhanced           # TDD+RAGAS with 1000 docs
make ragas-with-tdd              # Comprehensive test + report

# Validate and auto-setup pipelines
make validate-all-pipelines
make auto-setup-all

# See full list
make help
```

## 🤝 Contributing

We welcome contributions! Please see our [Developer Guide](docs/DEVELOPER_GUIDE.md) for:
- Development setup
- Code standards
- Testing requirements
- Submission process

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/intersystems-iris-rag/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/intersystems-iris-rag/discussions)

## 🗺️ Roadmap

- **Q1 2025**: Advanced RAG techniques (RAG-Fusion, Self-RAG)
- **Q2 2025**: Multi-modal support (images, audio)
- **Q3 2025**: Distributed processing and scaling
- **Q4 2025**: Enterprise features (audit, compliance)

---

**Built with ❤️ for the InterSystems IRIS community**
