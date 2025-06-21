# RAG Templates Library Consumption Framework

## Phase 1: Foundation Layer (Simple API and Core Services)

This implementation provides a zero-configuration Simple API that enables immediate RAG usage with sensible defaults, following clean architecture principles.

## 🚀 Quick Start

```python
from rag_templates import RAG

# Zero-configuration initialization
rag = RAG()

# Add documents
rag.add_documents([
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing enables computers to understand human language."
])

# Query for answers
answer = rag.query("What is machine learning?")
print(answer)
```

## 📋 Features Implemented

### ✅ Simple API RAG Class
- **Zero-config initialization**: `RAG()` works without any configuration
- **Simple document addition**: `add_documents()` accepts lists of strings or documents
- **String query responses**: `query()` returns simple string answers
- **Lazy initialization**: Expensive operations deferred until needed
- **No hard-coded secrets**: All sensitive values configurable via environment variables

### ✅ Enhanced Configuration Manager
- **Three-tier system**: Built-in defaults → Configuration files → Environment variables
- **Built-in defaults**: Sensible defaults for immediate operation
- **Environment variable support**: `RAG_DATABASE__IRIS__HOST` format
- **Configuration validation**: Type checking and required field validation
- **Fallback strategies**: Graceful handling of missing configuration

### ✅ Error Handling System
- **Error hierarchy**: `RAGFrameworkError`, `ConfigurationError`, `InitializationError`, etc.
- **Helpful error messages**: Clear descriptions with actionable suggestions
- **Fallback strategies**: Graceful degradation for common failures
- **Logging integration**: Structured logging with appropriate levels

## 🏗️ Architecture

```
rag_templates/
├── __init__.py              # Main package exports
├── simple.py                # Zero-config Simple API
├── core/
│   ├── __init__.py         # Core module exports
│   ├── config_manager.py   # Enhanced configuration management
│   └── errors.py           # Error handling system
└── README.md               # This file
```

## 🧪 Testing

All functionality is thoroughly tested using TDD methodology:

```bash
# Run all Phase 1 tests
pytest tests/test_simple_api_phase1.py -v

# Test specific functionality
pytest tests/test_simple_api_phase1.py::TestSimpleAPIPhase1::test_zero_config_initialization -v
```

### Test Coverage
- ✅ Zero-config initialization
- ✅ Simple query returns string
- ✅ Document addition from lists
- ✅ Default configuration loading
- ✅ Environment variable overrides
- ✅ No hardcoded secrets validation
- ✅ Error handling initialization
- ✅ Configuration error handling
- ✅ Lazy initialization pattern
- ✅ Three-tier configuration system
- ✅ Configuration validation
- ✅ Fallback strategies

## ⚙️ Configuration

### Built-in Defaults
The system provides sensible defaults for immediate operation:

```python
{
    "database": {
        "iris": {
            "host": "localhost",
            "port": 1972,
            "namespace": "USER",
            "username": None,  # Must be provided
            "password": None,  # Must be provided
            "timeout": 30
        }
    },
    "embeddings": {
        "model": "all-MiniLM-L6-v2",
        "dimension": 384,
        "provider": "sentence-transformers"
    },
    "pipelines": {
        "basic": {
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "default_top_k": 5
        }
    }
}
```

### Environment Variables
Override any configuration using environment variables:

```bash
export RAG_DATABASE__IRIS__HOST=my-iris-server
export RAG_DATABASE__IRIS__PORT=1972
export RAG_DATABASE__IRIS__USERNAME=myuser
export RAG_DATABASE__IRIS__PASSWORD=mypassword
export RAG_EMBEDDINGS__MODEL=all-mpnet-base-v2
```

### Configuration Files
Place a `config.yaml` file in your project root:

```yaml
database:
  iris:
    host: my-iris-server
    port: 1972
    username: myuser
    password: mypassword

embeddings:
  model: all-mpnet-base-v2
  dimension: 768
```

## 🔧 API Reference

### RAG Class

#### `__init__(config_path=None, **kwargs)`
Initialize the Simple RAG API with optional configuration.

#### `add_documents(documents, **kwargs)`
Add documents to the knowledge base.
- `documents`: List of strings or document dictionaries
- `**kwargs`: Additional processing options

#### `query(query_text, **kwargs) -> str`
Query the system and return a string answer.
- `query_text`: The question or query
- `**kwargs`: Query options (top_k, etc.)
- Returns: String answer

#### `get_config(key, default=None)`
Get a configuration value using dot notation.

#### `set_config(key, value)`
Set a configuration value.

#### `validate_config() -> bool`
Validate the current configuration.

### ConfigurationManager Class

#### `__init__(config_path=None, schema=None)`
Initialize with optional config file and validation schema.

#### `get(key_string, default=None)`
Retrieve configuration using colon notation (e.g., "database:iris:host").

#### `set(key_string, value)`
Set configuration value.

#### `validate()`
Validate configuration against schema and requirements.

## 🚨 Error Handling

The framework provides comprehensive error handling:

```python
from rag_templates.core.errors import RAGFrameworkError, ConfigurationError

try:
    rag = RAG()
    answer = rag.query("What is AI?")
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
    print(f"Suggestion: {e.suggestion}")
except RAGFrameworkError as e:
    print(f"Framework error: {e}")
```

## 🔄 Integration with Existing Infrastructure

The Simple API integrates seamlessly with the existing `iris_rag` infrastructure:

- Leverages existing `BasicRAGPipeline` for core functionality
- Uses existing `ConnectionManager` for database connections
- Compatible with current embedding and LLM functions
- Maintains existing configuration patterns

## 📈 Performance Considerations

- **Lazy initialization**: Database connections and model loading deferred until needed
- **Configuration caching**: Configuration loaded once and cached
- **Efficient defaults**: Sensible chunk sizes and batch sizes for optimal performance
- **Memory management**: Proper resource cleanup and connection pooling

## 🛡️ Security

- **No hardcoded secrets**: All sensitive values must be provided via environment variables or config files
- **Input validation**: All user inputs validated and sanitized
- **Error message safety**: Error messages don't expose sensitive information
- **Configuration validation**: Type checking and bounds validation for all config values

## 🔮 Future Phases

This Phase 1 implementation provides the foundation for future phases:

- **Phase 2**: Advanced API with streaming, async support, and advanced configuration
- **Phase 3**: Plugin system and extensibility framework
- **Phase 4**: Production deployment and monitoring tools

## 📝 Example Usage

See [`examples/simple_api_demo.py`](../examples/simple_api_demo.py) for a complete demonstration of the Simple API functionality.

## 🤝 Contributing

This implementation follows TDD principles:

1. **RED**: Write failing tests first
2. **GREEN**: Implement minimum code to pass tests
3. **REFACTOR**: Clean up code while keeping tests passing

All code must:
- Follow clean architecture principles
- Have no hardcoded secrets or environment values
- Be split into files under 500 lines
- Use configuration abstractions
- Include comprehensive test coverage

## 📄 License

This project is part of the RAG Templates framework and follows the same licensing terms.