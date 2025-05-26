# ObjectScript Integration for RAG Pipelines

## Overview

This document describes the ObjectScript integration implementation for the RAG Templates project, enabling seamless integration between IRIS ObjectScript and Python RAG pipelines through Embedded Python.

## Architecture

The ObjectScript integration consists of three main components:

### 1. ObjectScript Wrapper Classes

#### RAGDemo.Invoker.cls
The main interface class that provides ObjectScript methods for invoking Python RAG pipelines.

**Key Methods:**
- `InvokerExists()` - Health check method
- `InvokeBasicRAG(query, config)` - Invoke Basic RAG pipeline
- `InvokeColBERT(query, config)` - Invoke ColBERT pipeline
- `InvokeGraphRAG(query, config)` - Invoke GraphRAG pipeline
- `InvokeHyDE(query, config)` - Invoke HyDE pipeline
- `InvokeCRAG(query, config)` - Invoke CRAG pipeline
- `InvokeNodeRAG(query, config)` - Invoke NodeRAG pipeline
- `GetAvailablePipelines()` - Get list of available pipelines
- `HealthCheck()` - Comprehensive system health check

#### RAGDemo.TestBed.cls
Testing and validation class for comprehensive ObjectScript integration testing.

**Key Methods:**
- `TestBedExists()` - Health check method
- `RunAllRAGTests()` - Execute all RAG pipeline tests
- `BenchmarkAllPipelines()` - Run performance benchmarks
- `ValidatePipelineResults(results)` - Validate pipeline outputs

### 2. Python Bridge Module

The `objectscript/python_bridge.py` module provides the interface between ObjectScript and Python RAG implementations.

**Key Functions:**
- `health_check()` - System health verification
- `get_available_pipelines()` - Pipeline discovery
- `invoke_basic_rag(query, config)` - Basic RAG execution
- `invoke_colbert(query, config)` - ColBERT execution
- `invoke_graphrag(query, config)` - GraphRAG execution
- `invoke_hyde(query, config)` - HyDE execution
- `invoke_crag(query, config)` - CRAG execution
- `invoke_noderag(query, config)` - NodeRAG execution
- `run_benchmarks(pipeline_names)` - Benchmark execution
- `validate_results(results)` - Result validation

### 3. Error Handling and Validation

All functions include comprehensive error handling and return standardized JSON responses:

```json
{
  "success": true|false,
  "result": <actual_result>,
  "error": <error_message_if_failed>,
  "timestamp": "<ISO_timestamp>"
}
```

## Deployment

### Prerequisites

1. IRIS instance with Embedded Python enabled
2. Python RAG pipeline dependencies installed
3. Database schema initialized with vector search capabilities

### Deployment Steps

1. **Generate ObjectScript Classes:**
   ```bash
   python scripts/deploy_objectscript_classes.py
   ```

2. **Manual Deployment to IRIS:**
   - Copy `.cls` files from `objectscript/` directory
   - Import into IRIS Management Portal
   - Compile with 'ck' flags

3. **Verify Deployment:**
   ```sql
   SELECT RAGDemo.InvokerExists() AS test
   SELECT RAGDemo.TestBedExists() AS test
   SELECT RAGDemo.HealthCheck() AS health
   ```

## Usage Examples

### Basic RAG Query
```sql
SELECT RAGDemo.InvokeBasicRAG(
  'What are the effects of COVID-19?',
  '{"embedding_func": "default", "llm_func": "default"}'
) AS result
```

### Health Check
```sql
SELECT RAGDemo.HealthCheck() AS health
```

### Get Available Pipelines
```sql
SELECT RAGDemo.GetAvailablePipelines() AS pipelines
```

### Run Benchmarks
```sql
SELECT RAGDemo.BenchmarkAllPipelines() AS benchmarks
```

## Configuration

### Pipeline Configuration

Each pipeline accepts a JSON configuration string with the following structure:

```json
{
  "embedding_func": "function_name_or_callable",
  "llm_func": "function_name_or_callable",
  "colbert_query_encoder_func": "function_name_or_callable",
  "colbert_doc_encoder_func": "function_name_or_callable",
  "top_k": 5,
  "similarity_threshold": 0.7
}
```

### Environment Variables

The integration respects the following environment variables:
- `IRIS_CONNECTION_URL` - Full IRIS connection string
- `IRIS_HOST` - IRIS host (default: localhost)
- `IRIS_PORT` - IRIS port (default: 1972)
- `IRIS_NAMESPACE` - IRIS namespace (default: USER)
- `IRIS_USERNAME` - IRIS username (default: _SYSTEM)
- `IRIS_PASSWORD` - IRIS password (default: SYS)

## Testing

### TDD Approach

The integration was developed using Test-Driven Development (TDD) methodology:

1. **Red Phase:** Write failing tests for ObjectScript integration
2. **Green Phase:** Implement minimal code to make tests pass
3. **Refactor Phase:** Clean up and optimize implementation

### Test Coverage

- Python bridge module import and functionality
- Pipeline invocation with proper error handling
- Health checks and system validation
- Benchmark execution and result formatting
- Result validation and metrics calculation

### Running Tests

```bash
# Run ObjectScript integration tests
python -m pytest tests/test_objectscript_integration.py -v

# Run specific test class
python -m pytest tests/test_objectscript_integration.py::TestPythonBridge -v
```

## Performance Considerations

### Embedded Python Overhead

- Function calls between ObjectScript and Python have minimal overhead
- JSON serialization/deserialization is optimized for small to medium payloads
- Large result sets should be paginated or streamed

### Memory Management

- Python objects are automatically garbage collected
- IRIS connection pooling is handled by the bridge module
- Large embeddings are processed efficiently through string-based storage

### Scalability

- The integration supports concurrent requests
- Database connections are managed efficiently
- Pipeline instances can be cached for better performance

## Troubleshooting

### Common Issues

1. **ImportError: Module not found**
   - Ensure Python dependencies are installed in IRIS Python environment
   - Verify PYTHONPATH includes project directory

2. **Database Connection Errors**
   - Check IRIS connection parameters
   - Verify database schema is initialized
   - Ensure proper permissions for database operations

3. **Pipeline Execution Errors**
   - Verify embedding and LLM functions are properly configured
   - Check that required data is loaded in the database
   - Review error logs for specific pipeline issues

### Debugging

Enable detailed logging by setting environment variables:
```bash
export PYTHONPATH=/path/to/rag-templates
export LOG_LEVEL=DEBUG
```

## Integration with Existing Systems

### Enterprise Deployment

The ObjectScript integration enables:
- Integration with existing IRIS applications
- Seamless embedding in ObjectScript business logic
- Enterprise-grade error handling and logging
- Standardized JSON API for external systems

### API Gateway Integration

ObjectScript methods can be exposed through IRIS REST services:
```objectscript
Class RAGDemo.REST Extends %CSP.REST
{
  XData UrlMap
  {
    <Routes>
      <Route Url="/health" Method="GET" Call="HealthCheck"/>
      <Route Url="/pipelines" Method="GET" Call="GetPipelines"/>
      <Route Url="/query/:pipeline" Method="POST" Call="ExecuteQuery"/>
    </Routes>
  }
}
```

## Future Enhancements

### Planned Features

1. **Async Pipeline Execution**
   - Background job processing for long-running queries
   - Progress tracking and status updates

2. **Advanced Configuration Management**
   - Pipeline-specific configuration storage
   - Dynamic configuration updates

3. **Enhanced Monitoring**
   - Performance metrics collection
   - Real-time pipeline health monitoring

4. **Caching Layer**
   - Query result caching
   - Embedding cache management

## Conclusion

The ObjectScript integration provides a robust, enterprise-ready interface for RAG pipeline execution within IRIS environments. The TDD approach ensures reliability and maintainability, while the comprehensive error handling and validation systems provide production-ready stability.

For additional support or questions, refer to the test files in `tests/test_objectscript_integration.py` for usage examples and expected behaviors.