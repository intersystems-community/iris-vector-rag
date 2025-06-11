# Troubleshooting Guide

Common issues and solutions for RAG Templates installation, configuration, and operation.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [Database Connection Issues](#database-connection-issues)
- [Performance Troubleshooting](#performance-troubleshooting)
- [Error Message Explanations](#error-message-explanations)
- [Debugging Techniques](#debugging-techniques)
- [Known Issues and Workarounds](#known-issues-and-workarounds)

## Installation Issues

### Python Version Compatibility

**Problem**: Package installation fails with Python version errors.

**Solution**:
```bash
# Check Python version
python --version

# Ensure Python 3.11 or higher
# If using older version, install Python 3.11+
pyenv install 3.11.0
pyenv local 3.11.0

# Or use conda
conda create -n iris-rag python=3.11
conda activate iris-rag
```

### Missing Dependencies

**Problem**: Import errors for optional dependencies.

**Symptoms**:
```
ImportError: No module named 'sentence_transformers'
ImportError: No module named 'jaydebeapi'
```

**Solution**:
```bash
# Install specific dependency groups
pip install intersystems-iris-rag[embeddings]  # For embedding backends
pip install intersystems-iris-rag[jdbc]        # For JDBC support
pip install intersystems-iris-rag[all]         # For all optional dependencies

# Or install individually
pip install sentence-transformers      # For SentenceTransformers
pip install jaydebeapi jpype1         # For JDBC support
pip install transformers torch        # For Transformers backend
```

### JDBC Driver Issues

**Problem**: JDBC connection fails with driver not found.

**Symptoms**:
```
java.lang.ClassNotFoundException: com.intersystems.jdbc.IRISDriver
```

**Solution**:
```bash
# Download IRIS JDBC driver
curl -L -o intersystems-jdbc-3.8.4.jar \
  https://github.com/intersystems-community/iris-driver-distribution/raw/main/JDBC/JDK18/intersystems-jdbc-3.8.4.jar

# Set CLASSPATH
export CLASSPATH="$CLASSPATH:/path/to/intersystems-jdbc-3.8.4.jar"

# Or specify in configuration
RAG_DATABASE__IRIS__JDBC_JAR_PATH=/path/to/intersystems-jdbc-3.8.4.jar
```

### Virtual Environment Issues

**Problem**: Package not found after installation.

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Verify installation
pip list | grep intersystems-iris-rag

# Reinstall if necessary
pip uninstall intersystems-iris-rag
pip install -e .
```

## Configuration Problems

### Configuration File Not Found

**Problem**: [`ConfigurationManager`](../rag_templates/config/manager.py:10) cannot find configuration file.

**Symptoms**:
```
FileNotFoundError: Configuration file not found: config.yaml
```

**Solution**:
```bash
# Check file exists
ls -la config.yaml

# Use absolute path
export RAG_CONFIG_PATH=/absolute/path/to/config.yaml

# Or create default configuration
cat > config.yaml << EOF
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
EOF
```

### Environment Variable Override Issues

**Problem**: Environment variables not overriding configuration.

**Symptoms**: Configuration values not changing despite setting environment variables.

**Solution**:
```bash
# Check environment variable format
# Correct format: RAG_SECTION__SUBSECTION__KEY
export RAG_DATABASE__IRIS__HOST=localhost
export RAG_DATABASE__IRIS__PORT=1972

# Verify variables are set
env | grep RAG_

# Debug configuration loading
python -c "
from iris_rag.config.manager import ConfigurationManager
config = ConfigurationManager('config.yaml')
print('Host:', config.get('database:iris:host'))
print('Port:', config.get('database:iris:port'))
"
```

### YAML Syntax Errors

**Problem**: Configuration file has invalid YAML syntax.

**Symptoms**:
```
yaml.scanner.ScannerError: mapping values are not allowed here
```

**Solution**:
```bash
# Validate YAML syntax
python -c "
import yaml
with open('config.yaml', 'r') as f:
    try:
        yaml.safe_load(f)
        print('YAML is valid')
    except yaml.YAMLError as e:
        print(f'YAML error: {e}')
"

# Common YAML issues:
# 1. Incorrect indentation (use spaces, not tabs)
# 2. Missing colons after keys
# 3. Unquoted special characters
```

### Configuration Validation Errors

**Problem**: Configuration values fail validation.

**Symptoms**:
```
ConfigValidationError: Missing required config: database:iris:host
```

**Solution**:
```yaml
# Ensure all required fields are present
database:
  iris:
    host: localhost        # Required
    port: 1972            # Required
    namespace: USER       # Required
    username: demo        # Required
    password: demo        # Required
    driver: intersystems.jdbc  # Required

# Check configuration schema
python -c "
from iris_rag.config.manager import ConfigurationManager
config = ConfigurationManager('config.yaml')
try:
    config.validate()
    print('Configuration is valid')
except Exception as e:
    print(f'Validation error: {e}')
"
```

## Database Connection Issues

### IRIS Connection Failures

**Problem**: Cannot connect to InterSystems IRIS database.

**Symptoms**:
```
ConnectionError: Failed to connect to IRIS backend 'iris'
```

**Diagnostic Steps**:
```bash
# 1. Check IRIS is running
docker ps | grep iris
# or
iris list

# 2. Test network connectivity
telnet localhost 1972
nc -zv localhost 1972

# 3. Verify credentials
iris session iris -U USER
```

**Solutions**:

#### IRIS Not Running
```bash
# Start IRIS container
docker start iris-rag

# Or start new container
docker run -d \
  --name iris-rag \
  -p 1972:1972 \
  -p 52773:52773 \
  intersystemsdc/iris-community:latest
```

#### Wrong Connection Parameters
```yaml
# Update configuration
database:
  iris:
    host: localhost      # Check hostname
    port: 1972          # Check port (default 1972)
    namespace: USER     # Check namespace
    username: demo      # Check username
    password: demo      # Check password
```

#### Firewall Issues
```bash
# Check firewall rules
sudo ufw status
sudo iptables -L

# Allow IRIS ports
sudo ufw allow 1972
sudo ufw allow 52773
```

### Driver Import Errors

**Problem**: Database driver cannot be imported.

**Symptoms**:
```
ImportError: Failed to import database driver: intersystems.jdbc
```

**Solutions**:

#### JDBC Driver Issues
```bash
# Install Java dependencies
pip install jaydebeapi jpype1

# Download JDBC driver
wget https://github.com/intersystems-community/iris-driver-distribution/raw/main/JDBC/JDK18/intersystems-jdbc-3.8.4.jar

# Set environment variable
export RAG_DATABASE__IRIS__JDBC_JAR_PATH=/path/to/intersystems-jdbc-3.8.4.jar
```

#### DBAPI Driver Issues
```bash
# Install IRIS Python driver
pip install intersystems-iris

# Or use wheel file
pip install /path/to/intersystems_iris-3.2.0-py3-none-any.whl
```

### Connection Pool Exhaustion

**Problem**: Too many database connections.

**Symptoms**:
```
ConnectionError: Maximum number of connections exceeded
```

**Solution**:
```python
# Properly close connections
from iris_rag.core.connection import ConnectionManager

conn_mgr = ConnectionManager(config)
try:
    connection = conn_mgr.get_connection('iris')
    # Use connection
finally:
    conn_mgr.close_connection('iris')

# Or use context manager
with conn_mgr.get_connection('iris') as connection:
    # Use connection
    pass  # Connection automatically closed
```

### Schema Initialization Failures

**Problem**: Database schema creation fails.

**Symptoms**:
```
SQL Error: Table 'SourceDocuments' already exists
```

**Solution**:
```python
# Check existing schema
from iris_rag.storage.iris import IRISStorage

storage = IRISStorage(conn_mgr, config)

# Drop existing tables (caution: data loss)
storage.drop_schema()

# Recreate schema
storage.initialize_schema()

# Or use migration approach
storage.migrate_schema(from_version='1.0', to_version='2.0')
```

## Performance Troubleshooting

### Slow Query Performance

**Problem**: RAG queries take too long to execute.

**Diagnostic Steps**:
```python
# Enable query profiling
import time
import logging

logging.basicConfig(level=logging.DEBUG)

start_time = time.time()
result = pipeline.execute("What is machine learning?")
end_time = time.time()

print(f"Query took {end_time - start_time:.2f} seconds")
print(f"Retrieved {len(result['retrieved_documents'])} documents")
```

**Solutions**:

#### Use V2 Pipelines
```python
# V2 pipelines are significantly faster
from iris_rag.pipelines.basic_v2 import BasicRAGPipelineV2

# 2-6x performance improvement
pipeline = BasicRAGPipelineV2(conn_mgr, config)
```

#### Optimize Vector Search
```sql
-- Create HNSW index for faster vector search
CREATE INDEX idx_document_embedding_hnsw 
ON RAG.SourceDocuments_V2 (document_embedding_vector)
USING HNSW;

-- Check index usage
EXPLAIN SELECT TOP 5 * FROM RAG.SourceDocuments_V2 
WHERE VECTOR_DOT_PRODUCT(document_embedding_vector, ?) > 0.7;
```

#### Adjust Batch Sizes
```python
# Optimize for your system
config = {
    'pipelines': {
        'basic': {
            'sample_size': 500,      # Increase for better coverage
            'batch_size': 32,        # Adjust based on memory
            'top_k': 5              # Reduce if not needed
        }
    }
}
```

### Memory Issues

**Problem**: High memory usage or out-of-memory errors.

**Symptoms**:
```
MemoryError: Unable to allocate array
```

**Solutions**:

#### Reduce Batch Sizes
```python
# Reduce embedding batch size
config = {
    'embeddings': {
        'batch_size': 16,  # Reduce from default 32
        'max_length': 512  # Truncate long documents
    }
}
```

#### Use Streaming Processing
```python
# Process documents in chunks
def process_documents_streaming(documents, chunk_size=100):
    for i in range(0, len(documents), chunk_size):
        chunk = documents[i:i + chunk_size]
        pipeline.load_documents(chunk)
        yield f"Processed {i + len(chunk)} documents"
```

#### Monitor Memory Usage
```python
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")
    
    # Force garbage collection
    gc.collect()

# Call periodically during processing
monitor_memory()
```

### Embedding Generation Slowness

**Problem**: Embedding generation is slow.

**Solutions**:

#### Use GPU Acceleration
```python
# Configure for GPU usage
config = {
    'embeddings': {
        'sentence_transformers': {
            'device': 'cuda',  # Use GPU if available
            'model_name': 'all-MiniLM-L6-v2'
        }
    }
}
```

#### Cache Embeddings
```python
from functools import lru_cache

class CachedEmbeddingManager:
    @lru_cache(maxsize=10000)
    def embed_text(self, text):
        return self.embedding_func([text])[0]
```

#### Use Faster Models
```yaml
# Smaller, faster models
embeddings:
  sentence_transformers:
    model_name: all-MiniLM-L6-v2  # Fast, good quality
    # model_name: all-mpnet-base-v2  # Slower, better quality
```

## Error Message Explanations

### Common Error Patterns

#### ConfigValidationError
```
ConfigValidationError: Missing required config: database:iris:host
```
**Cause**: Required configuration parameter not provided.
**Fix**: Add missing parameter to configuration file or environment variables.

#### ConnectionError
```
ConnectionError: Failed to connect to IRIS backend 'iris': [Errno 111] Connection refused
```
**Cause**: IRIS database not running or not accessible.
**Fix**: Start IRIS database and verify network connectivity.

#### ImportError
```
ImportError: Failed to import database driver: intersystems.jdbc for backend iris
```
**Cause**: Required database driver not installed.
**Fix**: Install appropriate driver package and dependencies.

#### ValueError
```
ValueError: Unsupported database backend: postgres
```
**Cause**: Attempting to use unsupported database backend.
**Fix**: Use supported backend (currently only 'iris') or implement custom backend.

#### RAGPipelineError
```
RAGPipelineError: Embedding generation failed: Model not found
```
**Cause**: Embedding model not available or incorrectly configured.
**Fix**: Verify model name and ensure model is downloaded/accessible.

### SQL-Related Errors

#### Parameter Binding Issues
```
SQL Error: Invalid parameter binding
```
**Cause**: IRIS SQL parameter binding limitations.
**Fix**: Use V2 pipelines or JDBC driver for complex queries.

#### Vector Search Errors
```
SQL Error: VECTOR_DOT_PRODUCT function not found
```
**Cause**: Vector search functions not available in IRIS version.
**Fix**: Upgrade to IRIS 2025.1+ or use alternative similarity calculation.

#### Table Not Found
```
SQL Error: Table 'RAG.SourceDocuments' doesn't exist
```
**Cause**: Database schema not initialized.
**Fix**: Run schema initialization before using pipelines.

## Debugging Techniques

### Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Or configure specific loggers
logging.getLogger('iris_rag').setLevel(logging.DEBUG)
logging.getLogger('iris_rag.core.connection').setLevel(logging.DEBUG)
```

### Connection Debugging

```python
# Test database connection
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager

config = ConfigurationManager('config.yaml')
conn_mgr = ConnectionManager(config)

try:
    connection = conn_mgr.get_connection('iris')
    print("✅ Database connection successful")
    
    # Test query
    cursor = connection.cursor()
    cursor.execute("SELECT 1 as test")
    result = cursor.fetchone()
    print(f"✅ Query test successful: {result}")
    
except Exception as e:
    print(f"❌ Connection failed: {e}")
    import traceback
    traceback.print_exc()
```

### Pipeline Debugging

```python
# Debug pipeline execution
def debug_pipeline_execution(pipeline, query):
    print(f"🔍 Debugging query: {query}")
    
    try:
        # Test document retrieval
        documents = pipeline.query(query, top_k=3)
        print(f"✅ Retrieved {len(documents)} documents")
        
        # Test full execution
        result = pipeline.execute(query)
        print(f"✅ Generated answer: {result['answer'][:100]}...")
        
        return result
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

# Usage
result = debug_pipeline_execution(pipeline, "What is machine learning?")
```

### Configuration Debugging

```python
# Debug configuration loading
def debug_configuration(config_path):
    from iris_rag.config.manager import ConfigurationManager
    
    print(f"🔍 Loading configuration from: {config_path}")
    
    try:
        config = ConfigurationManager(config_path)
        
        # Check key configuration values
        db_config = config.get('database:iris')
        print(f"✅ Database config: {db_config}")
        
        embedding_config = config.get('embeddings')
        print(f"✅ Embedding config: {embedding_config}")
        
        # Validate configuration
        config.validate()
        print("✅ Configuration validation passed")
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        import traceback
        traceback.print_exc()

# Usage
debug_configuration('config.yaml')
```

### Performance Profiling

```python
import cProfile
import pstats
from io import StringIO

def profile_pipeline_execution(pipeline, query):
    """Profile pipeline execution to identify bottlenecks."""
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Execute pipeline
    result = pipeline.execute(query)
    
    profiler.disable()
    
    # Analyze results
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print("🔍 Performance Profile:")
    print(s.getvalue())
    
    return result

# Usage
result = profile_pipeline_execution(pipeline, "What is machine learning?")
```

## Known Issues and Workarounds

### IRIS SQL Parameter Binding

**Issue**: IRIS SQL has limitations with parameter binding in vector operations.

**Workaround**: Use V2 pipelines or JDBC driver:
```python
# Use V2 pipeline (recommended)
from iris_rag.pipelines.basic_v2 import BasicRAGPipelineV2

# Or configure JDBC driver
config = {
    'database': {
        'iris': {
            'driver': 'intersystems.jdbc',
            'jdbc_jar_path': '/path/to/intersystems-jdbc-3.8.4.jar'
        }
    }
}
```

### Vector Search Performance

**Issue**: Vector search can be slow on large datasets without proper indexing.

**Workaround**: Create HNSW indexes (requires IRIS Enterprise):
```sql
-- Create HNSW index for better performance
CREATE INDEX idx_document_embedding_hnsw 
ON RAG.SourceDocuments_V2 (document_embedding_vector)
USING HNSW;
```

### Memory Usage with Large Documents

**Issue**: Large documents can cause memory issues during embedding generation.

**Workaround**: Implement document chunking:
```python
def chunk_large_documents(documents, max_chunk_size=1000):
    """Split large documents into smaller chunks."""
    chunked_docs = []
    
    for doc in documents:
        if len(doc.page_content) > max_chunk_size:
            # Split into chunks
            chunks = [
                doc.page_content[i:i + max_chunk_size]
                for i in range(0, len(doc.page_content), max_chunk_size)
            ]
            
            for i, chunk in enumerate(chunks):
                chunked_docs.append(Document(
                    page_content=chunk,
                    metadata={**doc.metadata, 'chunk_id': i}
                ))
        else:
            chunked_docs.append(doc)
    
    return chunked_docs
```

### Embedding Model Download Issues

**Issue**: Embedding models may fail to download due to network issues.

**Workaround**: Pre-download models:
```python
from sentence_transformers import SentenceTransformer

# Pre-download model
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('/path/to/local/model')

# Configure to use local model
config = {
    'embeddings': {
        'sentence_transformers': {
            'model_name': '/path/to/local/model'
        }
    }
}
```

---

For additional help:
- Check the [Performance Guide](PERFORMANCE_GUIDE.md) for optimization tips
- Review the [Developer Guide](DEVELOPER_GUIDE.md) for architecture details
- See the [API Reference](API_REFERENCE.md) for detailed API documentation
- Visit the project's GitHub Issues for community support