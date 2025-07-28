# Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting information for the chunking system and ColBERT token embedding auto-population fixes. It covers common issues, error patterns, diagnostic procedures, and solutions to help developers quickly resolve problems and maintain system reliability.

## Quick Diagnostic Checklist

### System Health Check

Run this quick diagnostic to identify common issues:

```bash
# 1. Verify UV installation and project setup
uv --version
uv run python --version

# 2. Check IRIS database connectivity
uv run python -c "
from common.iris_connection_manager import get_iris_connection
try:
    conn = get_iris_connection()
    print('✅ IRIS connection successful')
    conn.close()
except Exception as e:
    print(f'❌ IRIS connection failed: {e}')
"

# 3. Verify core imports
uv run python -c "
try:
    from iris_rag.services.token_embedding_service import TokenEmbeddingService
    from tools.chunking.chunking_service import DocumentChunkingService
    from iris_rag.pipelines.basic import BasicRAGPipeline
    from iris_rag.pipelines.colbert.pipeline import ColBERTRAGPipeline
    print('✅ All core imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

# 4. Check database schema
uv run python -c "
from iris_rag.storage.schema_manager import SchemaManager
from iris_rag.config.manager import ConfigurationManager
from common.iris_connection_manager import get_iris_connection

config_manager = ConfigurationManager()
connection_manager = type('CM', (), {'get_connection': get_iris_connection})()
schema_manager = SchemaManager(connection_manager, config_manager)

try:
    schema_manager.ensure_table_schema('SourceDocuments')
    schema_manager.ensure_table_schema('DocumentTokenEmbeddings')
    print('✅ Database schema validation successful')
except Exception as e:
    print(f'❌ Schema validation failed: {e}')
"
```

## Common Issues and Solutions

### 1. ColBERT Auto-Population Issues

#### Issue: Dimension Mismatch Errors

**Symptoms:**
```
AssertionError: Expected 768D embeddings, got 384D
ValueError: Token embedding dimension mismatch
```

**Root Causes:**
- ColBERT interface misconfiguration
- Schema manager returning wrong dimensions
- Mixing document and token embedding functions

**Solutions:**

```python
# 1. Verify ColBERT interface configuration
from iris_rag.embeddings.colbert_interface import get_colbert_interface_from_config
from iris_rag.config.manager import ConfigurationManager

config_manager = ConfigurationManager()
connection_manager = type('CM', (), {'get_connection': get_iris_connection})()

colbert_interface = get_colbert_interface_from_config(config_manager, connection_manager)
print(f"ColBERT token dimension: {colbert_interface.get_token_dimension()}")

# Should output: ColBERT token dimension: 768
```

```python
# 2. Check schema manager configuration
from iris_rag.storage.schema_manager import SchemaManager

schema_manager = SchemaManager(connection_manager, config_manager)
token_dim = schema_manager.get_colbert_token_dimension()
doc_dim = schema_manager.get_vector_dimension("SourceDocuments")

print(f"Token embeddings: {token_dim}D")  # Should be 768
print(f"Document embeddings: {doc_dim}D")  # Should be 384
```

```yaml
# 3. Verify configuration file settings
# config/default.yaml
storage:
  base_embedding_dimension: 384      # Document embeddings
  colbert_token_dimension: 768       # Token embeddings
  colbert_backend: "native"          # or "huggingface"
```

#### Issue: Auto-Population Fails

**Symptoms:**
```
Error: Failed to auto-populate token embeddings
TokenEmbeddingService initialization failed
No token embeddings generated
```

**Root Causes:**
- Database connection issues
- Missing table permissions
- ColBERT interface initialization failure
- Insufficient memory or disk space

**Solutions:**

```python
# 1. Test TokenEmbeddingService initialization
from iris_rag.services.token_embedding_service import TokenEmbeddingService

try:
    service = TokenEmbeddingService(config_manager, connection_manager)
    print(f"✅ Service initialized with {service.token_dimension}D embeddings")
    
    # Test basic functionality
    stats = service.get_token_embedding_stats()
    print(f"Current coverage: {stats['coverage_percentage']:.1f}%")
    
except Exception as e:
    print(f"❌ Service initialization failed: {e}")
    # Check specific error details
    import traceback
    traceback.print_exc()
```

```sql
-- 2. Verify database permissions
-- Connect to IRIS and run:
SELECT PRIVILEGE_TYPE, IS_GRANTABLE 
FROM INFORMATION_SCHEMA.TABLE_PRIVILEGES 
WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentTokenEmbeddings';

-- Ensure INSERT, UPDATE, SELECT permissions exist
```

```python
# 3. Test ColBERT interface directly
try:
    interface = service.colbert_interface
    test_embeddings = interface.encode_query("test query")
    print(f"✅ ColBERT interface working: {len(test_embeddings)} tokens, {len(test_embeddings[0])}D")
except Exception as e:
    print(f"❌ ColBERT interface failed: {e}")
```

#### Issue: Pipeline Validation Failures

**Symptoms:**
```
ColBERT validation failed: No token embeddings available
Pipeline setup validation returned False
validate_setup() auto-population failed
```

**Root Causes:**
- Auto-population service not triggered
- Database transaction rollback
- Insufficient documents in database

**Solutions:**

```python
# 1. Force auto-population manually
from iris_rag.pipelines.colbert.pipeline import ColBERTRAGPipeline

pipeline = ColBERTRAGPipeline(iris_connector, config_manager)

# Force validation and auto-population
result = pipeline.validate_setup()
print(f"Validation result: {result}")

if not result:
    # Manual token embedding population
    from iris_rag.services.token_embedding_service import TokenEmbeddingService
    service = TokenEmbeddingService(config_manager, connection_manager)
    stats = service.ensure_token_embeddings_exist()
    print(f"Manual population: {stats.documents_processed} docs processed")
```

```python
# 2. Check document availability
connection = get_iris_connection()
cursor = connection.cursor()

cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
doc_count = cursor.fetchone()[0]
print(f"Available documents: {doc_count}")

if doc_count == 0:
    print("❌ No documents available for token embedding generation")
    print("Load documents first using pipeline.load_documents()")
```

### 2. Chunking System Issues

#### Issue: DocumentChunkingService Not Found

**Symptoms:**
```
ImportError: cannot import name 'DocumentChunkingService'
ModuleNotFoundError: No module named 'tools.chunking.chunking_service'
AttributeError: 'BasicRAGPipeline' object has no attribute 'chunking_service'
```

**Root Causes:**
- Missing chunking service module
- Incorrect import paths
- Service not initialized in pipeline

**Solutions:**

```python
# 1. Verify chunking service import
try:
    from tools.chunking.chunking_service import DocumentChunkingService
    print("✅ DocumentChunkingService import successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Check if tools/chunking/chunking_service.py exists")
```

```python
# 2. Test chunking service initialization
embedding_func = lambda texts: [[0.1] * 384 for _ in texts]  # Mock embedding
try:
    service = DocumentChunkingService(embedding_func=embedding_func)
    print("✅ DocumentChunkingService initialized successfully")
    
    # Test basic chunking
    chunks = service.chunk_document("test_doc", "Test content for chunking", "fixed_size")
    print(f"Generated {len(chunks)} chunks")
    
except Exception as e:
    print(f"❌ Service initialization failed: {e}")
```

```python
# 3. Verify pipeline integration
from iris_rag.pipelines.basic import BasicRAGPipeline

pipeline = BasicRAGPipeline(config_manager=config_manager)

# Check if chunking service is properly initialized
if hasattr(pipeline, 'chunking_service'):
    print("✅ Chunking service integrated in pipeline")
    print(f"Strategy: {pipeline.chunking_strategy}")
else:
    print("❌ Chunking service not found in pipeline")
    print("Check BasicRAGPipeline.__init__() method")
```

#### Issue: Chunking Strategy Errors

**Symptoms:**
```
ValueError: Unknown chunking strategy 'semantic'
KeyError: 'chunking_strategy' not found in configuration
AttributeError: 'DocumentChunkingService' object has no attribute 'semantic_chunking'
```

**Root Causes:**
- Invalid chunking strategy name
- Missing configuration
- Strategy not implemented in service

**Solutions:**

```python
# 1. Check available chunking strategies
from tools.chunking.chunking_service import DocumentChunkingService

service = DocumentChunkingService(embedding_func=lambda x: x)

# List available strategies (if method exists)
if hasattr(service, 'get_available_strategies'):
    strategies = service.get_available_strategies()
    print(f"Available strategies: {strategies}")
else:
    print("Available strategies: fixed_size, semantic, sentence, paragraph")
```

```yaml
# 2. Verify configuration
# config/basic_rag_example.yaml
pipelines:
  basic:
    chunking_strategy: "fixed_size"  # Use valid strategy name
    chunk_size: 1000
    chunk_overlap: 200
```

```python
# 3. Test strategy switching
pipeline = BasicRAGPipeline(config_manager=config_manager)

# Test different strategies
strategies = ["fixed_size", "semantic", "sentence"]
for strategy in strategies:
    try:
        pipeline.chunking_strategy = strategy
        print(f"✅ Strategy '{strategy}' set successfully")
    except Exception as e:
        print(f"❌ Strategy '{strategy}' failed: {e}")
```

#### Issue: Chunk Metadata Missing

**Symptoms:**
```
KeyError: 'chunk_index' not found in metadata
AttributeError: 'Document' object has no attribute 'id'
Missing chunk metadata in DocumentChunks table
```

**Root Causes:**
- Metadata not properly set during chunking
- Document ID generation issues
- Database insertion failures

**Solutions:**

```python
# 1. Test chunk metadata generation
from iris_rag.core.models import Document

test_doc = Document(
    page_content="Test document content for chunking validation",
    metadata={"source": "test.txt", "author": "Test"}
)

pipeline = BasicRAGPipeline(config_manager=config_manager)
chunks = pipeline._chunk_documents([test_doc])

# Verify metadata
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}:")
    print(f"  ID: {getattr(chunk, 'id', 'MISSING')}")
    print(f"  Metadata keys: {list(chunk.metadata.keys())}")
    
    required_keys = ['chunk_index', 'parent_document_id', 'chunking_strategy']
    missing_keys = [key for key in required_keys if key not in chunk.metadata]
    if missing_keys:
        print(f"  ❌ Missing metadata: {missing_keys}")
    else:
        print(f"  ✅ All required metadata present")
```

```python
# 2. Check DocumentChunks table structure
connection = get_iris_connection()
cursor = connection.cursor()

cursor.execute("""
    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentChunks'
    ORDER BY ORDINAL_POSITION
""")

columns = cursor.fetchall()
print("DocumentChunks table structure:")
for col in columns:
    print(f"  {col[0]}: {col[1]} ({'NULL' if col[2] == 'YES' else 'NOT NULL'})")
```

### 3. Database and Connection Issues

#### Issue: IRIS Connection Failures

**Symptoms:**
```
ConnectionError: Could not connect to IRIS database
jaydebeapi.DatabaseError: Connection failed
TimeoutError: Database connection timeout
```

**Root Causes:**
- IRIS server not running
- Incorrect connection parameters
- Network connectivity issues
- Authentication failures

**Solutions:**

```bash
# 1. Check IRIS server status
docker ps | grep iris
# Should show running IRIS container

# If not running, start IRIS
docker-compose up -d iris
```

```python
# 2. Test connection parameters
import os
from common.iris_connection_manager import get_iris_connection

# Check environment variables
print(f"IRIS_HOST: {os.getenv('IRIS_HOST', 'localhost')}")
print(f"IRIS_PORT: {os.getenv('IRIS_PORT', '1972')}")
print(f"IRIS_NAMESPACE: {os.getenv('IRIS_NAMESPACE', 'USER')}")
print(f"IRIS_USERNAME: {os.getenv('IRIS_USERNAME', 'demo')}")

# Test connection
try:
    connection = get_iris_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()
    print(f"✅ Connection successful: {result}")
    cursor.close()
    connection.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

```bash
# 3. Test IRIS connectivity from command line
# Using IRIS terminal
docker exec -it <iris-container> iris session iris

# Or using SQL client
docker exec -it <iris-container> iris sql iris
```

#### Issue: Table Schema Problems

**Symptoms:**
```
Table 'RAG.DocumentTokenEmbeddings' doesn't exist
Column 'token_embedding' not found
Schema validation failed
```

**Root Causes:**
- Tables not created
- Schema migration issues
- Incorrect table structure

**Solutions:**

```python
# 1. Force schema creation
from iris_rag.storage.schema_manager import SchemaManager

schema_manager = SchemaManager(connection_manager, config_manager)

# Create all required tables
tables = ['SourceDocuments', 'DocumentTokenEmbeddings', 'DocumentChunks']
for table in tables:
    try:
        schema_manager.ensure_table_schema(table)
        print(f"✅ Table {table} schema ensured")
    except Exception as e:
        print(f"❌ Table {table} failed: {e}")
```

```sql
-- 2. Manual table verification
-- Connect to IRIS and check tables
SELECT TABLE_NAME, TABLE_TYPE 
FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_SCHEMA = 'RAG';

-- Check specific table structure
DESCRIBE RAG.DocumentTokenEmbeddings;
```

```python
# 3. Reset schema if needed (CAUTION: This will delete data)
def reset_schema():
    """Reset database schema - USE WITH CAUTION"""
    connection = get_iris_connection()
    cursor = connection.cursor()
    
    # Drop tables in correct order (due to foreign keys)
    tables = ['DocumentTokenEmbeddings', 'DocumentChunks', 'SourceDocuments']
    for table in tables:
        try:
            cursor.execute(f"DROP TABLE IF EXISTS RAG.{table}")
            print(f"Dropped table {table}")
        except Exception as e:
            print(f"Failed to drop {table}: {e}")
    
    connection.commit()
    cursor.close()
    
    # Recreate schema
    schema_manager = SchemaManager(connection_manager, config_manager)
    for table in ['SourceDocuments', 'DocumentChunks', 'DocumentTokenEmbeddings']:
        schema_manager.ensure_table_schema(table)
        print(f"Recreated table {table}")

# Uncomment to use (DANGER: Will delete all data)
# reset_schema()
```

### 4. Performance and Memory Issues

#### Issue: Out of Memory During Processing

**Symptoms:**
```
MemoryError: Unable to allocate memory
OutOfMemoryError during token embedding generation
Process killed due to memory usage
```

**Root Causes:**
- Large batch sizes
- Memory leaks
- Insufficient system memory
- Large document processing

**Solutions:**

```python
# 1. Reduce batch sizes
config_manager.set("colbert.batch_size", 10)  # Reduce from default 50
config_manager.set("pipelines:basic:embedding_batch_size", 16)  # Reduce from 32

# 2. Process documents in smaller chunks
def process_large_dataset_safely(documents, batch_size=5):
    """Process large document sets with memory management."""
    
    pipeline = BasicRAGPipeline(config_manager=config_manager)
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
        
        try:
            pipeline.load_documents(documents=batch)
            
            # Force garbage collection
            import gc
            gc.collect()
            
        except MemoryError:
            print(f"Memory error in batch {i//batch_size + 1}, reducing batch size")
            # Process individually
            for doc in batch:
                pipeline.load_documents(documents=[doc])
                gc.collect()
```

```python
# 3. Monitor memory usage
import psutil
import os

def monitor_memory():
    """Monitor memory usage during processing."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    print(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
    print(f"Virtual memory: {memory_info.vms / 1024 / 1024:.1f} MB")
    
    # System memory
    system_memory = psutil.virtual_memory()
    print(f"System memory: {system_memory.percent}% used")
    
    return memory_info.rss / 1024 / 1024  # Return MB

# Use during processing
initial_memory = monitor_memory()
# ... processing code ...
final_memory = monitor_memory()
print(f"Memory increase: {final_memory - initial_memory:.1f} MB")
```

#### Issue: Slow Processing Performance

**Symptoms:**
```
Token embedding generation taking too long
Chunking process extremely slow
Pipeline timeouts
```

**Root Causes:**
- Inefficient batch processing
- Database connection overhead
- Large document sizes
- Suboptimal configuration

**Solutions:**

```python
# 1. Optimize batch processing
# Increase batch sizes for better throughput (if memory allows)
config_manager.set("colbert.batch_size", 100)
config_manager.set("pipelines:basic:embedding_batch_size", 64)

# 2. Use connection pooling
from common.iris_connection_manager import get_iris_connection

# Reuse connections instead of creating new ones
class ConnectionPool:
    def __init__(self):
        self.connection = get_iris_connection()
    
    def get_connection(self):
        return self.connection
    
    def close(self):
        self.connection.close()

# Use in services
connection_pool = ConnectionPool()
connection_manager = type('CM', (), {'get_connection': connection_pool.get_connection})()
```

```python
# 3. Profile performance bottlenecks
import time
import cProfile

def profile_pipeline_performance():
    """Profile pipeline performance to identify bottlenecks."""
    
    def test_function():
        pipeline = BasicRAGPipeline(config_manager=config_manager)
        
        # Load test documents
        test_docs = [Document(page_content=f"Test document {i}" * 100, metadata={}) 
                    for i in range(10)]
        
        start_time = time.time()
        pipeline.load_documents(documents=test_docs)
        load_time = time.time() - start_time
        
        start_time = time.time()
        result = pipeline.run("test query")
        query_time = time.time() - start_time
        
        print(f"Load time: {load_time:.2f}s")
        print(f"Query time: {query_time:.2f}s")
        
        return result
    
    # Profile the function
    cProfile.run('test_function()', 'profile_output.prof')
    
    # Analyze results
    import pstats
    stats = pstats.Stats('profile_output.prof')
    stats.sort_stats('cumulative').print_stats(20)

# Run profiling
profile_pipeline_performance()
```

## Diagnostic Tools and Scripts

### 1. System Health Check Script

```python
#!/usr/bin/env python3
"""
Comprehensive system health check for RAG Templates.
Run with: uv run python scripts/health_check.py
"""

import sys
import traceback
from typing import Dict, List, Tuple

def run_health_check() -> Dict[str, bool]:
    """Run comprehensive health check."""
    
    results = {}
    
    # Test 1: Basic imports
    try:
        from iris_rag.services.token_embedding_service import TokenEmbeddingService
        from tools.chunking.chunking_service import DocumentChunkingService
        from iris_rag.pipelines.basic import BasicRAGPipeline
        from iris_rag.pipelines.colbert.pipeline import ColBERTRAGPipeline
        results['imports'] = True
        print("✅ All imports successful")
    except Exception as e:
        results['imports'] = False
        print(f"❌ Import failed: {e}")
    
    # Test 2: Database connection
    try:
        from common.iris_connection_manager import get_iris_connection
        connection = get_iris_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        connection.close()
        results['database'] = True
        print("✅ Database connection successful")
    except Exception as e:
        results['database'] = False
        print(f"❌ Database connection failed: {e}")
    
    # Test 3: Schema validation
    try:
        from iris_rag.storage.schema_manager import SchemaManager
        from iris_rag.config.manager import ConfigurationManager
        
        config_manager = ConfigurationManager()
        connection_manager = type('CM', (), {'get_connection': get_iris_connection})()
        schema_manager = SchemaManager(connection_manager, config_manager)
        
        schema_manager.ensure_table_schema('SourceDocuments')
        results['schema'] = True
        print("✅ Schema validation successful")
    except Exception as e:
        results['schema'] = False
        print(f"❌ Schema validation failed: {e}")
    
    # Test 4: ColBERT interface
    try:
        from iris_rag.embeddings.colbert_interface import get_colbert_interface_from_config
        
        colbert_interface = get_colbert_interface_from_config(config_manager, connection_manager)
        token_dim = colbert_interface.get_token_dimension()
        
        if token_dim == 768:
            results['colbert'] = True
            print("✅ ColBERT interface working (768D)")
        else:
            results['colbert'] = False
            print(f"❌ ColBERT dimension mismatch: {token_dim}D (expected 768D)")
    except Exception as e:
        results['colbert'] = False
        print(f"❌ ColBERT interface failed: {e}")
    
    # Test 5: Chunking service
    try:
        embedding_func = lambda texts: [[0.1] * 384 for _ in texts]
        chunking_service = DocumentChunkingService(embedding_func=embedding_func)
        chunks = chunking_service.chunk_document("test", "Test content", "fixed_size")
        
        if len(chunks) > 0:
            results['chunking'] = True
            print("✅ Chunking service working")
        else:
            results['chunking'] = False
            print("❌ Chunking service returned no chunks")
    except Exception as e:
        results['chunking'] = False
        print(f"❌ Chunking service failed: {e}")
    
    return results

if __name__ == "__main__":
    print("Running RAG Templates Health Check...")
    print("=" * 50)
    
    results = run_health_check()
    
    print("\n" + "=" * 50)
    print("Health Check Summary:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test.capitalize()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All systems operational!")
        sys.exit(0)
    else:
        print("⚠️  Some systems need attention")
        sys.exit(1)
```

### 2. Performance Benchmark Script

```python
#!/usr/bin/env python3
"""
Performance benchmark for chunking and ColBERT systems.
Run with: uv run python scripts/performance_benchmark.py
"""

import time
import statistics
from typing import List, Dict

def benchmark_chunking_performance():
    """Benchmark chunking system performance."""
    
    from tools.chunking.chunking_service import DocumentChunkingService
    from iris_rag.core.models import Document
    
    # Setup
    embedding_func = lambda texts: [[0.1] * 384 for _ in texts]
    service = DocumentChunkingService(embedding_func=embedding_func)
    
    # Test documents of different sizes
    test_cases = [
        ("small", "Test content. " * 100),      # ~1KB
        ("medium", "Test content. " * 1000),    # ~10KB
        ("large", "Test content. " * 10000),    # ~100KB
    ]
    
    results = {}
    
    for size_name, content in test_cases:
        times = []
        
        for i in range(5):  # 5 runs for average
            start_time = time.time()
            chunks = service.chunk_document(f"test_{i}", content, "fixed_size")
            end_time = time.time()
            
            times.append(end_time - start_time)
        
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0
        
        results[size_name] = {
            'avg_time': avg_time,
            'std_dev': std_dev,
            'chunks_generated': len(chunks),
            'chars_per_second': len(content) / avg_time
        }
        
        print(f"{size_name.capitalize()} document ({len(content)} chars):")
        print(f"  Average time: {avg_time:.3f}s ± {std_dev:.3f}s")
        print(f"  Chunks generated: {len(chunks)}")
        print(f"  Processing rate: {results[size_name]['chars_per_second']:.0f} chars/sec")
        print()
    
    return results

def benchmark_colbert_performance():
    """Benchmark ColBERT auto-population performance."""
    
    from iris_rag.services.token_embedding_service import TokenEmbeddingService
    from iris_rag.config.manager import ConfigurationManager
    from common.iris_connection_manager import get_iris_connection
    
    # Setup
    config_manager = ConfigurationManager()
    connection_manager = type('CM', (), {'get_connection': get_iris_connection})()
    
    try:
        service = TokenEmbeddingService(config_manager, connection_manager)
        
        # Get current statistics
        start_stats = service.get_token_embedding_stats()
        
        print(f"Current token embedding coverage: {start_stats['coverage_percentage']:.1f}%")
        print(f"Documents with tokens: {start_stats['documents_with_token_embeddings']}")
        print(f"Total tokens: {start_stats['total_token_embeddings']}")
        
        if start_stats['documents_missing_token_embeddings'] > 0:
            print(f"\nBenchmarking auto-population for {start_stats['documents_missing_token_embeddings']} documents...")
            
            start_time = time.time()
            population_stats = service.ensure_token_embeddings_exist()
            end_time = time.time()
            
            print(f"Auto-population completed in {end_time - start_time:.2f}s")
            print(f"Documents processed: {population_stats.documents_processed}")
            print(f"Tokens generated: {population_stats.tokens_generated}")
            print(f"Processing rate: {population_stats.documents_processed / (end_time - start_time):.1f} docs/sec")
            print(f"Token generation rate: {population_stats.tokens_generated / (end_time - start_time):.1f} tokens/sec")
        else:
            print("No documents need token embedding population")
            
    except Exception as e:
        print(f"ColBERT benchmark failed: {e}")

if __name__ == "__main__":
    print("RAG Templates Performance Benchmark")
    print("=" * 50)
    
    print("1. Chunking Performance:")
    print("-" * 25)
    chunking_results = benchmark_chunking_performance()
    
    print("2. ColBERT Performance:")
    print("-" * 25)
    benchmark_colbert_performance()
```

### 3. Configuration Validator

```python
#!/usr/bin/env python3
"""
Configuration validation script.
Run with: uv run python scripts/validate_config.py
"""

def validate_configuration():
    """Validate system configuration."""
    
    from iris_rag.config.manager import ConfigurationManager
    
    config_manager = ConfigurationManager()
    issues = []
    
    # Check required configuration keys
    required_configs = {
        'storage:base_embedding_dimension': 384,
        'storage:colbert_token_dimension': 768,
        'storage:colbert_backend': ['native', 'huggingface'],
        'pipelines:basic:chunk_size': int,
        'pipelines:basic:chunk_overlap': int,
        'pipelines:basic:chunking_strategy': ['fixed_size', 'semantic', 'sentence'],
    }
    
    for key, expected in required_configs.items():
        try:
            value = config_manager.get(key)
            
            if isinstance(expected, list):
                if value not in expected:
                    issues.append(f"❌ {key}: '{value}' not in allowed values {expected}")
                else:
                    print(f"✅ {key}: {value}")
            elif isinstance(expected, type):
                if not isinstance(value, expected):
                    issues.append(f"❌ {key}: '{value}' is not of type {expected.__name__}")
                else:
                    print(f"