# JDBC Integration for Benchmarking Tests

## Overview

To use JDBC in benchmarking tests, you have two options:
1. **Minimal Change**: Keep existing pipelines, just swap the connection
2. **Full Migration**: Use JDBC-specific pipeline versions

## Option 1: Minimal Change (Recommended for Testing)

### Step 1: Create a JDBC-compatible iris_connector wrapper

```python
# common/iris_connector_jdbc.py
from jdbc_exploration.iris_jdbc_connector import get_iris_jdbc_connection

def get_iris_connection():
    """Drop-in replacement for ODBC connector using JDBC"""
    return get_iris_jdbc_connection()
```

### Step 2: Update imports in benchmark

```python
# In eval/enterprise_rag_benchmark_final.py
# Change:
from common.iris_connector import get_iris_connection
# To:
from common.iris_connector_jdbc import get_iris_connection
```

### Step 3: Ensure JDBC driver is available

```bash
# Download JDBC driver to project root
curl -L -o intersystems-jdbc-3.8.4.jar \
  https://github.com/intersystems-community/iris-driver-distribution/raw/main/JDBC/JDK18/intersystems-jdbc-3.8.4.jar
```

## Option 2: Full Migration (Better Performance)

### Step 1: Create JDBC versions of all pipelines

Already created:
- `basic_rag/pipeline_jdbc_v2.py` - Uses V2 tables with HNSW indexes

Need to create:
- `hyde/pipeline_jdbc_v2.py`
- `crag/pipeline_jdbc_v2.py`
- `colbert/pipeline_jdbc_v2.py`
- `noderag/pipeline_jdbc_v2.py`
- `graphrag/pipeline_jdbc_v2.py`
- `hybrid_ifind_rag/pipeline_jdbc_v2.py`

### Step 2: Update benchmark to use JDBC pipelines

```python
# eval/enterprise_rag_benchmark_jdbc.py
from basic_rag.pipeline_jdbc_v2 import BasicRAGPipelineJDBCV2
from hyde.pipeline_jdbc_v2 import HyDEPipelineJDBCV2
# ... etc
```

## Considerations for Benchmarking

### 1. JVM Startup Overhead

JDBC has a one-time JVM startup cost (~1-2 seconds). For fair benchmarking:

```python
# Pre-warm JVM before benchmarking
import jpype
if not jpype.isJVMStarted():
    jpype.startJVM(jpype.getDefaultJVMPath(), 
                  "-Djava.class.path=./intersystems-jdbc-3.8.4.jar")

# Now run benchmarks - JVM already started
```

### 2. Connection Pooling

For multiple benchmark runs, reuse connections:

```python
# Create single connection for all pipelines
jdbc_conn = get_iris_jdbc_connection()

# Pass to all pipelines
pipeline1 = BasicRAGPipelineJDBCV2(iris_connector=jdbc_conn, ...)
pipeline2 = HyDEPipelineJDBCV2(iris_connector=jdbc_conn, ...)
```

### 3. Use V2 Tables for Better Performance

V2 tables have HNSW indexes. Update queries to use:
- `RAG.SourceDocuments_V2` instead of `RAG.SourceDocuments`
- `RAG.DocumentChunks_V2` instead of `RAG.DocumentChunks`

### 4. Environment Setup

```bash
# Install JDBC dependencies in benchmark environment
pip install jaydebeapi jpype1

# Set environment variables
export IRIS_JDBC_DRIVER_PATH=./intersystems-jdbc-3.8.4.jar
```

## Quick Test Script

```python
# test_jdbc_benchmark.py
import time
from jdbc_exploration.iris_jdbc_connector import get_iris_jdbc_connection
from common.utils import get_embedding_func

# Test JDBC performance
conn = get_iris_jdbc_connection()
embedding_func = get_embedding_func()

# Generate test embedding
test_query = "diabetes symptoms"
query_embedding = embedding_func([test_query])[0]
query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])

# Benchmark query
start = time.time()
results = conn.execute("""
    SELECT TOP 5 doc_id, title,
           VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity
    FROM RAG.SourceDocuments_V2
    WHERE embedding IS NOT NULL
    ORDER BY similarity DESC
""", [query_embedding_str])
end = time.time()

print(f"JDBC query time: {end - start:.3f}s")
print(f"Results: {len(results)}")
```

## Expected Performance Impact

1. **Connection**: +1-2s for first connection (JVM startup)
2. **Queries**: Similar or slightly better than ODBC
3. **Safety**: Eliminates SQL injection risk
4. **V2 Tables**: Potential speedup from HNSW indexes

## Recommendation

For benchmarking:
1. Use Option 1 (minimal change) for quick testing
2. Pre-warm JVM before benchmark runs
3. Compare ODBC vs JDBC performance
4. If JDBC shows benefits, proceed with full migration

## Notes

- JDBC requires Java Runtime Environment
- First connection is slower due to JVM startup
- Subsequent queries perform similarly to ODBC
- Main benefit is parameter binding safety, not necessarily speed