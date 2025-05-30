# JDBC Exploration for IRIS Vector Support

## Overview

This directory contains proof-of-concept code for migrating from ODBC to JDBC to solve the vector function parameter binding issues in IRIS.

## Background

The IRIS Python driver (via ODBC) has several limitations:
1. **No parameter binding support** for vector functions (TO_VECTOR, VECTOR_COSINE)
2. **No native VECTOR data type support** in the Python driver
3. **SQL injection risks** due to required string concatenation

Since InterSystems has confirmed that DBAPI and Vector support won't be available in the Python driver anytime soon, and that JDBC is the only officially supported interface for vector operations, we need to explore using JDBC from Python.

## Solution: JayDeBeAPI

JayDeBeAPI allows Python to use JDBC drivers through the Java Native Interface (JNI). This gives us:
- ✅ Full parameter binding support for vector functions
- ✅ Better performance for vector operations
- ✅ Safer SQL queries (no string concatenation)
- ✅ Future-proof solution (JDBC is officially supported)

## Files in this Directory

1. **`test_jaydebeapi_connection.py`** - Basic connection test and vector operation validation
2. **`iris_jdbc_connector.py`** - JDBC connector wrapper that mimics the ODBC interface
3. **`requirements.txt`** - Python dependencies (jaydebeapi, jpype1)
4. **`JDBC_MIGRATION_PLAN.md`** - Detailed migration plan from ODBC to JDBC

## Quick Start

### 1. Install Dependencies

```bash
pip install jaydebeapi jpype1
```

### 2. Download IRIS JDBC Driver

Download the IRIS JDBC driver JAR file:
- From: https://github.com/intersystems-community/iris-driver-distribution
- Current version: `intersystems-jdbc-3.8.4.jar`

### 3. Set Environment Variables

```bash
export IRIS_HOST=localhost
export IRIS_PORT=1972
export IRIS_NAMESPACE=RAG
export IRIS_USERNAME=demo
export IRIS_PASSWORD=demo
export IRIS_JDBC_DRIVER_PATH=/path/to/intersystems-jdbc-3.8.4.jar
```

### 4. Test Connection

```bash
python jdbc_exploration/test_jaydebeapi_connection.py
```

### 5. Test JDBC Connector

```bash
python jdbc_exploration/iris_jdbc_connector.py
```

## Usage Example

```python
from jdbc_exploration.iris_jdbc_connector import get_iris_jdbc_connection

# Connect to IRIS via JDBC
conn = get_iris_jdbc_connection()

# Execute vector search with safe parameter binding
query_embedding = "0.1,0.2,0.3,..."  # 384 dimensions
results = conn.execute("""
    SELECT TOP ? doc_id, title,
           VECTOR_COSINE(
               TO_VECTOR(embedding, 'DOUBLE', 384),
               TO_VECTOR(?, 'DOUBLE', 384)
           ) as similarity
    FROM RAG.SourceDocuments
    WHERE embedding IS NOT NULL
    ORDER BY similarity DESC
""", [5, query_embedding])

# Process results
for doc_id, title, similarity in results:
    print(f"{doc_id}: {title} (similarity: {similarity})")
```

## Performance Comparison

Initial tests show:
- **Parameter Binding**: ✅ Works (vs ❌ broken in ODBC)
- **Query Performance**: Similar to ODBC
- **Connection Overhead**: +1-2 seconds for JVM startup
- **Memory Usage**: Higher due to JVM (monitor in production)

## Migration Status

- [x] Created JDBC connector wrapper
- [x] Tested vector operations
- [x] Created BasicRAG JDBC pipeline proof-of-concept
- [ ] Benchmark JDBC vs ODBC performance
- [ ] Migrate all 7 RAG pipelines
- [ ] Update data loaders and chunking services

## Next Steps

1. **Download JDBC driver** and test in your environment
2. **Run BasicRAG JDBC pipeline**: `python basic_rag/pipeline_jdbc.py`
3. **Compare performance** with ODBC version
4. **Decide on full migration** based on results

## Troubleshooting

### JVM Not Found
```
Error: Unable to find Java
Solution: Install Java JRE 8 or higher
```

### JDBC Driver Not Found
```
Error: IRIS JDBC driver not found
Solution: Download intersystems-jdbc-3.8.4.jar and update IRIS_JDBC_DRIVER_PATH
```

### Connection Failed
```
Error: Connection refused
Solution: Check IRIS is running and connection parameters are correct
```

## References

- [JayDeBeAPI Documentation](https://pypi.org/project/JayDeBeAPI/)
- [IRIS JDBC Documentation](https://docs.intersystems.com/iris/csp/docbook/DocBook.UI.Page.cls?KEY=BJAVA_jdbc)
- [IRIS Vector Search Documentation](https://docs.intersystems.com/iris/csp/docbook/DocBook.UI.Page.cls?KEY=GSQL_vecsearch)