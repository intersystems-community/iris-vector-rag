# JDBC Migration Plan for IRIS Vector Support

## Overview

Since IRIS DBAPI and native Vector support are not available in the Python driver, and only JDBC is officially supported for vector operations, we need to migrate from ODBC to JDBC using JayDeBeAPI.

## Current Issues with ODBC

1. **Parameter Binding Broken**: Vector functions don't work with parameter binding in ODBC
2. **No Native VECTOR Type Support**: Python driver lacks VECTOR data type support
3. **Complex Workarounds Required**: String concatenation creates SQL injection risks
4. **Performance Limitations**: Can't leverage HNSW indexes effectively

## JDBC Solution Benefits

1. **Full Vector Support**: JDBC driver has complete vector function support
2. **Parameter Binding Works**: Can use prepared statements with vector functions
3. **Better Performance**: Direct Java integration may offer better performance
4. **Future-Proof**: JDBC is the officially supported path for IRIS

## Migration Steps

### 1. Install Dependencies

```bash
pip install jaydebeapi jpype1
```

### 2. Download IRIS JDBC Driver

Download the latest IRIS JDBC driver JAR from:
- https://github.com/intersystems-community/iris-driver-distribution
- Or from WRC (Worldwide Response Center)

Current version: `intersystems-jdbc-3.8.4.jar`

### 3. Update Connection Code

Replace ODBC connections:

```python
# OLD: ODBC
from common.iris_connector import get_iris_connection
conn = get_iris_connection()

# NEW: JDBC
from jdbc_exploration.iris_jdbc_connector import get_iris_jdbc_connection
conn = get_iris_jdbc_connection()
```

### 4. Update SQL Queries

JDBC supports parameter binding with vectors:

```python
# OLD: String concatenation (SQL injection risk!)
sql = f"""
    SELECT VECTOR_COSINE(
        TO_VECTOR(embedding, 'DOUBLE', 384),
        TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)
    ) as similarity
    FROM RAG.SourceDocuments
"""
cursor.execute(sql)

# NEW: Safe parameter binding
sql = """
    SELECT VECTOR_COSINE(
        TO_VECTOR(embedding, 'DOUBLE', 384),
        TO_VECTOR(?, 'DOUBLE', 384)
    ) as similarity
    FROM RAG.SourceDocuments
"""
cursor.execute(sql, [query_embedding_str])
```

### 5. Update Pipeline Code

Example migration for BasicRAG pipeline:

```python
# basic_rag/pipeline_jdbc.py
from jdbc_exploration.iris_jdbc_connector import get_iris_jdbc_connection

class BasicRAGPipelineJDBC:
    def __init__(self):
        self.conn = get_iris_jdbc_connection()
    
    def search(self, query_embedding: str, top_k: int = 5):
        sql = """
            SELECT TOP ? 
                doc_id, 
                title, 
                text_content,
                VECTOR_COSINE(
                    TO_VECTOR(embedding, 'DOUBLE', 384),
                    TO_VECTOR(?, 'DOUBLE', 384)
                ) as similarity_score
            FROM RAG.SourceDocuments
            WHERE embedding IS NOT NULL
            ORDER BY similarity_score DESC
        """
        
        results = self.conn.execute(sql, [top_k, query_embedding])
        return results
```

## Implementation Priority

1. **Phase 1: Proof of Concept**
   - [x] Create JDBC connector wrapper
   - [ ] Test vector operations
   - [ ] Benchmark JDBC vs ODBC performance

2. **Phase 2: Pipeline Migration**
   - [ ] Migrate BasicRAG to JDBC
   - [ ] Test with real queries
   - [ ] Validate results match ODBC version

3. **Phase 3: Full Migration**
   - [ ] Update all 7 RAG pipelines
   - [ ] Update data loaders
   - [ ] Update chunking services
   - [ ] Run comprehensive tests

4. **Phase 4: Optimization**
   - [ ] Add connection pooling
   - [ ] Optimize batch operations
   - [ ] Implement retry logic

## Configuration

Set environment variables:

```bash
export IRIS_HOST=localhost
export IRIS_PORT=1972
export IRIS_NAMESPACE=RAG
export IRIS_USERNAME=demo
export IRIS_PASSWORD=demo
export IRIS_JDBC_DRIVER_PATH=/path/to/intersystems-jdbc-3.8.4.jar
```

## Testing

Run the test suite:

```bash
# Test JDBC connection
python jdbc_exploration/test_jaydebeapi_connection.py

# Test JDBC connector wrapper
python jdbc_exploration/iris_jdbc_connector.py

# Run pipeline tests
pytest tests/test_jdbc_pipelines.py
```

## Performance Comparison

Expected improvements:
- **Parameter Binding**: 100% success rate (vs 0% with ODBC)
- **Query Performance**: Similar or better than ODBC
- **Connection Overhead**: Slightly higher due to JVM startup
- **Memory Usage**: Higher due to JVM, but manageable

## Rollback Plan

If JDBC migration fails:
1. Keep ODBC code in parallel
2. Use feature flags to switch between ODBC/JDBC
3. Gradual rollout per pipeline

## Next Steps

1. **Immediate**: Download JDBC driver and test connection
2. **This Week**: Migrate BasicRAG pipeline as proof of concept
3. **Next Week**: Full migration if POC successful

## Notes

- JDBC requires Java Runtime Environment (JRE)
- JVM startup adds ~1-2 seconds to first connection
- Connection pooling recommended for production
- Monitor memory usage due to JVM overhead

## References

- [JayDeBeAPI Documentation](https://pypi.org/project/JayDeBeAPI/)
- [JPype Documentation](https://jpype.readthedocs.io/)
- [IRIS JDBC Documentation](https://docs.intersystems.com/iris/csp/docbook/DocBook.UI.Page.cls?KEY=BJAVA_jdbc)