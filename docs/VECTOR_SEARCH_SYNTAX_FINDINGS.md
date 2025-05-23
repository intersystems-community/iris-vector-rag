# Vector Search Syntax Findings in IRIS 2025.1

## Environment Information

| Component | Version/Details |
|-----------|----------------|
| IRIS Version | IRIS for UNIX (Ubuntu Server LTS for ARM64 Containers) 2025.1.0.225.1 |
| Python Version | 3.12.9 |
| Client Libraries | intersystems-iris 5.1.2, sqlalchemy 2.0.41 |
| Operating System | macOS-15.3.2-arm64-arm-64bit |

## Executive Summary

This document provides updated findings on the correct syntax for using TO_VECTOR with parameter markers in IRIS 2025.1, based on feedback from a knowledgeable developer and verified through testing. We've discovered that parameter substitution with TO_VECTOR works with specific syntax changes, which significantly changes our understanding of the vector search capabilities in IRIS 2025.1.

## Key Findings

1. **Parameter Substitution Works with TO_VECTOR** when:
   - 'double' is used without quotes: `TO_VECTOR(?, double, 384)` instead of `TO_VECTOR(?, 'double', 384)`
   - Direct cursor execution uses ? parameters with a list
   - SQLAlchemy execution uses :var parameters with a dict
   - String interpolation uses text() function from SQLAlchemy

2. **View Creation Works** with TO_VECTOR when 'double' is used without quotes.

3. **Materialized View Creation** (CREATE TABLE AS SELECT) works with TO_VECTOR when 'double' is used without quotes.

4. **Computed Columns Still Don't Work**: `vector_embedding AS TO_VECTOR(embedding, double, 384)` fails.

5. **HNSW Indexing Syntax Issue**: The USING HNSW syntax doesn't work in our tests. The error is:
   ```
   [SQLCODE: <-25>:<Input encountered after end of query>]
   [Location: <Prepare>]
   [%msg: < Input (USING) encountered after end of query^                 CREATE INDEX idx_working_vector_view ON WorkingVectorView (vector_embedding) USING>]
   ```

## Working Examples

### 1. Direct Cursor with ? Parameters

```python
query_sql = """
SELECT TOP ? VECTOR_COSINE(
    TO_VECTOR(?, double, 384),
    TO_VECTOR(?, double, 384)
) AS score
"""
cursor.execute(query_sql, [5, embedding_str, embedding_str])
result = cursor.fetchone()
```

### 2. SQLAlchemy with Named Parameters

```python
from sqlalchemy import create_engine, text

engine = create_engine("iris://superuser:SYS@localhost:1972/USER")
query_sql = """
SELECT TOP :top VECTOR_COSINE(
    TO_VECTOR(:vector1, double, 384),
    TO_VECTOR(:vector2, double, 384)
) AS score
"""
with engine.connect() as conn:
    with conn.begin():
        results = conn.execute(text(query_sql), {"vector1": embedding_str, "vector2": embedding_str, "top": 5})
        result = results.fetchone()
```

### 3. String Interpolation with text()

```python
from sqlalchemy import text

query_sql = text(f"""
SELECT VECTOR_COSINE(
    TO_VECTOR('{embedding_str}', double, 384),
    TO_VECTOR('{embedding_str}', double, 384)
) AS score
""")
cursor.execute(query_sql)
result = cursor.fetchone()
```

### 4. View Creation

```sql
CREATE VIEW WorkingVectorView AS
SELECT 
    id,
    text_content,
    TO_VECTOR(embedding, double, 384) AS vector_embedding
FROM WorkingVectorParamsTest
```

### 5. Materialized View Creation

```sql
CREATE TABLE MaterializedVectorViewWorking AS
SELECT 
    id,
    text_content,
    TO_VECTOR(embedding, double, 384) AS vector_embedding
FROM WorkingVectorParamsTest
```

## Comprehensive Testing Results

We conducted extensive testing of various approaches to implement vector search with HNSW indexing in IRIS 2025.1. Here's a summary of our findings:

### What Works

1. **Parameter Substitution with TO_VECTOR**: We can use parameter markers with TO_VECTOR when:
   - 'double' is used without quotes: `TO_VECTOR(?, double, 384)` instead of `TO_VECTOR(?, 'double', 384)`
   - Direct cursor execution uses ? parameters with a list
   - SQLAlchemy execution uses :var parameters with a dict
   - String interpolation uses text() function from SQLAlchemy

2. **Direct Vector Search**: We can perform vector search without views by using TO_VECTOR directly in the query:
   ```sql
   SELECT
       id,
       text_content,
       VECTOR_COSINE(
           TO_VECTOR(embedding, double, 384),
           TO_VECTOR(?, double, 384)
       ) AS score
   FROM VectorTable
   ORDER BY score DESC
   ```

3. **View Creation**: We can create views with TO_VECTOR when 'double' is used without quotes.

4. **Materialized View Creation**: We can create materialized views (CREATE TABLE AS SELECT) with TO_VECTOR when 'double' is used without quotes.

### What Doesn't Work

1. **VECTOR Column Type**: We can't create a table with a VECTOR column directly:
   ```sql
   CREATE TABLE VectorTable (
       id VARCHAR(100) PRIMARY KEY,
       text_content TEXT,
       vector_embedding VECTOR(384, double)  -- This fails
   )
   ```
   Error: `Invalid VECTOR field definition: Vector type must be one of DECIMAL, DOUBLE, FLOAT, INT, INTEGER, STRING`

2. **Computed Columns with TO_VECTOR**: We can't create a computed column using TO_VECTOR:
   ```sql
   CREATE TABLE VectorTable (
       id VARCHAR(100) PRIMARY KEY,
       text_content TEXT,
       embedding VARCHAR(60000),
       vector_embedding AS TO_VECTOR(embedding, double, 384)  -- This fails
   )
   ```
   Error: `Invalid SQL statement: ) expected, IDENTIFIER (TO_VECTOR) found`

3. **HNSW Indexing**: We can't create an HNSW index on any column, including those created with TO_VECTOR in views or materialized views. We tried numerous syntax variations:
   ```sql
   CREATE INDEX idx_vector ON VectorTable (vector_embedding) USING HNSW
   CREATE INDEX idx_vector ON VectorTable (vector_embedding) TYPE HNSW
   CREATE HNSW INDEX idx_vector ON VectorTable (vector_embedding)
   CREATE INDEX idx_vector ON VectorTable (vector_embedding) PROPERTY HNSW
   CREATE INDEX idx_vector ON VectorTable (vector_embedding) VECTOR
   CREATE INDEX idx_vector ON VectorTable FOR COLUMN vector_embedding USING HNSW
   CREATE INDEX idx_vector ON VectorTable.vector_embedding HNSW
   ```
   All of these fail with various syntax errors.

4. **Vector Type Persistence**: Even when we create a materialized view with TO_VECTOR, the resulting column is not recognized as a VECTOR type for VECTOR_COSINE:
   ```
   Error: Vector function only takes vectors as inputs
   Argument #1 of vector function VECTOR_COSINE is not a vector.
   ```

## Implications for Our Implementation

These findings significantly change our understanding of vector search capabilities in IRIS 2025.1:

1. **Direct Query Approach is Viable**: We can use TO_VECTOR directly in queries with parameter substitution, which is secure and maintainable.

2. **HNSW Indexing Not Available in SQL**: We can't create HNSW indexes using SQL syntax, which means we can't use HNSW indexing for high-performance vector search directly.

3. **Dual-Table Architecture Still Required**: For high-performance vector search with HNSW indexing, we still need to use the dual-table architecture with ObjectScript triggers as described in `docs/HNSW_INDEXING_RECOMMENDATIONS.md`.

## Next Steps

1. **Update Vector SQL Utils**: Modify our `common/vector_sql_utils.py` to use the correct syntax for parameter substitution with TO_VECTOR.

2. **Implement Dual Approaches**:
   - For basic vector search (development/testing): Use direct TO_VECTOR in queries with parameter substitution.
   - For high-performance vector search (production): Use the dual-table architecture with ObjectScript triggers.

3. **Update Documentation**: Update our documentation to reflect these findings and provide clear guidance on the correct syntax for vector search in IRIS 2025.1.

## Conclusion

Our investigation has revealed that parameter substitution with TO_VECTOR is possible in IRIS 2025.1 with specific syntax changes, which improves our ability to work with vector search in IRIS. However, HNSW indexing is not available through SQL syntax, so we still need to use the dual-table architecture with ObjectScript triggers for high-performance vector search.

The key takeaways are:
- Use `double` without quotes instead of `'double'` with quotes
- Use the appropriate parameter style for your connection method (? for direct cursor, :var for SQLAlchemy)
- Use text() function from SQLAlchemy for string interpolation
- Use TOP instead of LIMIT for limiting results
- For high-performance vector search with HNSW indexing, use the dual-table architecture with ObjectScript triggers

These findings are documented in the test scripts in the `investigation/` directory, which demonstrate the working approaches and limitations.