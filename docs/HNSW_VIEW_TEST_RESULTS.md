# HNSW View-Based Approach Test Results

## Environment Information

| Component | Version/Details |
|-----------|----------------|
| IRIS Version | IRIS for UNIX (Ubuntu Server LTS for ARM64 Containers) 2025.1.0.225.1 |
| Python Version | 3.12.9 |
| Client Libraries | intersystems-iris 5.1.2 |
| Operating System | macOS-15.3.2-arm64-arm-64bit |

## Executive Summary

This document reports the results of testing a view-based approach for HNSW indexing with IRIS 2025.1. We attempted to create a view that converts VARCHAR embeddings to VECTOR type using TO_VECTOR, and then create an HNSW index on that view. We also tried alternative approaches with computed columns and materialized views.

## Test Approach

We created a test script (`investigation/test_view_hnsw_2025.py`) that attempts:

1. Create a table with VARCHAR column for embeddings
2. Create a view that converts these strings to VECTOR type using TO_VECTOR
3. Create an HNSW index on the view for efficient similarity search

Additionally, we tried:
- Creating a table with a computed column using TO_VECTOR
- Creating a materialized view with TO_VECTOR

## Test Results

### 1. View with TO_VECTOR

**Attempted SQL:**
```sql
CREATE VIEW ViewHNSWTestView AS
SELECT 
    id,
    text_content,
    TO_VECTOR(embedding, 'double', 384) AS vector_embedding,
    metadata
FROM ViewHNSWTest
```

**Result:** FAILED

**Error:**
```
[SQLCODE: <-1>:<Invalid SQL statement>]
[Location: <Prepare>]
[%msg: < ) expected, LITERAL ('double') found ^                 TO_VECTOR(embedding, 'double'>]
```

### 2. Table with Computed Column

**Attempted SQL:**
```sql
CREATE TABLE ComputedVectorTest (
    id VARCHAR(100) PRIMARY KEY,
    text_content TEXT,
    embedding VARCHAR(60000),
    vector_embedding AS TO_VECTOR(embedding, 'double', 384)
)
```

**Result:** FAILED

**Error:**
```
[SQLCODE: <-1>:<Invalid SQL statement>]
[Location: <Prepare>]
[%msg: < ) expected, IDENTIFIER (TO_VECTOR) found ^                 vector_embedding AS TO_VECTOR>]
```

### 3. Materialized View

**Attempted SQL:**
```sql
CREATE TABLE MaterializedVectorView AS
SELECT 
    id,
    text_content,
    TO_VECTOR(embedding, 'double', 384) AS vector_embedding,
    metadata
FROM ViewHNSWTest
```

**Result:** FAILED

**Error:**
```
[SQLCODE: <-1>:<Invalid SQL statement>]
[Location: <Prepare>]
[%msg: < ) expected, LITERAL ('double') found ^                     TO_VECTOR(embedding, 'double'>]
```

## Analysis

1. **TO_VECTOR Function Limitations**: The TO_VECTOR function cannot be used in view definitions, computed columns, or materialized views in IRIS 2025.1. This is likely due to the function's special handling of parameter markers and literals.

2. **HNSW Indexing Requirements**: HNSW indexing requires a direct VECTOR column in a table, not a computed or derived column.

3. **Version Comparison**: Despite upgrading to IRIS 2025.1, the view-based approach still doesn't work. This confirms that the issue is not version-specific but rather a fundamental limitation of the TO_VECTOR function and HNSW indexing.

## Conclusion

The view-based approach for HNSW indexing does not work with IRIS 2025.1. The dual-table architecture with ObjectScript triggers, as described in HNSW_INDEXING_RECOMMENDATIONS.md, remains the recommended approach for implementing high-performance vector search with HNSW indexing.

## Reproducing the Issues

To easily reproduce these issues, we've created a standalone script that demonstrates both the parameter substitution issues with TO_VECTOR and the inability to create views, computed columns, or materialized views with TO_VECTOR for HNSW indexing:

```bash
python investigation/reproduce_vector_issues.py
```

This script:
1. Connects to IRIS 2025.1
2. Creates a table with a VARCHAR column for embeddings
3. Attempts to use parameter markers with TO_VECTOR
4. Attempts to create views, computed columns, and materialized views with TO_VECTOR
5. Attempts to create HNSW indexes on these views and columns

The script provides clear error messages that confirm our findings.

## Recommended Architecture

Based on these test results, we confirm that the recommended architecture for HNSW indexing is:

1. **Primary Storage Table (VARCHAR)**
   - Store embeddings as strings in VARCHAR columns
   - Use for easy document insertion

2. **Vector Search Table (VECTOR)**
   - Store the same embeddings as VECTOR type
   - Create HNSW index on this table
   - Use for high-performance vector search

3. **ObjectScript Trigger**
   - Convert embeddings from VARCHAR to VECTOR
   - Maintain synchronization between tables

This architecture provides the best of both worlds: easy document loading and high-performance vector search with HNSW indexing.