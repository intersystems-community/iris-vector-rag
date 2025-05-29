# IRIS Vector Search Colon Bug Report

## Executive Summary

IRIS SQL parser has a critical bug when processing vector search queries with the `TO_VECTOR()` function. The parser incorrectly interprets certain patterns within the vector string parameter as SQL parameter markers, causing query parsing failures. This affects all RAG pipelines attempting to use IRIS native vector search capabilities.

## Bug Details

### Environment
- IRIS Version: 2024.1+ (with vector search support)
- Python Driver: pyodbc
- Affected Feature: Vector search with `TO_VECTOR()` and `VECTOR_COSINE()` functions

### Issue Description

When executing vector similarity queries using `TO_VECTOR()`, IRIS SQL parser fails with the error:
```
[SQLCODE: <-1>:<Invalid SQL statement>]
[Location: <Prepare>]
[%msg: < ) expected, : found ^SELECT TOP :%qpar(1) doc_id , VECTOR_COSINE ( document_embedding_vector , TO_VECTOR ( :%qpar(2) , :%qpar>]
```

This occurs even when:
1. The embedding string contains no colons
2. Using parameterized queries with proper placeholders
3. Using direct string interpolation

### Root Cause Analysis

The IRIS SQL parser appears to have a bug in how it processes the `TO_VECTOR()` function parameters. It seems to be:
1. Incorrectly tokenizing the content within the TO_VECTOR string parameter
2. Misinterpreting certain numeric patterns (possibly scientific notation) as parameter markers
3. Converting the query internally to use `:%qpar` notation incorrectly

### Reproduction

```python
# This query fails even though embedding_str contains no colons
sql_query = f"""
    SELECT TOP 5 doc_id,
    VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{embedding_str}', 'DOUBLE')) AS similarity_score
    FROM RAG.SourceDocuments_V2
    WHERE document_embedding_vector IS NOT NULL
    ORDER BY similarity_score DESC
"""
cursor.execute(sql_query)  # Fails with colon error
```

### Impact

This bug prevents the use of IRIS native vector search capabilities, forcing developers to:
1. Use inefficient workarounds (e.g., BasicRAG's approach of loading all embeddings into memory)
2. Implement custom similarity calculations in application code
3. Lose the performance benefits of IRIS HNSW indexes

## Current Workarounds

### 1. BasicRAG Approach (Memory-Intensive)
```python
# Load documents into memory and calculate similarity in Python
sql = "SELECT TOP 100 doc_id, title, text_content, embedding FROM RAG.SourceDocuments"
cursor.execute(sql)
docs = cursor.fetchall()
# Calculate cosine similarity in Python
```

### 2. Stored Procedure Approach (Not Working)
Attempted to use stored procedures to bypass the parser, but the same error occurs within the procedure.

### 3. Temporary Table Approach (Not Tested)
Could potentially store query vectors in temporary tables, but this adds complexity and overhead.

## Recommended Solution

IRIS development team needs to:
1. Fix the SQL parser to correctly handle TO_VECTOR() parameters
2. Ensure parameter binding works correctly with vector functions
3. Add comprehensive test cases for vector search with various numeric formats

## Test Results

From `test_iris_vector_colon_bug.py`:
- Embedding dimensions: 384
- Embedding string length: 8054 characters
- Contains scientific notation: Yes (4 instances)
- Contains colons: No
- Parameterized query: Failed
- Direct interpolation: Failed
- Both fail with identical "colon found" error

## Business Impact

This bug is blocking the deployment of efficient RAG systems using IRIS vector search. All current implementations must use inefficient workarounds that:
- Increase memory usage
- Reduce query performance
- Cannot leverage HNSW indexes
- Limit scalability

## Severity: CRITICAL

This is a critical bug that makes the advertised vector search feature unusable in production environments.