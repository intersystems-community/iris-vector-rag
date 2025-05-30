# IRIS Python Driver Parameter Parsing Bug Report

## Issue Summary
The IRIS Python driver automatically converts SQL queries to use parameter placeholders (`:qpar`), even when the SQL is built as a complete string using f-strings. This prevents the use of vector search functions with embedded vector strings.

## Environment
- IRIS Version: 2025.1.0
- Python Driver: iris-python-3.10.1
- OS: macOS Sequoia

## Bug Description

### Expected Behavior
When executing SQL with f-strings that contain complete vector data:
```python
sql = f"""
    SELECT TOP 3 doc_id, title, text_content,
           VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)) as similarity_score
    FROM RAG.SourceDocuments_V2
    WHERE document_embedding_vector IS NOT NULL
    ORDER BY similarity_score DESC
"""
cursor.execute(sql)
```

The SQL should execute as-is without parameter conversion.

### Actual Behavior
The driver converts the SQL to use parameters, resulting in:
```
[SQLCODE: <-1>:<Invalid SQL statement>]
[Location: <Prepare>]
[%msg: < ) expected, : found ^SELECT TOP :%qpar(1) doc_id , title , text_content , VECTOR_COSINE ( document_embedding_vector , TO_VECTOR ( :%qpar(2) , :%qpar>]
```

### Root Cause
The IRIS Python driver appears to parse SQL strings and automatically convert patterns it recognizes as potential parameters, even when:
1. The SQL is built using f-strings with complete values
2. No parameters are passed to `cursor.execute()`
3. The embedded data contains special characters or patterns

## Impact
- Cannot use IRIS native vector search functions with HNSW indexes
- Forces fallback to inefficient manual similarity calculations
- Prevents leveraging IRIS vector search performance optimizations

## Reproduction Steps
1. Create a table with VECTOR column type
2. Build SQL query with embedded vector string using f-string
3. Execute query with `cursor.execute(sql)`
4. Observe parameter conversion error

## Workaround
Currently, the only workaround is to avoid vector search functions and use manual similarity calculations in Python, which is significantly slower.

## Suggested Fix
The driver should provide an option to disable automatic parameter parsing, such as:
```python
cursor.execute(sql, parse_params=False)
```

Or respect when no parameters are provided and execute the SQL as-is.

## Related Issues
- Scientific notation in embeddings (e.g., `1.23e-05`) may trigger parameter detection
- Large embedding strings (>5000 characters) may cause parsing issues
- The colon character in error messages suggests pattern matching on `:` characters