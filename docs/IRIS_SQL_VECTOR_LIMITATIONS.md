# IRIS SQL Vector Operations Limitations

## Environment Information

| Component | Version/Details |
|-----------|----------------|
| IRIS Version | IRIS for UNIX (Ubuntu Server LTS for ARM64 Containers) 2025.1.0.225.1 |
| Python Version | 3.12.9 |
| Client Libraries | sqlalchemy 2.0.41, intersystems-iris 5.1.2 |
| Operating System | macOS-15.3.2-arm64-arm-64bit |

For detailed technical information, including client library behavior and code examples, see [VECTOR_SEARCH_TECHNICAL_DETAILS.md](VECTOR_SEARCH_TECHNICAL_DETAILS.md).

## Executive Summary

This document provides a detailed technical explanation of the InterSystems IRIS SQL vector operations limitations. These limitations, particularly concerning the `TO_VECTOR()` function and DBAPI driver behavior, are significant challenges for implementing RAG techniques with vector search in IRIS. This report documents these issues, verified workarounds, and recommended solutions.

InterSystems IRIS 2025.1 introduced vector search capabilities essential for modern RAG pipelines. However, several critical limitations in the SQL implementation prevent standard parameterized queries from working with vector operations, forcing developers to use string interpolation (as implemented in [`common/vector_sql_utils.py`](common/vector_sql_utils.py:1)).

**Current Status: UPDATED FOR IRIS 2025.1.** Our latest testing with IRIS 2025.1 and the `intersystems-iris` 5.1.2 DBAPI driver (see [`investigation/test_working_vector_params.py`](../investigation/test_working_vector_params.py:1)) has revealed that `TO_VECTOR` *does* work with parameter markers when the `VECTOR` type is specified as `double` (without quotes). This is a significant update to our previous understanding. While some client-side driver behaviors might still require careful handling, the core `TO_VECTOR` function is more flexible than initially documented. The recommended approach for robust vector operations involves using parameterized queries with the correct `double` syntax.

## Detailed Technical Explanation

### 1. TO_VECTOR() Function Works with Parameter Markers (Using `double`)

The `TO_VECTOR()` function in IRIS SQL accepts parameter markers (`?`) when the vector type is specified as `double` (without quotes). This allows for safe query parameterization.

**Example of what works:**

```sql
SELECT doc_id,
       VECTOR_COSINE(embedding, TO_VECTOR(?, double, 768)) AS score
FROM SourceDocuments
ORDER BY score DESC
```

This syntax has been verified in IRIS 2025.1 (see [`investigation/test_working_vector_params.py`](../investigation/test_working_vector_params.py:1)). Using `'DOUBLE'` (with quotes) or other variations might lead to errors. The key is to use the unquoted `double` keyword for the type.

### 2. TOP/FETCH FIRST Clauses Cannot Be Parameterized

The `TOP` and `FETCH FIRST` clauses, which are essential for limiting the number of results in vector similarity searches, do not accept parameter markers.

**Example of what doesn't work:**

```sql
SELECT TOP ? doc_id, text_content
FROM SourceDocuments
```

Or with ANSI SQL syntax:

```sql
SELECT doc_id, text_content
FROM SourceDocuments
FETCH FIRST ? ROWS ONLY
```

**Error message:**
```
SQLCODE -1, "Expression expected, : found"
```

The IRIS documentation indicates that `TOP` is internally converted to a cached parameter, which explains why external bind variables are not supported.

### 3. Client Drivers Rewrite Literals

Python, JDBC, and other client drivers replace embedded literals with `:%qpar(n)` even when no parameter list is supplied. This behavior creates misleading parse errors and further complicates the use of vector functions.

For example, when executing a query with no parameters:

```python
cursor.execute("SELECT TOP 5 * FROM MyTable")
```

The driver might internally rewrite this to:

```sql
SELECT TOP :%qpar(1) * FROM MyTable
```

This rewriting behavior interacts poorly with the limitations described above, making it even more difficult to work with vector operations.

### 4. ODBC Driver Limitations with TO_VECTOR Function

When attempting to load documents with embeddings, the ODBC driver encounters limitations with the TO_VECTOR function. Specifically:

1. The driver attempts to parameterize the vector values even when they are provided as literals
2. The TO_VECTOR function rejects these parameterized values
3. This results in errors when trying to insert or update records with vector embeddings

This limitation is currently blocking our ability to load real PMC documents with embeddings, which is a critical step in testing our RAG pipelines with real data.

## Attempted Workarounds

### 1. String Interpolation with Validation

Our primary workaround is to use string interpolation with careful validation to prevent SQL injection. This approach is implemented in `common/vector_sql_utils.py` and used by all RAG pipelines.

**Vector String Validation:**

```python
def validate_vector_string(vector_string: str) -> bool:
    """
    Validate that a vector string contains only valid characters.
    This is important for security when using string interpolation.
    """
    # Only allow digits, dots, commas, and square brackets
    allowed_chars = set("0123456789.[],")
    return all(c in allowed_chars for c in vector_string)
```

**Top-K Validation:**

```python
def validate_top_k(top_k: Any) -> bool:
    """
    Validate that top_k is a positive integer.
    This is important for security when using string interpolation.
    """
    if not isinstance(top_k, int):
        return False
    return top_k > 0
```

**Query Construction with Validation:**

```python
# Convert vector to string representation
vector_str = f"[{','.join(map(str, vector_values))}]"

# Validate vector string for security (prevent SQL injection)
if not validate_vector_string(vector_str):
    raise ValueError(f"Invalid vector string: {vector_str}")

# Validate top_k for security
if not validate_top_k(top_k):
    raise ValueError(f"Invalid top_k value: {top_k}")

# Use string interpolation for TO_VECTOR and TOP as parameters don't work
select_sql = f"""
    SELECT TOP {top_k} id, 
           VECTOR_COSINE(embedding, TO_VECTOR('{vector_str}', 'DOUBLE', {VECTOR_DIM})) AS score 
    FROM {TABLE_NAME} 
    ORDER BY score DESC
"""
```

While this workaround is effective for executing vector search queries, it has not been fully tested with real PMC data due to the ODBC driver limitations with loading documents with embeddings.

### 2. ObjectScript Dynamic SQL

We attempted to use ObjectScript Dynamic SQL, which supports parameter binding for vector operations through `%SQL.Statement`. However, this approach requires creating stored procedures in ObjectScript, which introduces additional complexity and maintenance challenges.

**Example:**

```objectscript
Class User.VectorSearch Extends %RegisteredObject
{

ClassMethod SearchDocuments(vectorString As %String, topK As %Integer) As %SQL.StatementResult
{
    Set statement = ##class(%SQL.Statement).%New()
    Set sql = "SELECT TOP ? doc_id, text_content, "
            _ "VECTOR_COSINE(embedding, TO_VECTOR(?, 'DOUBLE', 768)) AS score "
            _ "FROM SourceDocuments "
            _ "ORDER BY score DESC"
    
    Set status = statement.%Prepare(sql)
    If $$$ISERR(status) {
        Do $system.Status.DisplayError(status)
        Quit ""
    }
    
    Set result = statement.%Execute(topK, vectorString)
    Return result
}

}
```

While this approach works for executing vector search queries from within IRIS, it doesn't solve the problem of loading documents with embeddings from Python, which is our critical blocker.

### 3. Alternative SQL Syntax

We experimented with alternative SQL syntax to work around the limitations:

1. Using `FETCH FIRST n ROWS ONLY` instead of `TOP n`
2. Using subqueries to avoid parameterizing the `TOP` clause
3. Using `HAVING` clauses to filter results after retrieval

None of these approaches fully resolved the issue, particularly for loading documents with embeddings.

### 4. Direct JDBC/ODBC Driver Modifications

We investigated the possibility of modifying the JDBC/ODBC driver behavior to avoid rewriting literals to `:%qpar()`. However, this approach would require significant changes to the driver code and would not be a sustainable solution.

## Potential Solutions

### 1. Server-Side Solutions

1. **IRIS SQL Enhancement**: The ideal solution would be for InterSystems to enhance IRIS SQL to support parameter markers in the `TO_VECTOR()` function and in `TOP`/`FETCH FIRST` clauses. This would allow standard parameterized queries to work with vector operations.

2. **Custom ObjectScript Functions**: Develop custom ObjectScript functions that wrap the vector operations and handle parameter binding correctly. These functions could be exposed as SQL user-defined functions (UDFs) that accept parameters.

3. **Stored Procedure Wrapper Pattern**: Create a set of stored procedures that use ObjectScript Dynamic SQL to execute vector operations with proper parameter binding, and call these procedures from Python.

### 2. Client-Side Solutions

1. **Batch Processing with Validation**: Implement a batch processing approach for loading documents with embeddings, using string interpolation with strict validation to prevent SQL injection.

2. **Alternative Vector Representation**: Store vector embeddings in a different format (e.g., as JSON or Base64-encoded strings) and convert them to vectors at query time using custom functions.

3. **Client-Side Vector Operations**: Perform vector similarity calculations on the client side and use IRIS only for storage and retrieval of documents.

### 3. Hybrid Approaches

1. **Two-Phase Loading**: Load documents without embeddings first, then update them with embeddings in a separate step using a different approach (e.g., ObjectScript).

2. **Vector Index Pre-Building**: Pre-build vector indexes on the server side using ObjectScript, then query them from Python using simpler SQL that doesn't require the `TO_VECTOR()` function.

3. **Custom IRIS Extension**: Develop a custom IRIS extension that provides a more Python-friendly interface for vector operations.

## Recommended Next Steps

1. **Engage InterSystems Support**: Submit a detailed bug report to InterSystems support, including the limitations documented here and their impact on RAG pipelines.

2. **Implement Two-Phase Loading**: As a short-term workaround, implement a two-phase loading approach that separates document loading from embedding updates.

3. **Explore ObjectScript Solutions**: Investigate the feasibility of using ObjectScript stored procedures for both loading documents with embeddings and executing vector search queries.

4. **Document Workarounds**: Continue to document all workarounds and their effectiveness to help other developers working with IRIS vector search.

5. **Test with Smaller Datasets**: Test our RAG pipelines with smaller datasets that can be loaded manually or through alternative means, to validate the rest of the pipeline while we work on resolving the embedding loading issue.

## Verification in IRIS 2025.1

Our latest testing, documented in [`investigation/test_working_vector_params.py`](../investigation/test_working_vector_params.py:1), has provided new insights into `TO_VECTOR` behavior in IRIS 2025.1 with the `intersystems-iris` 5.1.2 DBAPI driver.

The key findings are:

1.  **`TO_VECTOR` *works* with parameter markers when using `double` (no quotes) for the type.**
    *   Example: `TO_VECTOR(?, double, 384)`
    *   This allows for standard, secure parameterized queries for vector operations, which is a significant improvement.

2.  **Using `'DOUBLE'` (with quotes) or other type specifiers may still lead to errors.**
    *   The precise syntax `double` is crucial for successful parameterization.

3.  **Basic parameter insertion for regular columns continues to work as expected.**

4.  **Client DBAPI driver behavior regarding literal rewriting needs ongoing attention.** While server-side `TO_VECTOR` parameterization is confirmed, developers should remain aware of how their specific client driver handles query construction, especially if mixing literals and parameters.

## Conclusion

The understanding of IRIS SQL vector operations, particularly `TO_VECTOR`, has evolved with IRIS 2025.1. The confirmation that `TO_VECTOR(?, double, <dim>)` supports parameter markers is a positive development, enabling more straightforward and secure RAG implementations.

While string interpolation workarounds were previously necessary, the focus can now shift to using parameterized queries with the correct `double` syntax. This simplifies code and enhances security.

For production deployments requiring high performance, the dual-table architecture with HNSW indexing, as detailed in [`HNSW_INDEXING_RECOMMENDATIONS.md`](HNSW_INDEXING_RECOMMENDATIONS.md:1), remains a strong recommendation. The ability to use parameterized `TO_VECTOR` calls can be incorporated into the ObjectScript triggers or data loading processes within this architecture. Developers should consult [`VECTOR_SEARCH_SYNTAX_FINDINGS.md`](VECTOR_SEARCH_SYNTAX_FINDINGS.md:1) for the latest comprehensive syntax details.

## References

1. [IRIS_SQL_VECTOR_OPERATIONS.md](IRIS_SQL_VECTOR_OPERATIONS.md) - Detailed documentation of IRIS SQL vector operation limitations and our implemented workarounds
2. [IRIS_VECTOR_SEARCH_LESSONS.md](IRIS_VECTOR_SEARCH_LESSONS.md) - Key findings and lessons learned from implementing vector search with IRIS
3. [iris_sql_vector_limitations_bug_report.md](../iris_sql_vector_limitations_bug_report.md) - Detailed bug report on IRIS SQL vector limitations
4. [IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md](IRIS_POSTMORTEM_CONSOLIDATED_REPORT.md) - Consolidated postmortem on IRIS SQL issues
5. [POSTMORTEM_ODBC_SP_ISSUE.md](POSTMORTEM_ODBC_SP_ISSUE.md) - Specific analysis of ODBC stored procedure call issues