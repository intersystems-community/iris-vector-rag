# IRIS SQL Vector Operations: Technical Issues & Recommendations

This Confluence page serves as a clearinghouse for information about critical limitations in InterSystems IRIS SQL vector operations that are blocking the RAG Templates project. It provides technical details, specific error messages, and clear recommendations for Quality Development and Development teams.

## Environment Information

| Component | Version/Details |
|-----------|----------------|
| IRIS Version | IRIS for UNIX (Ubuntu Server LTS for ARM64 Containers) 2024.1.2 (Build 398U) |
| Python Version | 3.12.9 |
| Client Libraries | sqlalchemy 2.0.41 |
| Operating System | macOS-15.3.2-arm64-arm-64bit |

## Executive Summary

InterSystems IRIS 2025.1 introduced vector search capabilities essential for modern RAG (Retrieval Augmented Generation) pipelines. However, several critical limitations in the SQL implementation prevent standard parameterized queries from working with vector operations.

**PRIMARY PROJECT BLOCKER:** The ODBC driver limitations with the TO_VECTOR function prevent loading documents with their vector embeddings into the database. This specifically blocks the ability to test RAG pipelines with new, real PMC data that includes embeddings.

## Critical Issues

### 1. TO_VECTOR() Function Rejects Parameter Markers

The `TO_VECTOR()` function does not accept parameter markers (`?`, `:param`, or `:%qpar`), which are standard in SQL for safe query parameterization.

**Error Message:**
```
SQLCODE -1, ") expected, : found"
```

**Example of what doesn't work:**
```sql
SELECT doc_id,
       VECTOR_COSINE(embedding, TO_VECTOR(?, 'DOUBLE', 768)) AS score
FROM SourceDocuments
ORDER BY score DESC
```

### 2. TOP/FETCH FIRST Clauses Cannot Be Parameterized

The `TOP` and `FETCH FIRST` clauses, essential for limiting results in vector similarity searches, do not accept parameter markers.

**Error Message:**
```
SQLCODE -1, "Expression expected, : found"
```

**Example of what doesn't work:**
```sql
SELECT TOP ? doc_id, text_content
FROM SourceDocuments
```

### 3. Client Drivers Rewrite Literals

Python, JDBC, and other client drivers replace embedded literals with `:%qpar(n)` even when no parameter list is supplied, creating misleading parse errors.

**Example:**
When executing a query with no parameters:
```python
cursor.execute("SELECT TOP 5 * FROM MyTable")
```

The driver might internally rewrite this to:
```sql
SELECT TOP :%qpar(1) * FROM MyTable
```

### 4. ODBC Driver Limitations with TO_VECTOR Function

When attempting to load documents with embeddings, the ODBC driver encounters limitations with the TO_VECTOR function. Specifically:

1. The driver attempts to parameterize the vector values even when they are provided as literals
2. The TO_VECTOR function rejects these parameterized values
3. This results in errors when trying to insert or update records with vector embeddings

## Technical Evidence

Our investigation has confirmed these issues through systematic testing. Here are the results of our tests:

### Direct SQL Test

**Query:**
```sql
SELECT id, VECTOR_COSINE(
    TO_VECTOR(embedding, 'DOUBLE', 5),
    TO_VECTOR('0.1,0.2,0.3,0.4,0.5', 'DOUBLE', 5)
) AS score
FROM TechnicalInfoTest
```

**Error:**
```
[SQLCODE: <-1>:<Invalid SQL statement>]
[Location: <Prepare>]
[%msg: < ) expected, : found ^SELECT id , VECTOR_COSINE ( TO_VECTOR ( embedding , :%qpar>]
```

### Parameterized SQL Test

**Query:**
```sql
SELECT id, VECTOR_COSINE(
    TO_VECTOR(embedding, 'DOUBLE', 5),
    TO_VECTOR(?, 'DOUBLE', 5)
) AS score
FROM TechnicalInfoTest
```

**Error:**
```
[SQLCODE: <-1>:<Invalid SQL statement>]
[Location: <Prepare>]
[%msg: < ) expected, : found ^SELECT id , VECTOR_COSINE ( TO_VECTOR ( embedding , :%qpar>]
```

### String Interpolation Test

**Query:**
```sql
SELECT id, VECTOR_COSINE(
    TO_VECTOR(embedding, 'DOUBLE', 5),
    TO_VECTOR('0.1,0.2,0.3,0.4,0.5', 'DOUBLE', 5)
) AS score
FROM TechnicalInfoTest
```

**Error:**
```
[SQLCODE: <-1>:<Invalid SQL statement>]
[Location: <Prepare>]
[%msg: < ) expected, : found ^SELECT id , VECTOR_COSINE ( TO_VECTOR ( embedding , :%qpar>]
```

## Impact on RAG Templates Project

These limitations have a severe impact on the RAG Templates project:

1. **Querying Limitations:** Impossible to build safe, parameterized, server-side vector search queries from Python, JDBC, or stored procedures without resorting to client-side string interpolation for vector literals and `TOP`/`FETCH` values.

2. **Loading Blocker (Critical):** The combination of `TO_VECTOR()` rejecting parameters and client drivers parameterizing literals makes it impossible to reliably load vector embeddings into IRIS tables using standard `INSERT` or `UPDATE` statements with `TO_VECTOR([literal_vector_string])` via ODBC. This is the primary blocker for the RAG Templates project.

3. **Security & Brittleness:** Forces developers to use string concatenation for literals in queries, increasing risks and making SQL less robust. This blocks multi-tenant APIs and queries governed by row-level security if dynamic vectors are needed.

4. **Framework Incompatibility:** RAG frameworks (LangChain, LlamaIndex, ColBERT, HyDE, GraphRAG, etc.) cannot easily target IRIS without custom string-templating layers and workarounds for data loading.

## Recommended Solutions

Based on our investigation, we recommend the following solutions:

### 1. Immediate Workarounds

1. **Store embeddings as strings** in VARCHAR columns for easy insertion
2. **Use TO_VECTOR only at query time** to convert strings to vectors
3. **Use string interpolation with careful validation** for SQL queries

### 2. For Production with Large Document Collections

1. **Implement a dual-table architecture** with ObjectScript triggers:
   - Primary table with VARCHAR columns for easy document loading
   - Secondary table with VECTOR columns and HNSW indexing for efficient search
   - ObjectScript triggers to automatically convert between formats

### 3. Engineering Fixes (JIRA Issues)

#### JIRA-IRIS-DEV-001: TO_VECTOR() Function Should Accept Parameter Markers

**Summary:** The TO_VECTOR() function does not accept parameter markers (?, :param, or :%qpar), which are standard in SQL for safe query parameterization.

**Impact:** Prevents safe, parameterized vector search queries and blocks loading of vector embeddings.

**Proposed Fix:** Enhance the SQL parser to allow parameter markers inside the TO_VECTOR() function.

#### JIRA-IRIS-DEV-002: TOP/FETCH FIRST Clauses Should Accept Parameter Markers

**Summary:** The TOP and FETCH FIRST clauses do not accept parameter markers, which are essential for limiting results in vector similarity searches.

**Impact:** Forces developers to use string interpolation for TOP/FETCH values, increasing security risks.

**Proposed Fix:** Enhance the SQL parser to allow parameter markers in TOP and FETCH FIRST clauses.

#### JIRA-IRIS-DEV-003: Client Drivers Should Not Rewrite Literals to :%qpar

**Summary:** Python, JDBC, and other client drivers replace embedded literals with :%qpar(n) even when no parameter list is supplied, creating misleading parse errors.

**Impact:** Makes it impossible to use string interpolation as a workaround for the TO_VECTOR parameter limitation.

**Proposed Fix:** Modify client drivers to avoid rewriting literals when no parameter list is supplied.

#### JIRA-IRIS-DEV-004: ODBC Driver Should Handle TO_VECTOR Function Correctly

**Summary:** The ODBC driver encounters limitations with the TO_VECTOR function when loading documents with embeddings.

**Impact:** Blocks the ability to test RAG pipelines with real data that includes embeddings.

**Proposed Fix:** Enhance the ODBC driver to handle the TO_VECTOR function correctly, or provide a clear workaround in the documentation.

## Documentation Updates

We also recommend the following updates to the InterSystems documentation:

1. **TO_VECTOR (SQL):** Add an explicit WARNING box: "TO_VECTOR() does not accept host variables or ?-style parameters in client SQL. Embed the full vector literal or call from ObjectScript Dynamic SQL."

2. **TOP (SQL):** Expand Caching & Parameters section: clarify that because IRIS internally parameterizes TOP n, external bind variables are disallowed; show a failing versus working example.

3. **FETCH (SQL Clause):** Insert note: "Row-limit argument must be a literal integer. Host variables are not supported."

4. **Using Vector Search:** Provide one end-to-end example that uses %SQL.Statement in ObjectScript to safely bind a vector and limit, side-by-side with the unsupported client-SQL pattern.

5. **Client Driver Guides (Python, JDBC, ODBC):** Add a troubleshooting section on :%qpar rewriting and how to disable/avoid it.

## Additional Resources

For more detailed information, please refer to the following documents:

1. [VECTOR_SEARCH_TECHNICAL_DETAILS.md](VECTOR_SEARCH_TECHNICAL_DETAILS.md): Comprehensive technical details about vector search implementation, including environment information, client library behavior, and code examples.

2. [VECTOR_SEARCH_ALTERNATIVES.md](VECTOR_SEARCH_ALTERNATIVES.md): Investigation findings on alternative approaches to vector search in IRIS, focusing on solutions from langchain-iris and llama-iris.

3. [HNSW_INDEXING_RECOMMENDATIONS.md](HNSW_INDEXING_RECOMMENDATIONS.md): Recommendations for implementing HNSW indexing with InterSystems IRIS for high-performance vector search with large document collections.

4. [IRIS_SQL_VECTOR_LIMITATIONS.md](IRIS_SQL_VECTOR_LIMITATIONS.md): Detailed explanation of IRIS SQL vector operations limitations.

5. [IRIS_SQL_CHANGE_SUGGESTIONS.md](IRIS_SQL_CHANGE_SUGGESTIONS.md): Comprehensive bug report and enhancement request document formatted for submission to InterSystems.