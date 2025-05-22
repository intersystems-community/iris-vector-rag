# Vector Search Technical Details

This document provides comprehensive technical details about our vector search implementation, including environment information, client library behavior, and code examples. This information is intended to help Quality Development and Development teams understand the specific technical context of our vector search limitations and solutions.

## Environment Information

| Component | Version/Details |
|-----------|----------------|
| IRIS Version | IRIS for UNIX (Ubuntu Server LTS for ARM64 Containers) 2024.1.2 (Build 398U) Thu Oct 3 2024 14:29:04 EDT |
| Python Version | 3.12.9 |
| Operating System | macOS-15.3.2-arm64-arm-64bit |
| Platform | macOS-15.3.2-arm64-arm-64bit |

## Client Library Versions

| Library | Version |
|---------|--------|
| sqlalchemy | 2.0.41 |
| pyodbc | Not installed in test environment |
| sqlalchemy-iris | Not installed in test environment |
| langchain-iris | 0.2.1 (from investigation) |
| llama-iris | 0.5.0 (from investigation) |

## ODBC Driver Behavior

Our investigation has revealed specific behavior patterns in how the ODBC driver handles vector operations:

### SQL Query Test Results

#### Direct SQL

**Success**: No

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

#### Parameterized SQL

**Success**: No

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

#### String Interpolation

**Success**: No

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

### Key Observations

1. **Automatic Parameterization**: The ODBC driver automatically parameterizes parts of SQL statements, even when using string interpolation. This is evident from the error message showing `:%qpar` in the SQL statement.

2. **TO_VECTOR Function Limitation**: The TO_VECTOR function does not accept parameter markers, as shown by the error message `< ) expected, : found`.

3. **Consistent Behavior**: All three approaches (Direct SQL, Parameterized SQL, and String Interpolation) fail with the same error, indicating this is a fundamental limitation of the ODBC driver.

## Client Library Comparison

### DBAPI (Python Standard)

The Python DBAPI implementation we're using shows consistent behavior with the ODBC driver limitations:

```python
# This fails with the same error
cursor.execute("""
    SELECT TOP 3 id, text_content, 
           VECTOR_COSINE(
               TO_VECTOR(embedding, 'DOUBLE', 384),
               TO_VECTOR(?, 'DOUBLE', 384)
           ) AS score
    FROM SourceDocuments
    ORDER BY score ASC
""", (query_embedding_str,))
```

### SQLAlchemy with sqlalchemy-iris

The langchain-iris repository uses SQLAlchemy with a custom dialect (sqlalchemy-iris) that appears to handle vector operations differently:

```python
# From langchain-iris
results: Sequence[Row] = (
    session.query(
        self.table,
        (
            self.distance_strategy(embedding).label("distance")
            if self.native_vector
            else self.table.c.embedding.func(
                self.distance_strategy, embedding
            ).label("distance")
        ),
    )
    .filter(filter_by)
    .order_by(asc("distance"))
    .limit(k)
    .all()
)
```

This approach uses SQLAlchemy's query building capabilities, which may handle the TO_VECTOR function differently than direct SQL execution.

## Working vs. Non-Working Approaches

### What Works

1. **Storing Embeddings as Strings**:
   ```python
   # Convert embedding to comma-separated string
   embedding_str = ','.join(map(str, embedding))
   
   # Insert document with embedding as string
   cursor.execute(
       "INSERT INTO SourceDocuments (id, text_content, embedding) VALUES (?, ?, ?)",
       (doc_id, text, embedding_str)
   )
   ```

2. **Using langchain-iris with SQLAlchemy**:
   ```python
   vector_store = IRISVector.from_texts(
       texts=documents,
       embedding=embeddings,
       collection_name="langchain_test",
       connection_string=connection_string,
   )
   
   results = vector_store.similarity_search(query, k=3)
   ```

### What Doesn't Work

1. **Direct TO_VECTOR in SQL Queries**:
   ```python
   # This fails with ODBC driver parameterization issues
   cursor.execute(f"""
       SELECT TOP {top_k} id, text_content, 
              VECTOR_COSINE(
                  TO_VECTOR(embedding, 'DOUBLE', 384),
                  TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)
              ) AS score
       FROM SourceDocuments
       ORDER BY score ASC
   """)
   ```

2. **Parameterized TO_VECTOR**:
   ```python
   # This fails because TO_VECTOR doesn't accept parameter markers
   cursor.execute("""
       SELECT TOP ? id, text_content, 
              VECTOR_COSINE(
                  TO_VECTOR(embedding, 'DOUBLE', ?),
                  TO_VECTOR(?, 'DOUBLE', ?)
              ) AS score
       FROM SourceDocuments
       ORDER BY score ASC
   """, (top_k, embedding_dim, query_embedding_str, embedding_dim))
   ```

## Recommended Solutions

Based on our technical investigation, we recommend the following solutions:

### 1. Use langchain-iris (Easiest)

The langchain-iris library has successfully implemented workarounds for these limitations:

```python
from langchain_iris import IRISVector
from langchain_community.embeddings import FastEmbedEmbeddings

embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = IRISVector.from_texts(
    texts=documents,
    embedding=embeddings,
    collection_name="collection_name",
    connection_string=connection_string,
)

results = vector_store.similarity_search(query, k=3)
```

### 2. Implement Stored Procedures (Most Robust)

Create ObjectScript stored procedures for vector search:

```objectscript
CREATE PROCEDURE VectorSearch(
    IN p_table_name VARCHAR(100),
    IN p_query_embedding VARCHAR(60000),
    IN p_top_k INT,
    IN p_embedding_dim INT
)
LANGUAGE OBJECTSCRIPT
{
    Set sql = "SELECT TOP " _ p_top_k _ " id, text_content, " _
              "VECTOR_COSINE(" _
              "TO_VECTOR(embedding, 'DOUBLE', " _ p_embedding_dim _ ")," _
              "TO_VECTOR('" _ p_query_embedding _ "', 'DOUBLE', " _ p_embedding_dim _ ")" _
              ") AS score " _
              "FROM " _ p_table_name _ " " _
              "ORDER BY score ASC"
    
    Set tStatement = ##class(%SQL.Statement).%New()
    Set tStatus = tStatement.%Prepare(sql)
    If $$$ISOK(tStatus) {
        Set tResult = tStatement.%Execute()
        Return tResult
    }
    Else {
        Return $system.Status.GetErrorText(tStatus)
    }
}
```

Then call this procedure from Python:

```python
cursor.execute("CALL VectorSearch(?, ?, ?, ?)", 
               (table_name, query_embedding_str, top_k, embedding_dim))
```

### 3. Dual-Table Architecture with HNSW (Best Performance)

For large document collections, implement the dual-table architecture with HNSW indexing as described in [HNSW_INDEXING_RECOMMENDATIONS.md](HNSW_INDEXING_RECOMMENDATIONS.md).

## Conclusion

The technical details provided in this document confirm that the ODBC driver has fundamental limitations with the TO_VECTOR function. These limitations are consistent across different query approaches and are likely inherent to how the ODBC driver handles SQL statements.

The recommended solutions provide different trade-offs between ease of implementation and performance, with the dual-table architecture offering the best performance for large document collections.