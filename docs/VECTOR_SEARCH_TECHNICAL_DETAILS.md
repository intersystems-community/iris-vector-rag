# Vector Search Technical Details

This document provides comprehensive technical details about our vector search implementation, including environment information, client library behavior, and code examples. This information is intended to help Quality Development and Development teams understand the specific technical context of our vector search limitations and solutions.

## Environment Information

| Component | Version/Details |
|-----------|----------------|
| IRIS Version | IRIS for UNIX (Ubuntu Server LTS for ARM64 Containers) 2025.1.0.225.1 |
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
    TO_VECTOR(embedding, double, 5),
    TO_VECTOR('[0.1,0.2,0.3,0.4,0.5]', double, 5)
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
    TO_VECTOR(embedding, double, 5),
    TO_VECTOR(?, double, 5)
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
    TO_VECTOR(embedding, double, 5),
    TO_VECTOR('[0.1,0.2,0.3,0.4,0.5]', double, 5)
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

1. **`TO_VECTOR` with `double`**: The `TO_VECTOR` function works correctly with parameterized queries when `double` (no quotes) is used as the type specifier. The format for the vector string literal should be `[0.1,0.2,...]`.

2. **Client Driver Behavior**: While server-side parameterization with `TO_VECTOR(?, double, <dim>)` is confirmed, client drivers (ODBC, DBAPI) might still exhibit behaviors like attempting to re-parameterize parts of the query or having specific expectations for how vector data is passed. This requires careful testing with the specific driver in use.

3. **Importance of Correct Syntax**: Using `'DOUBLE'` (with quotes) or incorrect vector string formats will likely lead to errors. Adherence to the `TO_VECTOR(vector_data, double, dimension)` syntax is crucial.

## Client Library Comparison

### DBAPI (Python Standard)

The Python DBAPI implementation we're using shows consistent behavior with the ODBC driver limitations:

```python
# This fails with the same error
cursor.execute("""
    SELECT TOP 3 id, text_content, 
           VECTOR_COSINE(
               TO_VECTOR(embedding, double, 384),
               TO_VECTOR(?, double, 384)
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
                  TO_VECTOR(embedding, double, 384),
                  TO_VECTOR('{query_embedding_str}', double, 384)
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
                  TO_VECTOR(embedding, double, ?),
                  TO_VECTOR(?, double, ?)
              ) AS score
       FROM SourceDocuments
       ORDER BY score ASC
   """, (top_k, embedding_dim, query_embedding_str, embedding_dim))
   ```

## Recommended Solutions

Based on our updated technical investigation for IRIS 2025.1:

### 1. Parameterized Queries with `TO_VECTOR(?, double, <dim>)` (Preferred)

Directly use parameterized queries with the correct `TO_VECTOR` syntax. This is now the most straightforward and secure method.

```python
# Assuming 'conn' is an active DBAPI connection and 'cursor' is its cursor
# query_embedding_list is a Python list of floats, e.g., [0.1, 0.2, ...]
query_embedding_str = "[" + ",".join(map(str, query_embedding_list)) + "]" # Format as "[d1,d2,...]"
embedding_dim = len(query_embedding_list)
top_k = 3

sql = f"""
    SELECT TOP ? id, text_content,
           VECTOR_COSINE(
               embedding,  -- Assuming 'embedding' column is already VECTOR type
               TO_VECTOR(?, double, ?)
           ) AS score
    FROM SourceDocuments
    ORDER BY score ASC
"""
# Note: If 'embedding' column is VARCHAR, use TO_VECTOR(embedding, double, ?) for it too.
cursor.execute(sql, (top_k, query_embedding_str, embedding_dim))
results = cursor.fetchall()
```

**Key considerations:**
- Ensure the vector string passed as a parameter is in the format `"[d1,d2,d3,...]"`.
- The `embedding` column in the table should ideally be of `VECTOR` type. If it's `VARCHAR`, it also needs to be converted using `TO_VECTOR(embedding_column, double, dim)`.

### 2. Use `langchain-iris` (If using LangChain)

If your project leverages the LangChain ecosystem, `langchain-iris` likely incorporates the correct syntax or workarounds.

```python
from langchain_iris import IRISVector
from langchain_community.embeddings import FastEmbedEmbeddings # Or your preferred embedding model

# Ensure your IRISVector version is compatible with IRIS 2025.1 and uses correct TO_VECTOR syntax
embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = IRISVector.from_texts(
    texts=documents, # Your list of texts
    embedding=embeddings,
    collection_name="your_collection",
    connection_string="your_iris_connection_string", # Ensure this connects to IRIS 2025.1
)

results = vector_store.similarity_search(query_text, k=3)
```
Verify that `langchain-iris` is updated to handle the `double` syntax correctly for IRIS 2025.1.

### 3. ObjectScript Stored Procedures (For Complex Logic or Encapsulation)

For complex server-side logic or to encapsulate vector operations, ObjectScript stored procedures remain a robust option. Ensure they use the `TO_VECTOR(?, double, <dim>)` syntax internally for dynamic SQL.

```objectscript
CREATE PROCEDURE VectorSearch(
    IN p_table_name VARCHAR(100),
    IN p_query_embedding_str VARCHAR(60000), // Expects "[d1,d2,...]"
    IN p_top_k INT,
    IN p_embedding_dim INT
)
LANGUAGE OBJECTSCRIPT
{
    // Ensure p_query_embedding_str is correctly formatted before use
    Set sql = "SELECT TOP ? id, text_content, " _
              "VECTOR_COSINE(" _
              "TO_VECTOR(embedding, double, ?)," _ // Assuming 'embedding' column is VECTOR type
              "TO_VECTOR(?, double, ?)" _
              ") AS score " _
              "FROM " _ p_table_name _ " " _
              "ORDER BY score ASC"
    
    Set tStatement = ##class(%SQL.Statement).%New()
    Set tStatus = tStatement.%Prepare(sql)
    If $$$ISOK(tStatus) {
        // Parameters for %Execute: top_k, embedding_dim (for table's embedding), query_embedding_str, embedding_dim (for query_embedding)
        Set tResult = tStatement.%Execute(p_top_k, p_embedding_dim, p_query_embedding_str, p_embedding_dim)
        Return tResult
    }
    Else {
        Return $system.Status.GetErrorText(tStatus)
    }
}
```
Calling from Python:
```python
query_embedding_list = [0.1, 0.2, 0.3] # example
query_embedding_str = "[" + ",".join(map(str, query_embedding_list)) + "]"
embedding_dim = len(query_embedding_list)
top_k = 3
table_name = "SourceDocuments"

cursor.execute("{CALL VectorSearch(?, ?, ?, ?)}",
               (table_name, query_embedding_str, top_k, embedding_dim))
results = cursor.fetchall()
```

### 4. Dual-Table Architecture with HNSW (Best Performance for Large Scale)

This remains the top recommendation for performance with large datasets, as detailed in [`HNSW_INDEXING_RECOMMENDATIONS.md`](HNSW_INDEXING_RECOMMENDATIONS.md:1). The ObjectScript triggers in this architecture should be updated to use `TO_VECTOR(?, double, <dim>)`.

## Conclusion

With IRIS 2025.1, the primary method for vector operations should be direct parameterized SQL using `TO_VECTOR(?, double, <dim>)`. This simplifies development and enhances security compared to previous workarounds. Client-side library choices (`langchain-iris`) or architectural patterns (stored procedures, dual-table) should align with this updated understanding of `TO_VECTOR`'s capabilities. Always verify behavior with your specific client drivers and IRIS version.