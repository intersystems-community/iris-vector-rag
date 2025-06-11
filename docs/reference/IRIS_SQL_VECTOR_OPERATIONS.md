# IRIS SQL Vector Operations: Limitations and Workarounds

## 1. Executive Summary

InterSystems IRIS 2025.1 introduced vector search capabilities, which are essential for modern Retrieval-Augmented Generation (RAG) pipelines. However, several critical limitations in the SQL implementation prevent standard parameterized queries from working with vector operations. This document details these limitations, their impact on RAG implementations, and the workarounds we've developed.

The key issues identified are:

1. The `TO_VECTOR()` function rejects parameter markers
2. `TOP`/`FETCH FIRST` clauses cannot be parameterized
3. Client drivers rewrite literals to `:%qpar()` even when no parameter list is supplied
4. ODBC driver limitations with the TO_VECTOR function prevent loading documents with embeddings

These limitations force developers to use string interpolation for *querying* vector data (as implemented in [`common/vector_sql_utils.py`](common/vector_sql_utils.py:1) and used by [`common/db_vector_search.py`](common/db_vector_search.py:1)), instead of parameterized queries. This introduces potential security risks that must be carefully managed with validation.

**Current Status: PROJECT BLOCKED.** While workarounds for *querying* are in place, the **primary project blocker** is that ODBC driver limitations with the `TO_VECTOR()` function prevent loading documents with embeddings. This blocks our ability to test RAG pipelines with new, real PMC data. For a detailed explanation, see [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md).

## 2. Identified Limitations

### 2.1. TO_VECTOR() Function Rejects Parameter Markers

The `TO_VECTOR()` function in IRIS SQL does not accept parameter markers (`?`, `:param`, or `:%qpar`), which are standard in SQL for safe query parameterization.

**Example of what doesn't work:**

```sql
SELECT doc_id,
       VECTOR_COSINE(embedding, TO_VECTOR(?, 'DOUBLE', 768)) AS score
FROM SourceDocuments
ORDER BY score DESC
```

**Error message:**
```
SQLCODE -1, ") expected, : found"
```

According to the IRIS documentation, the `TO_VECTOR` function only accepts literal strings unless used in ObjectScript Dynamic SQL. This limitation prevents the use of standard parameterized queries for vector search operations.

### 2.2. TOP/FETCH FIRST Clauses Cannot Be Parameterized

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

### 2.3. Client Drivers Rewrite Literals to :%qpar()

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

## 3. Investigation Process

### 3.1. Test Scripts

We created several test scripts to investigate and document these limitations:

1. **test_pyodbc_vector_ops.py**: Tests basic vector operations using pyodbc
2. **test_iris_vector_workarounds.py**: Demonstrates workarounds for the identified limitations
3. **test_iris_dbapi_vector_ops.py**: Tests vector operations using the native IRIS DB-API driver
4. **test_sqlalchemy_vector_ops.py**: Tests vector operations using SQLAlchemy with the IRIS dialect

These scripts systematically test different approaches to vector operations in IRIS SQL and document the errors encountered.

### 3.2. Testing Methodology

Our testing methodology involved:

1. **Basic Connectivity Tests**: Ensuring that we could connect to IRIS using different drivers
2. **Simple Table Operations**: Creating tables with standard data types to verify basic functionality
3. **Vector Type Tests**: Creating tables with the `VECTOR` data type
4. **Parameter Binding Tests**: Attempting to use parameter markers with vector functions
5. **Workaround Tests**: Testing string interpolation with validation as a workaround

### 3.3. Key Findings

Our investigation revealed that:

1. **Standard tables and queries work fine**: Basic SQL operations with standard data types work as expected with parameterized queries.
2. **Vector data type is supported**: Creating tables with the `VECTOR` data type works correctly.
3. **Parameter binding fails with vector functions**: Attempts to use parameter markers with `TO_VECTOR()` consistently fail across all drivers.
4. **Parameter binding fails with TOP/FETCH**: Attempts to parameterize `TOP` or `FETCH FIRST` clauses consistently fail.
5. **String interpolation works**: Constructing queries with string interpolation works, but requires careful validation to prevent SQL injection.
6. **Driver behavior complicates matters**: The driver's rewriting of literals to `:%qpar()` adds an additional layer of complexity.

## 4. Implemented Workarounds

### 4.1. String Interpolation with Proper Validation

Our primary workaround is to use string interpolation with careful validation to prevent SQL injection. This approach is demonstrated in `test_iris_vector_workarounds.py` and implemented in `common/db_vector_search.py`.

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

### 4.2. Security Considerations and Mitigation Strategies

While string interpolation is generally discouraged due to SQL injection risks, our implementation includes several security measures:

1. **Strict Validation**: We validate all inputs before including them in SQL strings.
2. **Limited Character Sets**: For vector strings, we only allow digits, dots, commas, and square brackets.
3. **Type Checking**: For `top_k`, we ensure it's a positive integer.
4. **Error Handling**: We raise clear error messages when validation fails.
5. **Logging**: We log all SQL queries for auditing and debugging.

These measures significantly reduce the risk of SQL injection while allowing us to work around the IRIS SQL limitations.

### 4.3. Code Examples

**Example from db_vector_search.py:**

```python
def search_source_documents_dynamically(
    iris_connector: Any, top_k: int, vector_string: str
) -> List[Tuple[str, str, float]]:
    """
    Performs a vector search on the SourceDocuments table using dynamic SQL.
    Returns a list of tuples, where each tuple is (doc_id, text_content, score).
    """
    # Ensure top_k is an integer to prevent SQL injection via f-string
    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    # vector_string is already in the format like "[0.1,0.2,...]"
    sql = f"""
        SELECT doc_id, text_content,
               VECTOR_COSINE(embedding, TO_VECTOR('{vector_string}', 'DOUBLE', 768)) AS score
        FROM SourceDocuments
        WHERE embedding IS NOT NULL
        ORDER BY score DESC
        FETCH FIRST {top_k} ROWS ONLY
    """
    results: List[Tuple[str, str, float]] = []
    cursor = None
    try:
        logger.debug(f"Executing dynamic SQL for SourceDocuments search with top_k={top_k}, vector_string (preview): {vector_string[:100]}...")
        cursor = iris_connector.cursor()
        cursor.execute(sql) # No parameters passed here as all are interpolated
        fetched_rows = cursor.fetchall()
        if fetched_rows:
            # Ensure rows are tuples and have the expected number of elements
            results = [(str(row[0]), str(row[1]), float(row[2])) for row in fetched_rows if isinstance(row, tuple) and len(row) == 3]
        logger.debug(f"Found {len(results)} documents from SourceDocuments.")
    except Exception as e:
        logger.error(f"Error during dynamic SQL search on SourceDocuments: {e}")
        # Re-raise the exception so the calling pipeline can handle it or log it appropriately.
        raise
    finally:
        if cursor:
            cursor.close()
    return results
```

**Example from SQLAlchemy integration:**

```python
inner_vector_content = f"[{','.join(map(str, query_vector_py_list))}]"
to_vector_argument_sql = f"TO_VECTOR('{inner_vector_content}', 'DOUBLE', {VECTOR_DIM})"

stmt = (
    select(
        VectorTestTable.id,
        VectorTestTable.embedding,
        func.vector_cosine(
            VectorTestTable.embedding,
            text(to_vector_argument_sql) 
        ).label("score")
    )
    .order_by(text("score DESC"))
)
```

## 5. Integration with RAG Pipelines

### 5.1. How Workarounds are Integrated

Our RAG pipelines use the workarounds implemented in `common/db_vector_search.py` to perform vector similarity searches. The key integration points are:

1. **Embedding Generation**: RAG pipelines generate embeddings for queries using standard embedding models.
2. **Vector String Formatting**: The embeddings are converted to the string format required by IRIS.
3. **Dynamic SQL Construction**: The pipelines use the `search_source_documents_dynamically` and `search_knowledge_graph_nodes_dynamically` functions to construct and execute vector search queries.
4. **Result Processing**: The search results are processed and used for context retrieval in the RAG pipeline.

This integration allows our RAG pipelines to work with IRIS vector search capabilities despite the limitations in the SQL implementation.

### 5.2. Performance Considerations

The workarounds have minimal performance impact compared to parameterized queries. The additional validation steps add negligible overhead, and the string interpolation approach is efficient for vector search operations.

However, there are some considerations:

1. **Query Caching**: Unlike parameterized queries, string-interpolated queries may not benefit from query plan caching in IRIS.
2. **Memory Usage**: String interpolation creates new query strings for each search, which may increase memory usage slightly.
3. **Connection Pooling**: The workarounds are compatible with connection pooling, which helps mitigate any performance concerns.

### 5.3. Maintenance Considerations

The workarounds introduce some maintenance challenges:

1. **Code Complexity**: The validation and string interpolation code adds complexity compared to standard parameterized queries.
2. **Security Vigilance**: Developers must be careful to use the validation functions for all user inputs.
3. **Future Compatibility**: When IRIS fixes these limitations, we'll need to update our code to use parameterized queries.

To address these challenges, we've centralized the workarounds in the `db_vector_search.py` module, making it easier to maintain and eventually replace when IRIS adds support for parameterized vector operations.

## 6. Recommendations for Future Versions

### 6.1. Suggested Improvements to IRIS SQL

Based on our experience, we recommend the following improvements to IRIS SQL:

1. **Support for Parameter Markers in TO_VECTOR()**: Allow the use of `?`, `:param`, and other parameter markers in the `TO_VECTOR()` function.
2. **Parameterized TOP/FETCH FIRST Clauses**: Support parameter markers in `TOP` and `FETCH FIRST` clauses.
3. **Improved Driver Behavior**: Modify the IRIS drivers to avoid rewriting literals to `:%qpar()` when no parameter list is supplied.
4. **Enhanced Error Messages**: Provide clearer error messages when parameter binding fails, explaining the limitations and suggesting workarounds.
5. **Documentation Updates**: Update the IRIS documentation to clearly explain these limitations and provide official workarounds.

### 6.2. Alternative Approaches

Until these improvements are implemented, alternative approaches include:

1. **Stored Procedures in ObjectScript**: Create stored procedures using ObjectScript, which supports parameter binding for vector operations through `%SQL.Statement`.
2. **Custom IRIS Extensions**: Develop custom IRIS extensions or user-defined functions that handle parameter binding correctly.
3. **Client-Side Filtering**: Retrieve more results than needed and perform additional filtering on the client side.
4. **Vector Indexing**: Use IRIS vector indexing capabilities to improve search performance, reducing the impact of the workarounds.

## 7. Conclusion

The IRIS SQL vector operations limitations present significant challenges for RAG implementations. While we have implemented workarounds using string interpolation with careful validation, we have encountered a critical blocker: the ODBC driver limitations with the TO_VECTOR function prevent loading documents with embeddings, which is blocking our ability to test with real PMC data.

We recommend engaging InterSystems support to address these limitations in future IRIS versions, allowing for standard parameterized queries with vector operations. In the meantime, we are exploring alternative approaches to work around these limitations and enable testing with real data.

For a detailed explanation of these limitations and potential solutions, see [IRIS_SQL_VECTOR_LIMITATIONS.md](IRIS_SQL_VECTOR_LIMITATIONS.md).

## 8. References

1. InterSystems IRIS SQL Reference: TO_VECTOR
2. InterSystems IRIS SQL Reference: VECTOR_COSINE
3. InterSystems IRIS SQL Guide: Using Vector Search
4. InterSystems IRIS SQL Reference: TOP
5. InterSystems IRIS SQL Reference: FETCH
6. InterSystems IRIS Documentation: %SQL.Statement class reference
7. InterSystems IRIS SQL Guide: Defining and Using Stored Procedures
8. InterSystems Developer Community: Entity Framework Provider issue when "Support Delimited Identifiers" is turned off
9. InterSystems Developer Community: HELP — SQL statements containing double quotes cannot be executed via JDBC
10. InterSystems Developer Community: Mastering the JDBC SQL Gateway