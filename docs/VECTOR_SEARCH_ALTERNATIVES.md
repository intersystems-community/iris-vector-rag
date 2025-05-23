# Alternative Vector Search Approaches Investigation

## Environment Information

| Component | Version/Details |
|-----------|----------------|
| IRIS Version | IRIS for UNIX (Ubuntu Server LTS for ARM64 Containers) 2025.1.0.225.1 |
| Python Version | 3.12.9 |
| Client Libraries | sqlalchemy 2.0.41, langchain-iris 0.2.1, llama-iris 0.5.0 |
| Operating System | macOS-15.3.2-arm64-arm-64bit |

For detailed technical information, including client library behavior and code examples, see [VECTOR_SEARCH_TECHNICAL_DETAILS.md](VECTOR_SEARCH_TECHNICAL_DETAILS.md).

## Executive Summary

This document outlines an investigation into alternative approaches to vector search in InterSystems IRIS that may overcome the current TO_VECTOR function limitations. By examining successful implementations in external repositories (llama-iris and langchain-iris), we aim to identify techniques that could resolve our critical blocker: the inability to load documents with embeddings due to ODBC driver limitations with the TO_VECTOR function.

## Problem Statement

Our RAG templates project aims to leverage InterSystems IRIS SQL vector operations. Initial investigations with IRIS 2024.1 highlighted challenges with the `TO_VECTOR()` function, particularly concerning parameter markers and client driver behavior. With the advent of IRIS 2025.1, new findings (see [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md:1) and [`docs/VECTOR_SEARCH_SYNTAX_FINDINGS.md`](docs/VECTOR_SEARCH_SYNTAX_FINDINGS.md:1)) indicate that `TO_VECTOR` *does* support parameter markers when using the `double` (no quotes) type specifier: `TO_VECTOR(?, double, <dim>)`.

This document re-evaluates alternative approaches in light of these updated findings, focusing on robust and performant vector search solutions for IRIS 2025.1. The primary goal is to ensure efficient loading and querying of vector embeddings, compatible with HNSW indexing where appropriate.

## External Repositories Analysis

We have identified two external repositories that successfully implement vector search with InterSystems IRIS:

1. **llama-iris**: LlamaIndex integration with IRIS
   - Repository: https://github.com/caretdev/llama-iris
   - Provides IRISVectorStore for LlamaIndex

2. **langchain-iris**: LangChain integration with IRIS
   - Repository: https://github.com/caretdev/langchain-iris
   - Provides IRISVector for LangChain

Initial code review reveals several key differences in their approach compared to our current implementation:

### Key Architectural Differences

| Feature | External Repositories | Current RAG Templates |
|---------|----------------------|----------------------|
| **Database Access** | SQLAlchemy with custom dialect | Direct DBAPI connection |
| **Vector Type** | `IRISVectorType` or `IRISListBuild` | Direct SQL with `TO_VECTOR()` |
| **Query Construction** | SQLAlchemy query building | String interpolation with validation |
| **Fallback Mechanism** | Custom ObjectScript functions | Limited workarounds |
| **Container Approach** | Testcontainers | Dedicated Docker container |
| **IRIS Version** | 2024.1-preview or latest-cd | Not explicitly specified |

### Critical Differences in Vector Handling

1. **Vector Type Detection**:
   ```python
   # langchain-iris approach
   if conn.dialect.supports_vectors:
       self.native_vector = True
   ```

2. **Dynamic Vector Type Selection**:
   ```python
   # langchain-iris approach
   Column(
       "embedding",
       (
           IRISVectorType(self.dimension)
           if self.native_vector
           else IRISListBuild(self.dimension, float)
       ),
   )
   ```

3. **Custom ObjectScript Functions**:
   Both external repositories create custom ObjectScript functions for vector operations when native vector support isn't available:
   ```sql
   CREATE OR REPLACE FUNCTION langchain_cosine_distance(v1 VARBINARY, v2 VARBINARY)
   RETURNS NUMERIC(0,16)
   LANGUAGE OBJECTSCRIPT
   {
       set dims = $listlength(v1)
       set (distance, norm1, norm2, similarity) = 0
       
       for i=1:1:dims {
           set val1 = $list(v1, i)
           set val2 = $list(v2, i)
           
           set distance = distance + (val1 * val2)
           set norm1 = norm1 + (val1 * val1)
           set norm2 = norm2 + (val2 * val2)
       }
       
       set similarity = distance / $zsqr(norm1 * norm2)
       set similarity = $select(similarity > 1: 1, similarity < -1: -1, 1: similarity)
       quit 1 - similarity
   }
   ```

## Investigation Plan

### Phase 1: Environment Setup and Verification

1. **Set up test environments with both approaches**:
   - Create a simple test script using the `llama-iris` approach
   - Create a simple test script using our current approach
   - Use the same IRIS version (2024.1-preview) for both tests

2. **Verify vector operations in both environments**:
   - Test vector loading (embedding storage)
   - Test vector querying (similarity search)
   - Document any differences in behavior

### Phase 2: Detailed Analysis of Vector Operations

1. **Analyze vector loading mechanisms**:
   - Trace SQL queries generated by both approaches
   - Identify how `llama-iris` and `langchain-iris` handle embedding insertion
   - Compare with our current approach

2. **Analyze vector querying mechanisms**:
   - Trace SQL queries for similarity search
   - Identify how parameter binding is handled
   - Compare performance and reliability

3. **Examine SQLAlchemy dialect implementation**:
   - Review `sqlalchemy_iris` code to understand vector support detection
   - Identify how it handles vector types and operations
   - Determine if it's using a different approach for `TO_VECTOR()`

### Phase 3: Adaptation Strategy

1. **Develop proof-of-concept adaptations**:
   - Implement SQLAlchemy approach in a test branch
   - Test custom ObjectScript functions approach
   - Evaluate hybrid approach using our existing code with targeted improvements

2. **Benchmark and compare approaches**:
   - Measure performance of each approach
   - Evaluate reliability and compatibility
   - Document trade-offs

3. **Create migration plan**:
   - Outline steps to adapt successful approaches to our project
   - Identify required changes to existing code
   - Estimate effort and impact

## Test Script Design

We have designed a test script (`investigation/test_vector_approaches.py`) that will:

1. Set up an IRIS container using testcontainers
2. Test vector operations using the langchain-iris approach
3. Test vector operations using the llama-iris approach
4. Test vector operations using the current project's approach
5. Compare the results and behavior

The script focuses specifically on the vector loading issue, which is our critical blocker. It attempts to insert documents with embeddings using different approaches and reports on the success or failure of each method.

```python
# Key components of the test script
class VectorSearchInvestigation:
    def __init__(self, iris_image: str = "intersystemsdc/iris-community:2024.1-preview"):
        """Initialize the investigation with the specified IRIS image."""
        self.iris_image = iris_image
        self.container = None
        self.connection_string = None
        self.embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        
    # Methods for testing different approaches
    def test_langchain_iris_approach(self):
        """Test vector search using the langchain-iris approach."""
        # Implementation details...
    
    def test_llama_iris_approach(self):
        """Test vector search using the llama-iris approach."""
        # Implementation details...
    
    def test_current_project_approach(self):
        """Test vector search using the current project's approach."""
        # Implementation details...
    
    def run_investigation(self):
        """Run the full investigation."""
        # Implementation details...
```

## Investigation Findings

With IRIS 2025.1, the landscape for vector operations has changed significantly. Our latest tests (e.g., [`investigation/test_working_vector_params.py`](../investigation/test_working_vector_params.py:1)) confirm that parameterized queries with `TO_VECTOR(?, double, <dim>)` are viable.

This new understanding impacts the assessment of previous approaches:

1.  **Previous `langchain-iris` / `llama-iris` Workarounds (VARCHAR storage, `TO_VECTOR` at query time):**
    *   **Rationale then:** These libraries often stored embeddings as strings (e.g., in `VARCHAR` columns) and used `TO_VECTOR` only at query time. This was a workaround for the perceived inability to use `TO_VECTOR` with parameters during `INSERT` or `UPDATE` operations in older IRIS versions or with misconfigured syntax.
    *   **Relevance now (IRIS 2025.1):** While storing as `VARCHAR` and converting at query time still *works*, it's generally less efficient than storing directly as `VECTOR` type if the database supports it and if HNSW indexing is desired. The primary motivation for the `VARCHAR` approach (parameterization issues) is largely resolved with the correct `double` syntax. However, for systems where direct `VECTOR` type manipulation during ingest is complex, or for compatibility with tools expecting string representations, this can still be a fallback.

2.  **Direct `VECTOR` Type Usage with Parameterized `TO_VECTOR` (IRIS 2025.1):**
    *   **Storage:** Embeddings can be inserted/updated directly into columns of `VECTOR` type using `TO_VECTOR(?, double, <dim>)` within `INSERT` or `UPDATE` statements.
        ```sql
        -- Example INSERT with parameterized TO_VECTOR
        INSERT INTO MyTable (id, embedding_col) VALUES (?, TO_VECTOR(?, double, 384));
        -- Python: cursor.execute(sql, (my_id, "[0.1,0.2,...]",))
        ```
    *   **Querying:** Similarity searches can also use parameterized `TO_VECTOR` if the query vector is passed as a parameter.
        ```sql
        SELECT id, VECTOR_COSINE(embedding_col, TO_VECTOR(?, double, 384)) AS score
        FROM MyTable ORDER BY score DESC;
        -- Python: cursor.execute(sql, ("[0.5,0.6,...]",))
        ```
    *   **Advantages:**
        *   Allows for native `VECTOR` type storage, which is essential for HNSW indexing.
        *   Simplifies data loading compared to multi-step conversions or ObjectScript triggers if direct SQL is preferred.
        *   Utilizes standard SQL parameterization for security and clarity.

3.  **HNSW Indexing Considerations:**
    *   HNSW indexes can only be built on columns of `VECTOR` type.
    *   Therefore, if HNSW is a requirement, embeddings *must* ultimately reside in a `VECTOR` column. The `VARCHAR` storage approach is incompatible with direct HNSW indexing on that `VARCHAR` column.
    *   The dual-table architecture (see [`HNSW_INDEXING_RECOMMENDATIONS.md`](HNSW_INDEXING_RECOMMENDATIONS.md:1)) remains relevant if, for example, the ingestion pipeline prefers writing to a staging `VARCHAR` table before an ObjectScript trigger converts and moves data to a `VECTOR` table with an HNSW index. However, with working parameterized `TO_VECTOR`, direct insertion into the `VECTOR` table is more feasible.

## Proof of Concept Implementation (Updated for IRIS 2025.1)

A proof-of-concept should now focus on leveraging direct `VECTOR` type storage and parameterized `TO_VECTOR(?, double, <dim>)` calls. An example script like [`investigation/test_working_vector_params.py`](../investigation/test_working_vector_params.py:1) demonstrates this.

**Key aspects of a PoC:**

1.  **Table Definition:**
    ```sql
    CREATE TABLE MyVectorTable (
        id VARCHAR(255) PRIMARY KEY,
        text_content %Text,
        embedding VECTOR(EMBEDDING_DIMENSION, DOUBLE) WITH STORAGETYPE = 'STORE_VECTOR_AS_STRING' -- Or other storage types
    );
    -- Optionally, add HNSW index if needed
    CREATE HNSW INDEX idx_hnsw_embedding ON MyVectorTable (embedding) WITH (%PARALLEL);
    ```

2.  **Data Insertion (Python example):**
    ```python
    embedding_list = [0.1, 0.2, ..., 0.N] # Example embedding
    embedding_str = "[" + ",".join(map(str, embedding_list)) + "]" # Format: "[d1,d2,...]"
    doc_id = "doc1"
    text = "Some document text."

    insert_sql = "INSERT INTO MyVectorTable (id, text_content, embedding) VALUES (?, ?, TO_VECTOR(?, double, ?))"
    # cursor.execute(insert_sql, (doc_id, text, embedding_str, len(embedding_list)))
    # conn.commit()
    ```

3.  **Data Querying (Python example):**
    ```python
    query_embedding_list = [0.5, 0.6, ..., 0.M]
    query_embedding_str = "[" + ",".join(map(str, query_embedding_list)) + "]"
    query_dim = len(query_embedding_list)
    top_k = 5

    search_sql = f"""
    SELECT TOP ? id, text_content,
           VECTOR_COSINE(embedding, TO_VECTOR(?, double, ?)) AS score
    FROM MyVectorTable
    ORDER BY score DESC
    """
    # cursor.execute(search_sql, (top_k, query_embedding_str, query_dim))
    # results = cursor.fetchall()
    ```

## Recommended Implementation (IRIS 2025.1)

Given the confirmed support for `TO_VECTOR(?, double, <dim>)` in IRIS 2025.1:

1.  **Prioritize Native `VECTOR` Type Storage:**
    *   Define table columns intended for embeddings as `VECTOR(dimension, DOUBLE)`.
    *   This is crucial for HNSW indexing and optimal performance.

2.  **Use Parameterized Queries for Inserts/Updates and Queries:**
    *   Utilize `TO_VECTOR(?, double, <dim>)` for converting input vector data (formatted as `"[d1,d2,...]"`) during `INSERT` or `UPDATE` statements.
    *   Use the same for query vectors in `SELECT` statements.
    *   This approach enhances security and code clarity over string interpolation.

3.  **Update `common/vector_sql_utils.py` and `common/db_init.py`:**
    *   [`common/db_init.py`](common/db_init.py:1): Ensure table creation scripts define embedding columns as `VECTOR(dim, DOUBLE)`. Include HNSW index creation where appropriate.
    *   [`common/vector_sql_utils.py`](common/vector_sql_utils.py:1):
        *   Refactor functions to construct parameterized SQL for vector operations.
        *   Ensure helper functions correctly format Python list embeddings into the `"[d1,d2,...]"` string representation required by `TO_VECTOR` when passed as a parameter.
        *   Remove or deprecate utilities designed for older string-interpolation workarounds if no longer necessary.

4.  **Update RAG Pipelines:**
    *   Modify document loading logic to use parameterized `INSERT` statements with `TO_VECTOR(?, double, <dim>)`.
    *   Adjust vector search query construction to use parameterized `SELECT` statements.

5.  **HNSW Indexing:**
    *   For performance-critical applications, ensure HNSW indexes are created on the `VECTOR` columns.
    *   The dual-table architecture (staging `VARCHAR` table + ObjectScript trigger + final `VECTOR` table with HNSW) as described in [`HNSW_INDEXING_RECOMMENDATIONS.md`](HNSW_INDEXING_RECOMMENDATIONS.md:1) can still be used if the ingestion pipeline benefits from it, but the trigger logic should now also use the correct `TO_VECTOR(?, double, <dim>)` syntax if it's re-parsing string data. However, direct parameterized inserts into the `VECTOR` table are now more feasible.

## Code Example

```python
# Store embeddings as comma-separated strings
embedding_str = ','.join(map(str, embedding))

# Insert document with embedding as string
insert_sql = f"""
INSERT INTO {table_name} (id, text_content, embedding, metadata)
VALUES (?, ?, ?, ?)
"""
cursor.execute(insert_sql, (doc_id, text, embedding_str, "{}"))

# Query using TO_VECTOR at query time
search_sql = f"""
SELECT TOP {top_k} id, text_content, 
       VECTOR_COSINE(
           TO_VECTOR(embedding, double, 384), // Or just 'embedding' if it's already VECTOR type
           TO_VECTOR('{query_embedding_str}', double, 384)
       ) AS score
FROM {table_name}
ORDER BY score ASC
"""
```

## Expected Outcomes

This investigation is expected to yield:

1. **Clear understanding** of how external repositories successfully implement vector search with IRIS
2. **Identification of specific techniques** that overcome the TO_VECTOR function limitations
3. **Proof-of-concept implementation** of a working approach for loading embeddings
4. **Migration plan** for adapting our codebase to use the successful approach
5. **Documentation** of findings and recommendations

## Next Steps

1. **Execute the implementation plan**:
   - Update vector_sql_utils.py with the new approach
   - Modify db_init.py to use VARCHAR for embedding storage
   - Update RAG pipelines to use the new approach

2. **Test with real PMC data**:
   - Load real PMC documents with embeddings
   - Verify that the critical blocker is resolved
   - Run full RAG pipeline tests

3. **Update documentation**:
   - Update IRIS_SQL_VECTOR_OPERATIONS.md with the new approach
   - Update IRIS_SQL_VECTOR_LIMITATIONS.md with new insights
   - Create migration guide for future reference

## Performance Considerations for Large Document Collections

While our investigation has identified a viable solution for loading documents with embeddings, we must also consider performance optimization for large document collections. HNSW (Hierarchical Navigable Small World) indexing is essential for efficient vector search with large datasets, but it requires the VECTOR datatype.

**HNSW Indexing Requirement:** HNSW indexing requires the underlying column to be of `VECTOR` type. Storing embeddings as strings in `VARCHAR` columns is incompatible with direct HNSW indexing on that `VARCHAR` column. While views or computed columns using `TO_VECTOR` on a `VARCHAR` column might seem like a workaround, creating HNSW indexes on such derived vector structures is generally not supported or performant. For HNSW, data must reside in a native `VECTOR` column.

To address this requirement, we have created a separate document with detailed recommendations for implementing HNSW indexing: [HNSW_INDEXING_RECOMMENDATIONS.md](HNSW_INDEXING_RECOMMENDATIONS.md).

The recommended approach involves:
1. A dual-table architecture with VARCHAR storage for easy loading
2. VECTOR storage with HNSW indexing for efficient search
3. ObjectScript triggers to automatically convert between formats

This dual-table architecture is the only viable approach for implementing high-performance vector search with HNSW indexing. It provides the best of both worlds: easy document loading and high-performance vector search.

## Conclusion

The confirmation in IRIS 2025.1 that `TO_VECTOR` supports parameterized queries with the `double` type specifier (`TO_VECTOR(?, double, <dim>)`) significantly simplifies and improves vector search implementations.

**Key Takeaways for IRIS 2025.1:**

1.  **Direct Parameterized `TO_VECTOR` is Preferred:**
    *   This should be the default method for inserting, updating, and querying vector data.
    *   It allows for native `VECTOR` type storage, essential for HNSW indexing.
    *   It enhances security and code readability compared to older workarounds.

2.  **`VARCHAR` Storage as a Fallback:**
    *   Storing embeddings as strings in `VARCHAR` columns and converting with `TO_VECTOR` at query time is still possible but is a less optimal solution if native `VECTOR` type and HNSW are desired. It might be considered for compatibility or specific ingestion pipeline constraints.

3.  **HNSW Indexing Requires Native `VECTOR` Type:**
    *   To leverage HNSW for performance, embeddings must be stored in columns of `VECTOR` type.
    *   The dual-table architecture (detailed in [`HNSW_INDEXING_RECOMMENDATIONS.md`](HNSW_INDEXING_RECOMMENDATIONS.md:1)) remains a valid pattern if an intermediate staging/conversion step is beneficial, but direct parameterized inserts into `VECTOR` tables are now more feasible.

The recommended path forward is to adapt all RAG pipeline components (data loading, querying, table definitions) to use parameterized `TO_VECTOR(?, double, <dim>)` calls with native `VECTOR` type columns, enabling robust, secure, and performant vector search capabilities in IRIS 2025.1.