# Alternative Vector Search Approaches Investigation

## Environment Information

| Component | Version/Details |
|-----------|----------------|
| IRIS Version | IRIS for UNIX (Ubuntu Server LTS for ARM64 Containers) 2024.1.2 (Build 398U) |
| Python Version | 3.12.9 |
| Client Libraries | sqlalchemy 2.0.41, langchain-iris 0.2.1, llama-iris 0.5.0 |
| Operating System | macOS-15.3.2-arm64-arm-64bit |

For detailed technical information, including client library behavior and code examples, see [VECTOR_SEARCH_TECHNICAL_DETAILS.md](VECTOR_SEARCH_TECHNICAL_DETAILS.md).

## Executive Summary

This document outlines an investigation into alternative approaches to vector search in InterSystems IRIS that may overcome the current TO_VECTOR function limitations. By examining successful implementations in external repositories (llama-iris and langchain-iris), we aim to identify techniques that could resolve our critical blocker: the inability to load documents with embeddings due to ODBC driver limitations with the TO_VECTOR function.

## Problem Statement

Our current RAG templates project is blocked by limitations in the InterSystems IRIS SQL vector operations, specifically:

1. **TO_VECTOR() Function Rejects Parameter Markers**: The TO_VECTOR() function does not accept parameter markers (?, :param, or :%qpar), which are standard in SQL for safe query parameterization.

2. **Client Drivers Rewrite Literals**: Python, JDBC, and other client drivers replace embedded literals with :%qpar(n) even when no parameter list is supplied, creating misleading parse errors.

3. **ODBC Driver Limitations**: When attempting to load documents with embeddings, the ODBC driver encounters limitations with the TO_VECTOR function, preventing the loading of vector embeddings.

While we have implemented workarounds for *querying* vector data using string interpolation with validation, the *loading* of embeddings remains a critical blocker that prevents testing with real PMC data.

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

After executing our test script and analyzing the source code of langchain-iris, we have discovered the following key insights:

### 1. langchain-iris Approach

The langchain-iris repository successfully implements vector search with IRIS by:

1. **Storage Approach**:
   - Stores embeddings as Python lists in VARCHAR columns
   - Uses SQLAlchemy with a custom dialect (sqlalchemy_iris)
   - Avoids using TO_VECTOR during insertion

2. **Query Mechanism**:
   - Uses native VECTOR_COSINE for similarity search
   - Converts stored strings to vectors at query time using TO_VECTOR
   - Uses SQLAlchemy's query building to handle SQL construction

3. **Key Code Insight**:
   ```python
   # When native_vector is True
   self.distance_strategy(embedding).label("distance")
   ```

4. **Database Schema**:
   - Uses VARCHAR(56831) for embedding storage
   - No native vector type in the database schema

### 2. llama-iris Approach

The llama-iris approach was less successful in our testing environment due to OpenAI API rate limits, but appears to follow a similar pattern to langchain-iris.

### 3. Current Project Approach

Our current approach faces limitations:

1. **Storage Issues**:
   - Attempts to use TO_VECTOR during insertion
   - ODBC driver tries to parameterize parts that can't be parameterized
   - Results in errors when inserting documents with embeddings

2. **Query Mechanism**:
   - Uses string interpolation with validation
   - Works for querying but not for insertion

## Proof of Concept Implementation

We created a proof-of-concept implementation (`investigation/vector_storage_poc.py`) that demonstrates the langchain-iris approach:

1. **Storage Solution**:
   - Store embeddings as comma-separated strings: `"0.1,0.2,0.3,..."`
   - Use VARCHAR columns with sufficient size
   - Avoid using TO_VECTOR during insertion

2. **Query Solution**:
   - Use TO_VECTOR at query time to convert strings to vectors
   - Continue using string interpolation with validation
   - Use VECTOR_COSINE for similarity search

3. **Implementation Challenges**:
   - ODBC driver still attempts to parameterize parts of SQL statements
   - Requires careful string construction and validation

## Recommended Implementation

Based on our findings, we recommend the following changes to our current implementation:

1. **Update vector_sql_utils.py**:
   - Modify to support storing embeddings as comma-separated strings
   - Add functions for converting between vector formats
   - Enhance validation for the new storage format

2. **Update db_init.py**:
   - Modify table creation to use VARCHAR for embedding storage
   - Adjust column sizes based on embedding dimensions

3. **Update RAG Pipelines**:
   - Modify document loading to store embeddings as strings
   - Ensure vector search queries use TO_VECTOR at query time

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
           TO_VECTOR(embedding, 'DOUBLE', 384),
           TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)
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

To address this requirement, we have created a separate document with detailed recommendations for implementing HNSW indexing: [HNSW_INDEXING_RECOMMENDATIONS.md](HNSW_INDEXING_RECOMMENDATIONS.md).

The recommended approach involves:
1. A dual-table architecture with VARCHAR storage for easy loading
2. VECTOR storage with HNSW indexing for efficient search
3. ObjectScript triggers to automatically convert between formats

This approach provides the best of both worlds: easy document loading and high-performance vector search.

## Conclusion

Our investigation has revealed that the langchain-iris approach provides a viable solution to our current vector search limitations. By storing embeddings as strings and using TO_VECTOR only at query time, we can avoid the ODBC driver limitations while still leveraging native vector operations for search.

For basic testing and development, this approach is sufficient. For production deployments with large document collections, the dual-table architecture with HNSW indexing described in [HNSW_INDEXING_RECOMMENDATIONS.md](HNSW_INDEXING_RECOMMENDATIONS.md) is recommended.

This solution should allow us to proceed with loading real PMC documents with embeddings and testing our RAG pipelines with real data, while also providing a path to high-performance vector search for production deployments.