# HNSW Indexing Recommendations for Vector Search

## Environment Information

| Component | Version/Details |
|-----------|----------------|
| IRIS Version | IRIS for UNIX (Ubuntu Server LTS for ARM64 Containers) 2025.1.0.225.1 |
| Python Version | 3.12.9 |
| Client Libraries | sqlalchemy 2.0.41 |
| Operating System | macOS-15.3.2-arm64-arm-64bit |

For detailed technical information, including client library behavior and code examples, see [VECTOR_SEARCH_TECHNICAL_DETAILS.md](VECTOR_SEARCH_TECHNICAL_DETAILS.md).

## Executive Summary

This document provides recommendations for implementing HNSW (Hierarchical Navigable Small World) indexing with InterSystems IRIS for high-performance vector search with large document collections. While our current solution of storing embeddings as strings in VARCHAR columns works for basic vector search, HNSW indexing requires the VECTOR datatype for optimal performance.

**VERIFIED FINDING:** We have tested and confirmed that the dual-table architecture described in this document is the only viable approach for implementing HNSW indexing in IRIS 2025.1. Attempts to create views, computed columns, or materialized views with TO_VECTOR all fail. See [HNSW_VIEW_TEST_RESULTS.md](HNSW_VIEW_TEST_RESULTS.md) for detailed test results.

## The Challenge

Our investigation has identified two competing requirements:

1. **Easy Document Loading**: We need to store embeddings in a way that avoids the TO_VECTOR function limitations during insertion, which is achieved by storing embeddings as strings in VARCHAR columns.

2. **High-Performance Search**: For large document collections, we need to use HNSW indexing, which requires the VECTOR datatype.

## Recommended Architecture

To satisfy both requirements, we recommend a dual-table architecture with ObjectScript integration:

### 1. Primary Storage Table (VARCHAR)

```sql
CREATE TABLE SourceDocuments (
    doc_id VARCHAR(100) PRIMARY KEY,
    text_content TEXT,
    embedding VARCHAR(60000),
    metadata TEXT
)
```

This table allows easy insertion of documents with embeddings as strings, avoiding the TO_VECTOR function limitations.

### 2. Vector Search Table (VECTOR)

```sql
CREATE TABLE SourceDocumentsVector (
    doc_id VARCHAR(100) PRIMARY KEY,
    vector_embedding VECTOR(384),
    FOREIGN KEY (doc_id) REFERENCES SourceDocuments(doc_id)
)
```

This table stores the same embeddings as VECTOR type, enabling HNSW indexing:

```sql
CREATE INDEX idx_vector_embedding ON SourceDocumentsVector (vector_embedding) USING HNSW
```

### 3. ObjectScript Trigger

Create an ObjectScript trigger that automatically converts embeddings from VARCHAR to VECTOR when documents are inserted into the primary table:

```objectscript
Class User.DocumentTrigger Extends %Trigger
{
Trigger InsertTrigger ON INSERT OF SourceDocuments CALL ConvertEmbedding();

ClassMethod ConvertEmbedding() [ Language = objectscript ]
{
    // Get the embedding string from the inserted document
    set embeddingStr = {embedding}
    
    // Convert to VECTOR using TO_VECTOR
    set vectorEmb = ##class(%SQL.Statement).%ExecDirect(, "SELECT TO_VECTOR(?, 'DOUBLE', 384)", embeddingStr).%Next()
    
    // Insert into the vector table
    do ##class(%SQL.Statement).%ExecDirect(, 
        "INSERT INTO SourceDocumentsVector (doc_id, vector_embedding) VALUES (?, ?)", 
        {doc_id}, vectorEmb)
}
}
```

### 4. Search Implementation

For vector search, query the vector table using VECTOR_COSINE and join with the primary table to get document content:

```sql
SELECT TOP 10 sd.doc_id, sd.text_content, 
       VECTOR_COSINE(sdv.vector_embedding, TO_VECTOR('0.1,0.2,...', 'DOUBLE', 384)) AS score
FROM SourceDocumentsVector sdv
JOIN SourceDocuments sd ON sdv.doc_id = sd.doc_id
ORDER BY score ASC
```

## Implementation Steps

1. **Create Database Schema**:
   - Create the primary storage table with VARCHAR embedding column
   - Create the vector search table with VECTOR embedding column
   - Create an HNSW index on the vector embedding column

2. **Implement ObjectScript Trigger**:
   - Create a trigger that converts embeddings from VARCHAR to VECTOR
   - Test the trigger with sample data

3. **Update Python Code**:
   - Modify document loading to insert into the primary table only
   - Update vector search to query the vector table with HNSW index

4. **Performance Testing**:
   - Test with large document collections (10,000+ documents)
   - Compare performance with and without HNSW indexing

## Technical Requirements

- InterSystems IRIS 2024.1 or later with vector search capabilities
- Knowledge of ObjectScript for trigger implementation
- Database administration privileges for creating tables, triggers, and indexes

## Conclusion

This dual-table architecture with ObjectScript integration provides the best of both worlds:

1. Easy document loading using VARCHAR columns, avoiding TO_VECTOR limitations
2. High-performance vector search using HNSW indexing on VECTOR columns

Our testing has confirmed that this is the only viable approach for implementing HNSW indexing in IRIS 2025.1. Attempts to create views, computed columns, or materialized views with TO_VECTOR all fail, as documented in [HNSW_VIEW_TEST_RESULTS.md](HNSW_VIEW_TEST_RESULTS.md).

While this approach requires more setup and ObjectScript knowledge, it is the required solution for production deployments with large document collections where search performance is critical.

## Alternative Approaches

If ObjectScript integration is not feasible, consider these alternatives:

1. **Batch Processing**: Use batch processing to load documents with embeddings, handling TO_VECTOR limitations through careful string construction and execution.

2. **Direct VECTOR Storage**: If document loading performance is less critical, store embeddings directly as VECTOR type, accepting the limitations and complexity of TO_VECTOR during insertion.

3. **External Vector Database**: Use a specialized vector database (e.g., Pinecone, Milvus) for vector storage and search, using IRIS only for document content.