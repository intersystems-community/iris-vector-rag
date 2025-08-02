# IRIS SQL Vector Operations Reference

## Overview

This document provides a comprehensive reference for performing vector operations using SQL in InterSystems IRIS within this RAG templates project. It covers the proper usage of vector functions, storage patterns, and the mandatory utility functions that ensure consistent vector handling across the codebase.

## Table of Contents

1. [Vector Storage in IRIS](#vector-storage-in-iris)
2. [Mandatory Vector Insertion Utility](#mandatory-vector-insertion-utility)
3. [Vector Search Operations](#vector-search-operations)
4. [IRIS SQL Vector Functions](#iris-sql-vector-functions)
5. [Table Schemas](#table-schemas)
6. [Python Integration](#python-integration)
7. [Performance Considerations](#performance-considerations)
8. [Best Practices](#best-practices)
9. [Common Patterns](#common-patterns)
10. [Troubleshooting](#troubleshooting)

## Vector Storage in IRIS

### Storage Format

In this project, vectors are stored as comma-separated strings in VARCHAR columns due to IRIS Community Edition limitations. The format is:

```
"0.1,0.2,0.3,0.4,0.5"
```

### Key Tables

- **`RAG.SourceDocuments`**: Main document storage with embeddings
- **`RAG.DocumentTokenEmbeddings`**: Token-level embeddings for ColBERT
- **`RAG.KnowledgeGraphNodes`**: Graph node embeddings
- **`RAG.DocumentChunks`**: Chunked document embeddings

## Mandatory Vector Insertion Utility

### Critical Rule from `.clinerules`

**ALL vector insertions MUST use the [`common.db_vector_utils.insert_vector()`](common/db_vector_utils.py:6) utility function.** Direct INSERT statements with vector data are prohibited.

### Function Signature

```python
def insert_vector(
    cursor: Any,
    table_name: str,
    vector_column_name: str,
    vector_data: List[float],
    target_dimension: int,
    key_columns: Dict[str, Any],
    additional_data: Optional[Dict[str, Any]] = None
) -> bool
```

### Parameters

- **`cursor`**: Database cursor object
- **`table_name`**: Target table (e.g., "RAG.DocumentTokenEmbeddings")
- **`vector_column_name`**: Column storing the vector
- **`vector_data`**: Raw embedding vector as list of floats
- **`target_dimension`**: Target vector dimension (truncates/pads as needed)
- **`key_columns`**: Primary key or identifying columns
- **`additional_data`**: Optional additional column data

### Usage Example

```python
from common.db_vector_utils import insert_vector

# Insert a document token embedding
success = insert_vector(
    cursor=cursor,
    table_name="RAG.DocumentTokenEmbeddings",
    vector_column_name="embedding",
    vector_data=[0.1, 0.2, 0.3, ...],  # 768-dimensional vector
    target_dimension=768,
    key_columns={
        "doc_id": "PMC123456",
        "token_index": 0
    },
    additional_data={
        "token_text": "diabetes"
    }
)
```

### Why This Utility is Mandatory

1. **Consistent Vector Formatting**: Handles proper TO_VECTOR() syntax
2. **Dimension Management**: Automatically truncates or pads vectors
3. **Error Handling**: Provides consistent error handling across the codebase
4. **Security**: Prevents SQL injection through proper parameterization
5. **Maintainability**: Centralizes vector insertion logic

## Vector Search Operations

### Using Vector Search Utilities

The project provides utilities in [`common/vector_sql_utils.py`](common/vector_sql_utils.py:1) for safe vector search operations:

```python
from common.vector_sql_utils import format_vector_search_sql, execute_vector_search

# Format a vector search query
sql = format_vector_search_sql(
    table_name="SourceDocuments",
    vector_column="embedding",
    vector_string="[0.1,0.2,0.3]",
    embedding_dim=768,
    top_k=10,
    id_column="doc_id",
    content_column="text_content"
)

# Execute the search
cursor = connection.cursor()
results = execute_vector_search(cursor, sql)
```

### High-Level Search Functions

Use the functions in [`common/db_vector_search.py`](common/db_vector_search.py:1):

```python
from common.db_vector_search import search_source_documents_dynamically

results = search_source_documents_dynamically(
    iris_connector=connection,
    top_k=10,
    vector_string="[0.1,0.2,0.3,...]"
)
```

## IRIS SQL Vector Functions

### TO_VECTOR()

Converts string representations to vector format:

```sql
TO_VECTOR('0.1,0.2,0.3', 'FLOAT', 3)
TO_VECTOR('[0.1,0.2,0.3]', 'DOUBLE', 3)
```

**Parameters:**
- Vector string (comma-separated values)
- Data type: `'FLOAT'` or `'DOUBLE'`
- Dimension count

### Vector Similarity Functions

#### VECTOR_COSINE()
```sql
VECTOR_COSINE(vector1, vector2)
```
Returns cosine similarity (higher = more similar).

#### VECTOR_DOT_PRODUCT()
```sql
VECTOR_DOT_PRODUCT(vector1, vector2)
```
Returns dot product of two vectors.

#### VECTOR_L2_DISTANCE()
```sql
VECTOR_L2_DISTANCE(vector1, vector2)
```
Returns Euclidean distance (lower = more similar).

### Example Vector Search Query

```sql
SELECT TOP 10 doc_id, text_content,
       VECTOR_COSINE(
           TO_VECTOR(embedding, 'FLOAT', 768),
           TO_VECTOR('[0.1,0.2,0.3,...]', 'FLOAT', 768)
       ) AS similarity_score
FROM RAG.SourceDocuments
WHERE embedding IS NOT NULL
ORDER BY similarity_score DESC
```

## Table Schemas

### RAG.SourceDocuments

```sql
CREATE TABLE RAG.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(1000),
    text_content CLOB,
    embedding VARCHAR(32000),  -- Comma-separated vector string
    metadata VARCHAR(4000),    -- JSON metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### RAG.DocumentTokenEmbeddings

```sql
CREATE TABLE RAG.DocumentTokenEmbeddings (
    doc_id VARCHAR(255),
    token_index INTEGER,
    token_text VARCHAR(500),
    embedding VARCHAR(32000),  -- Comma-separated vector string
    PRIMARY KEY (doc_id, token_index)
);
```

### RAG.KnowledgeGraphNodes

```sql
CREATE TABLE RAG.KnowledgeGraphNodes (
    node_id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(100),
    properties VARCHAR(4000),  -- JSON properties
    embedding VARCHAR(32000)   -- Comma-separated vector string
);
```

## Python Integration

### Using IRISVectorStore

The [`iris_rag.storage.vector_store_iris.IRISVectorStore`](iris_rag/storage/vector_store_iris.py:28) class provides a high-level interface:

```python
from iris_rag.storage.vector_store_iris import IRISVectorStore
from iris_rag.core.models import Document

# Initialize vector store
vector_store = IRISVectorStore(connection_manager, config_manager)

# Add documents with embeddings
documents = [Document(id="doc1", page_content="content", metadata={})]
embeddings = [[0.1, 0.2, 0.3, ...]]  # 768-dimensional vectors
vector_store.add_documents(documents, embeddings)

# Perform similarity search
results = vector_store.similarity_search(
    query_embedding=[0.1, 0.2, 0.3, ...],
    top_k=10,
    filter={"category": "medical"}
)
```

### Connection Management

Use the [`iris_rag.core.connection.ConnectionManager`](iris_rag/core/connection.py:1):

```python
from iris_rag.core.connection import ConnectionManager

connection_manager = ConnectionManager(config)
connection = connection_manager.get_connection("iris")
cursor = connection.cursor()
```

## Performance Considerations

### Query Optimization

1. **Use TOP instead of LIMIT**: IRIS SQL requires `SELECT TOP n` syntax
2. **Filter NULL embeddings**: Always include `WHERE embedding IS NOT NULL`
3. **Index on key columns**: Create indexes on frequently queried columns

### Vector Dimension Management

1. **Consistent dimensions**: Ensure all vectors have the same dimension
2. **Truncation/padding**: Use [`insert_vector()`](common/db_vector_utils.py:6) for automatic handling
3. **Memory usage**: Consider vector dimension impact on storage and performance

### Connection Pooling

```python
# Use connection pooling for better performance
connection_manager = ConnectionManager(config)
with connection_manager.get_connection("iris") as connection:
    # Perform operations
    pass
```

## Best Practices

### 1. Always Use Utility Functions

```python
# ✅ CORRECT: Use the mandatory utility
from common.db_vector_utils import insert_vector
success = insert_vector(cursor, table_name, column_name, vector_data, dimension, keys)

# ❌ WRONG: Direct SQL insertion
cursor.execute("INSERT INTO table (embedding) VALUES (TO_VECTOR(?, 'FLOAT', 768))", [vector_str])
```

### 2. Validate Inputs

```python
from common.vector_sql_utils import validate_vector_string, validate_top_k

# Validate before using in queries
if not validate_vector_string(vector_str):
    raise ValueError("Invalid vector string")

if not validate_top_k(top_k):
    raise ValueError("Invalid top_k value")
```

### 3. Use Proper Error Handling

```python
try:
    results = search_source_documents_dynamically(connection, top_k, vector_string)
except Exception as e:
    logger.error(f"Vector search failed: {e}")
    # Handle error appropriately
```

### 4. Follow SQL Rules

- Use `TOP` instead of `LIMIT`
- Always filter `WHERE embedding IS NOT NULL`
- Use proper column validation for security

## Common Patterns

### Document Similarity Search

```python
def find_similar_documents(query_embedding: List[float], top_k: int = 10):
    vector_string = "[" + ",".join(map(str, query_embedding)) + "]"
    
    return search_source_documents_dynamically(
        iris_connector=connection,
        top_k=top_k,
        vector_string=vector_string
    )
```

### Token-Level Search (ColBERT)

```python
def search_token_embeddings(doc_id: str, query_tokens: List[List[float]]):
    results = []
    for token_embedding in query_tokens:
        vector_string = "[" + ",".join(map(str, token_embedding)) + "]"
        
        sql = format_vector_search_sql(
            table_name="RAG.DocumentTokenEmbeddings",
            vector_column="embedding",
            vector_string=vector_string,
            embedding_dim=768,
            top_k=5,
            id_column="doc_id",
            content_column="token_text",
            additional_where=f"doc_id = '{doc_id}'"
        )
        
        cursor = connection.cursor()
        token_results = execute_vector_search(cursor, sql)
        results.extend(token_results)
        cursor.close()
    
    return results
```

### Batch Vector Insertion

```python
def insert_document_embeddings(doc_id: str, embeddings: List[List[float]], tokens: List[str]):
    cursor = connection.cursor()
    try:
        for i, (embedding, token) in enumerate(zip(embeddings, tokens)):
            success = insert_vector(
                cursor=cursor,
                table_name="RAG.DocumentTokenEmbeddings",
                vector_column_name="embedding",
                vector_data=embedding,
                target_dimension=768,
                key_columns={"doc_id": doc_id, "token_index": i},
                additional_data={"token_text": token}
            )
            if not success:
                logger.warning(f"Failed to insert embedding for token {i}")
        
        connection.commit()
    except Exception as e:
        connection.rollback()
        raise
    finally:
        cursor.close()
```

## Troubleshooting

### Common Issues

1. **"Invalid vector string" errors**
   - Ensure vector strings contain only digits, dots, commas, and brackets
   - Use [`validate_vector_string()`](common/vector_sql_utils.py:36) before queries

2. **Dimension mismatches**
   - Use [`insert_vector()`](common/db_vector_utils.py:6) for automatic dimension handling
   - Verify target_dimension parameter matches your model

3. **SQL injection concerns**
   - Always use the provided utility functions
   - Never construct SQL with direct string interpolation of user input

4. **Performance issues**
   - Add indexes on frequently queried columns
   - Use connection pooling
   - Consider vector dimension optimization

### Debugging Vector Operations

```python
import logging
logging.getLogger('common.db_vector_utils').setLevel(logging.DEBUG)
logging.getLogger('common.vector_sql_utils').setLevel(logging.DEBUG)

# Enable detailed logging for vector operations
```

### Validation Helpers

```python
from common.vector_sql_utils import validate_vector_string, validate_top_k

# Test vector string format
vector_str = "[0.1,0.2,0.3]"
assert validate_vector_string(vector_str), "Invalid vector format"

# Test top_k parameter
assert validate_top_k(10), "Invalid top_k value"
```

## Migration Notes

### From Direct SQL to Utilities

If you have existing code with direct vector SQL:

```python
# OLD: Direct SQL (prohibited)
cursor.execute(
    "INSERT INTO table (embedding) VALUES (TO_VECTOR(?, 'FLOAT', 768))",
    [vector_string]
)

# NEW: Use mandatory utility
from common.db_vector_utils import insert_vector
insert_vector(
    cursor=cursor,
    table_name="table",
    vector_column_name="embedding",
    vector_data=vector_list,  # List[float], not string
    target_dimension=768,
    key_columns={"id": doc_id}
)
```

### Vector Format Migration

```python
# Convert string format to list for utility functions
vector_string = "0.1,0.2,0.3"
vector_list = [float(x) for x in vector_string.split(",")]

# Use with insert_vector utility
insert_vector(cursor, table, column, vector_list, dimension, keys)
```

## References

- [InterSystems IRIS SQL Reference: TO_VECTOR](https://docs.intersystems.com/)
- [InterSystems IRIS SQL Reference: Vector Functions](https://docs.intersystems.com/)
- [Project Vector Utilities](common/vector_sql_utils.py:1)
- [Project Vector Store Implementation](iris_rag/storage/vector_store_iris.py:1)
- [Project Rules (.clinerules)](.clinerules:1)