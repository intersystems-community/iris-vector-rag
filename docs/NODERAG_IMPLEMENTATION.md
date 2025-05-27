# NodeRAG Implementation

This document outlines the implementation of NodeRAG (Node-based Retrieval Augmented Generation) in the RAG templates framework.

## Current Project Status & Blocker Impact

**IMPORTANT:** As of May 21, 2025, full functionality of NodeRAG with newly loaded real PMC data, particularly aspects reliant on vector embeddings (e.g., building graph structures from new embeddings, or initial seed node selection using embeddings), is **BLOCKED**.

This is due to a critical limitation with the InterSystems IRIS ODBC driver and the `TO_VECTOR()` SQL function, which prevents the successful loading of document and node embeddings into the database. While the NodeRAG pipeline logic and SQL CTEs are implemented, their operation with fresh, real-data embeddings cannot be fully tested or utilized until this blocker is resolved.

For more details on this blocker, please refer to [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md).

## Overview

NodeRAG is an advanced retrieval technique that uses a heterogeneous knowledge graph with multiple node types (Entity, Document, Concept, Summary) to find and retrieve information. Unlike traditional vector search approaches, NodeRAG leverages both the structural relationships in the graph and semantic similarity to locate relevant information.

The key distinguishing features of NodeRAG are:
1. **Heterogeneous Graph Structure** - Using different node types to represent various kinds of information
2. **Multi-hop Traversal** - Finding related information by traversing connections in the graph
3. **Hybrid Scoring** - Combining graph structure and vector similarity for ranking
4. **Node Type-aware Context Assembly** - Organizing information by node type for better LLM understanding

## SQL-Based Implementation Approach

While the original NodeRAG concept typically uses Python-based graph libraries like NetworkX, our implementation leverages SQL recursive CTEs (Common Table Expressions) for efficient in-database graph traversal. This approach offers several advantages:

1. **Performance** - Keeping traversal operations in the database reduces data transfer and leverages database optimization
2. **Scalability** - SQL engines are designed to handle large datasets efficiently
3. **Integration** - Tighter integration with IRIS database capabilities
4. **Simplicity** - Reduces dependencies on external graph libraries

## Components Implemented

### 1. SQL Functions (Removed - `common/noderag_cte.sql`)
**Note**: SQL UDF files have been removed as they were not used by the current Python implementation.
- `NodeRAGTraversal`: Core recursive CTE for multi-hop graph traversal (now implemented in Python)
- `GetNodeRAGSeedNodes`: Helper function to identify starting nodes
- `GetNodeContents`: Function to retrieve content for nodes
- `GetNodeCentralityScores`: Function to calculate node importance

### 2. NodeRAG Pipeline (`noderag/pipeline.py`)
- `NodeRAGPipeline` class implementing the full retrieval pipeline
- Methods for initial node identification, graph traversal, and answer generation
- Support for both mock implementations (for testing) and real database connections

### 3. Demo Script (`scripts_to_review/demo_noderag.py`)
- Command-line interface for testing the NodeRAG pipeline. (Note: This script is in `scripts_to_review/`; its canonical status should be confirmed. The main pipeline script [`noderag/pipeline.py`](noderag/pipeline.py:1) may also be runnable for demos if it includes an `if __name__ == '__main__':` block.)
- Support for both single queries and sample query sets
- Pretty printing of results organized by node type

### 4. Test Suite ([`tests/test_noderag.py`](tests/test_noderag.py:1))
- Comprehensive test suite for the NodeRAG pipeline
- Tests for each component of the pipeline
- Both mock-based unit tests and simulated real connection tests

## The Knowledge Graph Structure

The NodeRAG knowledge graph is stored in two main tables:

1. **KnowledgeGraphNodes**
   ```sql
   CREATE TABLE KnowledgeGraphNodes (
       node_id VARCHAR(255) PRIMARY KEY,
       node_type VARCHAR(50),
       node_name VARCHAR(255),
       content CLOB,
       embedding VECTOR,
       metadata_json CLOB
   )
   ```

2. **KnowledgeGraphEdges**
   ```sql
   CREATE TABLE KnowledgeGraphEdges (
       edge_id VARCHAR(255) PRIMARY KEY,
       source_node_id VARCHAR(255),
       target_node_id VARCHAR(255),
       relationship_type VARCHAR(50),
       weight FLOAT,
       properties_json CLOB,
       FOREIGN KEY (source_node_id) REFERENCES KnowledgeGraphNodes(node_id),
       FOREIGN KEY (target_node_id) REFERENCES KnowledgeGraphNodes(node_id)
   )
   ```

## Recursive CTE for Graph Traversal

The core of our implementation is the SQL recursive CTE that performs graph traversal.
**Note on Parameters in CTE:** The SQL snippet below uses placeholders like `:query_embedding_str`, `:start_node_count`, `:score_decay`, `:hybrid_weight`, and `:max_depth`. Due to IRIS SQL limitations where `TO_VECTOR()` and `LIMIT` (or `TOP`) do not accept direct parameter markers, these placeholders are substituted with validated values when the final SQL query is constructed in the Python code (e.g., using f-strings or other string formatting). The `common/vector_sql_utils.py` module provides utilities for safe construction of such dynamic SQL, especially for the vector string part.

```sql
WITH RECURSIVE NodePath (start_node, current_node, path, depth, score) AS (
    -- Base case: start with seed nodes from vector similarity
    SELECT
        n.node_id,
        n.node_id,
        CAST(n.node_id AS VARCHAR(1000)),
        0,
        VECTOR_COSINE_SIMILARITY(n.embedding, TO_VECTOR(:query_embedding_str)) AS score -- :query_embedding_str is replaced by validated string literal
    FROM KnowledgeGraphNodes n
    WHERE n.node_type IN ('Entity', 'Document', 'Concept', 'Summary')
    ORDER BY score DESC
    LIMIT :start_node_count -- :start_node_count is replaced by validated integer literal
    
    UNION ALL
    
    -- Recursive case: traverse to connected nodes
    SELECT
        p.start_node,
        e.target_node_id,
        p.path || ',' || e.target_node_id,
        p.depth + 1,
        -- Hybrid scoring that combines path score and direct relevance
        p.score * :score_decay + -- :score_decay is replaced by validated numeric literal
        VECTOR_COSINE_SIMILARITY(n2.embedding, TO_VECTOR(:query_embedding_str)) * :hybrid_weight -- :hybrid_weight is replaced by validated numeric literal
    FROM NodePath p
    JOIN KnowledgeGraphEdges e ON p.current_node = e.source_node_id
    JOIN KnowledgeGraphNodes n2 ON e.target_node_id = n2.node_id
    WHERE p.depth < :max_depth -- :max_depth is replaced by validated integer literal
      -- Filter relevant relationship types
      AND CASE
            WHEN e.relationship_type = 'IS_PART_OF' THEN 1
            WHEN e.relationship_type = 'RELATED_TO' THEN 1
            WHEN e.relationship_type = 'MENTIONS' THEN 1
            WHEN e.relationship_type = 'CITES' THEN 1
            WHEN e.relationship_type = 'DEFINES' THEN 1
            ELSE 0
          END = 1
      -- Avoid cycles by checking path
      AND NOT POSITION(',' || e.target_node_id || ',' IN ',' || p.path || ',') > 0
)
```

## Usage

1. **Graph Building:**
   Before using NodeRAG with new data, you need to build and populate the knowledge graph tables (`KnowledgeGraphNodes`, `KnowledgeGraphEdges`).
   **Note on Embedding Loading:** Storing new `document_embedding` or `entity_embedding` values into the `embedding VECTOR` column of `KnowledgeGraphNodes` is currently **BLOCKED** by the `TO_VECTOR`/ODBC issue. The graph can be built with text content, but vector-based operations on new nodes will be affected.

   ```python
   # Example code for building a knowledge graph (simplified)
   # This would typically be done in a data loading/indexing stage
   
   # Create nodes
   for document in documents:
       # Extract entities, concepts, etc.
       entities = extract_entities(document.text)
       
       # Create document node
       doc_node_id = f"doc_{document.id}"
       # store_node would attempt to INSERT into KnowledgeGraphNodes.
       # Storing document_embedding (VECTOR type) is currently problematic for new data.
       store_node(doc_node_id, "Document", document.text, document_embedding)
       
       # Create entity nodes and edges
       for entity in entities:
           entity_node_id = f"entity_{entity.id}"
           store_node(entity_node_id, "Entity", entity.description, entity_embedding)
           
           # Create edge between document and entity
           store_edge(doc_node_id, entity_node_id, "MENTIONS", 1.0)
   ```

2. **Query Execution:**
   ```python
   from noderag.pipeline import NodeRAGPipeline
   from common.iris_connector import get_iris_connection
   from common.embedding_utils import get_embedding_model
   
   # Create the pipeline
   iris_connector = get_iris_connection()
   embedding_model = get_embedding_model()
   embedding_func = lambda text: embedding_model.encode(text)
   
   pipeline = NodeRAGPipeline(
       iris_connector=iris_connector,
       embedding_func=embedding_func,
       llm_func=your_llm_function
   )
   
   # Run query
   result = pipeline.run("What is the relationship between diabetes and insulin?")
   
   # Process results
   answer = result["answer"]
   retrieved_nodes = result["retrieved_documents"]
   ```

## Key Advantages of NodeRAG

- **Rich Context Structure**: Organizes information by node type, providing more structure to the LLM
- **Multi-hop Information Retrieval**: Can discover information through connections that might not be found by direct vector similarity
- **Knowledge Integration**: Leverages both structured knowledge (graph) and unstructured content (text)
- **Flexible Traversal**: Can be tuned to prioritize different relationship types or traversal depths based on query characteristics

## Implementation Notes

- The hybrid scoring approach balances both the graph structure (path from initial nodes) and direct relevance (vector similarity to the query)
- Node types are preserved throughout the pipeline and used to structure the context for the LLM
- The implementation includes detailed logging for better debugging and performance analysis
- Mock implementations allow for testing without requiring a real knowledge graph
