# GraphRAG Implementation

This document outlines the implementation of GraphRAG (Graph-based Retrieval Augmented Generation) in the RAG templates framework.

## Current Project Status & Blocker Impact

**IMPORTANT:** As of May 21, 2025, full functionality of GraphRAG with newly loaded real PMC data, particularly aspects reliant on vector embeddings (e.g., building graph structures from new embeddings, or initial seed node selection using embeddings), is **BLOCKED**.

This is due to a critical limitation with the InterSystems IRIS ODBC driver and the `TO_VECTOR()` SQL function, which prevents the successful loading of document and node embeddings into the database. While the GraphRAG pipeline logic and SQL CTEs are implemented, their operation with fresh, real-data embeddings cannot be fully tested or utilized until this blocker is resolved.

For more details on this blocker, refer to [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md).

## Overview

GraphRAG is an advanced retrieval technique that uses a knowledge graph structure to improve information retrieval for question answering. Unlike traditional vector search methods, GraphRAG leverages both the semantic similarity and the graph structure (relationships between entities) to find relevant information.

The key distinguishing features of GraphRAG are:
1. **Graph Traversal** - Starting from seed nodes, navigate through connections to discover related information
2. **Hybrid Scoring** - Combining graph proximity with semantic similarity for node ranking
3. **Path-based Relevance** - Considering the entire path to a node, not just point-to-point similarity
4. **Multi-hop Discovery** - Uncovering information that requires multiple reasoning steps

## SQL-Based Implementation Approach

Our implementation leverages SQL recursive CTEs (Common Table Expressions) for efficient in-database graph traversal. This approach offers several advantages:

1. **Performance** - Keeping traversal operations in the database reduces data transfer and leverages database optimization
2. **Scalability** - SQL engines are designed to handle large datasets efficiently
3. **Integration** - Tighter integration with IRIS database capabilities
4. **Consistency** - Following the same SQL-based pattern as our NodeRAG implementation

## Components Implemented

### 1. SQL Functions (Removed - `common/graphrag_cte.sql`)
**Note**: SQL UDF files have been removed as they were not used by the current Python implementation.
- `GraphRAGTraversal`: Core recursive CTE for multi-hop graph traversal (now implemented in Python)
- `FindGraphRAGStartNodes`: Helper function to identify starting nodes via vector similarity
- `GetGraphNodeDetails`: Function to retrieve details for nodes
- `GetConnectedNodes`: Function to get connections for a specific node

### 2. GraphRAG Pipeline (`graphrag/pipeline.py`)
- `GraphRAGPipeline` class implementing the full retrieval pipeline
- Functions for initial node identification, graph traversal, and answer generation
- Support for both mock implementations (for testing) and real database connections

### 3. Demo Script (`scripts_to_review/demo_graphrag.py`)
- Command-line interface for testing the GraphRAG pipeline. (Note: This script is in `scripts_to_review/`; its canonical status should be confirmed. The main pipeline script [`graphrag/pipeline.py`](graphrag/pipeline.py:1) may also be runnable for demos if it includes an `if __name__ == '__main__':` block.)
- Support for both single queries and sample query sets
- Visualization of graph traversal paths
- Pretty printing of results organized by node type

### 4. Tests ([`tests/test_graphrag.py`](tests/test_graphrag.py:1))
- Comprehensive test suite for the GraphRAG pipeline
- Tests for each component of the pipeline
- Both mock-based unit tests and simulated real connection tests

## The Knowledge Graph Structure

The GraphRAG knowledge graph is stored in two main tables:

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
**Note on Parameters in CTE:** The SQL snippet below uses placeholders like `:query_embedding_str`, `:score_decay`, `:hybrid_weight`, and `:max_depth`. The `StartNodes` subquery implies initial seed nodes are provided. Due to IRIS SQL limitations where `TO_VECTOR()` does not accept direct parameter markers, these placeholders (especially `:query_embedding_str`) are substituted with validated values when the final SQL query is constructed in the Python code (e.g., using f-strings or other string formatting). The `common/vector_sql_utils.py` module provides utilities for safe construction of such dynamic SQL.

```sql
WITH RECURSIVE PathCTE (start_node, current_node, path, depth, score) AS (
    -- Base case: start with seed nodes from vector similarity
    SELECT
        n.node_id,
        n.node_id,
        CAST(n.node_id AS VARCHAR(1000)),
        0,
        VECTOR_COSINE_SIMILARITY(n.embedding, TO_VECTOR(:query_embedding_str)) AS score -- :query_embedding_str is replaced by validated string literal
    FROM KnowledgeGraphNodes n
    WHERE n.node_id IN (SELECT node_id FROM StartNodes) -- StartNodes generated dynamically
    
    UNION ALL
    
    -- Recursive case: traverse connected nodes
    SELECT
        p.start_node,
        e.target_node_id,
        p.path || ',' || e.target_node_id,
        p.depth + 1,
        -- Calculate hybrid score combining path score and direct relevance
        p.score * :score_decay + -- :score_decay is replaced by validated numeric literal
        VECTOR_COSINE_SIMILARITY(
            target_node.embedding,
            TO_VECTOR(:query_embedding_str)
        ) * :hybrid_weight -- :hybrid_weight is replaced by validated numeric literal
    FROM PathCTE p
    JOIN KnowledgeGraphEdges e ON p.current_node = e.source_node_id
    JOIN KnowledgeGraphNodes target_node ON e.target_node_id = target_node.node_id
    WHERE p.depth < :max_depth -- :max_depth is replaced by validated integer literal
      -- Avoid cycles in traversal
      AND NOT POSITION(',' || e.target_node_id || ',' IN ',' || p.path || ',') > 0
)
```

## Usage

1. **Graph Building:**
   Before using GraphRAG with new data, you need to build and populate the knowledge graph tables (`KnowledgeGraphNodes`, `KnowledgeGraphEdges`).
   **Note on Embedding Loading:** Storing new `embedding` values into the `embedding VECTOR` column of `KnowledgeGraphNodes` (line 59) is currently **BLOCKED** by the `TO_VECTOR`/ODBC issue. The graph can be built with text content, but vector-based operations on new nodes will be affected.

   ```python
   # Example code for building a knowledge graph (simplified)
   # This would typically be done in a data loading/indexing stage
   
   # Add nodes
   for entity in entities:
       # Create entity node
       # Storing entity.embedding (VECTOR type) is currently problematic for new data.
       execute_sql("""
           INSERT INTO KnowledgeGraphNodes
           (node_id, node_type, node_name, content, embedding)
           VALUES (?, ?, ?, ?, ?)
       """, (entity.id, "Entity", entity.name, entity.description, entity.embedding))
   
   # Add edges
   for relationship in relationships:
       # Create relationship edge
       execute_sql("""
           INSERT INTO KnowledgeGraphEdges 
           (edge_id, source_node_id, target_node_id, relationship_type, weight) 
           VALUES (?, ?, ?, ?, ?)
       """, (relationship.id, relationship.source, relationship.target, relationship.type, relationship.weight))
   ```

2. **Query Execution:**
   ```python
   from graphrag.pipeline import GraphRAGPipeline
   from common.iris_connector import get_iris_connection
   from common.embedding_utils import get_embedding_model
   
   # Get components
   iris_connector = get_iris_connection()
   embedding_model = get_embedding_model()
   embedding_func = lambda text: embedding_model.encode(text)
   
   # Create pipeline
   pipeline = GraphRAGPipeline(
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

## Key Advantages of GraphRAG

- **Relationship Discovery**: Finds information connected through relationships that might not be found by direct vector similarity
- **Multi-hop Reasoning**: Can discover information that requires multiple steps or connections to reach
- **Explanatory Paths**: The traversal path provides a natural explanation of how information is connected
- **Complex Query Support**: Better equipped to handle complex queries requiring integration of multiple information sources

## Differences from NodeRAG

While both GraphRAG and NodeRAG use graph structures, they have different emphases:

- **NodeRAG** organizes nodes by type (Entity, Document, Concept) and focuses on presenting information by node type
- **GraphRAG** focuses more on the path through the graph, with less emphasis on node types
- **NodeRAG** uses node types to guide traversal behavior
- **GraphRAG** uses relationship types and weights more prominently in traversal decisions

## Implementation Notes

- The hybrid scoring approach balances both the graph structure (path from initial nodes) and direct relevance (vector similarity to the query)
- The recursive CTE approach allows efficient traversal entirely within the database
- Mock implementations are provided for testing without requiring a real knowledge graph
- The implementation includes detailed logging for better debugging and performance analysis
