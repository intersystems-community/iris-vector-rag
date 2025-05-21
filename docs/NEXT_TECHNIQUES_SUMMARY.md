# Next RAG Techniques Implementation Summary (ARCHIVAL - OUTDATED)

**IMPORTANT NOTICE: This document is outdated and kept for historical reference only. It describes an early plan for implementing ColBERT, NodeRAG, and GraphRAG.**

**All six RAG techniques mentioned in this project (BasicRAG, HyDE, CRAG, ColBERT, NodeRAG, and GraphRAG) have since been implemented.**

For current information, please refer to:
- Main project status: [`README.md`](README.md:1)
- Individual RAG technique implementation details:
  - [`docs/COLBERT_IMPLEMENTATION.md`](docs/COLBERT_IMPLEMENTATION.md:1)
  - [`docs/NODERAG_IMPLEMENTATION.md`](docs/NODERAG_IMPLEMENTATION.md:1)
  - [`docs/GRAPHRAG_IMPLEMENTATION.md`](docs/GRAPHRAG_IMPLEMENTATION.md:1)

---

(Original content below for historical context only)

This document outlines the implementation plan for three advanced RAG techniques - ColBERT, NodeRAG, and GraphRAG - which will build upon our successfully implemented CRAG (Corrective Retrieval Augmented Generation) pipeline.

## 1. ColBERT (Contextualized Late Interaction over BERT)

### Core Mechanism
- **Multi-Vector Embeddings**:
  - Document Encoding: Generate token-level embeddings for **every token** in each document
  - Query Encoding: Generate token-level embeddings for the query tokens
- **Late Interaction (MaxSim)**:
  - For each query token, find its maximum similarity with all document tokens
  - Calculate the sum of these maximum similarities as the document's score
- **Retrieval & Answer Generation**:
  - Retrieve top-k documents based on the MaxSim scores
  - Generate answer using retrieved documents as context

### Implementation Details
- **Requirements**:
  - Token-level embeddings table: `DocumentTokenEmbeddings`
  - Efficient MaxSim calculation (either as IRIS UDF or client-side function)
  - Token-level embeddings for both documents and queries

### Implementation Challenges
- Database volume from token-level embeddings
- Efficient MaxSim computation (could be computationally expensive)
- Potential for two-phase retrieval (first finding candidate documents, then detailed MaxSim)

## 2. NodeRAG

### Core Mechanism
- **Heterograph Construction**:
  - Build a graph with multiple node types (Entity, Event, Document, etc.)
  - Process includes decomposition, augmentation, and enrichment
- **Multi-hop Graph Search**:
  - Identify relevant starting nodes based on the query
  - Traverse the graph considering node/edge types and weights
- **Hybrid Retrieval**:
  - Combine graph structural information with vector similarity
  - Collect content from traversed nodes for context

### Implementation Details
- **Requirements**:
  - `KnowledgeGraphNodes` and `KnowledgeGraphEdges` tables
  - Graph traversal algorithms (Python-centric using NetworkX or similar)
  - Query-to-node matching (initial node identification)

### Implementation Challenges
- Entity extraction and relationship identification
- Optimizing graph traversal for performance
- Balancing between structural and vector-based relevance

## 3. GraphRAG

### Core Mechanism
- **Knowledge Graph Construction**:
  - Extract entities and relationships from documents
  - Store in a structured knowledge graph
- **Initial Node Identification**:
  - Find starting nodes relevant to the query using vector search or entity linking
- **Graph Traversal**:
  - Use recursive CTEs in SQL to traverse the graph efficiently
  - Find related nodes and paths from starting points
- **Context Assembly & Answer Generation**:
  - Collect information from traversed nodes
  - Generate answer based on the graph-derived context

### Implementation Details
- **Requirements**:
  - Same tables as NodeRAG but with focus on SQL-based traversal
  - Recursive CTE implementations for graph exploration
  - `kg_edges` view for SQL-friendly graph access

### Implementation Challenges
- Efficient recursive CTE formulation
- Balancing traversal depth vs breadth
- Handling potentially circular paths or unreasonable traversals

## Technical Requirements

### Common Infrastructure
- **Vector Operations**: All techniques require IRIS vector operations using `TO_VECTOR` and vector similarity functions
- **HNSW Index**: Vector indices for efficient similarity search:
  ```sql
  CREATE INDEX HNSWIndex ON TABLE SourceDocuments (embedding) 
    AS HNSW(Distance='Cosine')
  ```
- **Query Time Functions**: Properly formatted SQL for vector operations:
  ```sql
  SELECT TOP ? doc_id, text_content, 
         VECTOR_COSINE_SIMILARITY(embedding, TO_VECTOR(?)) AS similarity_score
  FROM SourceDocuments
  ORDER BY similarity_score DESC
  ```

### Testing Approach
- Unit tests for component logic
- SQL validation tests against real IRIS
- End-to-end metrics tests (recall, faithfulness, latency)
- Comparative benchmarking between techniques

## Implementation Sequence
1. **Data Layer**: Enhance PMC document parsing and entity extraction
2. **ColBERT Implementation**: Focus on token-level embeddings and MaxSim
3. **Node/Graph Common Components**: Shared knowledge graph construction
4. **NodeRAG Specific**: Python-based graph traversal
5. **GraphRAG Specific**: SQL recursive CTE traversal
6. **Evaluation**: Comparative metrics across all techniques
