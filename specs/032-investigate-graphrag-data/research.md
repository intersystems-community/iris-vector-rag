# Research: GraphRAG Data/Setup Investigation

**Feature**: 032-investigate-graphrag-data | **Date**: 2025-10-06
**Phase**: 0 (Research) | **Input**: Architecture analysis from codebase

## R1: GraphRAG Architecture Analysis

### Question
How does HybridGraphRAGPipeline retrieve documents? What data sources does it query?

### Findings

**Source File**: `iris_rag/pipelines/hybrid_graphrag.py`

**Pipeline Hierarchy**:
- `HybridGraphRAGPipeline` extends `GraphRAGPipeline`
- `GraphRAGPipeline` extends `RAGPipeline` (base class)
- HybridGraphRAGPipeline adds iris_graph_core integration (RRF fusion, HNSW optimization)

**Data Sources Used**:
1. **Vector Store**: IRIS vector search via `self.vector_store`
2. **Knowledge Graph**: Entity/relationship/community tables (if iris_graph_core available)
3. **Text Search**: iFind text search (if iris_graph_core available)

**Retrieval Flow** (from code analysis):
```python
# HybridGraphRAGPipeline initialization:
# 1. Initializes parent GraphRAGPipeline with entity extraction service
# 2. Discovers iris_graph_core modules (optional)
# 3. Creates IRIS connection for graph operations
# 4. Initializes HybridRetrievalMethods if iris_graph_core available
```

**Key Dependencies**:
- `EntityExtractionService` (from `iris_rag.services.entity_extraction`)
- `iris_graph_core` modules (optional, gracefully degrades)
- IRIS database connection for knowledge graph tables

**Critical Observation**: HybridGraphRAGPipeline falls back to standard GraphRAGPipeline if iris_graph_core not available, BUT still requires knowledge graph data from entity extraction.

## R2: Entity Extraction Pipeline

### Question
When and how does entity extraction occur during document ingestion?

### Findings

**Source File**: `iris_rag/services/entity_extraction.py`

**Entity Extraction Service** (`OntologyAwareEntityExtractor`):
- **Extraction Method**: Configured via `config_manager.get("entity_extraction", {})`
- **Default Method**: `"ontology_hybrid"` (rule-based + ontology mapping)
- **Entity Types**: Configurable, defaults to `["ENTITY", "CONCEPT", "PROCESS"]`
- **Storage**: Uses `EntityStorageAdapter` to persist entities to IRIS

**Extraction Configuration**:
```python
# From entity_extraction.py:
self.method = self.config.get("method", "ontology_hybrid")
self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
self.enabled_types = set(self.config.get("entity_types", ["ENTITY", "CONCEPT", "PROCESS"]))
self.max_entities_per_doc = self.config.get("max_entities", 100)
```

**Ontology Integration**:
- Uses `GeneralOntologyPlugin` for domain-agnostic entity extraction
- Supports auto-detection of domain from ontology content
- Requires ontology to be loaded and enabled in config

**Storage Mechanism**:
- `EntityStorageAdapter` writes to IRIS knowledge graph tables
- Entities linked to source documents via document IDs
- Relationships extracted from entity co-occurrence patterns

**Critical Observation**: Entity extraction is initialized in GraphRAGPipeline constructor but **requires explicit invocation during document loading**. Service exists but may not be called automatically during `load_documents()`.

## R3: Knowledge Graph Schema

### Question
What tables/schema are required for GraphRAG? What's the expected state after loading 10 documents?

### Findings

**Source File**: `iris_rag/storage/schema_manager.py`

**Schema Manager** (`SchemaManager`):
- Central authority for all vector dimensions and schema management
- Tracks schema versions and configuration changes
- Provides automatic migration support

**Knowledge Graph Tables** (from schema inspection):

Expected tables:
1. **RAG.Entities**: Entity storage with types, names, descriptions, embeddings
2. **RAG.Relationships**: Entity-to-entity relationships with types and strength
3. **RAG.Communities**: Entity clusters for multi-hop reasoning

**Schema Structure** (expected):
```sql
-- Entities table
CREATE TABLE RAG.Entities (
    id VARCHAR PRIMARY KEY,
    name VARCHAR,
    type VARCHAR,
    description CLOB,
    document_id VARCHAR,  -- Link to source document
    embedding VECTOR(DOUBLE, 384),  -- Entity embedding
    metadata JSON
)

-- Relationships table
CREATE TABLE RAG.Relationships (
    id VARCHAR PRIMARY KEY,
    source_entity_id VARCHAR,
    target_entity_id VARCHAR,
    relationship_type VARCHAR,
    strength DOUBLE,
    metadata JSON
)

-- Communities table
CREATE TABLE RAG.Communities (
    id VARCHAR PRIMARY KEY,
    entity_ids JSON,  -- Array of entity IDs
    summary CLOB,
    metadata JSON
)
```

**Expected State After 10 Documents**:
- **Entities**: 50-200 entities (5-20 per document average)
- **Relationships**: 100-500 relationships (entity co-occurrences)
- **Communities**: 10-50 communities (semantic clusters)

**Schema Initialization**:
- SchemaManager ensures tables exist via `ensure_schema_metadata_table()`
- Table creation managed by `SchemaManager.create_schema_manager(pipeline_type, ...)`
- GraphRAG requires schema manager to create knowledge graph tables

**Critical Observation**: Schema manager exists but **knowledge graph tables may not be created if entity extraction is disabled or if schema_manager not invoked during pipeline setup**.

## R4: Comparison with Working Pipelines

### Question
What data do basic/crag pipelines access vs GraphRAG?

### Findings

**Pipeline Data Requirements Comparison**:

| Pipeline | Vector Store | Metadata Tables | Knowledge Graph | Text Index |
|----------|--------------|-----------------|-----------------|------------|
| BasicRAGPipeline | ✅ Required | ❌ Optional | ❌ Not used | ❌ Not used |
| BasicRAGRerankingPipeline | ✅ Required | ✅ Required | ❌ Not used | ❌ Not used |
| CRAGPipeline | ✅ Required | ✅ Required | ❌ Not used | ❌ Not used |
| GraphRAGPipeline | ✅ Required | ✅ Required | ✅ **REQUIRED** | ❌ Not used |
| HybridGraphRAGPipeline | ✅ Required | ✅ Required | ✅ **REQUIRED** | ✅ Optional |

**Key Differences**:

1. **Basic/CRAG**: Only require vector embeddings in vector store
   - Documents → Chunks → Embeddings → Vector table
   - Retrieval: Vector similarity search only

2. **GraphRAG**: Requires vector embeddings **AND** knowledge graph
   - Documents → Chunks → Embeddings → Vector table
   - Documents → Entity Extraction → Entities/Relationships → Knowledge graph tables
   - Retrieval: Vector search + Graph traversal

**Data Loading Flow Comparison**:

**BasicRAGPipeline.load_documents()**:
```python
# 1. Chunk documents
# 2. Generate embeddings
# 3. Store in vector table
# ✅ COMPLETE - No additional steps needed
```

**GraphRAGPipeline.load_documents()**:
```python
# 1. Chunk documents
# 2. Generate embeddings
# 3. Store in vector table
# 4. Extract entities from documents  ← CRITICAL STEP
# 5. Store entities in knowledge graph tables  ← CRITICAL STEP
# 6. Extract relationships between entities  ← CRITICAL STEP
# 7. Generate communities from graph structure  ← CRITICAL STEP
# ❓ ARE STEPS 4-7 ACTUALLY EXECUTED?
```

**Critical Discovery**: Working pipelines (basic, crag) only need vector data. GraphRAG requires **additional entity extraction and knowledge graph construction** during document loading.

**Hypothesis**: GraphRAG returns zero results because steps 4-7 (entity extraction, relationship extraction, community generation) are **not being executed** during `make load-data`.

## Research Summary

### Root Cause Hypothesis

GraphRAG pipeline has 0% retrieval because:

1. **Knowledge graph tables may not exist** (schema not initialized for GraphRAG)
2. **Entity extraction is not triggered** during `make load-data` workflow
3. **GraphRAG queries empty knowledge graph** and finds no entities/relationships
4. **Retrieval returns empty results** because graph traversal finds nothing

### Evidence Supporting Hypothesis

1. **Other pipelines work**: Basic/CRAG only need vector data (which IS loaded)
2. **GraphRAG shows empty contexts**: No documents retrieved = no graph data
3. **No crashes**: Pipeline executes successfully but finds no data
4. **Entity extraction service exists**: Code is present but may not be invoked
5. **Schema manager exists**: But GraphRAG tables may not be created

### Next Steps (Phase 1)

1. Create diagnostic scripts to verify:
   - Do knowledge graph tables exist?
   - Are tables empty or populated?
   - Is entity extraction called during load_data?

2. Identify the exact missing step in data loading workflow

3. Document fix path (either fix load_data or add separate graphrag setup step)

### Key Files for Investigation

- `iris_rag/pipelines/hybrid_graphrag.py` - GraphRAG retrieval logic
- `iris_rag/services/entity_extraction.py` - Entity extraction service
- `iris_rag/storage/schema_manager.py` - Schema management
- `iris_rag/storage/enterprise_storage.py` - Entity storage (likely location)
- `Makefile` - load-data target implementation
- `data/loader_fixed.py` - Document loading logic

---

**Research Phase Complete** - Ready for Phase 1 (Design & Contracts)
