# Data Model: IRIS EMBEDDING Support

**Feature**: 051-add-native-iris
**Date**: 2025-01-06
**Source**: Extracted from spec.md Key Entities section

## Entity Definitions

### 1. Embedding Configuration

**Purpose**: Represents IRIS %Embedding.Config table entries with model settings and optional entity extraction configuration.

**Fields**:
- `name` (string, required): Unique identifier for this configuration
- `model_name` (string, required): HuggingFace model ID (e.g., "sentence-transformers/all-MiniLM-L6-v2")
- `hf_cache_path` (string, required): Local path to HuggingFace model cache directory
- `python_path` (string, required): Path to Python executable with required packages
- `embedding_class` (string, required): IRIS class name (e.g., "%Embedding.SentenceTransformers")
- `description` (string, optional): Human-readable description
- `enable_entity_extraction` (boolean, default=False): Whether to extract entities during vectorization
- `entity_types` (list[string], optional): Entity types to extract (e.g., ["Disease", "Symptom", "Medication"])
- `batch_size` (int, default=32): Documents per batch for embedding generation
- `device_preference` (string, default="auto"): GPU preference ("cuda", "mps", "cpu", "auto")

**Validation Rules**:
- `model_name` must be valid HuggingFace model ID or local path
- `hf_cache_path` must be writable directory
- `python_path` must be executable file
- `batch_size` must be positive integer
- `device_preference` must be one of: "cuda", "mps", "cpu", "auto"
- If `enable_entity_extraction=True`, `entity_types` must not be empty

**State Transitions**: N/A (configuration is read-only during runtime)

**Relationships**:
- Has many EMBEDDING Columns (1:N)
- Has one Cached Model Instance (1:1 during runtime)

**Storage**: IRIS %Embedding.Config table + extended JSON configuration field

### 2. EMBEDDING Column

**Purpose**: Represents SQL column definition that auto-vectorizes a source text column using specified Embedding Configuration.

**Fields**:
- `table_name` (string, required): Name of IRIS table containing this column
- `column_name` (string, required): Name of the EMBEDDING column
- `source_column` (string, required): Name of text column to vectorize
- `config_name` (string, required): Reference to Embedding Configuration
- `vector_dimension` (int, derived): Dimension of embedding vectors (from model)
- `created_at` (timestamp, auto): When column was added to table
- `last_updated` (timestamp, auto): When column definition was last modified

**Validation Rules**:
- `table_name` must exist in IRIS database
- `column_name` must not already exist in table
- `source_column` must exist in table and be text type
- `config_name` must reference existing Embedding Configuration
- `vector_dimension` must match model output dimension

**State Transitions**: N/A (column definition is static after creation)

**Relationships**:
- Belongs to one Embedding Configuration (N:1)
- Generates many Vectorized Documents (1:N)

**Storage**: IRIS table schema metadata + column definition

### 3. Cached Model Instance

**Purpose**: Represents in-memory embedding model with device allocation, reference count, and performance metrics.

**Fields**:
- `config_name` (string, required): Reference to Embedding Configuration
- `model` (SentenceTransformer, required): Loaded model instance
- `device` (string, required): Actual device ("cuda:0", "mps", "cpu")
- `load_time_ms` (float, required): Time to load model in milliseconds
- `reference_count` (int, default=0): Number of active references
- `last_access_time` (timestamp, required): Last time model was used
- `memory_usage_mb` (float, required): Model memory footprint
- `cache_hits` (int, default=0): Number of times model was reused from cache
- `cache_misses` (int, default=0): Number of times model had to be loaded
- `total_embeddings_generated` (int, default=0): Count of embeddings generated

**Validation Rules**:
- `device` must be one of: "cuda:0", "mps", "cpu"
- `reference_count` must be non-negative
- `memory_usage_mb` must be positive
- Cache hit rate = cache_hits / (cache_hits + cache_misses) should be >95%

**State Transitions**:
```
UNLOADED → LOADING → LOADED → EVICTING → UNLOADED
```
- UNLOADED: No model in memory
- LOADING: Model being loaded from disk/network
- LOADED: Model ready for embedding generation
- EVICTING: Model being removed from cache (memory pressure)

**Relationships**:
- Belongs to one Embedding Configuration (1:1)
- Used by many concurrent requests (1:N)

**Storage**: In-memory only (Python dictionary, not persisted)

### 4. Vectorized Document

**Purpose**: Represents text content with generated embedding vector, source metadata, and optional extracted entities.

**Fields**:
- `doc_id` (UUID, required): Unique document identifier
- `text_content` (string, required): Original text that was vectorized
- `embedding_vector` (vector, required): Generated embedding (float array)
- `config_name` (string, required): Configuration used for vectorization
- `source_table` (string, required): IRIS table name
- `source_column` (string, required): IRIS column name
- `row_id` (int, required): IRIS row identifier
- `extracted_entities` (list[EntityExtractionResult], optional): Entities found in text
- `extraction_timestamp` (timestamp, required): When vectorization occurred
- `embedding_time_ms` (float, required): Time to generate embedding
- `entity_extraction_time_ms` (float, optional): Time to extract entities

**Validation Rules**:
- `text_content` must not be empty
- `embedding_vector` dimension must match config model dimension
- `row_id` must reference valid row in source_table
- If `extracted_entities` not empty, `entity_extraction_time_ms` must be set

**State Transitions**: N/A (documents are immutable after vectorization)

**Relationships**:
- Belongs to one EMBEDDING Column (N:1)
- Has many Entity Extraction Results (1:N, optional)

**Storage**: IRIS table with EMBEDDING column (vectors stored in IRIS)

### 5. Entity Extraction Result

**Purpose**: Represents entities extracted during vectorization with type, text span, confidence, and relationships.

**Fields**:
- `entity_id` (UUID, required): Unique entity identifier
- `doc_id` (UUID, required): Reference to source document
- `entity_type` (string, required): Type of entity (e.g., "Disease", "Medication")
- `entity_text` (string, required): Actual entity text (e.g., "diabetes", "insulin")
- `text_span_start` (int, required): Character offset where entity starts
- `text_span_end` (int, required): Character offset where entity ends
- `confidence_score` (float, required): Extraction confidence (0.0-1.0)
- `relationships` (list[dict], optional): Relationships to other entities
- `extraction_method` (string, required): How entity was extracted ("llm_batch", "llm_single")
- `extraction_timestamp` (timestamp, required): When entity was extracted

**Validation Rules**:
- `entity_type` must be in configured entity_types list
- `entity_text` must not be empty
- `text_span_end` must be > `text_span_start`
- `confidence_score` must be between 0.0 and 1.0
- Entity text must match substring at specified span

**State Transitions**: N/A (entities are immutable after extraction)

**Relationships**:
- Belongs to one Vectorized Document (N:1)
- Has many relationships to other Entity Extraction Results (N:N)

**Storage**: GraphRAG knowledge graph tables (iris-vector-graph schema)

### 6. Model Cache Statistics

**Purpose**: Represents aggregate performance metrics for monitoring and optimization.

**Fields**:
- `config_name` (string, required): Configuration being monitored
- `measurement_window_start` (timestamp, required): Start of measurement period
- `measurement_window_end` (timestamp, required): End of measurement period
- `total_cache_hits` (int, required): Cache hits during window
- `total_cache_misses` (int, required): Cache misses during window
- `cache_hit_rate` (float, derived): Hits / (Hits + Misses)
- `avg_embedding_time_ms` (float, required): Average time per embedding
- `p95_embedding_time_ms` (float, required): 95th percentile embedding time
- `gpu_utilization_pct` (float, optional): Average GPU utilization
- `memory_usage_peak_mb` (float, required): Peak memory usage
- `model_load_count` (int, required): Times model was loaded
- `model_eviction_count` (int, required): Times model was evicted
- `total_embeddings_generated` (int, required): Count of embeddings

**Validation Rules**:
- `measurement_window_end` must be >= `measurement_window_start`
- `cache_hit_rate` must be between 0.0 and 1.0
- Target: `cache_hit_rate` >= 0.95 (95%)
- Target: `p95_embedding_time_ms` < 50ms (for cached models)
- Target: `gpu_utilization_pct` >= 0.80 (80%) when GPU available

**State Transitions**: N/A (statistics are snapshots)

**Relationships**:
- Belongs to one Embedding Configuration (N:1)
- Aggregates data from one Cached Model Instance (N:1)

**Storage**: Structured logs (JSON format), optionally in monitoring database

## Entity Relationship Diagram

```
┌─────────────────────────┐
│ Embedding Configuration │
│ (IRIS %Embedding.Config)│
└────────┬────────────────┘
         │ 1
         │
         │ N
┌────────▼────────┐       ┌──────────────────────┐
│ EMBEDDING Column│       │ Cached Model Instance│
│ (IRIS Schema)   │       │ (In-Memory Only)     │
└────────┬────────┘       └──────────────────────┘
         │ 1                      │ 1
         │                        │
         │ N                      │ N (concurrent)
┌────────▼──────────────┐         │
│ Vectorized Document   │◄────────┘
│ (IRIS Table Rows)     │
└────────┬──────────────┘
         │ 1
         │
         │ N (optional)
┌────────▼───────────────────┐
│ Entity Extraction Result   │
│ (GraphRAG Knowledge Graph) │
└────────────────────────────┘

┌─────────────────────────┐
│ Model Cache Statistics  │
│ (Logs/Monitoring)       │
└─────────────────────────┘
    Aggregates from Cached Model Instance
```

## Data Flow

1. **Configuration Setup**:
   - Admin creates Embedding Configuration in %Embedding.Config
   - System validates configuration (model exists, paths writable)
   - Configuration stored in IRIS

2. **Table Creation**:
   - Developer creates table with EMBEDDING column
   - EMBEDDING column references configuration by name
   - System creates Cached Model Instance (lazy load)

3. **Document Insertion**:
   - User inserts row with text data
   - IRIS triggers vectorization via EMBEDDING column
   - Python function loads/reuses Cached Model Instance
   - Model generates embedding vector
   - (Optional) Entity extraction occurs if enabled
   - Vector stored in EMBEDDING column
   - Entities stored in GraphRAG tables

4. **Cache Management**:
   - Model stays in cache for reuse (high hit rate)
   - Cache eviction on memory pressure (LRU)
   - Statistics logged periodically

5. **Query**:
   - RAG pipeline queries EMBEDDING column
   - IRIS performs vector similarity search
   - Results include both vectors and extracted entities

## Performance Characteristics

| Entity | Read Pattern | Write Pattern | Size Estimate |
|--------|--------------|---------------|---------------|
| Embedding Configuration | Read-heavy (cached) | Rare (setup only) | <1KB per config |
| EMBEDDING Column | Read-heavy (queries) | Moderate (inserts) | Metadata only (~100B) |
| Cached Model Instance | Very read-heavy | Write on load/evict | 200-400MB per model |
| Vectorized Document | Read-heavy (queries) | Write-heavy (bulk loads) | ~1-4KB per document |
| Entity Extraction Result | Read-heavy (GraphRAG) | Moderate (extraction) | ~100-500B per entity |
| Model Cache Statistics | Append-only | Write on interval | ~500B per measurement |

## Constraints & Invariants

1. **Cache Invariant**: At most 2 Cached Model Instances in memory simultaneously (4GB limit)
2. **Config Invariant**: EMBEDDING Columns can only reference existing, valid configurations
3. **Dimension Invariant**: All vectors for same configuration must have same dimension
4. **Entity Invariant**: Extracted entities must reference valid documents
5. **Performance Invariant**: Cache hit rate must be >= 95% during normal operation

---

**Data Model Complete**: 2025-01-06
**Next**: Contract generation (contracts/*.yaml)
