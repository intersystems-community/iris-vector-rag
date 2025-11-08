# Tasks: IRIS EMBEDDING Support with Optimized Model Caching

**Feature**: 051-add-native-iris
**Input**: Design documents from `/Users/tdyar/ws/rag-templates/specs/051-add-native-iris/`
**Prerequisites**: plan.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓, quickstart.md ✓

---

## Execution Flow

```
1. ✓ Load plan.md: Python 3.11+, sentence-transformers, torch, IRIS 2025.3+
2. ✓ Load data-model.md: 6 entities (EmbeddingConfig, CachedModelInstance, etc.)
3. ✓ Load contracts/: 2 contracts (iris_embedding, entity_extraction)
4. ✓ Load research.md: 6 technical decisions documented
5. → Generate 27 tasks (setup → tests → implementation → validation)
6. → Apply TDD ordering (tests before implementation)
7. → Mark [P] for parallelizable tasks (different files, no dependencies)
```

---

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no shared dependencies)
- Includes exact file paths for each task

---

## Phase 3.1: Setup & Prerequisites

### T001: ✅ Verify dependencies and update pyproject.toml
**File**: `/Users/tdyar/ws/rag-templates/pyproject.toml`
**Actions**:
- Add `sentence-transformers>=2.2.0` to dependencies
- Add `torch>=2.0.0` to dependencies
- Ensure `iris-vector-graph>=2.0.0` is present (GraphRAG requirement)
- Update optional dependencies group `[embedding]` for IRIS EMBEDDING support
**Verification**: Run `uv sync` and verify no conflicts

### T002: ✅ Create IRIS database schema extensions
**File**: `/Users/tdyar/ws/rag-templates/iris_rag/storage/iris_embedding_schema.sql`
**Actions**:
- Create SQL script to extend %Embedding.Config table with JSON configuration field
- Add indexes for config_name lookups
- Document schema in inline comments
**Verification**: SQL script can be executed via `iris_rag/storage/iris_vector_store.py`

---

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### T003 [P] ✅ Contract test: IRIS EMBEDDING integration
**File**: `/Users/tdyar/ws/rag-templates/tests/contract/test_iris_embedding_contract.py`
**Actions**:
- Load contract from `specs/051-add-native-iris/contracts/iris_embedding_contract.yaml`
- Implement 9 test scenarios:
  1. `test_load_valid_config`: Load configuration from %Embedding.Config
  2. `test_config_not_found`: Handle missing configuration
  3. `test_validate_valid_config`: Validate correct configuration
  4. `test_validate_missing_model`: Detect missing model file
  5. `test_embed_texts_cache_hit`: Cache hit performance (<50ms)
  6. `test_embed_texts_cache_miss`: First-time model load (<5000ms)
  7. `test_embed_texts_empty_text`: Handle empty text error
  8. `test_cache_hit_rate_target`: Verify 95% cache hit rate
  9. `test_gpu_fallback`: CPU fallback on GPU OOM
- All tests MUST FAIL initially (implementation doesn't exist yet)
**Verification**: `pytest tests/contract/test_iris_embedding_contract.py` → 9 failures

### T004 [P] ✅ Contract test: Entity extraction
**File**: `/Users/tdyar/ws/rag-templates/tests/contract/test_entity_extraction_contract.py`
**Actions**:
- Load contract from `specs/051-add-native-iris/contracts/entity_extraction_contract.yaml`
- Implement 10 test scenarios:
  1. `test_extract_entities_batch_medical_domain`: Batch extraction for 3 documents
  2. `test_entity_accuracy_medical_domain`: 85% accuracy target
  3. `test_batch_vs_single_extraction_performance`: 10x efficiency
  4. `test_entity_relationship_extraction`: Extract relationships
  5. `test_store_entities_in_knowledge_graph`: Store in GraphRAG tables
  6. `test_extraction_disabled`: Handle disabled extraction
  7. `test_empty_entity_types`: Handle missing entity types
  8. `test_llm_api_retry`: Retry with exponential backoff
  9. `test_configure_custom_entity_types`: Domain-specific entities
  10. `test_get_entities_for_document`: Retrieve entities
- Mock LLM API to avoid costs
- All tests MUST FAIL initially
**Verification**: `pytest tests/contract/test_entity_extraction_contract.py` → 10 failures

### T005 [P] Integration test: End-to-end EMBEDDING workflow
**File**: `/Users/tdyar/ws/rag-templates/tests/integration/test_iris_embedding_e2e.py`
**Actions**:
- Test complete workflow from quickstart.md:
  1. Configure embedding model
  2. Validate configuration
  3. Create table with EMBEDDING column (SQL)
  4. Insert documents (auto-vectorization)
  5. Query with BasicRAGPipeline
  6. Verify cache statistics
- Use live IRIS database (constitutional requirement)
- Test with 10 sample documents
- MUST FAIL initially (no implementation)
**Verification**: `pytest tests/integration/test_iris_embedding_e2e.py` → failure

### T006 [P] Integration test: GraphRAG entity extraction workflow
**File**: `/Users/tdyar/ws/rag-templates/tests/integration/test_entity_extraction_graphrag.py`
**Actions**:
- Test entity extraction during vectorization:
  1. Configure embedding with entity extraction enabled
  2. Insert medical documents with entities (Disease, Symptom, Medication)
  3. Verify entities extracted and stored in GraphRAG tables
  4. Query with HybridGraphRAGPipeline using extracted entities
  5. Verify entity-based retrieval works
- Use mocked LLM for entity extraction
- MUST FAIL initially
**Verification**: `pytest tests/integration/test_entity_extraction_graphrag.py` → failure

---

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### T007 [P] ✅ Model: EmbeddingConfig data class
**File**: `/Users/tdyar/ws/rag-templates/iris_rag/config/embedding_config.py`
**Actions**:
- Create `EmbeddingConfig` dataclass with fields from data-model.md:
  - name, model_name, hf_cache_path, python_path, embedding_class, description
  - enable_entity_extraction, entity_types, batch_size, device_preference
- Add validation rules (from data-model.md lines 25-31)
- Create `ValidationResult` dataclass for validation responses
- Implement `validate_embedding_config(config: EmbeddingConfig) -> ValidationResult`
  - Check model file exists at hf_cache_path
  - Verify Python path executable
  - Test model load (dry run)
  - Validate entity_types if extraction enabled
**Verification**: Contract test `test_validate_valid_config` passes

### T008 [P] ✅ Model: EntityExtractionResult data class
**File**: `/Users/tdyar/ws/rag-templates/iris_rag/embeddings/entity_extractor.py` (models section)
**Actions**:
- Create `EntityExtractionResult` dataclass with fields from data-model.md:
  - entity_id, doc_id, entity_type, entity_text
  - text_span_start, text_span_end, confidence_score
  - relationships, extraction_method, extraction_timestamp
- Add validation: entity_text matches text_span in source document
- Create `BatchEntityExtractionResult` dataclass for batch responses
- Create `DocumentEntityResult` dataclass for per-document results
**Verification**: Model can be instantiated with valid data

### T009 ✅ Extend CachedModelInstance in embeddings manager
**File**: `/Users/tdyar/ws/rag-templates/iris_rag/embeddings/manager.py`
**Actions**:
- Extend existing `CachedModelInstance` (from Feature 050) with new fields:
  - reference_count (int, default=0)
  - cache_hits, cache_misses (int, default=0)
  - total_embeddings_generated (int, default=0)
- Update double-checked locking pattern to track cache hits/misses
- Add `get_cache_stats(config_name: str = None) -> CacheStatistics` function
- Add `clear_cache(config_name: str = None) -> ClearCacheResult` function
**Verification**: Cache statistics can be retrieved
**Dependencies**: Requires T007 (EmbeddingConfig model)

### T010 ✅ IRIS EMBEDDING integration layer
**File**: `/Users/tdyar/ws/rag-templates/iris_rag/embeddings/iris_embedding.py`
**Actions**:
- Implement `get_config(config_name: str) -> EmbeddingConfig`:
  - Query %Embedding.Config table via IRISVectorStore
  - Parse Configuration JSON field
  - Return EmbeddingConfig instance
  - Raise CONFIG_NOT_FOUND error if not found
- Implement `embed_texts(config_name: str, texts: list[str]) -> EmbeddingResult`:
  - Load config via get_config()
  - Pass to embeddings/manager.py cache (reuse Feature 050)
  - Generate embeddings using cached model
  - Track cache hits/misses, timing metrics
  - Handle GPU OOM with CPU fallback
  - Return embeddings with performance metadata
- Implement `configure_embedding(...)` helper for configuration creation
**Verification**: Contract tests `test_load_valid_config`, `test_embed_texts_cache_hit` pass
**Dependencies**: Requires T007 (EmbeddingConfig), T009 (cache manager)

### T011 Entity extraction implementation
**File**: `/Users/tdyar/ws/rag-templates/iris_rag/embeddings/entity_extractor.py`
**Actions**:
- Implement `extract_entities_batch(texts: list[str], config: EmbeddingConfig) -> BatchEntityExtractionResult`:
  - Check enable_entity_extraction flag
  - Batch texts (up to 10 per LLM call per FR-018)
  - Call LLM API with entity types from config
  - Parse LLM response into EntityExtractionResult instances
  - Extract relationships between entities
  - Return batch result with timing metrics
- Implement `store_entities(doc_id: UUID, entities: list[EntityExtractionResult]) -> EntityStorageResult`:
  - Insert entities into GraphRAG knowledge graph tables
  - Create relationships in graph.relationships table
  - Handle duplicate entities gracefully
- Implement `configure_entity_types(config_name: str, entity_types: list[str]) -> ConfigurationResult`
- Implement `get_entities(doc_id: UUID) -> DocumentEntities`
- Add exponential backoff retry for LLM API errors (3 attempts)
**Verification**: Contract tests `test_extract_entities_batch_medical_domain`, `test_store_entities_in_knowledge_graph` pass
**Dependencies**: Requires T008 (EntityExtractionResult model)

### T012 EMBEDDING column operations
**File**: `/Users/tdyar/ws/rag-templates/iris_rag/storage/iris_embedding_ops.py`
**Actions**:
- Implement SQL helpers for EMBEDDING columns:
  - `create_embedding_column(table_name: str, column_name: str, source_column: str, config_name: str)`
    - Generates CREATE TABLE SQL with EMBEDDING column
    - References %Embedding.Config by name
  - `validate_embedding_table(table_name: str) -> ValidationResult`
    - Verify table exists with valid EMBEDDING column
    - Check config_name references exist
  - `bulk_insert_with_embedding(table_name: str, documents: list[dict])`
    - Insert documents, auto-triggers vectorization
    - Track vectorization performance
- Integrate with iris_embedding.py for vectorization callbacks
**Verification**: Integration test `test_iris_embedding_e2e` passes (create table step)
**Dependencies**: Requires T010 (IRIS EMBEDDING integration)

### T013 Update BasicRAGPipeline for EMBEDDING support
**File**: `/Users/tdyar/ws/rag-templates/iris_rag/pipelines/basic_rag.py`
**Actions**:
- Add `embedding_config: str = None` parameter to `__init__()`
- If embedding_config provided:
  - Use IRIS EMBEDDING-based retrieval instead of manual vectorization
  - Query EMBEDDING column directly via IRISVectorStore
  - Skip manual embedding generation in load_documents()
- Maintain backward compatibility (existing manual vectorization still works)
- Update docstring with EMBEDDING usage example
**Verification**: Can create pipeline with `embedding_config="medical_embeddings"`
**Dependencies**: Requires T010 (IRIS EMBEDDING integration)

### T014 Update CRAGPipeline for EMBEDDING support
**File**: `/Users/tdyar/ws/rag-templates/iris_rag/pipelines/crag.py`
**Actions**:
- Add `embedding_config: str = None` parameter to `__init__()`
- Integrate EMBEDDING-based retrieval for self-evaluation step
- Maintain backward compatibility
**Verification**: CRAG pipeline works with EMBEDDING configuration
**Dependencies**: Requires T010 (IRIS EMBEDDING integration)

### T015 Update HybridGraphRAGPipeline for EMBEDDING + entity extraction
**File**: `/Users/tdyar/ws/rag-templates/iris_rag/pipelines/graphrag.py`
**Actions**:
- Add `embedding_config: str = None` parameter to `__init__()`
- If embedding_config provided AND entity extraction enabled:
  - Use extracted entities from EMBEDDING vectorization
  - Query knowledge graph tables populated by entity_extractor.py
  - Hybrid retrieval: vector + text + graph (using EMBEDDING entities)
- Skip manual entity extraction if entities already extracted during vectorization
- Update docstring with GraphRAG + EMBEDDING usage example
**Verification**: Integration test `test_entity_extraction_graphrag` passes
**Dependencies**: Requires T010 (IRIS EMBEDDING), T011 (entity extraction)

### T016 Update PyLateColBERTPipeline for EMBEDDING support
**File**: `/Users/tdyar/ws/rag-templates/iris_rag/pipelines/pylate_colbert.py`
**Actions**:
- Add `embedding_config: str = None` parameter to `__init__()`
- Integrate EMBEDDING-based retrieval for ColBERT late interaction
- Maintain backward compatibility
**Verification**: PyLateColBERT pipeline works with EMBEDDING configuration
**Dependencies**: Requires T010 (IRIS EMBEDDING integration)

### T017 Update BasicRAGRerankingPipeline for EMBEDDING support
**File**: `/Users/tdyar/ws/rag-templates/iris_rag/pipelines/basic_rerank.py`
**Actions**:
- Add `embedding_config: str = None` parameter to `__init__()`
- Integrate EMBEDDING-based retrieval before reranking step
- Maintain backward compatibility
**Verification**: Reranking pipeline works with EMBEDDING configuration
**Dependencies**: Requires T010 (IRIS EMBEDDING integration)

---

## Phase 3.4: Integration & Configuration

### T018 Update IRISVectorStore for %Embedding.Config queries
**File**: `/Users/tdyar/ws/rag-templates/iris_rag/storage/iris_vector_store.py`
**Actions**:
- Add `query_embedding_config(config_name: str) -> dict` method
  - Query %Embedding.Config table
  - Return configuration JSON
- Add `list_embedding_configs() -> list[dict]` method
  - List all available embedding configurations
- Update connection handling for EMBEDDING column operations
**Verification**: Can query %Embedding.Config table successfully
**Dependencies**: Requires T002 (schema extensions)

### T019 Update default configuration with EMBEDDING settings
**File**: `/Users/tdyar/ws/rag-templates/iris_rag/config/default_config.yaml`
**Actions**:
- Add `embedding:` section with defaults:
  ```yaml
  embedding:
    enabled: false  # Enable EMBEDDING-based vectorization
    default_config: null  # Default %Embedding.Config name
    cache:
      max_memory_mb: 4096  # 4GB max for cached models
      max_models: 2  # Max 2 models in cache simultaneously
      eviction_policy: "lru"  # Least Recently Used
    entity_extraction:
      enabled: false
      batch_size: 10  # Documents per LLM call
      retry_attempts: 3
      retry_backoff_ms: 1000
  ```
**Verification**: Configuration loads without errors

### T020 Add structured logging for EMBEDDING operations
**File**: `/Users/tdyar/ws/rag-templates/iris_rag/embeddings/iris_embedding.py` (extend)
**Actions**:
- Add structured logging for:
  - Model load events (INFO level)
  - Cache hits/misses (DEBUG level)
  - Cache statistics summary every 100 embeddings (INFO level)
  - Errors with full context: row ID, text hash, model name (ERROR level)
  - GPU OOM events with fallback info (WARN level)
- Use Python `logging` module with JSON formatting
**Verification**: Logs appear in expected format during vectorization

---

## Phase 3.5: Performance Tests & Benchmarks

### T021 [P] Performance test: Cache hit rate target (95%)
**File**: `/Users/tdyar/ws/rag-templates/tests/integration/test_embedding_cache_perf.py` (extend existing)
**Actions**:
- Generate 1,000 embeddings using same configuration
- Measure cache hits/misses via get_cache_stats()
- Assert cache_hit_rate >= 0.95
- Assert avg_embedding_time_ms < 50 (when cached)
- Assert model_load_count <= 2 (should only load once or twice)
**Verification**: Test passes with >95% cache hit rate
**Dependencies**: Requires T010 (IRIS EMBEDDING), T009 (cache stats)

### T022 [P] Performance benchmark: 50x speedup target
**File**: `/Users/tdyar/ws/rag-templates/tests/integration/test_bulk_vectorization.py`
**Actions**:
- Implement benchmark from contract `benchmark_cache_hit_performance`:
  - Vectorize 1,746 documents (exact count from DP-442038)
  - Measure total time
  - Assert total_time < 30 seconds (target from FR-002)
  - Assert model loads exactly once
  - Compare against baseline (would be 20 minutes without cache)
- Log performance metrics: embeddings/sec, GPU utilization
**Verification**: 1,746 documents vectorize in <30 seconds
**Dependencies**: Requires T010 (IRIS EMBEDDING), T012 (bulk operations)

### T023 [P] Performance benchmark: 10K enterprise scale test
**File**: `/Users/tdyar/ws/rag-templates/tests/integration/test_10k_documents.py`
**Actions**:
- Implement benchmark from contract `benchmark_10k_documents`:
  - Generate 10,000 test documents
  - Bulk insert with EMBEDDING column
  - Measure total vectorization time
  - Assert total_time < 600 seconds (10 minutes)
  - Assert model loads exactly once
  - Assert cache_hit_rate > 0.95
**Verification**: 10K documents vectorize in <10 minutes
**Dependencies**: Requires T010 (IRIS EMBEDDING), T012 (bulk operations)

### T024 [P] Performance benchmark: Entity extraction batch efficiency
**File**: `/Users/tdyar/ws/rag-templates/tests/integration/test_entity_extraction_performance.py`
**Actions**:
- Implement benchmark from contract `benchmark_batch_extraction_efficiency`:
  - Extract entities from 100 medical documents
  - Compare batch (10 LLM calls) vs single (100 LLM calls)
  - Assert batch_time < single_time / 5
  - Measure cost savings (LLM API calls)
- Mock LLM to avoid actual API costs during testing
**Verification**: Batch extraction >5x faster than single
**Dependencies**: Requires T011 (entity extraction)

### T025 [P] Performance benchmark: Entity extraction accuracy (85%)
**File**: `/Users/tdyar/ws/rag-templates/tests/integration/test_entity_accuracy.py`
**Actions**:
- Implement benchmark from contract `benchmark_entity_extraction_accuracy`:
  - Use 100 documents with ground-truth entity annotations
  - Extract entities using entity_extractor.py
  - Calculate Precision, Recall, F1 Score
  - Assert precision >= 0.85, recall >= 0.85, f1 >= 0.85
- Use real LLM API for accuracy testing (not mocked)
**Verification**: Entity extraction meets 85% accuracy target
**Dependencies**: Requires T011 (entity extraction)

---

## Phase 3.6: Documentation & Polish

### T026 [P] Update quickstart.md with actual implementation details
**File**: `/Users/tdyar/ws/rag-templates/specs/051-add-native-iris/quickstart.md` (verify)
**Actions**:
- Verify all code examples in quickstart.md work with implementation
- Add actual import paths (now known from implementation)
- Update troubleshooting section with real error messages
- Add performance benchmark results (from T021-T025)
- Ensure all 7 steps can be executed without errors
**Verification**: Execute quickstart.md end-to-end, all steps work

### T027 [P] Create migration guide from manual to EMBEDDING vectorization
**File**: `/Users/tdyar/ws/rag-templates/docs/embedding_migration_guide.md`
**Actions**:
- Document migration workflow from quickstart.md "Workflow 1"
- Step-by-step guide:
  1. Assess existing manual vectorization setup
  2. Create EMBEDDING configuration
  3. Create new table with EMBEDDING column
  4. Migrate data (copy from old table → auto-vectorizes)
  5. Update RAG pipeline configuration
  6. Verify performance improvements
  7. Decommission old tables
- Include rollback procedures
- Add troubleshooting for common migration issues
**Verification**: Documentation is complete and clear

---

## Dependencies

**Critical Path** (TDD):
```
T001-T002 (Setup)
  ↓
T003-T006 (Tests - MUST FAIL)
  ↓
T007-T008 (Models)
  ↓
T009-T012 (Core Implementation)
  ↓
T013-T017 (Pipeline Integration)
  ↓
T018-T020 (Configuration & Logging)
  ↓
T021-T025 (Performance Validation)
  ↓
T026-T027 (Documentation)
```

**Blocking Dependencies**:
- T003-T006 BLOCK T007-T027 (tests must exist and fail first - TDD)
- T007 (EmbeddingConfig) BLOCKS T009, T010, T011
- T008 (EntityExtractionResult) BLOCKS T011
- T009 (Cache manager) BLOCKS T010
- T010 (IRIS EMBEDDING) BLOCKS T012, T013-T017, T021-T023
- T011 (Entity extraction) BLOCKS T015, T024-T025
- T012 (EMBEDDING ops) BLOCKS T022-T023
- T013-T017 (Pipelines) BLOCK T026 (can't verify quickstart without pipelines)

**Parallel Groups** ([P] tasks):
- Group 1: T003, T004, T005, T006 (contract tests, different files)
- Group 2: T007, T008 (models, different files)
- Group 3: T013, T014, T016, T017 (pipeline updates, different files)
- Group 4: T021, T022, T023, T024, T025 (performance tests, different files)
- Group 5: T026, T027 (documentation, different files)

---

## Parallel Execution Examples

### Launch Contract Tests (T003-T006) Together:
```bash
# Terminal 1
pytest tests/contract/test_iris_embedding_contract.py -v

# Terminal 2
pytest tests/contract/test_entity_extraction_contract.py -v

# Terminal 3
pytest tests/integration/test_iris_embedding_e2e.py -v

# Terminal 4
pytest tests/integration/test_entity_extraction_graphrag.py -v
```

### Launch Pipeline Updates (T013-T017) Together:
```bash
# All pipelines can be updated in parallel (different files)
# Task agent commands:
Task: "Update BasicRAGPipeline for EMBEDDING support in iris_rag/pipelines/basic_rag.py"
Task: "Update CRAGPipeline for EMBEDDING support in iris_rag/pipelines/crag.py"
Task: "Update PyLateColBERTPipeline for EMBEDDING support in iris_rag/pipelines/pylate_colbert.py"
Task: "Update BasicRAGRerankingPipeline for EMBEDDING support in iris_rag/pipelines/basic_rerank.py"
# Note: T015 (HybridGraphRAG) depends on T011 completing first
```

### Launch Performance Benchmarks (T021-T025) Together:
```bash
# Terminal 1
pytest tests/integration/test_embedding_cache_perf.py -v

# Terminal 2
pytest tests/integration/test_bulk_vectorization.py -v

# Terminal 3
pytest tests/integration/test_10k_documents.py -v

# Terminal 4
pytest tests/integration/test_entity_extraction_performance.py -v

# Terminal 5
pytest tests/integration/test_entity_accuracy.py -v
```

---

## Validation Checklist
*GATE: All items must be checked before tasks are considered complete*

- [x] All contracts have corresponding tests (T003: iris_embedding, T004: entity_extraction)
- [x] All entities have model tasks (T007: EmbeddingConfig, T008: EntityExtractionResult, T009: CachedModelInstance)
- [x] All tests come before implementation (T003-T006 before T007-T025)
- [x] Parallel tasks truly independent (verified: different files, no shared state)
- [x] Each task specifies exact file path (all tasks include full paths)
- [x] No task modifies same file as another [P] task (verified)
- [x] Performance targets specified (95% cache hit, 50x speedup, 85% entity accuracy)
- [x] All 5 pipelines updated for EMBEDDING support (T013-T017)
- [x] Integration tests validate quickstart.md workflows (T005, T006)
- [x] Documentation tasks included (T026, T027)

---

## Notes

- **TDD Critical**: Tests T003-T006 MUST be written and MUST FAIL before starting T007
- **[P] tasks**: Different files, can run concurrently
- **Sequential tasks**: Same file or dependency chain, must run in order
- **Commit strategy**: Commit after each task completion
- **Constitutional compliance**: Live IRIS database required for integration tests (T005, T006, T021-T023)
- **Performance validation**: All benchmarks must pass targets (FR-002, FR-003, FR-017)
- **Backward compatibility**: Existing manual vectorization workflows must continue to work (FR-013)

---

**Task Generation Complete**: 27 tasks ready for execution
**Next Command**: `/implement` or begin manual execution starting with T001
