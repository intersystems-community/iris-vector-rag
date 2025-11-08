# Research: IRIS EMBEDDING Support with Optimized Model Caching

**Feature**: 051-add-native-iris
**Date**: 2025-01-06
**Status**: Complete

## Research Questions

### 1. IRIS EMBEDDING Integration Pattern

**Question**: How should we integrate with IRIS %Embedding.Config table and EMBEDDING column type to solve the 720x slowdown?

**Research Findings**:
- IRIS EMBEDDING columns call Python via %Embedding.SentenceTransformers class
- Current implementation reloads model for each row (DP-442038 root cause)
- %Embedding.Config table stores: Name, Configuration (JSON with modelName, hfCachePath, pythonPath), EmbeddingClass, Description
- Configuration JSON example: `{"hfCachePath":"/path/to/cache","modelName":"sentence-transformers/all-MiniLM-L6-v2","pythonPath":"/path/to/python"}`

**Decision**: Create `iris_rag/embeddings/iris_embedding.py` module that:
1. Reads %Embedding.Config entries via SQL queries
2. Passes model configuration to existing `embeddings/manager.py` cache (Feature 050)
3. Provides Python callable that IRIS %Embedding.SentenceTransformers can invoke
4. Returns vectors directly to IRIS without intermediate storage

**Rationale**: Leverages existing model cache infrastructure from Feature 050 (448x speedup proven), extends it for IRIS EMBEDDING integration rather than reimplementing caching.

**Alternatives Considered**:
- ❌ Modify IRIS %Embedding.SentenceTransformers directly (requires IRIS codebase changes, outside project scope)
- ❌ Implement separate cache in IRIS ObjectScript (duplicates Python work, harder to maintain)
- ✅ Extend iris_rag embeddings manager with IRIS EMBEDDING awareness (reuses proven caching, clean Python interface)

### 2. Entity Extraction Integration Point

**Question**: When and how should entity extraction occur during EMBEDDING vectorization?

**Research Findings**:
- GraphRAG requires entities (Disease, Symptom, Medication) extracted from documents
- Current GraphRAG entity extraction is post-vectorization (separate step)
- EMBEDDING columns trigger on INSERT/UPDATE - ideal hook point for entity extraction
- Batch extraction more efficient: 10 documents per LLM call vs per-document (FR-018)

**Decision**: Implement `iris_rag/embeddings/entity_extractor.py` with:
1. Configurable entity types per %Embedding.Config (medical vs general domain)
2. Batch accumulation: buffer 10 texts, extract in single LLM call
3. Storage: insert extracted entities into GraphRAG knowledge graph tables
4. Optional: enable/disable via config flag (addresses Q2 from spec open questions)

**Rationale**: Vectorization and entity extraction happen together = single database write, consistent state, better performance.

**Alternatives Considered**:
- ❌ Keep entity extraction separate (requires re-reading documents, inefficient)
- ❌ Extract per-row during vectorization (10x more LLM calls, expensive)
- ✅ Batch extraction during vectorization with configurable buffer (optimal LLM usage, fresh data)

### 3. Thread-Safe Model Caching Pattern

**Question**: How to ensure cache safety across concurrent IRIS processes calling Python?

**Research Findings**:
- IRIS can have multiple processes simultaneously calling Python embedding functions
- Feature 050 already implements double-checked locking pattern for thread safety
- Python GIL provides some protection but not sufficient for true concurrency
- Need process-level locking for cache initialization

**Decision**: Extend Feature 050 cache with:
1. Keep existing double-checked locking (thread-safe within process)
2. Add file-based lock for cache initialization across processes
3. Use `fcntl` (Unix) or `msvcrt` (Windows) for cross-process locks
4. Lock only during initialization, not during embedding generation (minimal overhead)

**Rationale**: Combines proven thread-safe pattern with process-safe locking for complete concurrency safety.

**Alternatives Considered**:
- ❌ No process locks (risk of duplicate model loads, wasted memory)
- ❌ Lock every cache access (severe performance penalty)
- ✅ Lock only initialization with file-based semaphore (safe + fast)

### 4. GPU Memory Management

**Question**: How to handle GPU memory exhaustion during bulk vectorization?

**Research Findings**:
- GPU memory limited (8GB M1, 16GB M3, varies on CUDA)
- Embedding models: ~200-400MB per model
- Batch size: 32-128 documents per GPU call
- OOM errors crash entire process if not handled

**Decision**: Implement graceful degradation:
1. Detect GPU type (CUDA/MPS/CPU) via torch.cuda.is_available(), torch.backends.mps.is_available()
2. Start with GPU, catch OOM errors
3. Reduce batch size on OOM (128 → 64 → 32 → 1)
4. Fall back to CPU if GPU consistently fails
5. Log GPU utilization metrics (FR-022)

**Rationale**: Maximizes GPU usage when available, prevents crashes, maintains progress on CPU when GPU exhausted.

**Alternatives Considered**:
- ❌ GPU-only (fails when memory exhausted)
- ❌ CPU-only (leaves GPU unused, slower)
- ✅ GPU-first with automatic CPU fallback (best performance with reliability)

### 5. Configuration Validation Approach

**Question**: How to validate %Embedding.Config entries before table creation (FR-010)?

**Research Findings**:
- Invalid configs cause runtime errors during first INSERT
- Errors include: missing model files, invalid paths, wrong JSON format
- Better to fail early during CREATE TABLE with clear error message

**Decision**: Implement pre-flight validation in `iris_rag/config/embedding_config.py`:
1. Parse Configuration JSON, validate required fields (modelName, hfCachePath)
2. Check model file exists at hfCachePath or can be downloaded
3. Verify Python path executable and has required packages
4. Test model load (dry run) before returning success
5. Raise ValidationError with actionable message if invalid

**Rationale**: Fail fast with clear errors prevents user confusion and wasted time debugging INSERT failures.

**Alternatives Considered**:
- ❌ No validation (users discover errors late, poor UX)
- ❌ Validate only on first INSERT (table exists but unusable)
- ✅ Validate before table creation (clean failure, immediate feedback)

### 6. Performance Monitoring & Metrics

**Question**: What metrics to collect for cache performance and debugging (FR-022)?

**Research Findings**:
- Need to verify >95% cache hit rate target
- Track model load time (<5s target)
- Monitor GPU utilization (>80% target)
- Debug cache eviction issues

**Decision**: Implement structured logging in `iris_rag/embeddings/manager.py`:
```python
cache_stats = {
    "cache_hits": int,
    "cache_misses": int,
    "hit_rate": float,
    "avg_embedding_time_ms": float,
    "model_load_count": int,
    "gpu_utilization_pct": float,
    "memory_usage_mb": float
}
```

Log on:
- Model load events (INFO level)
- Cache misses (DEBUG level)
- Stats summary every 100 embeddings (INFO level)
- Errors with full context (ERROR level with row ID, text hash)

**Rationale**: Enables performance validation, troubleshooting, and optimization based on real usage patterns.

**Alternatives Considered**:
- ❌ No metrics (can't validate performance targets)
- ❌ Metrics to database (adds overhead, complicates setup)
- ✅ Structured logging with periodic summaries (observable, low overhead)

## Open Questions Resolved

From spec Open Questions section:

**Q1: Cache eviction policy?**
**Decision**: LRU (Least Recently Used) with timestamp tracking.
**Rationale**: Simple, proven, handles memory limits gracefully. Default: evict when memory >4GB or >2 models cached.

**Q2: Entity extraction mandatory?**
**Decision**: Optional, controlled by config flag `enable_entity_extraction: bool` in %Embedding.Config JSON.
**Rationale**: Not all use cases need entities (e.g., basic RAG), making it optional reduces LLM costs for non-GraphRAG pipelines.

**Q3: Memory budget?**
**Decision**: 4GB max memory, max 2 models cached simultaneously.
**Rationale**: Fits typical server configs, leaves headroom for other processes. Configurable via environment variable.

**Q4: Custom entity extraction models?**
**Decision**: Phase 1 supports LLM-based extraction only. Custom models deferred to Phase 2.
**Rationale**: LLM-based extraction covers 85%+ of use cases, adding custom model support adds complexity better addressed after validating core functionality.

## Technology Stack Summary

**Core Technologies**:
- sentence-transformers: Embedding model library
- torch: GPU acceleration (CUDA/MPS/CPU)
- InterSystems IRIS 2025.3+: Database with EMBEDDING column support
- iris-vector-graph: GraphRAG knowledge graph integration

**Integration Patterns**:
- SQL queries for %Embedding.Config access (standard IRIS utilities)
- Python callable interface for IRIS %Embedding.SentenceTransformers
- JSON configuration in IRIS tables
- File-based process locks for cross-process cache safety

**Testing Approach**:
- Contract tests with YAML specifications (TDD)
- Integration tests with live IRIS database (constitutional requirement)
- Performance tests with 10K+ documents
- Mocked LLM for entity extraction tests (avoid API costs)

## Implementation Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| IRIS %Embedding.SentenceTransformers API changes | High - breaks integration | Document exact API version used, test against specific IRIS version |
| GPU memory exhaustion crashes | Medium - lost progress | Implement batch size reduction + CPU fallback |
| LLM API costs for entity extraction | Medium - budget overrun | Batch extraction (10 docs/call), make extraction optional |
| Model cache eviction thrashing | Low - performance degradation | Monitor cache hit rate, alert if <95% |
| Concurrent process cache corruption | High - data integrity | File-based locks + atomic cache updates |

## References

- IRIS Vector Search Docs: https://docs.intersystems.com/iris20252/csp/docbook/DocBook.UI.Page.cls?KEY=GSQL_vecsearch
- Jira DP-442038: https://usjira.iscinternal.com/browse/DP-442038
- Feature 050 (Model Cache): `/Users/tdyar/ws/rag-templates/specs/050-fix-embedding-model/`
- SentenceTransformers Docs: https://www.sbert.net/
- PyTorch GPU Memory Management: https://pytorch.org/docs/stable/notes/cuda.html

---

**Research Complete**: 2025-01-06
**Next Phase**: Design (data-model.md, contracts, quickstart.md)
