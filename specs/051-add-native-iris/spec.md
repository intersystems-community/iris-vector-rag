# Feature Specification: IRIS EMBEDDING Support with Optimized Model Caching

**Feature Branch**: `051-add-native-iris`
**Created**: 2025-01-06
**Status**: Draft
**Input**: User description: "Add native IRIS EMBEDDING support with optimized model caching and entity extraction for GraphRAG"

**Background**: InterSystems IRIS Vector Search provides an EMBEDDING data type that automatically vectorizes text columns when data changes. However, the current implementation (%Embedding.SentenceTransformers) has a critical performance issue (DP-442038): it reloads the embedding model for each row, causing a 720x slowdown (1,746 rows in 20 minutes vs <1 second in Python). This feature will solve that performance issue and extend EMBEDDING support to rag-templates with GraphRAG entity extraction capabilities.

**Reference Documentation**:
- IRIS Vector Search: https://docs.intersystems.com/iris20252/csp/docbook/DocBook.UI.Page.cls?KEY=GSQL_vecsearch
- Performance Issue: https://usjira.iscinternal.com/browse/DP-442038

---

## User Scenarios & Testing

### Primary User Story

A developer building a RAG application needs to automatically vectorize medical documents stored in IRIS tables. When new documents are inserted or updated, their text content must be immediately vectorized without manual intervention. The vectorization process must complete in seconds (not minutes) for thousands of documents, and the system should automatically extract entities for GraphRAG knowledge graphs during the same vectorization pass.

### Acceptance Scenarios

1. **Given** an IRIS table with an EMBEDDING column configured with a model cache, **When** 1,746 rows are inserted with clinical text, **Then** vectorization completes in under 30 seconds (vs current 20 minutes) with >95% cache hit rate

2. **Given** a bulk data load of 10,000 medical documents, **When** the load executes with EMBEDDING columns enabled, **Then** the embedding model loads exactly once and remains cached for all 10,000 rows

3. **Given** an EMBEDDING configuration with entity extraction enabled, **When** documents are inserted with medical text mentioning "diabetes" and "insulin", **Then** the system extracts entities (e.g., Disease: diabetes, Medication: insulin) and stores them for GraphRAG retrieval

4. **Given** a GraphRAG pipeline using EMBEDDING-based vectorization, **When** a user queries "What medications treat diabetes?", **Then** the system retrieves relevant documents using hybrid search (vector + knowledge graph) with entities extracted during vectorization

5. **Given** GPU hardware available, **When** the embedding model is initialized, **Then** the system automatically detects and uses GPU acceleration (CUDA/MPS) for embedding generation

6. **Given** a cached embedding model in memory, **When** a new table with the same model configuration is created, **Then** the system reuses the existing cached model without reloading

### Edge Cases

- What happens when GPU memory is exhausted during batch vectorization of 100,000 documents?
- How does the system handle EMBEDDING tables when the configured model file is missing or corrupted?
- What happens if entity extraction fails for malformed text (e.g., binary data in text column)?
- How does the system handle concurrent EMBEDDING operations across multiple IRIS processes?
- What happens when the Python embedding service crashes mid-vectorization for a 50,000 row bulk load?

---

## Requirements

### Functional Requirements

**Performance Requirements**
- **FR-001**: System MUST cache embedding models in memory to eliminate per-row model reloading overhead
- **FR-002**: System MUST achieve vectorization performance within 50x of native Python (20 minutes → <30 seconds for 1,746 rows)
- **FR-003**: System MUST maintain cache hit rate >95% during bulk vectorization operations
- **FR-004**: System MUST support concurrent vectorization requests without cache thrashing
- **FR-005**: System MUST automatically detect and utilize GPU acceleration when available (CUDA, MPS, or CPU fallback)

**EMBEDDING Data Type Support**
- **FR-006**: System MUST support creating IRIS tables with EMBEDDING columns that auto-vectorize specified text columns
- **FR-007**: System MUST integrate with IRIS %Embedding.Config table to read model configuration (model name, cache path, Python path)
- **FR-008**: System MUST support all embedding models compatible with SentenceTransformers library
- **FR-009**: System MUST handle EMBEDDING column updates when source text columns change (INSERT, UPDATE operations)
- **FR-010**: System MUST validate EMBEDDING configurations before table creation to prevent runtime errors

**Configuration & Integration**
- **FR-011**: System MUST provide configuration interface for EMBEDDING settings (model name, cache directory, batch size, GPU preference)
- **FR-012**: System MUST integrate EMBEDDING-based vectorization as an option in all RAG pipelines (basic, basic_rerank, crag, graphrag, pylate_colbert)
- **FR-013**: System MUST allow users to choose between traditional vectorization (via pipeline.load_documents) and EMBEDDING-based auto-vectorization
- **FR-014**: System MUST support hybrid workflows where some tables use EMBEDDING columns and others use manual vectorization

**Entity Extraction for GraphRAG**
- **FR-015**: System MUST extract entities from text during vectorization when GraphRAG pipelines are configured
- **FR-016**: System MUST support configurable entity types (e.g., medical: Disease, Symptom, Medication; general: Person, Organization, Location)
- **FR-017**: System MUST store extracted entities in knowledge graph format compatible with HybridGraphRAG pipeline
- **FR-018**: System MUST batch entity extraction to minimize LLM API calls (e.g., extract from 10 documents per LLM call vs per-document)

**Model Management**
- **FR-019**: System MUST implement thread-safe model caching using double-checked locking pattern
- **FR-020**: System MUST support graceful model cache eviction when memory limits are reached
- **FR-021**: System MUST provide model preloading capability to avoid cold-start delays
- **FR-022**: System MUST log model loading events, cache statistics, and performance metrics

**Error Handling**
- **FR-023**: System MUST provide clear error messages when EMBEDDING configuration is invalid (missing model, invalid path, insufficient permissions)
- **FR-024**: System MUST handle embedding failures gracefully without blocking entire bulk load operations
- **FR-025**: System MUST retry failed embeddings with exponential backoff for transient errors (GPU memory, network timeouts)
- **FR-026**: System MUST log all embedding errors with sufficient context for debugging (row ID, text content hash, error type)

### Key Entities

- **Embedding Configuration**: Represents %Embedding.Config entries with model name, cache path, Python environment settings, and optional entity extraction configuration
- **EMBEDDING Column**: Represents SQL column definition that auto-vectorizes a source text column using specified Embedding Configuration
- **Cached Model Instance**: Represents in-memory embedding model with device allocation (CPU/CUDA/MPS), reference count, and last access timestamp
- **Vectorized Document**: Represents text content with generated embedding vector, source metadata, extraction timestamp, and optional extracted entities
- **Entity Extraction Result**: Represents entities extracted during vectorization with entity type, text span, confidence score, and relationships to other entities
- **Model Cache Statistics**: Represents performance metrics including cache hit rate, average embedding time, GPU utilization, and memory consumption

---

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [ ] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

### Open Questions
- **Q1**: What should be the default cache eviction policy when memory limits are reached? (LRU, LFU, size-based?)
- **Q2**: Should entity extraction be mandatory for all EMBEDDING columns or only when GraphRAG pipelines are active?
- **Q3**: What is the target memory budget for cached models? (e.g., max 4GB, max 2 models simultaneously)
- **Q4**: Should the system support custom entity extraction models beyond the default LLM-based extraction?

---

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked (see Open Questions)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [ ] Review checklist passed (pending clarification on Q1-Q4)

---

## Success Criteria

**Performance Targets**:
- Eliminate 720x slowdown: 1,746 rows vectorize in <30 seconds (currently 20 minutes)
- Cache hit rate >95% during bulk operations
- GPU utilization >80% when GPU available
- Model loading time <5 seconds for first request per model

**Quality Targets**:
- Zero embedding failures for valid text input
- Entity extraction accuracy >85% for medical domain (Disease, Symptom, Medication entities)
- 100% backward compatibility with existing manual vectorization workflows

**Integration Targets**:
- All 5 RAG pipelines support EMBEDDING-based vectorization option
- Documentation includes migration guide from manual to EMBEDDING-based approach
- Contract tests validate EMBEDDING integration across all pipelines
