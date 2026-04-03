# Feature Specification: Attach Existing Corpus

**Feature Branch**: `069-attach-existing-corpus`
**Created**: 2026-04-03
**Status**: Draft
**Input**: Attach existing corpus to graph — zero-copy bridge from RAG.SourceDocuments or any IRIS table to IVG graph nodes + vector search, reusing existing HNSW indexes without re-embedding or data duplication

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Attach RAG corpus to graph (Priority: P1)

A user has 10,000 documents already ingested in `RAG.SourceDocuments` with pre-computed VECTOR embeddings and an HNSW index. They want to run graph queries (`MATCH (d:Document)`) over these documents and use vector search without re-embedding or copying data.

They call `pipeline.attach_existing_corpus(source_table="RAG.SourceDocuments", ...)` and immediately get graph traversal + vector similarity search over the existing data — zero new tables, zero re-embedding, zero data duplication.

**Why this priority**: This is the primary use case. Every IVR user with existing `RAG.SourceDocuments` data hits this wall today — they must manually create graph nodes and copy embeddings. This eliminates that friction.

**Independent Test**: Can be tested by inserting 100 documents into `RAG.SourceDocuments`, calling `attach_existing_corpus`, then verifying both `MATCH (d:Document)` and `vector_search("RAG.SourceDocuments", ...)` return results.

**Acceptance Scenarios**:

1. **Given** `RAG.SourceDocuments` has 100 rows with `doc_id`, `text_content`, and `embedding` (VECTOR(FLOAT,768)) columns, **When** the user calls `pipeline.attach_existing_corpus(source_table="RAG.SourceDocuments", id_col="doc_id", text_col="text_content", embedding_col="embedding", graph_label="Document")`, **Then** `engine.query("MATCH (d:Document) RETURN d LIMIT 5")` returns 5 documents and `engine.vector_search("RAG.SourceDocuments", "embedding", query_vec, top_k=5)` returns 5 results with scores > 0.
2. **Given** the same attached corpus, **When** new rows are inserted into `RAG.SourceDocuments`, **Then** graph queries and vector searches immediately include the new rows (no re-attach required).
3. **Given** the same attached corpus, **When** the user calls `attach_existing_corpus` a second time with the same parameters, **Then** the operation succeeds idempotently (no error, no duplicate mappings).

---

### User Story 2 - Attach custom IRIS table (Priority: P2)

A user has clinical data in a custom IRIS table (e.g., `HS.FHIRServer.Storage.Json.FHIRDocumentData` or `MyApp.PatientNotes`) with a VECTOR column. They want to make this searchable via IVG graph queries without ETL into `RAG.SourceDocuments`.

**Why this priority**: Extends the same pattern beyond RAG-specific tables. Healthcare customers with FHIR data, custom schemas, or HealthShare tables benefit directly — no ETL step, no data duplication.

**Independent Test**: Create a custom table with 50 rows, a text column, and a VECTOR column. Call `attach_existing_corpus` and verify graph queries work.

**Acceptance Scenarios**:

1. **Given** a table `MyApp.ClinicalNotes` with columns `note_id`, `note_text`, `note_embedding VECTOR(FLOAT,384)`, **When** the user calls `pipeline.attach_existing_corpus(source_table="MyApp.ClinicalNotes", id_col="note_id", text_col="note_text", embedding_col="note_embedding", graph_label="ClinicalNote")`, **Then** `engine.query("MATCH (n:ClinicalNote) RETURN n LIMIT 5")` returns 5 results.
2. **Given** the attached table, **When** the user calls `engine.vector_search("MyApp.ClinicalNotes", "note_embedding", query_vec)`, **Then** results come from the existing HNSW index with no re-indexing.

---

### User Story 3 - Dimension mismatch safety (Priority: P3)

A user attaches a table with 768-dim embeddings but later queries with a 384-dim vector. The system should fail fast with a clear error rather than returning garbage results.

**Why this priority**: Prevents a class of silent-failure bugs that are hard to diagnose. Dimension mismatches between query vectors and stored vectors produce IRIS runtime errors or meaningless similarity scores.

**Independent Test**: Attach a table with 768-dim vectors, then call vector_search with a 384-dim query vector. Verify a clear DimensionMismatchError is raised before the SQL is executed.

**Acceptance Scenarios**:

1. **Given** an attached table with `VECTOR(FLOAT,768)` embeddings, **When** the user calls `vector_search` with a 384-dim query vector, **Then** a `DimensionMismatchError` is raised with a message stating "Query vector dimension (384) does not match table embedding dimension (768)".
2. **Given** a table where the embedding column is NULL for all rows, **When** the user calls `attach_existing_corpus`, **Then** a warning is logged ("No non-NULL embeddings found — vector search will return no results until embeddings are populated") but the operation succeeds.

---

### Edge Cases

- What happens when the source table doesn't exist? → `ValueError` with table name
- What happens when the embedding column doesn't exist? → `ValueError` with column name
- What happens when the table has 0 rows? → Success with warning (graph queries return empty, vector search returns empty)
- What happens when the table has no HNSW index on the vector column? → Success with warning ("No HNSW index found — vector search will use brute force. Consider running BUILD INDEX for better performance.")
- What happens when `attach_existing_corpus` is called twice with different labels for the same table? → Both labels work (one table, two graph labels)
- What happens when `attach_existing_corpus` is called with the same label but a different table? → The label silently re-points to the new table (upsert semantics, matching IVG `map_sql_table` behavior). No explicit detach method in v1.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST expose a single method `attach_existing_corpus(source_table, id_col, text_col, embedding_col, graph_label)` on `HybridGraphRAGPipeline`
- **FR-002**: System MUST validate that `source_table` exists and contains `id_col`, `text_col`, and `embedding_col` columns before creating any mappings
- **FR-003**: System MUST call `engine.map_sql_table(source_table, id_col, graph_label)` to create the graph-to-SQL bridge (zero-copy, no data duplication)
- **FR-004**: System MUST call `engine.validate_vector_table(source_table, embedding_col)` to detect embedding dimension and row count
- **FR-005**: System MUST store the detected embedding dimension so that subsequent `vector_search` calls can validate query vector dimensions before executing SQL
- **FR-006**: System MUST be idempotent — calling `attach_existing_corpus` with the same parameters produces the same result; calling with the same label but a different table silently re-points the label (upsert semantics)
- **FR-007**: System MUST NOT copy data, create new tables, or re-compute embeddings
- **FR-008**: System MUST return a summary dict: `{table, label, id_col, embedding_col, dimension, row_count, has_hnsw_index}` where `has_hnsw_index` is `True`, `False`, or `None` (detection not supported on this IRIS build)

### Key Entities

- **CorpusBridge**: A registered mapping between an existing IRIS SQL table and IVG graph nodes — stored in `Graph_KG.table_mappings` (IVG's existing mapping table)
- **VectorRegistration**: The association of a specific VECTOR column on a mapped table for use with `engine.vector_search()` — stored as metadata alongside the table mapping

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can attach an existing 10,000-row corpus to graph + vector search in under 2 seconds (no data copy, no re-embedding). 10K rows is the acceptance test ceiling — the operation is O(1) metadata calls regardless of table size.
- **SC-002**: Graph queries over attached corpus return results immediately without any manual node creation
- **SC-003**: Vector similarity search over attached corpus uses the existing HNSW index with no re-indexing
- **SC-004**: Dimension mismatches are caught before SQL execution with a clear error message
- **SC-005**: The operation is idempotent — running it twice produces no errors or duplicates

## Assumptions

- IVG >= 1.27.0 is installed (provides `map_sql_table`, `validate_vector_table`, `vector_search`)
- The source table already has a VECTOR column with embeddings populated (pre-computed by IVR ingest, external tools, or IRIS `%Embedding`)
- The source table's VECTOR column may or may not have an HNSW index — the bridge works either way (HNSW just makes it faster)
- `Graph_KG.table_mappings` table exists (created by `engine.initialize_schema()`)

## Clarifications

### Session 2026-04-03

- Q: When a user re-attaches the same label to a different table, what happens? → A: Upsert — silently re-points the label to the new table (no explicit detach method in v1)
- Q: What is the maximum table size for acceptance testing? → A: 10K rows is the ceiling — the operation is O(1) metadata calls, testing beyond 10K adds no signal
