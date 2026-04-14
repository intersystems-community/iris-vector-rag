# Feature Specification: ObjectScript SDK for IVR

**Feature Branch**: `070-objectscript-sdk`
**Created**: 2026-04-13
**Status**: Draft
**Input**: ObjectScript SDK for IVR — pure ObjectScript+SQL classes for schema management, document ingest, vector/text/hybrid search, existing table bridge, and RAGAS evaluation

## Problem Statement

IVR is Python-only. An ObjectScript developer cannot create RAG tables, ingest documents, run vector search, or evaluate pipeline quality without writing Python. This is a gap:
- IVG (iris-vector-graph) has 110 pure ObjectScript ClassMethods — customers can use the graph layer entirely from ObjectScript
- IVR has zero ObjectScript API — customers must context-switch to Python for every RAG operation
- Internal status reports falsely claimed ObjectScript integration existed (June 2025) — the referenced test files and source directories were never created

The RAG tables (`RAG.SourceDocuments`, `RAG.Entities`, etc.) are plain IRIS SQL. There is no technical barrier to ObjectScript operating on them directly.

## User Scenarios & Testing *(mandatory)*

### User Story 1 — Schema Management (Priority: P1)

An ObjectScript developer wants to create the RAG schema, check its status, and tear it down — all from ObjectScript without touching Python.

**Independent Test**: Call `##class(RAG.SDK.Schema).Initialize(384)`, verify tables exist via `$SYSTEM.SQL.Schema.TableExists()`, call `Status()`, call `Drop()`, verify tables gone.

**Acceptance Scenarios**:

1. **Given** a clean IRIS namespace with no RAG tables, **When** the developer calls `Do ##class(RAG.SDK.Schema).Initialize(384)`, **Then** `RAG.SourceDocuments`, `RAG.DocumentChunks`, and `RAG.Entities` tables are created with a `VECTOR(DOUBLE, 384)` embedding column on SourceDocuments.
2. **Given** an initialized schema, **When** the developer calls `Set json = ##class(RAG.SDK.Schema).Status()`, **Then** the returned JSON contains `{"tables": ["RAG.SourceDocuments", ...], "embeddingDimension": 384, "documentCount": 0}`.
3. **Given** an initialized schema, **When** `Initialize(384)` is called again, **Then** it succeeds idempotently (no error, no duplicate tables).
4. **Given** an initialized schema with data, **When** `Drop()` is called, **Then** all RAG.* tables are dropped.

---

### User Story 2 — Document Ingest (Priority: P1)

An ObjectScript developer wants to insert documents with pre-computed vector embeddings using `TO_VECTOR()`, or optionally generate embeddings via Embedded Python when no pre-computed vector is available.

**Independent Test**: Insert 10 documents with pre-computed 384-dim vectors, verify row count, verify vector search returns results.

**Acceptance Scenarios**:

1. **Given** an initialized schema, **When** the developer calls `Do ##class(RAG.SDK.Pipeline).AddDocument("doc1", "Patient presents with chest pain...", "{""source"":""ER""}", "0.12,0.34,...")`, **Then** a row is inserted into `RAG.SourceDocuments` with the text, metadata, and `TO_VECTOR()` embedding.
2. **Given** an initialized schema, **When** the developer calls `Do ##class(RAG.SDK.Pipeline).AddDocumentBatch(jsonArray)` with a JSON array of 100 documents (each with id, text, metadata, embedding), **Then** all 100 rows are inserted in a single batch.
3. **Given** an initialized schema and the `sentence-transformers` package installed in the IRIS Python environment, **When** the developer calls `Do ##class(RAG.SDK.Pipeline).AddDocumentWithEmbed("doc2", "Metformin dosage...", "{}")`, **Then** the embedding is generated via `Language=python` using `all-MiniLM-L6-v2` (384-dim, constitution default) and the row is inserted with the computed vector. An optional `model` parameter overrides the default.

---

### User Story 3 — Search (Priority: P1)

An ObjectScript developer wants to run vector similarity search, text search, and hybrid search — all returning JSON results.

**Independent Test**: Insert 50 documents, run `VectorSearch` with a query vector, verify results are returned sorted by score descending.

**Acceptance Scenarios**:

1. **Given** 50 documents with embeddings in `RAG.SourceDocuments`, **When** the developer calls `Set results = ##class(RAG.SDK.Search).VectorSearch(queryVecStr, 10)`, **Then** results is a JSON array of up to 10 documents with `id`, `text`, `score` fields, sorted by score descending. The SQL uses `VECTOR_COSINE(embedding, TO_VECTOR(?, DOUBLE, N))`.
2. **Given** 50 documents, **When** the developer calls `Set results = ##class(RAG.SDK.Search).TextSearch("chest pain", 10)`, **Then** results is a JSON array of matching documents using SQL `%CONTAINS` or `LIKE` text matching.
3. **Given** 50 documents, **When** the developer calls `Set results = ##class(RAG.SDK.Search).HybridSearch(queryVecStr, "chest pain", 10, "RRF")`, **Then** results combine vector and text scores using RRF (Reciprocal Rank Fusion) as the default strategy. The `strategy` parameter accepts `"RRF"` (default), `"linear"` (weighted combination), or a custom strategy name registered via a future extension point.

---

### User Story 4 — Existing Table Bridge (Priority: P2)

An ObjectScript developer has data in an existing IRIS table (e.g., `MyApp.ClinicalNotes`) with a VECTOR column. They want two things: (a) make it searchable via IVG graph queries without copying data, AND (b) point the IVR SDK's `VectorSearch`, `TextSearch`, and `HybridSearch` operations at that table so they don't need to use `RAG.SourceDocuments` at all.

**Independent Test**: Create a custom table, call `AttachTable`, call `SetDefaultTable`, then call `VectorSearch` and verify it searches the custom table — not `RAG.SourceDocuments`.

**Acceptance Scenarios**:

1. **Given** `MyApp.ClinicalNotes` with columns `note_id`, `note_text`, `embedding VECTOR(DOUBLE, 384)`, **When** the developer calls `Do ##class(RAG.SDK.Bridge).AttachTable("MyApp.ClinicalNotes", "note_id", "note_text", "embedding", "ClinNote")`, **Then** `##class(Graph.KG.Traversal).BFSFastJson(...)` can traverse the mapped nodes.
2. **Given** the same table, **When** the developer calls `Do ##class(RAG.SDK.Bridge).SetDefaultTable("MyApp.ClinicalNotes", "note_id", "note_text", "embedding")`, **Then** subsequent calls to `##class(RAG.SDK.Search).VectorSearch(queryVec, 5)` search `MyApp.ClinicalNotes` instead of `RAG.SourceDocuments` — the developer never touches `RAG.SourceDocuments` at all.
3. **Given** `SetDefaultTable` has been called, **When** `##class(RAG.SDK.Search).GetDefaultTable()` is called, **Then** it returns `{"table":"MyApp.ClinicalNotes","id_col":"note_id","text_col":"note_text","embedding_col":"embedding"}`.
4. **Given** no `SetDefaultTable` has been called, **When** any search method is called, **Then** it defaults to `RAG.SourceDocuments` (backward compatible).
5. **Given** `SetDefaultTable` has been called with a table that has 10K rows and embeddings, **When** `VectorSearch` is called, **Then** results come from the custom table with correct scores — proving zero ETL was required.

---

### User Story 5 — RAGAS Evaluation (Priority: P3)

An ObjectScript developer wants to evaluate RAG quality using RAGAS metrics. This is the one operation that legitimately requires `Language=python` — RAGAS is a Python library.

**Independent Test**: Insert 5 documents, define 3 evaluation questions with ground truths, call `RunRAGAS`, verify JSON output contains faithfulness, answer_relevancy, context_precision, context_recall scores.

**Acceptance Scenarios**:

1. **Given** documents in the RAG schema and `ragas`, `datasets` packages installed in IRIS Python env (`Do $system.Python.Install("ragas")`), **When** the developer calls `Set scores = ##class(RAG.SDK.Evaluate).RunRAGAS(questionsJson, groundTruthsJson, 5, "my-llm-config")`, **Then** scores is a JSON object `{"faithfulness": 0.92, "answer_relevancy": 0.87, "context_precision": 0.81, "context_recall": 0.78, "per_question": [...]}`.
2. **Given** the `ragas` package is NOT installed, **When** `RunRAGAS` is called, **Then** a clear error message states: "RAGAS requires Python packages: Do $system.Python.Install(""ragas"") and Do $system.Python.Install(""datasets"")".

---

### Edge Cases

- Schema already exists when `Initialize()` called → idempotent success
- `AddDocument` with empty embedding string → insert row with NULL embedding, log warning
- `VectorSearch` with wrong dimension query vector → IRIS SQL error caught and returned as clear JSON error
- `AttachTable` with non-existent table → `ValueError` style JSON error
- `RunRAGAS` without Python packages → clear prerequisites error message
- `AddDocumentWithEmbed` without sentence-transformers installed → clear prerequisites error

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: All classes MUST be in the `RAG.SDK` package namespace
- **FR-002**: All methods MUST be ClassMethods (no instance state needed — SQL is the state)
- **FR-003**: All methods MUST accept and return JSON strings (no `%DynamicObject` in signatures — keeps it wire-friendly for REST/MCP)
- **FR-004**: Schema management MUST use SQL DDL (`CREATE TABLE`, `DROP TABLE`) read from a shared `.sql` file (`sql/schema.sql`) that both Python IVR and ObjectScript SDK reference — single source of truth, no hardcoded DDL in either language
- **FR-005**: Document ingest MUST use `TO_VECTOR()` for pre-computed embeddings
- **FR-006**: Search MUST use `VECTOR_COSINE()` or `VECTOR_DOT_PRODUCT()` SQL functions
- **FR-007**: Bridge MUST register the table mapping by writing directly to `Graph_KG.table_mappings` SQL table (the same table IVG's Python `map_sql_table` writes to) — `Graph.KG.Meta` does not support table-level graph traversal registration
- **FR-008**: RAGAS evaluation MUST use `Language=python` and clearly document prerequisites
- **FR-009**: No method except `AddDocumentWithEmbed` and `RunRAGAS` may use `Language=python`
- **FR-010**: SDK MUST operate on the same `RAG.*` tables as Python IVR — shared data, not a parallel schema
- **FR-011**: `RAG.SDK.Bridge` MUST provide `SetDefaultTable(table, idCol, textCol, embCol)` that stores the table config in `Graph_KG.sdk_config` (or equivalent persistent store) so all subsequent `RAG.SDK.Search` calls use that table instead of `RAG.SourceDocuments`
- **FR-012**: `RAG.SDK.Search` methods MUST check for a configured default table first (from `SetDefaultTable`) and fall back to `RAG.SourceDocuments` if none is set — backward compatible

### Key Entities

- **RAG.SDK.Schema** — DDL wrapper (Initialize, Drop, Status)
- **RAG.SDK.Pipeline** — document CRUD (AddDocument, AddDocumentBatch, AddDocumentWithEmbed)
- **RAG.SDK.Search** — query operations (VectorSearch, TextSearch, HybridSearch)
- **RAG.SDK.Bridge** — existing table bridge (AttachTable, SetDefaultTable, GetDefaultTable)
- **RAG.SDK.Evaluate** — RAGAS evaluation (RunRAGAS) [Language=python]

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An ObjectScript developer can point the SDK at an existing IRIS table and run vector search against it with zero ETL — all in < 10 lines of ObjectScript (`SetDefaultTable` + `VectorSearch`)
- **SC-002**: `VectorSearch` returns results in < 100ms for 10K documents (same SQL as Python IVR)
- **SC-003**: ObjectScript SDK and Python IVR operate on identical tables — a document inserted from ObjectScript is searchable from Python and vice versa
- **SC-004**: RAGAS evaluation runs from ObjectScript with clear prerequisite documentation and returns interpretable JSON scores
- **SC-005**: All 5 `.cls` files total < 500 lines of ObjectScript (thin SQL wrappers, not reimplementation)

## Assumptions

- IRIS 2025.1+ (VECTOR type, VECTOR_COSINE, TO_VECTOR available on all license tiers)
- IVG >= 1.27.0 installed for Bridge functionality (map_sql_table)
- For `AddDocumentWithEmbed`: sentence-transformers installed in IRIS Python env
- For `RunRAGAS`: ragas + datasets installed in IRIS Python env
- The `RAG.*` schema namespace is shared between Python IVR and ObjectScript SDK

## Clarifications

### Session 2026-04-13

- Q: How should HybridSearch combine vector and text scores? → A: RRF (Reciprocal Rank Fusion) as default, but pluggable — strategy parameter accepts "RRF", "linear", or custom name
- Q: Should the SDK hardcode DDL or share it with Python IVR? → A: Shared `.sql` file (`sql/schema.sql`) — single source of truth for both languages
- Q: What embedding model should AddDocumentWithEmbed default to? → A: `all-MiniLM-L6-v2` (384-dim) — constitution default, configurable via optional `model` parameter
