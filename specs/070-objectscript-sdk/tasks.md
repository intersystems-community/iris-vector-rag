# Tasks: ObjectScript SDK for IVR

**Input**: Design documents from `/specs/070-objectscript-sdk/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Included (TDD — %UnitTest per CLAUDE.md mandate).

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to

---

## Phase 1: Setup

**Purpose**: Create directory structure and shared DDL file.

- [ ] T001 Create directory `iris_src/src/RAG/SDK/` for ObjectScript class files
- [ ] T002 Create directory `tests/objectscript/RAG/SDK/Test/` for %UnitTest classes
- [ ] T003 Create `sql/schema.sql` with `CREATE TABLE IF NOT EXISTS` DDL for RAG.SourceDocuments (doc_id VARCHAR(64) PK, title VARCHAR(500), text_content LONGVARCHAR, metadata LONGVARCHAR, embedding VECTOR(DOUBLE, 384)), RAG.DocumentChunks (chunk_id VARCHAR(64) PK, source_doc_id VARCHAR(64) FK, chunk_text LONGVARCHAR, chunk_embedding VECTOR(DOUBLE, 384)), and RAG.Entities (entity_id VARCHAR(64) PK, entity_name VARCHAR(500), entity_type VARCHAR(100), source_doc_id VARCHAR(64) FK) — statements separated by GO delimiters, all IF NOT EXISTS
- [ ] T004 Refactor Python IVR's schema creation to read DDL from `sql/schema.sql` instead of hardcoded strings — update `iris_vector_rag/storage/enterprise_storage.py` and any other file with inline CREATE TABLE statements for RAG tables to use `open("sql/schema.sql").read()` and execute each GO-delimited statement

**Checkpoint**: Directory structure exists, sql/schema.sql is the single DDL source, Python IVR reads from it.

---

## Phase 2: Foundational

**Purpose**: Deploy ObjectScript classes to IRIS and set up test infrastructure.

- [ ] T005 Create `scripts/deploy_sdk.sh` — copies `iris_src/src/RAG/SDK/*.cls` to the IRIS container and loads them with `$SYSTEM.OBJ.Load()`, following the pattern of `scripts/deploy_vecindex.sh`
- [ ] T006 Create `scripts/deploy_sdk_tests.sh` — copies `tests/objectscript/RAG/SDK/Test/*.cls` to the IRIS container for `%UnitTest.Manager.RunTest()`

**Checkpoint**: Deploy scripts exist. ObjectScript classes can be loaded into IRIS.

---

## Phase 3: User Story 1 — Schema Management (Priority: P1) 🎯 MVP

**Goal**: ObjectScript developer can create, inspect, and drop the RAG schema without Python.

**Independent Test**: Call `Initialize(384)`, verify tables exist, call `Status()`, call `Drop()`, verify tables gone.

### Tests for US1

- [ ] T007 [US1] Write `tests/objectscript/RAG/SDK/Test/SchemaTest.cls` as `%UnitTest.TestCase` subclass with: `TestInitialize` (calls `Initialize(384)`, asserts tables exist via `$SYSTEM.SQL.Schema.TableExists()`), `TestInitializeIdempotent` (calls twice, expects no error), `TestStatus` (parses returned JSON, asserts `embeddingDimension=384`), `TestDrop` (calls `Drop()`, asserts tables gone)

### Implementation for US1

- [ ] T008 [US1] Implement `iris_src/src/RAG/SDK/Schema.cls` with three ClassMethods:
  - `Initialize(dim As %Integer = 384) As %String` — reads `sql/schema.sql` via `##class(%Stream.FileCharacter).%New()`, splits on `GO`, executes each DDL statement via `%SQL.Statement`, returns `{}`-JSON status
  - `Drop() As %String` — executes `DROP TABLE IF EXISTS` for each RAG table in reverse FK order
  - `Status() As %String` — queries INFORMATION_SCHEMA for RAG tables, counts rows in SourceDocuments, returns `{"tables":[...],"embeddingDimension":N,"documentCount":N}`

**Checkpoint**: Schema management works from ObjectScript — create, inspect, drop.

---

## Phase 4: User Story 2 — Document Ingest (Priority: P1)

**Goal**: Insert documents with pre-computed vectors from ObjectScript using TO_VECTOR().

**Independent Test**: Insert 10 documents, query `SELECT COUNT(*) FROM RAG.SourceDocuments`, assert 10.

### Tests for US2

- [ ] T009 [US2] Write `tests/objectscript/RAG/SDK/Test/PipelineTest.cls` with: `TestAddDocument` (calls AddDocument with 384-dim comma-vector, asserts row exists via SQL), `TestAddDocumentBatch` (inserts 10 at once, asserts count=10), `TestAddDocumentNullEmbedding` (empty embedding string, asserts row inserted with NULL embedding)

### Implementation for US2

- [ ] T010 [US2] Implement `iris_src/src/RAG/SDK/Pipeline.cls` with three ClassMethods:
  - `AddDocument(id, text, metadata="{}",embedding="") As %String` — if embedding non-empty: `INSERT INTO RAG.SourceDocuments ... TO_VECTOR(:embedding, DOUBLE, 384)`, else insert with NULL; returns `{"id":id,"status":"ok"}`
  - `AddDocumentBatch(jsonArray As %String) As %String` — parses JSON array via `%DynamicArray.%FromJSON()`, loops calling `AddDocument` for each; returns `{"inserted":N,"errors":0}`
  - `AddDocumentWithEmbed(id, text, metadata="{}", model="all-MiniLM-L6-v2") As %String [Language=python]` — imports `sentence_transformers`, loads model, encodes text, returns comma-joined vector string; ObjectScript wrapper inserts via `AddDocument`

**Checkpoint**: Documents with pre-computed vectors insert correctly. Batch insert works.

---

## Phase 5: User Story 3 — Search (Priority: P1)

**Goal**: Vector, text, and hybrid search returning sorted JSON results from ObjectScript.

**Independent Test**: Insert 50 docs, call VectorSearch with a query vector, assert results array non-empty and sorted by score descending.

### Tests for US3

- [ ] T011 [US3] Write `tests/objectscript/RAG/SDK/Test/SearchTest.cls` with: `TestVectorSearch` (insert 10 docs, search with matching vector, assert top result has highest score), `TestTextSearch` (insert docs with known keyword, assert TextSearch finds them), `TestHybridSearchRRF` (assert HybridSearch with strategy="RRF" returns merged results), `TestHybridSearchLinear` (strategy="linear")

### Implementation for US3

- [ ] T012 [US3] Implement `iris_src/src/RAG/SDK/Search.cls` with three ClassMethods:
  - `VectorSearch(queryVec As %String, topK As %Integer = 10) As %String` — detect dimension from queryVec (count commas + 1); execute `SELECT TOP :topK doc_id, text_content, VECTOR_COSINE(embedding, TO_VECTOR(?, DOUBLE, :dim)) AS score FROM RAG.SourceDocuments ORDER BY score DESC` via `%SQL.Statement`; returns JSON array `[{"id":"...","text":"...","score":0.92},...]`
  - `TextSearch(text As %String, topK As %Integer = 10) As %String` — `SELECT TOP :topK ... WHERE text_content %CONTAINS(:text)` with fallback to `LIKE '%'_text_'%'`; returns same JSON shape
  - `HybridSearch(queryVec As %String, text As %String, topK As %Integer = 10, strategy As %String = "RRF") As %String` — calls `VectorSearch` and `TextSearch`, dispatches to private `FuseRRF` or `FuseLinear` based on strategy param; returns merged JSON array

- [ ] T013 [P] [US3] Add private ClassMethods to `iris_src/src/RAG/SDK/Search.cls`:
  - `FuseRRF(vecResults As %DynamicArray, txtResults As %DynamicArray, k As %Integer = 60) As %DynamicArray [Private]` — RRF formula `1/(k+rank_v) + 1/(k+rank_t)`, sort descending, return top N
  - `FuseLinear(vecResults As %DynamicArray, txtResults As %DynamicArray, alpha As %Double = 0.7) As %DynamicArray [Private]` — `alpha * vec_score + (1-alpha) * txt_score`

**Checkpoint**: All three search modes work and return ranked JSON results.

---

## Phase 6: User Story 4 — Existing Table Bridge (Priority: P2)

**Goal**: Attach any existing IRIS table with a VECTOR column to the graph without copying data.

**Independent Test**: Create test table, call AttachTable, verify IVG graph traversal works on the mapped nodes.

### Tests for US4

- [ ] T014 [US4] Write `tests/objectscript/RAG/SDK/Test/BridgeTest.cls` with: `TestAttachTable` (creates `Test070.Notes` with VECTOR column, calls AttachTable, asserts returned JSON has rowCount>0), `TestAttachNonExistentTable` (asserts JSON error returned), `TestAttachWrongColumn` (non-existent column, asserts error)

### Implementation for US4

- [ ] T015 [US4] Implement `iris_src/src/RAG/SDK/Bridge.cls` with one ClassMethod:
  - `AttachTable(table, idCol, textCol, embCol, label) As %String` —
    1) validate: `SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=:schema AND TABLE_NAME=:tbl AND COLUMN_NAME IN (:idCol,:textCol,:embCol)` — return JSON error if any missing;
    2) detect dimension: count commas in first non-NULL value of embCol + 1;
    3) register: `INSERT INTO Graph_KG.table_mappings (table_name, id_col, text_col, vector_col, label) VALUES (?,?,?,?,?)` (same table IVG's Python map_sql_table writes to — enables graph traversal);
    4) return `{"table":table,"label":label,"dimension":N,"rowCount":N}`

**Checkpoint**: Existing IRIS tables with VECTOR columns are bridged to graph with one ClassMethod call.

---

## Phase 7: User Story 5 — RAGAS Evaluation (Priority: P3)

**Goal**: Run RAGAS metrics from ObjectScript with clear prerequisite documentation.

**Independent Test**: With ragas+datasets installed, insert 5 docs, call RunRAGAS with 3 questions, assert JSON output has faithfulness/answer_relevancy/context_precision/context_recall keys.

### Tests for US5

- [ ] T016 [US5] Write `tests/objectscript/RAG/SDK/Test/EvaluateTest.cls` with: `TestRunRAGASMissingPackage` (mock missing ragas, assert clear error JSON), `TestRunRAGASOutputShape` (with packages installed, assert returned JSON has all 4 metric keys as floats 0-1)

### Implementation for US5

- [ ] T017 [US5] Implement `iris_src/src/RAG/SDK/Evaluate.cls` with one ClassMethod:
  - `RunRAGAS(questionsJson, groundTruthsJson, topK=5, llmConfig="") As %String [Language=python]` — imports `ragas`, `datasets`; for each question calls `RAG.SDK.Search.VectorSearch` to retrieve contexts; builds `Dataset.from_list(data)`; calls `ragas.evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])`; returns JSON scores. On `ImportError`, returns `{"error":"RAGAS requires: Do $system.Python.Install(\"ragas\") and Do $system.Python.Install(\"datasets\")"}`

**Checkpoint**: RAGAS evaluation runs from ObjectScript with interpretable JSON output.

---

## Phase 8: Polish & Cross-Cutting Concerns

- [ ] T018 [P] Write `iris_src/src/RAG/SDK/README.md` documenting all ClassMethods, prerequisites, and quickstart examples from `specs/070-objectscript-sdk/quickstart.md`
- [ ] T019 [P] Add `make deploy-sdk` target to `Makefile` that calls `scripts/deploy_sdk.sh` followed by `scripts/deploy_sdk_tests.sh`
- [ ] T020 [P] Add `make test-sdk` target to `Makefile` that runs `%UnitTest.Manager.RunTest("RAG.SDK.Test", "/noload")` against the IRIS container
- [ ] T020a [P] Write `tests/objectscript/RAG/SDK/Test/PerfTest.cls` — insert 10,000 documents via `AddDocumentBatch`, run `VectorSearch` 20 times, assert p50 latency < 100ms — satisfies SC-002
- [ ] T021 Run full %UnitTest suite via `make test-sdk` — all 5 test classes pass
- [ ] T022 Verify interoperability: insert 5 documents via ObjectScript `AddDocument`, query via Python `pipeline.query()`, assert documents returned

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 (needs sql/schema.sql and directories)
- **US1 Schema (Phase 3)**: Depends on Phase 2 — BLOCKS all other user stories
- **US2 Ingest (Phase 4)**: Depends on US1 (needs schema initialized)
- **US3 Search (Phase 5)**: Depends on US2 (needs documents to search)
- **US4 Bridge (Phase 6)**: Depends on Phase 2 only (independent of US1-3)
- **US5 Evaluate (Phase 7)**: Depends on US3 (needs search to retrieve contexts)
- **Polish (Phase 8)**: Depends on all user stories

### Parallel Opportunities

- T007 (schema tests) and T003 (sql/schema.sql) can be written in parallel
- T009 (pipeline tests), T011 (search tests), T014 (bridge tests) can be written in parallel after T007 passes
- T013 (RRF/Linear private methods) can be written in parallel with T012 body
- T018, T019, T020 (polish) can all run in parallel

---

## Implementation Strategy

### MVP (User Stories 1-3 only — Phases 1-5)

1. Complete Phases 1-2 (setup + deploy)
2. US1 Schema: tests → `RAG.SDK.Schema.cls` → validate
3. US2 Ingest: tests → `RAG.SDK.Pipeline.cls` (AddDocument + Batch only, skip WithEmbed) → validate
4. US3 Search: tests → `RAG.SDK.Search.cls` → validate
5. **STOP and VALIDATE**: ObjectScript dev can create schema, insert docs with vectors, and search — in pure ObjectScript

### Incremental Delivery

1. MVP (US1+2+3) → ships core ObjectScript RAG capability
2. US4 Bridge → zero-copy path for existing IRIS tables
3. US5 Evaluate → RAGAS from ObjectScript
4. Polish → Makefile targets, README, interop test

---

## Parallel Example: User Story 3

```
# In parallel:
Task T011 — Write SearchTest.cls
Task T013 — Write FuseRRF + FuseLinear private methods

# Then sequentially:
Task T012 — Implement Search.cls public methods (depends on tests)
```
