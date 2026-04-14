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

- [X] T001 Create directory `iris_src/src/RAG/SDK/` for ObjectScript class files
- [X] T002 Create directory `tests/objectscript/RAG/SDK/Test/` for %UnitTest classes
- [X] T003 Create `sql/schema.sql` with `CREATE TABLE IF NOT EXISTS` DDL for RAG.SourceDocuments (doc_id VARCHAR(64) PK, title VARCHAR(500), text_content LONGVARCHAR, metadata LONGVARCHAR, embedding VECTOR(DOUBLE, 384)), RAG.DocumentChunks (chunk_id VARCHAR(64) PK, source_doc_id VARCHAR(64) FK, chunk_text LONGVARCHAR, chunk_embedding VECTOR(DOUBLE, 384)), and RAG.Entities (entity_id VARCHAR(64) PK, entity_name VARCHAR(500), entity_type VARCHAR(100), source_doc_id VARCHAR(64) FK) — statements separated by GO delimiters, all IF NOT EXISTS
- [X] T004 Refactor Python IVR's schema creation to read DDL from `sql/schema.sql` instead of hardcoded strings — update `iris_vector_rag/storage/enterprise_storage.py` and any other file with inline CREATE TABLE statements for RAG tables to use `open("sql/schema.sql").read()` and execute each GO-delimited statement

**Checkpoint**: Directory structure exists, sql/schema.sql is the single DDL source, Python IVR reads from it.

---

## Phase 2: Foundational

**Purpose**: Deploy ObjectScript classes to IRIS and set up test infrastructure.

- [X] T005 Create `scripts/deploy_sdk.sh` — copies `iris_src/src/RAG/SDK/*.cls` to the IRIS container and loads them with `$SYSTEM.OBJ.Load()`, following the pattern of `scripts/deploy_vecindex.sh`
- [X] T006 Create `scripts/deploy_sdk_tests.sh` — copies `tests/objectscript/RAG/SDK/Test/*.cls` to the IRIS container for `%UnitTest.Manager.RunTest()`

**Checkpoint**: Deploy scripts exist. ObjectScript classes can be loaded into IRIS.

---

## Phase 3: User Story 1 — Schema Management (Priority: P1) 🎯 MVP

**Goal**: ObjectScript developer can create, inspect, and drop the RAG schema without Python.

**Independent Test**: Call `Initialize(384)`, verify tables exist, call `Status()`, call `Drop()`, verify tables gone.

### Tests for US1

- [X] T007 [US1] Write `tests/objectscript/RAG/SDK/Test/SchemaTest.cls` as `%UnitTest.TestCase` subclass. Each test must compile and pass independently — no mocking. Specific assertions required:
  - `TestInitialize`: call `##class(RAG.SDK.Schema).Initialize(384)`, assert `$SYSTEM.SQL.Schema.TableExists("RAG","SourceDocuments")=1`, assert `$SYSTEM.SQL.Schema.TableExists("RAG","DocumentChunks")=1`, assert `$SYSTEM.SQL.Schema.TableExists("RAG","Entities")=1`
  - `TestEmbeddingColumnExists`: after Initialize, query `SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA='RAG' AND TABLE_NAME='SourceDocuments' AND COLUMN_NAME='embedding'` — assert =1
  - `TestInitializeIdempotent`: call Initialize(384) twice, assert no SQLCODE error on second call
  - `TestStatus`: parse JSON from `##class(RAG.SDK.Schema).Status()`, assert `embeddingDimension=384`, assert `tables` array contains `"RAG.SourceDocuments"`, assert `documentCount=0`
  - `TestDrop`: call Drop(), assert `$SYSTEM.SQL.Schema.TableExists("RAG","SourceDocuments")=0`
  - `TestSchemaParity`: query INFORMATION_SCHEMA for RAG.SourceDocuments columns, assert column names and types match those defined in `sql/schema.sql` — catches schema drift between ObjectScript SDK and Python IVR

### ⛔ PHASE GATE: DO NOT PROCEED TO PHASE 4 UNTIL ALL SchemaTest methods pass

### Implementation for US1

- [X] T008 [US1] Implement `iris_src/src/RAG/SDK/Schema.cls` with three ClassMethods:
  - `Initialize(dim As %Integer = 384) As %String` — reads `sql/schema.sql` via `##class(%Stream.FileCharacter).%New()`, splits on `GO`, executes each DDL statement via `%SQL.Statement`, returns `{}`-JSON status
  - `Drop() As %String` — executes `DROP TABLE IF EXISTS` for each RAG table in reverse FK order
  - `Status() As %String` — queries INFORMATION_SCHEMA for RAG tables, counts rows in SourceDocuments, returns `{"tables":[...],"embeddingDimension":N,"documentCount":N}`

**Checkpoint**: Schema management works from ObjectScript — create, inspect, drop.

---

## Phase 4: User Story 2 — Document Ingest (Priority: P1)

**Goal**: Insert documents with pre-computed vectors from ObjectScript using TO_VECTOR().

**Independent Test**: Insert 10 documents, query `SELECT COUNT(*) FROM RAG.SourceDocuments`, assert 10.

### Tests for US2

- [X] T009 [US2] Write `tests/objectscript/RAG/SDK/Test/PipelineTest.cls`. All assertions must execute live SQL against IRIS — no mocking:
  - `TestAddDocumentWithVector`: call `AddDocument("test-001","Patient chest pain","{}","0.1,0.2,0.3,...")` (384 comma-separated floats), then `SELECT COUNT(*) FROM RAG.SourceDocuments WHERE doc_id='test-001'` — assert =1; `SELECT text_content FROM RAG.SourceDocuments WHERE doc_id='test-001'` — assert ="Patient chest pain"; verify embedding IS NOT NULL via `SELECT CASE WHEN embedding IS NULL THEN 0 ELSE 1 END FROM RAG.SourceDocuments WHERE doc_id='test-001'` — assert =1
  - `TestAddDocumentNullEmbedding`: call `AddDocument("test-002","No vector","{}","")`, assert row inserted with NULL embedding, assert returned JSON has `"status":"ok"` not an error
  - `TestAddDocumentBatch`: build JSON array of 10 docs with distinct doc_ids and 384-dim vectors, call `AddDocumentBatch`, assert `SELECT COUNT(*) FROM RAG.SourceDocuments WHERE doc_id LIKE 'batch-%'` = 10
  - `TestAddDocumentDuplicateId`: call `AddDocument("test-001",...)` twice — second call must not crash; assert exactly 1 row with doc_id='test-001'
  - `TestVectorRoundtrip`: after `AddDocument` with known vector "1.0,0.0,0.0,...", query `SELECT VECTOR_COSINE(embedding, TO_VECTOR('1.0,0.0,0.0,...', DOUBLE, 384)) FROM RAG.SourceDocuments WHERE doc_id='test-rt'` — assert score > 0.99 (self-similarity)

### ⛔ PHASE GATE: DO NOT PROCEED TO PHASE 5 UNTIL ALL PipelineTest methods pass

### Implementation for US2

- [X] T010 [US2] Implement `iris_src/src/RAG/SDK/Pipeline.cls` with three ClassMethods:
  - `AddDocument(id, text, metadata="{}",embedding="") As %String` — if embedding non-empty: `INSERT INTO RAG.SourceDocuments ... TO_VECTOR(:embedding, DOUBLE, 384)`, else insert with NULL; returns `{"id":id,"status":"ok"}`
  - `AddDocumentBatch(jsonArray As %String) As %String` — parses JSON array via `%DynamicArray.%FromJSON()`, loops calling `AddDocument` for each; returns `{"inserted":N,"errors":0}`
  - `AddDocumentWithEmbed(id, text, metadata="{}", model="all-MiniLM-L6-v2") As %String [Language=python]` — imports `sentence_transformers`, loads model, encodes text, returns comma-joined vector string; ObjectScript wrapper inserts via `AddDocument`

**Checkpoint**: Documents with pre-computed vectors insert correctly. Batch insert works.

---

## Phase 5: User Story 3 — Search (Priority: P1)

**Goal**: Vector, text, and hybrid search returning sorted JSON results from ObjectScript.

**Independent Test**: Insert 50 docs, call VectorSearch with a query vector, assert results array non-empty and sorted by score descending.

### Tests for US3

- [X] T011 [US3] Write `tests/objectscript/RAG/SDK/Test/SearchTest.cls`. Tests must insert their own known data and verify exact results — not just "non-empty array":
  - `TestVectorSearchReturnsTopResult`: insert doc "diabetes-001" with vector V1 and doc "diabetes-002" with vector V2 where V1 is more similar to query Q. Call `VectorSearch(Q,5)`. Parse returned JSON array. Assert `results[0].id = "diabetes-001"` (top result is most similar). Assert `results[0].score > results[1].score` (sorted descending). Assert all returned scores are in range 0.0-1.0.
  - `TestVectorSearchDimensionDetection`: insert doc with 384-dim vector, call `VectorSearch` with 384-dim query — assert succeeds. Call with 128-dim query — assert returns JSON error, not an IRIS crash.
  - `TestVectorSearchTopKRespected`: insert 20 docs, call `VectorSearch(Q,5)` — assert returned array has exactly 5 elements.
  - `TestTextSearch`: insert doc with text "metformin diabetes treatment", call `TextSearch("metformin",5)` — assert returned JSON array contains a result with `id` matching the inserted doc.
  - `TestHybridSearchRRF`: insert 5 docs, call `HybridSearch(Q,"diabetes",5,"RRF")` — assert returned JSON array is non-empty, all scores > 0, array is sorted descending by score.
  - `TestHybridSearchLinear`: call `HybridSearch(Q,"diabetes",5,"linear")` — assert non-empty results, different ordering than RRF (insert data where the two strategies produce different top results to distinguish them).
  - `TestDefaultTableFallback`: with no SetDefaultTable configured, call `VectorSearch` — assert results come from `RAG.SourceDocuments` (backward compat, no error)

### ⛔ PHASE GATE: DO NOT PROCEED TO PHASE 6 UNTIL ALL SearchTest methods pass

### Implementation for US3

- [X] T012 [US3] Implement `iris_src/src/RAG/SDK/Search.cls` with three ClassMethods:
  - `VectorSearch(queryVec As %String, topK As %Integer = 10) As %String` — 1) detect dimension from queryVec (count commas + 1); 2) call `##class(RAG.SDK.Bridge).GetDefaultTable()` to get target table config (falls back to RAG.SourceDocuments if not set); 3) execute `SELECT TOP :topK :idCol, :textCol, VECTOR_COSINE(:embCol, TO_VECTOR(?, DOUBLE, :dim)) AS score FROM :table ORDER BY score DESC` via `%SQL.Statement`; returns JSON array `[{"id":"...","text":"...","score":0.92},...]`
  - `TextSearch(text As %String, topK As %Integer = 10) As %String` — `SELECT TOP :topK ... WHERE text_content %CONTAINS(:text)` with fallback to `LIKE '%'_text_'%'`; returns same JSON shape
  - `HybridSearch(queryVec As %String, text As %String, topK As %Integer = 10, strategy As %String = "RRF") As %String` — calls `VectorSearch` and `TextSearch`, dispatches to private `FuseRRF` or `FuseLinear` based on strategy param; returns merged JSON array

- [X] T013 [P] [US3] Add private ClassMethods to `iris_src/src/RAG/SDK/Search.cls`:
  - `FuseRRF(vecResults As %DynamicArray, txtResults As %DynamicArray, k As %Integer = 60) As %DynamicArray [Private]` — RRF formula `1/(k+rank_v) + 1/(k+rank_t)`, sort descending, return top N
  - `FuseLinear(vecResults As %DynamicArray, txtResults As %DynamicArray, alpha As %Double = 0.7) As %DynamicArray [Private]` — `alpha * vec_score + (1-alpha) * txt_score`

**Checkpoint**: All three search modes work and return ranked JSON results.

---

## Phase 6: User Story 4 — Existing Table Bridge (Priority: P2)

**Goal**: Attach any existing IRIS table with a VECTOR column to the graph without copying data.

**Independent Test**: Create test table, call AttachTable, verify IVG graph traversal works on the mapped nodes.

### Tests for US4

- [X] T014 [US4] Write `tests/objectscript/RAG/SDK/Test/BridgeTest.cls`. Must create a real custom table in IRIS and verify actual graph mapping AND overlay search:
  - `Setup`: create `Test070.ClinicalNotes` with columns `note_id VARCHAR(50) PK, note_text VARCHAR(2000), embedding VECTOR(DOUBLE,384)`, insert 5 rows with known 384-dim vectors
  - `TestAttachTableSuccess`: call `##class(RAG.SDK.Bridge).AttachTable("Test070.ClinicalNotes","note_id","note_text","embedding","TestNote")`, parse returned JSON, assert `dimension=384`, assert `rowCount=5`, assert `label="TestNote"`
  - `TestAttachTableRegisteredInMappings`: after AttachTable, query `SELECT COUNT(*) FROM Graph_KG.table_mappings WHERE label='TestNote'` — assert =1
  - `TestSetDefaultTable`: call `##class(RAG.SDK.Bridge).SetDefaultTable("Test070.ClinicalNotes","note_id","note_text","embedding")`, then call `GetDefaultTable()`, parse JSON, assert `table="Test070.ClinicalNotes"` — proves config persisted
  - `TestOverlaySearch`: after SetDefaultTable, call `##class(RAG.SDK.Search).VectorSearch(queryVec, 5)` — parse results, assert ALL returned ids are from `Test070.ClinicalNotes` (i.e., match format "note-XXX" not "doc-XXX") — proves search hit the custom table, not RAG.SourceDocuments
  - `TestDefaultTableFallback`: call `##class(Graph.KG.Meta).Delete("rag_sdk_default_table")` to clear config, then call `VectorSearch` — assert it searches `RAG.SourceDocuments` (backward compat)
  - `TestAttachNonExistentTable`: call `AttachTable("NoSuch.Table",...)` — assert returned JSON contains `"error"` key
  - `TestAttachMissingColumn`: call `AttachTable("Test070.ClinicalNotes","note_id","note_text","no_such_col","X")` — assert JSON error
  - `TestAttachIdempotent`: call AttachTable twice — assert `SELECT COUNT(*) FROM Graph_KG.table_mappings WHERE label='TestNote'` = 1
  - `Teardown`: drop `Test070.ClinicalNotes`, delete from `Graph_KG.table_mappings WHERE label='TestNote'`, clear `rag_sdk_default_table` meta key

### ⛔ PHASE GATE: DO NOT PROCEED TO PHASE 7 UNTIL ALL BridgeTest methods pass (including TestOverlaySearch)

### Implementation for US4

- [X] T015 [US4] Implement `iris_src/src/RAG/SDK/Bridge.cls` with three ClassMethods:
  - `AttachTable(table, idCol, textCol, embCol, label) As %String` —
    1) validate: `SELECT COUNT(*) FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=:schema AND TABLE_NAME=:tbl AND COLUMN_NAME IN (:idCol,:textCol,:embCol)` — return JSON error if any missing;
    2) detect dimension: count commas in first non-NULL value of embCol + 1;
    3) register: `INSERT INTO Graph_KG.table_mappings (table_name, id_col, text_col, vector_col, label) VALUES (?,?,?,?,?)` (same table IVG's Python map_sql_table writes to — enables graph traversal);
    4) return `{"table":table,"label":label,"dimension":N,"rowCount":N}`
  - `SetDefaultTable(table As %String, idCol As %String, textCol As %String, embCol As %String) As %String` — persists table config via `##class(Graph.KG.Meta).Set("rag_sdk_default_table", table)` + Set for each col; returns `{"table":table,"status":"ok"}`. This lets `RAG.SDK.Search` use any IRIS table as the search target without touching `RAG.SourceDocuments`.
  - `GetDefaultTable() As %String` — reads from `Graph.KG.Meta`, returns `{"table":"...","id_col":"...","text_col":"...","embedding_col":"..."}` or `{"table":"RAG.SourceDocuments","id_col":"doc_id","text_col":"text_content","embedding_col":"embedding"}` as default if not set

**Checkpoint**: Existing IRIS tables with VECTOR columns are bridged to graph with one ClassMethod call.

---

## Phase 7: User Story 5 — RAGAS Evaluation (Priority: P3)

**Goal**: Run RAGAS metrics from ObjectScript with clear prerequisite documentation.

**Independent Test**: With ragas+datasets installed, insert 5 docs, call RunRAGAS with 3 questions, assert JSON output has faithfulness/answer_relevancy/context_precision/context_recall keys.

### Tests for US5

- [X] T016 [US5] Write `tests/objectscript/RAG/SDK/Test/EvaluateTest.cls`:
  - `TestRunRAGASMissingPackage`: temporarily rename `ragas` import to `ragas_NOTINSTALLED` to simulate missing package (or use a try/except check) — assert returned JSON contains `"error"` key with installation instructions containing the string `$system.Python.Install`
  - `TestRunRAGASOutputShape`: with ragas+datasets installed, insert 5 documents with known content, call `RunRAGAS` with 3 questions and matching ground truths — parse returned JSON, assert all four keys present: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`; assert each value is a float between 0.0 and 1.0 inclusive; assert `per_question` key is a JSON array with 3 elements

### ⛔ PHASE GATE: DO NOT PROCEED TO PHASE 8 UNTIL ALL EvaluateTest methods pass

### Implementation for US5

- [X] T017 [US5] Implement `iris_src/src/RAG/SDK/Evaluate.cls` with one ClassMethod:
  - `RunRAGAS(questionsJson, groundTruthsJson, topK=5, llmConfig="") As %String [Language=python]` — imports `ragas`, `datasets`; for each question calls `RAG.SDK.Search.VectorSearch` to retrieve contexts; builds `Dataset.from_list(data)`; calls `ragas.evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])`; returns JSON scores. On `ImportError`, returns `{"error":"RAGAS requires: Do $system.Python.Install(\"ragas\") and Do $system.Python.Install(\"datasets\")"}`

**Checkpoint**: RAGAS evaluation runs from ObjectScript with interpretable JSON output.

---

## Phase 8: Polish & Cross-Cutting Concerns

- [X] T018 [P] Write `iris_src/src/RAG/SDK/README.md` documenting all ClassMethods, prerequisites, and quickstart examples from `specs/070-objectscript-sdk/quickstart.md`
- [X] T019 [P] Add `make deploy-sdk` target to `Makefile` that calls `scripts/deploy_sdk.sh` followed by `scripts/deploy_sdk_tests.sh`
- [X] T020 [P] Add `make test-sdk` target to `Makefile` that runs `%UnitTest.Manager.RunTest("RAG.SDK.Test", "/noload")` against the IRIS container
- [X] T020a [P] Write `tests/objectscript/RAG/SDK/Test/PerfTest.cls` — insert 10,000 documents via `AddDocumentBatch`, run `VectorSearch` 20 times, assert p50 latency < 100ms — satisfies SC-002
- [X] T021 Run full %UnitTest suite via `make test-sdk` — all 5 test classes (SchemaTest, PipelineTest, SearchTest, BridgeTest, EvaluateTest) pass with 0 failures and 0 errors. A single failing method in any class is a hard stop.
- [X] T022 Write and run `tests/test_sdk_e2e.py` — full cross-language E2E test in Python:
  - Step 1: Call `##class(RAG.SDK.Schema).Drop()` then `Initialize(384)` — clean slate
  - Step 2: Insert 10 documents via `##class(RAG.SDK.Pipeline).AddDocument(...)` — known vectors
  - Step 3: Call Python IVR `pipeline.query("diabetes treatment", top_k=5)` — assert returns ObjectScript-inserted docs
  - Step 4: Insert 5 more documents via Python IVR `pipeline.add_documents([...])` — distinct text
  - Step 5: Call `##class(RAG.SDK.Search).TextSearch("distinct_keyword", 5)` from ObjectScript — assert returns Python-inserted docs
  - Step 6: **Overlay scenario**: call `##class(RAG.SDK.Bridge).SetDefaultTable("RAG.SourceDocuments","doc_id","text_content","embedding")` (points SDK at the default IVR table), then call `##class(RAG.SDK.Search).VectorSearch(queryVec,5)` — assert returns the same 15 docs from Steps 2+4
  - Step 7: Assert `SELECT COUNT(*) FROM RAG.SourceDocuments` = 15
  - This test is the final gate — if it fails, the SDK is not shippable

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
