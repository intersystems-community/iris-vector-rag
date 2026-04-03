# Tasks: Attach Existing Corpus

**Input**: Design documents from `/specs/069-attach-existing-corpus/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/

**Tests**: Included (test-first per CLAUDE.md mandate).

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup

**Purpose**: Ensure IVG engine is available and graph schema initialized

- [X] T001 Verify iris-vector-graph >= 1.27.0 is installed and `map_sql_table`, `validate_vector_table`, `vector_search` are importable in iris_vector_rag/pipelines/hybrid_graphrag.py
- [X] T002 Add `DimensionMismatchError` exception class and `AttachResult` TypedDict to iris_vector_rag/pipelines/hybrid_graphrag.py (from contracts/attach_existing_corpus.py)

---

## Phase 2: Foundational

**Purpose**: Add `_attached_corpora` dict and HNSW detection helper to `HybridGraphRAGPipeline`

- [X] T003 Add `self._attached_corpora: Dict[str, AttachResult] = {}` to `HybridGraphRAGPipeline.__init__()` in iris_vector_rag/pipelines/hybrid_graphrag.py
- [X] T004 Add `_detect_hnsw_index(self, source_table: str, embedding_col: str) -> Optional[bool]` private method to `HybridGraphRAGPipeline` in iris_vector_rag/pipelines/hybrid_graphrag.py — queries INFORMATION_SCHEMA.INDEXES for a VECTOR index on the column; returns True/False if detected, None if detection not supported on this IRIS build

**Checkpoint**: Foundation ready — user story implementation can begin.

---

## Phase 3: User Story 1 — Attach RAG corpus to graph (Priority: P1) 🎯 MVP

**Goal**: Call `attach_existing_corpus("RAG.SourceDocuments", ...)` and get graph + vector search over existing data with zero data copy.

**Independent Test**: Insert 100 docs into RAG.SourceDocuments, call attach, verify MATCH and vector_search both return results.

### Tests for User Story 1

- [X] T005 [P] [US1] Write test_attach_corpus_basic in tests/test_attach_corpus.py (repo root — not colbert_iris/, since this feature applies to any IRIS table) — creates a test table with 100 rows + VECTOR(FLOAT,384), calls `attach_existing_corpus`, asserts `AttachResult` has correct dimension/row_count/label and `has_hnsw_index` is a bool or None, verifies `engine.query("MATCH (d:TestDoc)")` returns rows
- [X] T006 [P] [US1] Write test_attach_corpus_idempotent in tests/test_attach_corpus.py — calls `attach_existing_corpus` twice with same params, asserts no error and same result
- [X] T007 [P] [US1] Write test_attach_corpus_vector_search in tests/test_attach_corpus.py — attaches table, runs `engine.vector_search()` with a random query vector matching the table dimension, asserts results returned with scores > 0
- [X] T008 [P] [US1] Write test_attach_new_rows_visible in tests/test_attach_corpus.py — attaches table, inserts new row, verifies graph query includes it without re-attach

### Implementation for User Story 1

- [X] T009 [US1] Implement `attach_existing_corpus(self, source_table, id_col, text_col, embedding_col, graph_label) -> AttachResult` on `HybridGraphRAGPipeline` in iris_vector_rag/pipelines/hybrid_graphrag.py — validate table exists, call `engine.validate_vector_table()`, call `engine.map_sql_table()`, detect HNSW, store in `_attached_corpora`, return AttachResult
- [X] T010 [US1] Add validation: if `source_table` or any column doesn't exist, raise `ValueError` with table/column name in iris_vector_rag/pipelines/hybrid_graphrag.py
- [X] T011 [US1] Add warning logging: if embedding column is all NULL, log warning but succeed; if no HNSW index, log suggestion to BUILD INDEX in iris_vector_rag/pipelines/hybrid_graphrag.py

**Checkpoint**: User Story 1 complete — attach + graph query + vector search works on RAG.SourceDocuments.

---

## Phase 4: User Story 2 — Attach custom IRIS table (Priority: P2)

**Goal**: Same `attach_existing_corpus` works on any IRIS table (HS.*, MyApp.*, etc.) not just RAG.SourceDocuments.

**Independent Test**: Create a custom table `TestApp.ClinicalNotes`, attach it, verify graph queries work.

### Tests for User Story 2

- [X] T012 [P] [US2] Write test_attach_custom_table in tests/test_attach_corpus.py — creates `TestApp.Notes` with 50 rows, different column names (note_id, note_text, note_emb), calls attach with graph_label="Note", verifies MATCH works

### Implementation for User Story 2

- [X] T013 [US2] Verify no code changes needed — US1 implementation should already handle any table name. If schema-prefixed tables (e.g., `MyApp.ClinicalNotes`) need special handling in `validate_vector_table` or `map_sql_table`, fix in iris_vector_rag/pipelines/hybrid_graphrag.py

**Checkpoint**: Custom tables work identically to RAG tables.

---

## Phase 5: User Story 3 — Dimension mismatch safety (Priority: P3)

**Goal**: Fail fast with `DimensionMismatchError` when query vector dimension != attached table dimension.

**Independent Test**: Attach 768-dim table, search with 384-dim vector, verify error raised before SQL.

### Tests for User Story 3

- [X] T014 [P] [US3] Write test_dimension_mismatch_error in tests/test_attach_corpus.py — attaches 384-dim table, calls vector_search with 768-dim query, asserts `DimensionMismatchError` raised with message containing both dimensions
- [X] T015 [P] [US3] Write test_attach_all_null_embeddings in tests/test_attach_corpus.py — creates table with all-NULL VECTOR column, calls attach, asserts success with warning logged

### Implementation for User Story 3

- [X] T016 [US3] Add `_validate_query_dimension(self, graph_label: str, query_vec)` private method on `HybridGraphRAGPipeline` in iris_vector_rag/pipelines/hybrid_graphrag.py — looks up `_attached_corpora[graph_label]`, checks `len(query_vec) == dimension`, raises `DimensionMismatchError` if mismatch. Not a public API — callers use `engine.vector_search()` directly; this is a guard wired into the pipeline's existing search methods

**Checkpoint**: Dimension mismatches are caught with clear errors.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T017 [P] Write test_attach_nonexistent_table in tests/test_attach_corpus.py — verifies ValueError for missing table
- [X] T018 [P] Write test_attach_nonexistent_column in tests/test_attach_corpus.py — verifies ValueError for missing column
- [X] T019 [P] Write test_upsert_repoint_label in tests/test_attach_corpus.py — attaches label to table A, then same label to table B, verifies graph queries now hit table B
- [X] T020 Run full test suite: `pytest tests/test_attach_corpus.py -v` — all tests pass
- [X] T021 Run quickstart.md validation against live IRIS

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Setup — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Foundational — this is the MVP
- **US2 (Phase 4)**: Depends on US1 (verifies same code handles custom tables)
- **US3 (Phase 5)**: Depends on US1 (needs attached corpus to test dimension check)
- **Polish (Phase 6)**: Depends on US1-US3 complete

### Parallel Opportunities

- T005-T008 (US1 tests) can all be written in parallel
- T012 (US2 test) can be written in parallel with T014-T015 (US3 tests)
- T017-T019 (Polish tests) can all be written in parallel

---

## Parallel Example: User Story 1

```bash
# Write all US1 tests in parallel:
Task: "T005 — test_attach_corpus_basic in tests/test_attach_corpus.py"
Task: "T006 — test_attach_corpus_idempotent in tests/test_attach_corpus.py"
Task: "T007 — test_attach_corpus_vector_search in tests/test_attach_corpus.py"
Task: "T008 — test_attach_new_rows_visible in tests/test_attach_corpus.py"

# Then implement sequentially:
Task: "T009 — Implement attach_existing_corpus method"
Task: "T010 — Add validation for missing tables/columns"
Task: "T011 — Add warning logging for NULL embeddings / missing HNSW"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T002)
2. Complete Phase 2: Foundational (T003-T004)
3. Complete Phase 3: User Story 1 (T005-T011)
4. **STOP and VALIDATE**: All US1 tests pass, graph + vector search work
5. This is a deployable increment

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. US1 → Attach RAG tables ✓ (MVP)
3. US2 → Custom tables verified ✓
4. US3 → Dimension safety ✓
5. Polish → Edge cases, full validation ✓
