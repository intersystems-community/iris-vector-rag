# Tasks: Fix GraphRAG Vector Retrieval Logic

**Feature Branch**: `033-fix-graphrag-vector`
**Input**: Design documents from `/Users/intersystems-community/ws/rag-templates/specs/033-fix-graphrag-vector/`
**Prerequisites**: plan.md, research.md, data-model.md, quickstart.md, contracts/ (all ✅ available)

## Execution Flow

```
1. Load plan.md → Tech stack: Python 3.12, iris_rag, IRIS 2025.3.0, RAGAS
2. Load contracts/ → 4 contract specifications (VSC-001, DVC-002, RAG-003, LOG-004)
3. Load data-model.md → Vector search entities (QueryEmbedding, DocumentEmbedding, SimilarityScore, RetrievedDocument)
4. Load quickstart.md → Validation procedures (smoke tests, RAGAS evaluation)
5. Generate tasks by TDD workflow:
   → Phase 3.1: Setup (prerequisites check)
   → Phase 3.2: Contract tests [MUST FAIL before implementation]
   → Phase 3.3: Investigation (identify root cause)
   → Phase 3.4: Implementation (fix vector search)
   → Phase 3.5: Validation (verify fix works)
   → Phase 3.6: Documentation (update findings)
6. Apply task rules:
   → Contract tests can run in parallel [P]
   → Investigation tasks sequential (share findings)
   → Implementation tasks modify same file (graphrag.py) → sequential
   → Validation tests can run in parallel [P]
7. Dependencies:
   → T001-T005 (contract tests) → T006-T009 (investigation) → T010-T014 (implementation) → T015-T018 (validation) → T019-T021 (docs)
8. Return: 21 tasks ready for TDD execution
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- All file paths are absolute from repository root

## Path Conventions

This is a **single project** structure (Python framework extension):
- **Source code**: `iris_rag/pipelines/graphrag.py`, `iris_rag/pipelines/graphrag_merged.py`
- **Tests**: `tests/contract/`, `tests/integration/`, `tests/e2e/`
- **Common utilities**: `common/vector_sql_utils.py`, `common/iris_sql_utils.py`
- **Scripts**: `scripts/simple_working_ragas.py`, `scripts/test_graphrag_validation.py`

---

## Phase 3.1: Setup & Prerequisites Check

- [x] **T001** Verify IRIS database running on port 11972 and test data loaded (2,376 documents with embeddings)
  - **File**: Check via `docker ps | grep iris` and database query
  - **Success**: IRIS accessible, RAG.SourceDocuments has 2,376 rows with non-null embeddings
  - **Blocker**: Without test data, all tests will fail meaninglessly

- [x] **T002** Create contract test directory structure
  - **File**: `tests/contract/test_vector_search_contract.py` (create empty)
  - **File**: `tests/contract/test_dimension_validation_contract.py` (create empty)
  - **File**: `tests/contract/test_ragas_validation_contract.py` (create empty)
  - **File**: `tests/contract/test_diagnostic_logging_contract.py` (create empty)
  - **Success**: 4 empty test files exist

- [x] **T003** Create integration test file structure
  - **File**: `tests/integration/test_graphrag_vector_search.py` (create empty)
  - **Success**: Integration test file exists

---

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests (Write First, Expect FAIL)

- [x] **T004** [P] Contract test: Vector search returns documents (VSC-001)
  - **File**: `tests/contract/test_vector_search_contract.py`
  - **Contract**: `specs/033-fix-graphrag-vector/contracts/vector_search_contract.md`
  - **Requirements**: FR-001, FR-002, FR-003
  - **Tests**:
    - `test_vector_search_returns_documents` (FR-001)
    - `test_vector_search_respects_top_k` (FR-002)
    - `test_vector_search_results_have_scores`
    - `test_vector_search_results_sorted_descending`
    - `test_vector_search_returns_relevant_documents`
    - `test_vector_search_works_with_384d_embeddings` (FR-003)
  - **Expected**: **FAIL** (vector search currently returns 0 results)
  - **Verify**: Run `pytest tests/contract/test_vector_search_contract.py -v` → all FAIL

- [x] **T005** [P] Contract test: Dimension validation (DVC-002)
  - **File**: `tests/contract/test_dimension_validation_contract.py`
  - **Contract**: `specs/033-fix-graphrag-vector/contracts/dimension_validation_contract.md`
  - **Requirements**: FR-005
  - **Tests**:
    - `test_query_embedding_is_384_dimensions`
    - `test_dimension_validation_before_search`
    - `test_dimension_mismatch_raises_clear_error`
    - `test_dimension_error_suggests_actionable_fix`
    - `test_document_embedding_dimension_check`
    - `test_mismatched_embeddings_prevented`
  - **Expected**: **PARTIAL PASS** (embeddings are 384D, but validation not implemented)
  - **Verify**: Run `pytest tests/contract/test_dimension_validation_contract.py -v` → 2 pass, 4 skip

- [x] **T006** [P] Contract test: RAGAS acceptance (RAG-003)
  - **File**: `tests/contract/test_ragas_validation_contract.py`
  - **Contract**: `specs/033-fix-graphrag-vector/contracts/ragas_validation_contract.md`
  - **Requirements**: FR-019, FR-020, FR-021, FR-022
  - **Tests**:
    - `test_context_precision_above_30_percent` (FR-019)
    - `test_context_recall_above_20_percent` (FR-020)
    - `test_overall_performance_improved_from_baseline` (FR-022)
    - `test_all_queries_retrieve_documents` (FR-021)
    - `test_success_rate_is_100_percent`
    - `test_failed_queries_is_zero`
    - `test_faithfulness_maintained`
    - `test_answer_relevancy_maintained`
  - **Expected**: **FAIL** (context precision/recall currently 0%)
  - **Verify**: Run `pytest tests/contract/test_ragas_validation_contract.py -v` → 5 fail, 3 pass
  - **Note**: Takes 2-5 minutes to run (executes RAGAS evaluation)

- [x] **T007** [P] Contract test: Diagnostic logging (LOG-004)
  - **File**: `tests/contract/test_diagnostic_logging_contract.py`
  - **Contract**: `specs/033-fix-graphrag-vector/contracts/diagnostic_logging_contract.md`
  - **Requirements**: FR-004
  - **Tests**:
    - `test_logs_zero_results_message`
    - `test_logs_query_embedding_dimensions`
    - `test_logs_total_documents`
    - `test_logs_documents_with_embeddings`
    - `test_logs_sql_query_executed`
    - `test_logs_top_k_parameter`
    - `test_logs_similarity_scores_when_zero_results`
    - `test_logging_level_info_shows_high_level_status`
    - `test_logging_level_debug_shows_detailed_diagnostics`
  - **Expected**: **PARTIAL PASS** (basic logging exists, but not comprehensive)
  - **Verify**: Run `pytest tests/contract/test_diagnostic_logging_contract.py -v` → 1 pass, 8 skip

- [x] **T008** [P] Integration test: GraphRAG vector search workflow
  - **File**: `tests/integration/test_graphrag_vector_search.py`
  - **Tests**:
    - `test_graphrag_vector_search_retrieval` (FR-001)
    - `test_graphrag_top_k_configuration` (FR-006)
    - `test_graphrag_dimension_validation` (FR-005)
    - `test_graphrag_diagnostic_logging` (FR-004)
    - `test_graphrag_embedding_consistency` (FR-003)
  - **Expected**: **FAIL** (integration tests verify end-to-end workflow)
  - **Verify**: Run `pytest tests/integration/test_graphrag_vector_search.py -v` → all FAIL

---

## Phase 3.3: Investigation (Root Cause Analysis)

**Goal**: Identify exact difference between working BasicRAG and broken GraphRAG vector search

- [ ] **T009** Compare GraphRAG vector search implementation with BasicRAG
  - **Files to analyze**:
    - `iris_rag/pipelines/graphrag.py` (broken vector search)
    - `iris_rag/pipelines/basic.py` (working vector search)
  - **Focus areas**:
    - Query embedding generation
    - VECTOR_DOT_PRODUCT SQL query construction
    - Parameter binding for vector search
    - Result retrieval and ranking
  - **Deliverable**: Document differences in investigation notes (inline comments or separate file)
  - **Success**: Identified specific line(s) causing 0 results in GraphRAG

- [ ] **T010** Identify VECTOR_DOT_PRODUCT SQL usage differences
  - **Files to analyze**:
    - `iris_rag/pipelines/graphrag.py` (current implementation)
    - `common/vector_sql_utils.py` (proven helper functions)
    - `common/iris_sql_utils.py` (parameterized query helpers)
  - **Focus areas**:
    - TO_VECTOR conversion syntax
    - Query vector string formatting
    - Parameterized query placeholders
    - ORDER BY score DESC clause
  - **Deliverable**: Exact SQL difference causing failure
  - **Success**: Root cause SQL identified (e.g., missing TO_VECTOR, wrong parameter format)

- [ ] **T011** Review common/vector_sql_utils.py for proven patterns
  - **File**: `common/vector_sql_utils.py`
  - **Goal**: Understand established patterns for IRIS vector search
  - **Extract**:
    - Vector string formatting functions
    - VECTOR_DOT_PRODUCT query templates
    - Parameter binding approaches
    - Error handling patterns
  - **Deliverable**: Reusable patterns for GraphRAG fix
  - **Success**: Identified proven pattern to apply to GraphRAG

- [ ] **T012** Document root cause in investigation findings
  - **File**: Create investigation notes (can be comments in graphrag.py or separate doc)
  - **Content**:
    - Exact root cause (SQL syntax, parameter binding, etc.)
    - Why BasicRAG works but GraphRAG doesn't
    - Proven pattern to apply
    - Expected behavior after fix
  - **Success**: Clear documentation of root cause and fix approach

---

## Phase 3.4: Core Implementation (ONLY after tests are failing)

**Goal**: Fix vector search retrieval logic in GraphRAG pipeline

**Note**: Tasks T013-T017 modify the same file (`iris_rag/pipelines/graphrag.py`) → MUST be sequential (no [P])

- [ ] **T013** Fix vector search SQL query in GraphRAGPipeline.query()
  - **File**: `iris_rag/pipelines/graphrag.py`
  - **Method**: `GraphRAGPipeline.query()` or `GraphRAGPipeline._retrieve_contexts()`
  - **Changes**:
    - Apply proven VECTOR_DOT_PRODUCT SQL pattern from BasicRAG
    - Fix TO_VECTOR conversion syntax
    - Correct parameter binding for query vector
    - Ensure ORDER BY score DESC clause
  - **Reference**: `common/vector_sql_utils.py` for helpers
  - **Success**: Vector search query matches proven pattern from BasicRAG
  - **Verify**: Run T004 contract test → should PASS after this fix

- [ ] **T014** Add embedding dimension validation (FR-005)
  - **File**: `iris_rag/pipelines/graphrag.py`
  - **Add method**: `GraphRAGPipeline._validate_dimensions(embedding, expected_dims=384)`
  - **Changes**:
    - Validate query embedding is 384D before search
    - Raise `DimensionMismatchError` with clear message if mismatch
    - Include both actual and expected dimensions in error
    - Suggest actionable fix (model verification or re-indexing)
  - **Success**: Dimension validation prevents mismatched searches
  - **Verify**: Run T005 contract test → all dimension tests PASS

- [ ] **T015** Add diagnostic logging when 0 results (FR-004)
  - **File**: `iris_rag/pipelines/graphrag.py`
  - **Changes**:
    - Add INFO log: "Vector search returned 0 results" when len(results) == 0
    - Add DEBUG logs:
      - Query embedding dimensions
      - Total documents in RAG.SourceDocuments
      - Documents with embeddings count
      - SQL query executed
      - Top-K parameter value
      - Sample similarity scores (or "None returned")
  - **Success**: Comprehensive diagnostics logged when 0 results
  - **Verify**: Run T007 contract test → all logging tests PASS

- [ ] **T016** Make top-K configurable (FR-006)
  - **File**: `iris_rag/pipelines/graphrag.py`
  - **Changes**:
    - Read top_k from config: `self.config.get("retrieval.top_k", 10)`
    - Use configurable K in vector search SQL query
    - Default K=10 if not specified
  - **Success**: Top-K parameter configurable via config file
  - **Verify**: Run quickstart test with custom K=5 → returns ≤5 documents

- [ ] **T017** Apply same fixes to HybridGraphRAGPipeline
  - **File**: `iris_rag/pipelines/graphrag_merged.py` (or `iris_rag/pipelines/hybrid_graphrag.py`)
  - **Changes**:
    - Apply same vector search SQL fix from T013
    - Apply same dimension validation from T014
    - Apply same diagnostic logging from T015
    - Apply same top-K configuration from T016
  - **Success**: HybridGraphRAG benefits from same fixes
  - **Verify**: HybridGraphRAG pipeline also retrieves documents successfully

---

## Phase 3.5: Validation (Verify Fix Works)

**Goal**: Confirm all contract tests pass and RAGAS metrics meet targets

- [ ] **T018** [P] Run all contract tests → expect PASS
  - **Command**: `pytest tests/contract/test_vector_search_contract.py tests/contract/test_dimension_validation_contract.py tests/contract/test_diagnostic_logging_contract.py -v`
  - **Expected**:
    - `test_vector_search_contract.py`: 6/6 PASS (FR-001, FR-002, FR-003)
    - `test_dimension_validation_contract.py`: 6/6 PASS (FR-005)
    - `test_diagnostic_logging_contract.py`: 9/9 PASS (FR-004)
  - **Success**: All 21 contract tests pass
  - **Blocker**: If any FAIL, return to Phase 3.4 to fix implementation

- [ ] **T019** [P] Run RAGAS evaluation → expect >30% precision, >20% recall
  - **Command**: `IRIS_HOST=localhost IRIS_PORT=11972 RAGAS_PIPELINES="graphrag" .venv/bin/python scripts/simple_working_ragas.py`
  - **Expected metrics** (FR-019, FR-020, FR-021, FR-022):
    - Context precision >30% (target: ~42%)
    - Context recall >20% (target: ~28%)
    - Overall performance >14.4% baseline (target: ~45%)
    - Success rate 100% (all 5 queries retrieve documents)
  - **Output**: `outputs/reports/ragas_evaluations/simple_ragas_report_*.json`
  - **Success**: All RAGAS targets met
  - **Verify**: Run T006 contract test → all RAGAS tests PASS

- [ ] **T020** [P] Run smoke tests on 5 sample queries
  - **Queries** (from quickstart.md):
    - "What are the symptoms of diabetes?"
    - "How is diabetes diagnosed?"
    - "What are the treatments for type 2 diabetes?"
    - "What are the complications of untreated diabetes?"
    - "What is the difference between type 1 and type 2 diabetes?"
  - **For each query**:
    - Verify contexts retrieved >0
    - Verify answer is relevant (not "No relevant documents")
    - Verify answer mentions expected topics
  - **Success**: All 5 queries retrieve documents and generate relevant answers

- [ ] **T021** [P] Verify no regression in other pipelines
  - **Pipelines to test**:
    - BasicRAG: `create_pipeline("basic").query("What are the symptoms of diabetes?")`
    - CRAG: `create_pipeline("crag").query("What are the symptoms of diabetes?")`
    - BasicRerank: `create_pipeline("basic_rerank").query("What are the symptoms of diabetes?")`
  - **Expected**: All pipelines still retrieve documents (no regression)
  - **Success**: BasicRAG, CRAG, BasicRerank unaffected by GraphRAG fix

---

## Phase 3.6: Documentation & Cleanup

- [ ] **T022** Update Feature 032 investigation findings with resolution
  - **File**: `specs/032-investigate-graphrag-data/investigation/FINDINGS.md`
  - **Add section**: "Feature 033 Resolution"
  - **Content**:
    - Root cause identified (exact SQL/logic issue)
    - Fix applied (vector search SQL corrected)
    - Validation results (RAGAS metrics improved)
    - Status: Vector search fixed, entity-document linking still pending (Feature 034)
  - **Success**: FINDINGS.md updated with resolution

- [ ] **T023** Add vector search debugging guide to docs/
  - **File**: `docs/troubleshooting/vector_search_debugging.md` (create new)
  - **Content**:
    - How to debug 0 results from vector search
    - Diagnostic logging interpretation
    - Common issues (dimension mismatch, SQL syntax, embedding model)
    - Troubleshooting steps
  - **Success**: Debugging guide available for future issues

- [ ] **T024** Document top-K configuration in README
  - **File**: `README.md` or `iris_rag/config/README.md`
  - **Add section**: "Retrieval Configuration"
  - **Content**:
    - How to configure top_k parameter
    - Default values (K=10)
    - Example config override
  - **Success**: Top-K configuration documented

---

## Dependencies

### Critical Path (TDD Workflow)

```
Phase 3.1 (Setup)
  ↓
Phase 3.2 (Contract Tests - MUST FAIL)
  T004, T005, T006, T007, T008 [all parallel]
  ↓
Phase 3.3 (Investigation - Sequential)
  T009 → T010 → T011 → T012
  ↓
Phase 3.4 (Implementation - Sequential, same file)
  T013 → T014 → T015 → T016 → T017
  ↓
Phase 3.5 (Validation - Parallel)
  T018, T019, T020, T021 [all parallel]
  ↓
Phase 3.6 (Documentation - Independent)
  T022, T023, T024 [can be parallel]
```

### Detailed Dependencies

- **T001** (IRIS running) blocks ALL tests (T004-T008, T018-T021)
- **T002, T003** (test files) block T004-T008 (writing tests)
- **T004-T008** (contract tests) block T009 (must fail first for TDD)
- **T009-T012** (investigation) blocks T013 (need root cause before fixing)
- **T013** (SQL fix) blocks T014-T017 (dimension validation needs working query)
- **T013-T017** (implementation) blocks T018-T021 (validation needs fixed code)
- **T018** (contract tests pass) required for Feature 033 acceptance
- **T019** (RAGAS metrics) required for Feature 033 acceptance (FR-019, FR-020, FR-021, FR-022)
- **T022-T024** (docs) can run after T019 (have resolution to document)

---

## Parallel Execution Examples

### Example 1: Launch all contract tests together (T004-T008)

```bash
# After T002, T003 complete, run all contract tests in parallel:
pytest tests/contract/test_vector_search_contract.py \
       tests/contract/test_dimension_validation_contract.py \
       tests/contract/test_ragas_validation_contract.py \
       tests/contract/test_diagnostic_logging_contract.py \
       tests/integration/test_graphrag_vector_search.py \
       -v
```

**Expected**: All tests FAIL (proving tests are written correctly and implementation is broken)

### Example 2: Validation tasks in parallel (T018-T021)

After implementation complete, run all validation in parallel:

```bash
# Terminal 1: Contract tests
pytest tests/contract/ -v

# Terminal 2: RAGAS evaluation
RAGAS_PIPELINES="graphrag" python scripts/simple_working_ragas.py

# Terminal 3: Smoke tests
python -c "
from iris_rag import create_pipeline
pipeline = create_pipeline('graphrag')
queries = [
    'What are the symptoms of diabetes?',
    'How is diabetes diagnosed?',
    'What are the treatments for type 2 diabetes?',
    'What are the complications of untreated diabetes?',
    'What is the difference between type 1 and type 2 diabetes?'
]
for q in queries:
    result = pipeline.query(q)
    print(f'{q}: {len(result.contexts)} contexts')
"

# Terminal 4: Regression tests
pytest tests/e2e/test_basic_pipeline_e2e.py \
       tests/e2e/test_crag_pipeline_e2e.py \
       -v
```

### Example 3: Documentation tasks in parallel (T022-T024)

```bash
# After T019 complete, write docs in parallel:
# Person 1: Update FINDINGS.md
# Person 2: Write debugging guide
# Person 3: Update README.md
```

---

## Notes

- **[P] tasks** = different files, no dependencies, can run in parallel
- **TDD critical**: Verify tests FAIL before implementing (T004-T008 must fail before T013)
- **Investigation before coding**: T009-T012 identify root cause before T013 fixes it
- **Same file conflict**: T013-T017 all modify `graphrag.py` → must be sequential
- **Commit after each task**: Enables rollback if needed
- **RAGAS takes time**: T006 and T019 each take 2-5 minutes (evaluates 5 queries)

---

## Validation Checklist

*GATE: Feature 033 complete when all criteria met*

### Contract Test Coverage
- [x] VSC-001: Vector search returns documents (6 tests in T004)
- [x] DVC-002: Dimension validation (6 tests in T005)
- [x] RAG-003: RAGAS acceptance (8 tests in T006)
- [x] LOG-004: Diagnostic logging (9 tests in T007)
- [x] Integration: End-to-end workflow (5 tests in T008)

### Requirements Coverage
- [x] FR-001: Vector search returns documents → T004, T013
- [x] FR-002: Top-K retrieval → T004, T016
- [x] FR-003: Works with 384D embeddings → T004, T013
- [x] FR-004: Diagnostic logging → T007, T015
- [x] FR-005: Dimension validation → T005, T014
- [x] FR-006: Configurable top-K → T016
- [x] FR-019: Context precision >30% → T006, T019
- [x] FR-020: Context recall >20% → T006, T019
- [x] FR-021: All queries retrieve documents → T006, T019
- [x] FR-022: Overall performance >14.4% → T006, T019

### Task Ordering Validation
- [x] Setup before tests (T001-T003 → T004-T008)
- [x] Tests before implementation (T004-T008 → T013-T017)
- [x] Investigation before coding (T009-T012 → T013)
- [x] Implementation before validation (T013-T017 → T018-T021)
- [x] Validation before docs (T018-T021 → T022-T024)

### Parallel Task Safety
- [x] T004-T008: Different test files, no shared state ✓
- [x] T018-T021: Different validation approaches, no conflicts ✓
- [x] T022-T024: Different documentation files ✓
- [x] T013-T017: **Same file (graphrag.py) → SEQUENTIAL** ✓

### File Path Specificity
- [x] All tasks specify exact file paths
- [x] No vague tasks like "fix vector search" without file
- [x] Test files match contract specifications

---

## Estimation

**Total effort**: ~8-12 hours (TDD cycle)

- **Phase 3.1 Setup**: 0.5h (T001-T003)
- **Phase 3.2 Contract Tests**: 2h (T004-T008) - write 29 tests total
- **Phase 3.3 Investigation**: 2-3h (T009-T012) - identify exact root cause
- **Phase 3.4 Implementation**: 3-4h (T013-T017) - fix SQL, add validation, logging
- **Phase 3.5 Validation**: 1-2h (T018-T021) - verify all tests pass, RAGAS metrics met
- **Phase 3.6 Documentation**: 1h (T022-T024) - update findings, write guides

**Critical path**: Setup → Contract Tests (2h) → Investigation (3h) → Implementation (4h) → Validation (2h) = ~11h

**Parallelization savings**: Contract tests (T004-T008) can run concurrently, validation (T018-T021) can run concurrently → saves ~1-2h

---

## Success Criteria

Feature 033 is **COMPLETE** when:

1. ✅ All 29 contract tests pass (T004-T008, verified in T018)
2. ✅ RAGAS metrics meet targets (T019):
   - Context precision >30%
   - Context recall >20%
   - Overall performance >14.4%
   - Success rate 100%
3. ✅ All 5 smoke test queries retrieve documents (T020)
4. ✅ No regression in BasicRAG, CRAG, BasicRerank (T021)
5. ✅ Investigation findings updated with resolution (T022)
6. ✅ Debugging guide and configuration docs added (T023-T024)

**Acceptance**: Run `pytest tests/contract/ -v` → 29/29 PASS + RAGAS evaluation shows >30% precision, >20% recall

---

**Tasks Status**: ✅ READY FOR EXECUTION
**Next Step**: Begin with T001 (verify IRIS running) and proceed through TDD workflow
