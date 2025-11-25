# Implementation Tasks: DSPy Optimization Integration for HippoRAG2

**Feature**: DSPy Optimization Integration for HippoRAG2
**Branch**: `063-dspy-optimization`
**Target Repository**: `/Users/tdyar/ws/hipporag2-pipeline`
**Date**: 2025-11-24

## Task Summary

| Phase | User Story | Task Count | Parallel Tasks | Completion Criteria |
|-------|-----------|------------|----------------|---------------------|
| Setup | N/A | 3 | 1 | Repository prepared with optimized extractor file |
| Foundational | N/A | 4 | 3 | Test infrastructure ready for TDD implementation |
| US1: Enable Pre-Optimized Entity Extraction | P1 | 8 | 3 | Multi-word entities extracted with 85%+ recall, HotpotQA F1 improves to 0.7+ |
| US2: Verify Optimization Impact | P2 | 5 | 2 | Baseline and optimized F1 scores documented, 31.8%+ improvement verified |
| US3: Graceful Degradation | P3 | 4 | 2 | Fallback works correctly with clear logging for all error scenarios |
| Polish | N/A | 3 | 2 | Documentation complete, .env updated |
| **TOTAL** | **3 stories** | **27** | **13** | **All success criteria met, zero breaking changes** |

---

## Implementation Strategy

**MVP Scope**: User Story 1 (Enable Pre-Optimized Entity Extraction)
- Delivers core value: 31.8% F1 improvement and correct multi-word entity extraction
- Independently testable via HotpotQA evaluation
- Provides foundation for US2 verification and US3 robustness

**Incremental Delivery**:
1. **Sprint 1**: US1 (P1) - Core optimization functionality
2. **Sprint 2**: US2 (P2) - Verification and metrics
3. **Sprint 3**: US3 (P3) - Robustness and error handling polish

**Key Dependencies**:
- US2 depends on US1 (needs working optimization to measure)
- US3 is independent (tests fallback behavior when optimization fails)

---

## Phase 1: Setup (Repository Preparation)

**Goal**: Prepare hipporag2-pipeline repository with DSPy optimization assets

**Tasks**:

- [ ] T001 Switch to hipporag2-pipeline repository and checkout branch 002-performance-optimization in /Users/tdyar/ws/hipporag2-pipeline
- [ ] T002 Copy entity_extractor_optimized.json from branch 002 to repository root /Users/tdyar/ws/hipporag2-pipeline/entity_extractor_optimized.json
- [ ] T003 [P] Verify optimized extractor file integrity (check SHA256 checksum matches expected value from DSPY_OPTIMIZATION_RESULTS.md)

**Completion Criteria**: `entity_extractor_optimized.json` available in hipporag2-pipeline root

---

## Phase 2: Foundational (Test Infrastructure)

**Goal**: Set up test infrastructure for TDD implementation (red-green-refactor)

**Tasks**:

- [ ] T004 [P] Create unit test file /Users/tdyar/ws/hipporag2-pipeline/tests/unit/test_dspy_optimization_loading.py with test structure for Contract 1 (Optimized Extractor Loading)
- [ ] T005 [P] Create integration test file /Users/tdyar/ws/hipporag2-pipeline/tests/integration/test_hotpotqa_with_optimization.py with test structure for HotpotQA evaluation
- [ ] T006 [P] Create test fixture helper module /Users/tdyar/ws/hipporag2-pipeline/tests/fixtures/dspy_fixtures.py for mock DSPy extractors and configuration
- [ ] T007 Install dspy-ai dependency if not already present (add to pyproject.toml or requirements.txt)

**Completion Criteria**: Test files created, all tests initially fail (red phase), dspy-ai dependency available

---

## Phase 3: User Story 1 - Enable Pre-Optimized Entity Extraction (P1)

**Story Goal**: Researcher can enable DSPy optimization via environment variable to achieve 85%+ multi-word entity recall and 0.7+ F1 on bridge questions

**Independent Test**: Set `DSPY_OPTIMIZED_EXTRACTOR_PATH` environment variable, index documents, verify multi-word entities extracted correctly, run HotpotQA evaluation to confirm F1 ≥ 0.7

### Contract Tests (TDD Red Phase)

- [ ] T008 [US1] Write failing test: load_optimized_extractor with valid path and API key returns loaded DSPy program in /Users/tdyar/ws/hipporag2-pipeline/tests/unit/test_dspy_optimization_loading.py
- [ ] T009 [US1] Write failing test: extract_entities_optimized uses DSPy when loaded, returns list of (entity_text, entity_type) tuples in /Users/tdyar/ws/hipporag2-pipeline/tests/unit/test_dspy_optimization_loading.py
- [ ] T010 [US1] Write failing test: extract_batch_with_optimization wraps single-doc extractor for batch processing, preserves document order in /Users/tdyar/ws/hipporag2-pipeline/tests/unit/test_dspy_optimization_loading.py

### Core Implementation (TDD Green Phase)

- [ ] T011 [P] [US1] Create DSPy signature definition matching optimization training in /Users/tdyar/ws/hipporag2-pipeline/src/hipporag2/pipeline/dspy_signatures.py (copy from scripts/optimize_entity_extraction.py)
- [ ] T012 [P] [US1] Implement load_optimized_extractor function in /Users/tdyar/ws/hipporag2-pipeline/src/hipporag2/pipeline/dspy_loader.py to load DSPy ChainOfThought from file
- [ ] T013 [P] [US1] Implement parse_dspy_entities function in /Users/tdyar/ws/hipporag2-pipeline/src/hipporag2/pipeline/dspy_loader.py to convert DSPy output to (entity_text, entity_type) tuples
- [ ] T014 [US1] Modify HippoRAG2Pipeline.__init__ in /Users/tdyar/ws/hipporag2-pipeline/src/hipporag2/pipeline/hipporag2_pipeline.py to detect DSPY_OPTIMIZED_EXTRACTOR_PATH environment variable
- [ ] T015 [US1] Add optimized extractor loading logic to HippoRAG2Pipeline.__init__ in /Users/tdyar/ws/hipporag2-pipeline/src/hipporag2/pipeline/hipporag2_pipeline.py (call load_optimized_extractor if env var set)

**Verify Tests Pass**: Run T008-T010 tests, confirm green phase

### Integration & Verification

- [ ] T016 [US1] Write integration test: HotpotQA evaluation with optimization enabled achieves F1 ≥ 0.7 on bridge questions in /Users/tdyar/ws/hipporag2-pipeline/tests/integration/test_hotpotqa_with_optimization.py
- [ ] T017 [US1] Run HotpotQA evaluation with optimization enabled (DSPY_OPTIMIZED_EXTRACTOR_PATH=entity_extractor_optimized.json python examples/hotpotqa_evaluation.py 2), verify F1 ≥ 0.7 and "Chief of Protocol" question succeeds
- [ ] T018 [US1] Verify multi-word entity recall ≥ 85% by examining extracted entities from HotpotQA documents

**US1 Completion Criteria**:
- ✅ Multi-word entities like "Chief of Protocol of the United States" extracted as complete phrases
- ✅ HotpotQA bridge question F1 score ≥ 0.7 (improvement from baseline ~0.0)
- ✅ Optimization enabled via environment variable DSPY_OPTIMIZED_EXTRACTOR_PATH
- ✅ All contract tests (T008-T010) passing

---

## Phase 4: User Story 2 - Verify Optimization Impact via Evaluation (P2)

**Story Goal**: Data scientist can measure F1 improvement by running HotpotQA evaluation before/after enabling optimization

**Independent Test**: Run HotpotQA with baseline config (no optimization), record F1. Run with optimization, record F1. Verify 31.8%+ improvement.

### Baseline Measurement

- [ ] T019 [US2] Run HotpotQA evaluation without optimization (comment out DSPY_OPTIMIZED_EXTRACTOR_PATH) in /Users/tdyar/ws/hipporag2-pipeline and save results to /tmp/hotpotqa_baseline.log
- [ ] T020 [US2] Extract baseline F1 score from /tmp/hotpotqa_baseline.log, verify "Chief of Protocol" question fails, document in /Users/tdyar/ws/hipporag2-pipeline/docs/OPTIMIZATION_RESULTS.md

### Optimized Measurement & Comparison

- [ ] T021 [P] [US2] Run HotpotQA evaluation with optimization enabled and save results to /tmp/hotpotqa_optimized.log (already completed in T017 if US1 done first)
- [ ] T022 [P] [US2] Extract optimized F1 score from /tmp/hotpotqa_optimized.log, verify "Chief of Protocol" question succeeds
- [ ] T023 [US2] Compare baseline vs optimized F1 scores, calculate improvement percentage, verify ≥ 31.8% improvement, document comparison in /Users/tdyar/ws/hipporag2-pipeline/docs/OPTIMIZATION_RESULTS.md

**US2 Completion Criteria**:
- ✅ Baseline F1 score documented
- ✅ Optimized F1 score documented
- ✅ F1 improvement ≥ 31.8% verified (0.294 → 0.387+)
- ✅ Multi-word entity recall improvement documented (50% → 85%+)
- ✅ Results documented in OPTIMIZATION_RESULTS.md

---

## Phase 5: User Story 3 - Graceful Degradation Without Optimization (P3)

**Story Goal**: User can deploy HippoRAG2 without optimization configuration and system falls back gracefully with clear logging

**Independent Test**: Start HippoRAG2 without DSPY_OPTIMIZED_EXTRACTOR_PATH, verify standard extraction works. Test with invalid path, missing API key, verify fallback and logging.

### Error Handling Tests (TDD Red Phase)

- [ ] T024 [US3] Write failing test: load_optimized_extractor with missing file returns error with clear message in /Users/tdyar/ws/hipporag2-pipeline/tests/unit/test_dspy_optimization_loading.py
- [ ] T025 [US3] Write failing test: load_optimized_extractor with missing API key returns error and warning in /Users/tdyar/ws/hipporag2-pipeline/tests/unit/test_dspy_optimization_loading.py
- [ ] T026 [US3] Write failing test: HippoRAG2Pipeline.__init__ with no DSPY_OPTIMIZED_EXTRACTOR_PATH uses standard extraction and logs info message in /Users/tdyar/ws/hipporag2-pipeline/tests/unit/test_dspy_optimization_loading.py

### Graceful Fallback Implementation (TDD Green Phase)

- [ ] T027 [P] [US3] Add error handling to load_optimized_extractor in /Users/tdyar/ws/hipporag2-pipeline/src/hipporag2/pipeline/dspy_loader.py for file not found, invalid JSON, missing API key, DSPy import errors
- [ ] T028 [P] [US3] Add logging statements (info/warning) to HippoRAG2Pipeline.__init__ in /Users/tdyar/ws/hipporag2-pipeline/src/hipporag2/pipeline/hipporag2_pipeline.py for optimization status (enabled/disabled/failed)
- [ ] T029 [US3] Add try-except wrapper in HippoRAG2Pipeline.__init__ to catch all extractor loading errors and fall back to standard EntityExtractionService in /Users/tdyar/ws/hipporag2-pipeline/src/hipporag2/pipeline/hipporag2_pipeline.py

**Verify Tests Pass**: Run T024-T026 tests, confirm green phase

### Verification

- [ ] T030 [US3] Test HippoRAG2 startup without DSPY_OPTIMIZED_EXTRACTOR_PATH set, verify standard extraction works and "no optimization" info message logged
- [ ] T031 [US3] Test HippoRAG2 startup with invalid DSPY_OPTIMIZED_EXTRACTOR_PATH (nonexistent file), verify fallback to standard extraction and warning logged

**US3 Completion Criteria**:
- ✅ Fallback to standard extraction works within 5 seconds
- ✅ Clear info message logged when optimization not configured
- ✅ Clear warning logged when optimization configuration invalid
- ✅ System continues to work without optimization (zero breaking changes)
- ✅ All error handling tests (T024-T026) passing

---

## Phase 6: Polish & Cross-Cutting Concerns

**Goal**: Complete documentation, update configuration examples, final verification

**Tasks**:

- [ ] T032 [P] Update /Users/tdyar/ws/hipporag2-pipeline/.env.example with DSPY_OPTIMIZED_EXTRACTOR_PATH and usage comment
- [ ] T033 [P] Update /Users/tdyar/ws/hipporag2-pipeline/docs/DSPY_OPTIMIZATION_INTEGRATION.md with implementation notes, testing results, and quickstart instructions
- [ ] T034 Run full test suite (pytest tests/) to verify zero breaking changes and all new tests passing

**Completion Criteria**: Documentation complete, .env.example updated, all tests passing

---

## Dependency Graph

```text
User Story Completion Order (by priority):

1. Setup Phase → Foundational Phase (blocking)
2. US1 (P1) → Independent (can start after foundational)
3. US2 (P2) → Depends on US1 (needs working optimization)
4. US3 (P3) → Independent (tests fallback, can run in parallel with US2)
5. Polish Phase → Depends on all user stories complete
```

**Parallel Opportunities**:
- Setup: T003 (verify checksum) can run parallel with T001-T002
- Foundational: T004, T005, T006 (test file creation) all parallel
- US1: T011, T012, T013 (DSPy helpers) all parallel
- US2: T021, T022 (measurements) parallel if US1 complete
- US3: T027, T028 (error handling) parallel
- Polish: T032, T033 (documentation) parallel

---

## Parallel Execution Examples

### Sprint 1 (US1 - Core Optimization)

**Week 1**: Setup + Foundational
```bash
# Parallel batch 1
task T001 && task T002 &
task T003 &
wait

# Parallel batch 2
task T004 &
task T005 &
task T006 &
wait

task T007  # Sequential (dependency install)
```

**Week 2**: US1 Contract Tests + Implementation
```bash
# Sequential (TDD red phase)
task T008
task T009
task T010

# Parallel batch (implementation)
task T011 &
task T012 &
task T013 &
wait

# Sequential (integration)
task T014
task T015
task T016
task T017
task T018
```

### Sprint 2 (US2 - Verification + US3 - Robustness)

**US2 + US3 in parallel** (independent stories):
```bash
# US2 baseline
task T019
task T020

# US3 tests + implementation (parallel with US2)
task T024 &
task T025 &
task T026 &
wait

task T027 &
task T028 &
wait

task T029

# US2 + US3 verification
task T021 &  # US2
task T030 &  # US3
wait

task T022
task T023
task T031
```

### Sprint 3 (Polish)

```bash
# Parallel documentation
task T032 &
task T033 &
wait

task T034  # Final verification
```

---

## Testing Strategy

**TDD Workflow** (per user story):
1. Write failing contract tests (red phase)
2. Implement minimum code to pass tests (green phase)
3. Refactor for clarity and performance
4. Run integration tests to verify end-to-end behavior

**Test Coverage**:
- Unit tests: Contract behavior (loading, extraction, parsing)
- Integration tests: HotpotQA evaluation with real documents
- Error handling tests: All fallback scenarios

**Success Verification**:
- US1: F1 ≥ 0.7, multi-word recall ≥ 85%
- US2: F1 improvement ≥ 31.8% documented
- US3: Fallback works with clear logging
- All: Zero breaking changes (existing tests pass)

---

## File Modification Summary

### New Files Created
- `/Users/tdyar/ws/hipporag2-pipeline/src/hipporag2/pipeline/dspy_signatures.py` - DSPy signature definitions
- `/Users/tdyar/ws/hipporag2-pipeline/src/hipporag2/pipeline/dspy_loader.py` - Loading and parsing logic
- `/Users/tdyar/ws/hipporag2-pipeline/tests/unit/test_dspy_optimization_loading.py` - Unit tests
- `/Users/tdyar/ws/hipporag2-pipeline/tests/integration/test_hotpotqa_with_optimization.py` - Integration tests
- `/Users/tdyar/ws/hipporag2-pipeline/tests/fixtures/dspy_fixtures.py` - Test fixtures
- `/Users/tdyar/ws/hipporag2-pipeline/docs/OPTIMIZATION_RESULTS.md` - Measurement documentation

### Modified Files
- `/Users/tdyar/ws/hipporag2-pipeline/src/hipporag2/pipeline/hipporag2_pipeline.py` - Add optimization loading
- `/Users/tdyar/ws/hipporag2-pipeline/.env.example` - Add DSPY_OPTIMIZED_EXTRACTOR_PATH
- `/Users/tdyar/ws/hipporag2-pipeline/docs/DSPY_OPTIMIZATION_INTEGRATION.md` - Update integration guide

### Copied Files
- `/Users/tdyar/ws/hipporag2-pipeline/entity_extractor_optimized.json` - From branch 002

---

## MVP Definition

**Minimum Viable Product = User Story 1 (P1)**

**Scope**: Enable pre-optimized entity extraction via environment variable
- Core value: 31.8% F1 improvement, correct multi-word entity extraction
- Deliverable: Working optimization with HotpotQA F1 ≥ 0.7
- Timeline: Sprint 1 (2 weeks)
- Tasks: T001-T018 (18 tasks, 6 parallel opportunities)

**MVP Acceptance**:
✅ Environment variable DSPY_OPTIMIZED_EXTRACTOR_PATH enables optimization
✅ Multi-word entities extracted with 85%+ recall
✅ HotpotQA bridge questions achieve F1 ≥ 0.7
✅ Zero breaking changes to existing workflows

**Post-MVP Enhancements**:
- Sprint 2: Add verification metrics (US2) and error handling polish (US3)
- Sprint 3: Complete documentation and final polish

---

**Total Tasks**: 34
**Parallel Tasks**: 13 (38% parallelizable)
**Estimated Effort**: 3 sprints (6 weeks total, 2 weeks for MVP)
