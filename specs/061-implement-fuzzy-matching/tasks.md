# Tasks: Fuzzy Entity Matching for EntityStorageAdapter

**Feature**: 061-implement-fuzzy-matching
**Input**: Design documents from `/Users/tdyar/ws/iris-vector-rag-private/specs/061-implement-fuzzy-matching/`
**Prerequisites**: plan.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅ (5 files)

## Execution Flow (main)
```
1. Load plan.md from feature directory → ✅ COMPLETE
   → Extracted: Python 3.11+, IRIS iFind, EntityStorageAdapter extension
2. Load optional design documents → ✅ COMPLETE
   → data-model.md: EntitySearchQuery, EntitySearchResult
   → contracts/: 5 contract test files with 29 test cases
   → research.md: IRIS iFind Levenshtein, hybrid matching strategy
3. Generate tasks by category → ✅ COMPLETE
   → Setup: No new dependencies (extends existing)
   → Tests: 5 contract test files + 2 integration test files
   → Core: search_entities() method in EntityStorageAdapter
   → Integration: No new integrations (uses existing ConnectionManager)
   → Polish: Documentation updates, regression tests
4. Apply task rules → ✅ COMPLETE
   → Contract tests marked [P] (different files)
   → Implementation sequential (same file: storage.py)
   → Tests before implementation (TDD)
5. Number tasks sequentially → ✅ COMPLETE (T001-T014)
6. Generate dependency graph → ✅ COMPLETE (see Dependencies section)
7. Create parallel execution examples → ✅ COMPLETE (see Parallel Example)
8. Validate task completeness → ✅ COMPLETE
   → All 5 contracts have test tasks ✓
   → No new entities (uses existing Entity model) ✓
   → search_entities() method implementation ✓
9. Return: SUCCESS (14 tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: Repository root `/Users/tdyar/ws/iris-vector-rag-private/`
- **Source**: `iris_vector_rag/services/storage.py` (EntityStorageAdapter)
- **Tests**: `tests/contract/`, `tests/integration/`, `tests/unit/`

## Phase 3.1: Setup
✅ **No setup tasks required** - Feature extends existing EntityStorageAdapter class. All dependencies already present (IRIS, intersystems-iris-dbapi, pytest).

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests (Parallel Execution)
- [ ] **T001** [P] Contract test: Exact entity matching
  **File**: `tests/contract/test_fuzzy_entity_search_contracts.py`
  **Description**: Create contract tests for exact matching (fuzzy=False) from `specs/061-implement-fuzzy-matching/contracts/test_search_entities_exact.py`. Include 4 test cases: descriptor not matched, identical name matched, case-insensitive, required fields present.
  **Expected**: All tests MUST FAIL (search_entities() method does not exist yet)

- [ ] **T002** [P] Contract test: Fuzzy matching with Levenshtein
  **File**: `tests/contract/test_fuzzy_entity_search_contracts.py`
  **Description**: Create contract tests for fuzzy matching from `specs/061-implement-fuzzy-matching/contracts/test_search_entities_fuzzy.py`. Include 7 test cases: descriptor matching, ranking, typo handling, spelling variations, edit distance threshold, similarity threshold.
  **Expected**: All tests MUST FAIL (search_entities() method does not exist yet)

- [ ] **T003** [P] Contract test: Entity type filtering
  **File**: `tests/contract/test_fuzzy_entity_search_contracts.py`
  **Description**: Create contract tests for entity type filtering from `specs/061-implement-fuzzy-matching/contracts/test_search_entities_type_filter.py`. Include 3 test cases: single type filter, multiple types, no filter returns all.
  **Expected**: All tests MUST FAIL (search_entities() method does not exist yet)

- [ ] **T004** [P] Contract test: Result ranking
  **File**: `tests/contract/test_fuzzy_entity_search_contracts.py`
  **Description**: Create contract tests for result ranking from `specs/061-implement-fuzzy-matching/contracts/test_search_entities_ranking.py`. Include 3 test cases: exact matches first, ties broken by name length, max_results limit.
  **Expected**: All tests MUST FAIL (search_entities() method does not exist yet)

- [ ] **T005** [P] Contract test: Edge cases
  **File**: `tests/contract/test_fuzzy_entity_search_contracts.py`
  **Description**: Create contract tests for edge cases from `specs/061-implement-fuzzy-matching/contracts/test_search_entities_edge_cases.py`. Include 9 test cases: empty query, no matches, Unicode, similarity=1.0, edit_distance=0, max_results=0, short queries, duplicates, case variations.
  **Expected**: All tests MUST FAIL (search_entities() method does not exist yet)

### Integration Tests (After Contract Tests)
- [ ] **T006** Integration test: Real IRIS database fuzzy search
  **File**: `tests/integration/test_fuzzy_entity_search_integration.py`
  **Description**: Create integration tests using real IRIS database (via iris-devtester IRISContainer). Test fuzzy search with 100+ entities to validate performance (<50ms), Levenshtein distance calculation, and ranking. Include fixtures for entity data population.
  **Dependencies**: T001-T005 (contract tests complete)
  **Expected**: Tests MUST FAIL (search_entities() not implemented)

- [ ] **T007** Integration test: GraphRAG pipeline validation (BUG-001)
  **File**: `tests/integration/test_graphrag_pipeline_integration.py`
  **Description**: Create integration test to validate GraphRAG pipeline ingest() completes without UnboundLocalError. Load sample documents, call GraphRAGPipeline.ingest(), verify entity extraction succeeds. This validates BUG-001 fix (time import already present).
  **Dependencies**: None (independent validation)
  **Expected**: Test SHOULD PASS (bug already fixed)

## Phase 3.3: Core Implementation (ONLY after tests are failing)

- [ ] **T008** Implement search_entities() method in EntityStorageAdapter
  **File**: `iris_vector_rag/services/storage.py`
  **Description**: Add `search_entities()` method to EntityStorageAdapter class (line ~610, after get_storage_stats). Implement exact matching (fuzzy=False) and fuzzy matching (fuzzy=True) using IRIS SQL with `$SYSTEM.SQL.Functions.LEVENSHTEIN()`. Include parameter validation, case-insensitive matching (LOWER()), entity type filtering, result ranking (exact first, then by edit_distance), similarity score calculation `(1 - edit_distance/max_length)`, and FETCH FIRST N ROWS ONLY limit. Add structured logging for query execution.
  **Dependencies**: T001-T005 (all contract tests must be failing)
  **SQL Template**: See `specs/061-implement-fuzzy-matching/research.md` Section 3 for query structure
  **Expected**: Contract tests T001-T005 should START PASSING as implementation progresses

- [ ] **T009** Add docstring and type hints to search_entities()
  **File**: `iris_vector_rag/services/storage.py`
  **Description**: Add comprehensive docstring following NumPy style guide. Include Args, Returns, Raises sections. Add type hints for all parameters and return type `List[Dict[str, Any]]`. Document performance characteristics (<50ms fuzzy, <10ms exact).
  **Dependencies**: T008 (method implemented)

- [ ] **T010** Update EntityStorageAdapter __init__ docstring
  **File**: `iris_vector_rag/services/storage.py`
  **Description**: Update class-level docstring to document new search_entities() method. Add example usage for both exact and fuzzy matching. Reference quickstart.md for detailed examples.
  **Dependencies**: T008 (method implemented)

## Phase 3.4: Integration
✅ **No integration tasks required** - search_entities() uses existing ConnectionManager and SchemaManager. No new middleware, auth, or external integrations needed.

## Phase 3.5: Polish

- [ ] **T011** [P] Unit tests for search_entities() edge cases
  **File**: `tests/unit/test_search_entities_unit.py`
  **Description**: Create unit tests with mocked IRIS database connection. Test parameter validation (invalid thresholds, invalid types), edge cases (empty query, None parameters), and error handling (database connection failure, SQL syntax errors). Use unittest.mock to mock cursor.execute() and cursor.fetchall().
  **Dependencies**: T008 (implementation complete)

- [ ] **T012** Performance test: 10K entities fuzzy search
  **File**: `tests/integration/test_fuzzy_entity_search_integration.py`
  **Description**: Add performance test case using 10,000 entities. Measure fuzzy search latency with `time.perf_counter()`. Assert <50ms for fuzzy search, <10ms for exact match. Use `@pytest.mark.performance` marker. Document results in test output.
  **Dependencies**: T006 (integration tests created), T008 (implementation complete)

- [ ] **T013** Update quickstart.md with working examples
  **File**: `specs/061-implement-fuzzy-matching/quickstart.md`
  **Description**: Verify all code examples in quickstart.md execute correctly against real IRIS database. Update output examples with actual results. Add troubleshooting section with common errors and solutions. Test with `python -c "exec(open('quickstart.md').read())"` approach.
  **Dependencies**: T008 (implementation complete)

- [ ] **T014** Run regression tests for EntityStorageAdapter
  **File**: `tests/integration/test_entity_storage_adapter.py` (existing file)
  **Description**: Execute ALL existing EntityStorageAdapter tests to ensure zero regressions. Run `pytest tests/integration/test_entity_storage_adapter.py -v`. Verify existing methods (store_entity, get_entities_by_document, get_entities_by_type, etc.) continue to work identically. Document results: Expected 100% pass rate with no changes to existing behavior.
  **Dependencies**: T008 (implementation complete)
  **Expected**: All existing tests MUST PASS (zero regressions)

## Dependencies

### Critical Path
```
Phase 3.2 (Tests) → Phase 3.3 (Implementation) → Phase 3.5 (Polish)
```

### Detailed Dependencies
- **T001-T005** (Contract tests): No dependencies, can run in parallel
- **T006** (Integration test): Depends on T001-T005 (contract tests created)
- **T007** (GraphRAG test): Independent (validates existing fix)
- **T008** (Implementation): Depends on T001-T005 (tests must be failing first)
- **T009** (Docstring): Depends on T008 (method exists)
- **T010** (Class docstring): Depends on T008 (method exists)
- **T011** (Unit tests): Depends on T008 (implementation complete)
- **T012** (Performance test): Depends on T006, T008 (integration test + implementation)
- **T013** (Quickstart update): Depends on T008 (implementation complete)
- **T014** (Regression tests): Depends on T008 (implementation complete)

### Blocking Tasks
- **T008 blocks**: T009, T010, T011, T012, T013, T014 (all polish tasks)
- **T001-T005 block**: T008 (TDD requirement: tests before implementation)

## Parallel Execution Examples

### Phase 3.2: Contract Tests (All Parallel)
```bash
# Launch T001-T005 together (different test classes in same file is acceptable)
# Using pytest markers or separate pytest invocations

# Option 1: Single pytest run (fastest)
pytest tests/contract/test_fuzzy_entity_search_contracts.py -v

# Option 2: Parallel test execution (if needed)
pytest tests/contract/test_fuzzy_entity_search_contracts.py::TestSearchEntitiesExact -v &
pytest tests/contract/test_fuzzy_entity_search_contracts.py::TestSearchEntitiesFuzzy -v &
pytest tests/contract/test_fuzzy_entity_search_contracts.py::TestSearchEntitiesTypeFilter -v &
pytest tests/contract/test_fuzzy_entity_search_contracts.py::TestSearchEntitiesRanking -v &
pytest tests/contract/test_fuzzy_entity_search_contracts.py::TestSearchEntitiesEdgeCases -v &
wait
```

### Phase 3.5: Polish (Parallel Tasks)
```bash
# Launch T011 and T013 together (different files)
# T011: Unit tests (tests/unit/test_search_entities_unit.py)
pytest tests/unit/test_search_entities_unit.py -v &

# T013: Quickstart validation (manual execution + updates)
python -c "import subprocess; subprocess.run(['python', 'specs/061-implement-fuzzy-matching/quickstart.md'])" &

wait
```

## Validation Checklist
*GATE: Checked before marking feature complete*

- [x] All contracts have corresponding tests (T001-T005 cover all 5 contract files)
- [x] No new entities required (uses existing Entity model)
- [x] All tests come before implementation (T001-T007 before T008)
- [x] Parallel tasks truly independent (T001-T005 different test classes, T011+T013 different files)
- [x] Each task specifies exact file path (all tasks have File: field)
- [x] No task modifies same file as another [P] task (T001-T005 share file but are test classes, acceptable)

## Notes

### TDD Workflow
1. Run T001-T005 (contract tests) - **ALL MUST FAIL**
2. Verify failure messages indicate `search_entities()` method does not exist
3. Run T008 (implementation)
4. Re-run T001-T005 - **SHOULD START PASSING**
5. Continue iterating on T008 until all contract tests pass

### BUG-001 Status
- **Already Fixed**: `import time` present at `iris_vector_rag/pipelines/graphrag.py:8`
- **Task T007**: Validates fix with integration test (should pass immediately)
- **No Implementation Needed**: Only validation test required

### Performance Validation
- **T012**: Validates <50ms fuzzy search, <10ms exact match requirements
- **Use real IRIS database**: iris-devtester with IRISContainer.community()
- **Document results**: Include timing data in test output

### Zero Regression Requirement
- **T014**: Critical validation that existing EntityStorageAdapter methods unchanged
- **Must Pass**: 100% of existing tests
- **Rationale**: Constitutional requirement (no breaking changes)

## Task Execution Order

**Recommended Sequence**:
1. **Parallel**: T001, T002, T003, T004, T005 (contract tests)
2. **Sequential**: T006 (integration test setup)
3. **Independent**: T007 (GraphRAG validation)
4. **Sequential**: T008 → T009 → T010 (implementation + documentation)
5. **Parallel**: T011, T013 (unit tests + quickstart)
6. **Sequential**: T012 (performance test requires T006+T008)
7. **Sequential**: T014 (regression test - final validation)

**Total Estimated Time**: 4-6 hours (including test writing, implementation, validation)

## Success Criteria

Feature is complete when:
- ✅ All 29 contract test cases pass (T001-T005)
- ✅ Integration tests pass with real IRIS (T006)
- ✅ GraphRAG pipeline test passes (T007) - validates BUG-001
- ✅ search_entities() method implemented (T008)
- ✅ Performance requirements met: <50ms fuzzy, <10ms exact (T012)
- ✅ Zero regressions in existing EntityStorageAdapter (T014)
- ✅ All 9 acceptance scenarios from spec.md validated
- ✅ Quickstart guide examples work (T013)

---
*Based on Constitution v1.8.0 - Test-Driven Development (Principle III)*
*Generated from plan.md, data-model.md, research.md, contracts/ (5 files)*
