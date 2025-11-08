# Tasks: Fill in Testing Gaps for HybridGraphRAG Query Paths

**Input**: Design documents from `/Users/intersystems-community/ws/rag-templates/specs/034-fill-in-gaps/`
**Prerequisites**: plan.md ✓, research.md ✓, contracts/ ✓, quickstart.md ✓

## Execution Flow (main)
```
1. Load plan.md from feature directory → ✓ Loaded
   → Tech stack: Python 3.11+, pytest, pytest-mock, iris-vector-graph (optional)
   → Structure: Single project (tests/ directory)
2. Load optional design documents:
   → research.md: ✓ 6 technical decisions documented
   → contracts/: ✓ 8 contract specifications (7 new + 1 existing reference)
   → data-model.md: N/A (testing-only feature)
3. Generate tasks by category:
   → Setup: IRIS connectivity, test data validation, fixture review
   → Tests: 7 contract test files [P] + 1 integration test file
   → Core: Fixture enhancements for mocking scenarios
   → Integration: N/A (tests only)
   → Polish: Validation and coverage verification
4. Apply task rules:
   → Different test files = mark [P] for parallel
   → Fixture updates = sequential (single conftest.py)
   → Tests validate existing implementation (Feature 033)
5. Number tasks sequentially (T001-T015)
6. Generate dependency graph (Setup → Contract Tests → Integration → Validation)
7. Create parallel execution examples for contract tests
8. Validate task completeness:
   → All 8 contracts have test tasks? ✓
   → All 28 FRs covered? ✓
   → TDD approach followed? ✓ (tests validate existing code)
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions
- All paths are absolute from repository root

## Phase 3.1: Setup & Prerequisites

- [x] **T001** Verify IRIS database connectivity and health ✅
  - **Path**: Run `python evaluation_framework/test_iris_connectivity.py`
  - **Verify**: IRIS container running on localhost:11972 or localhost:21972 ✅
  - **Verify**: Connection succeeds with _SYSTEM/SYS credentials ✅
  - **Verify**: RAG schema exists and is accessible ✅
  - **Dependency**: Must pass before any tests can execute
  - **Estimated Time**: 2 minutes

- [x] **T002** Verify test data loaded (2,376 documents with embeddings) ✅
  - **Path**: Query `SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL`
  - **Verify**: Count >= 2,376 documents ✅ (exactly 2,376)
  - **Verify**: All documents have 384-dimensional embeddings ✅
  - **Verify**: Documents contain biomedical content ✅
  - **Dependency**: T001 must pass (IRIS connectivity)
  - **Estimated Time**: 2 minutes

- [x] **T003** Review existing test fixtures in conftest.py ✅
  - **Path**: Read `/Users/intersystems-community/ws/rag-templates/tests/conftest.py`
  - **Review**: Available fixtures (graphrag_pipeline, config_manager, embedding_manager) ✅
  - **Review**: Pytest markers (@pytest.mark.requires_database) ✅
  - **Review**: Fixture scopes and lifecycle ✅
  - **Note**: Will add new fixtures in T012
  - **Dependency**: None
  - **Estimated Time**: 5 minutes

## Phase 3.2: Contract Tests (TDD) ⚠️ WRITE TESTS FIRST
**CRITICAL: These tests validate existing HybridGraphRAG implementation from Feature 033**

- [ ] **T004 [P]** Contract test: Hybrid Fusion Query Path
  - **Path**: Create `/Users/intersystems-community/ws/rag-templates/tests/contract/test_hybrid_fusion_contract.py`
  - **Contract**: `contracts/hybrid_fusion_contract.md`
  - **Requirements**: FR-001, FR-002, FR-003
  - **Test Cases**:
    - TC-001: test_hybrid_fusion_executes_successfully (method="hybrid" success)
    - TC-002: test_hybrid_fusion_fallback_on_zero_results (0 results → fallback)
    - TC-003: test_hybrid_fusion_fallback_on_exception (exception → fallback)
  - **Assertions**: RAGResponse, retrieval_method metadata, fallback logging
  - **Markers**: @pytest.mark.requires_database
  - **Mocking**: Use pytest-mock to simulate 0 results and exceptions
  - **Dependency**: T001-T003 complete
  - **Estimated Time**: 20 minutes

- [ ] **T005 [P]** Contract test: RRF (Reciprocal Rank Fusion) Path
  - **Path**: Create `/Users/intersystems-community/ws/rag-templates/tests/contract/test_rrf_contract.py`
  - **Contract**: `contracts/rrf_contract.md`
  - **Requirements**: FR-004, FR-005, FR-006
  - **Test Cases**:
    - TC-004: test_rrf_executes_successfully (method="rrf" success)
    - TC-005: test_rrf_fallback_on_zero_results (0 results → fallback)
    - TC-006: test_rrf_fallback_on_exception (exception → fallback)
  - **Assertions**: RRF fusion results, vector_fallback metadata, logging
  - **Markers**: @pytest.mark.requires_database
  - **Mocking**: Mock iris_graph_core RRF method
  - **Dependency**: T001-T003 complete
  - **Estimated Time**: 20 minutes

- [ ] **T006 [P]** Contract test: Enhanced Text Search Path
  - **Path**: Create `/Users/intersystems-community/ws/rag-templates/tests/contract/test_text_search_contract.py`
  - **Contract**: `contracts/text_search_contract.md`
  - **Requirements**: FR-007, FR-008, FR-009
  - **Test Cases**:
    - TC-007: test_text_search_executes_successfully (method="text" success)
    - TC-008: test_text_search_fallback_on_zero_results (0 results → fallback)
    - TC-009: test_text_search_fallback_on_exception (exception → fallback)
  - **Assertions**: iFind text search results, fallback behavior, logging
  - **Markers**: @pytest.mark.requires_database
  - **Mocking**: Mock iris_graph_core text search method
  - **Dependency**: T001-T003 complete
  - **Estimated Time**: 20 minutes

- [ ] **T007 [P]** Contract test: HNSW Vector Search Path
  - **Path**: Create `/Users/intersystems-community/ws/rag-templates/tests/contract/test_hnsw_vector_contract.py`
  - **Contract**: `contracts/hnsw_vector_contract.md`
  - **Requirements**: FR-010, FR-011, FR-012
  - **Test Cases**:
    - TC-010: test_hnsw_vector_executes_successfully (method="vector" success)
    - TC-011: test_hnsw_vector_fallback_on_zero_results (0 results → fallback)
    - TC-012: test_hnsw_vector_fallback_on_exception (exception → fallback)
  - **Assertions**: HNSW vector search results, IRISVectorStore fallback, logging
  - **Markers**: @pytest.mark.requires_database
  - **Mocking**: Mock iris_graph_core HNSW method
  - **Dependency**: T001-T003 complete
  - **Estimated Time**: 20 minutes

- [ ] **T008 [P]** Contract test: Knowledge Graph Traversal Path
  - **Path**: Create `/Users/intersystems-community/ws/rag-templates/tests/contract/test_kg_traversal_contract.py`
  - **Contract**: `contracts/kg_traversal_contract.md`
  - **Requirements**: FR-013, FR-014, FR-015
  - **Test Cases**:
    - TC-013: test_kg_traversal_executes_successfully (method="kg" success)
    - TC-014: test_kg_seed_entity_finding (entity identification)
    - TC-015: test_kg_multi_hop_depth_limits (depth limit enforcement)
  - **Assertions**: Graph traversal results, seed entities, depth limits
  - **Markers**: @pytest.mark.requires_database
  - **Note**: Tests inherited GraphRAGPipeline functionality
  - **Dependency**: T001-T003 complete
  - **Estimated Time**: 25 minutes

- [ ] **T009 [P]** Contract test: Fallback Mechanism Validation
  - **Path**: Create `/Users/intersystems-community/ws/rag-templates/tests/contract/test_fallback_mechanism_contract.py`
  - **Contract**: `contracts/fallback_mechanism_contract.md`
  - **Requirements**: FR-016, FR-017, FR-018, FR-019
  - **Test Cases**:
    - TC-016: test_fallback_retrieves_documents_successfully (IRISVectorStore fallback)
    - TC-017: test_fallback_logs_diagnostic_messages (logging validation)
    - TC-018: test_fallback_metadata_indicates_vector_fallback (metadata check)
    - TC-019: test_graceful_degradation_without_iris_graph_core (unavailable dependency)
  - **Assertions**: Fallback execution, diagnostic logs, metadata, graceful degradation
  - **Markers**: @pytest.mark.requires_database
  - **Mocking**: Mock iris_graph_core unavailable
  - **Dependency**: T001-T003 complete
  - **Estimated Time**: 25 minutes

- [ ] **T010 [P]** Contract test: Error Handling and Edge Cases
  - **Path**: Create `/Users/intersystems-community/ws/rag-templates/tests/contract/test_error_handling_contract.py`
  - **Contract**: `contracts/error_handling_contract.md`
  - **Requirements**: FR-023, FR-024, FR-025
  - **Test Cases**:
    - TC-023: test_missing_required_tables_handled (missing RDF_EDGES, kg_NodeEmbeddings_optimized)
    - TC-024: test_iris_graph_core_connection_failure_handled (connection exception)
    - TC-025: test_system_continues_after_fallback (multiple queries post-fallback)
  - **Assertions**: Error handling, logging, fallback activation, state consistency
  - **Markers**: @pytest.mark.requires_database
  - **Mocking**: Mock missing tables and connection failures
  - **Dependency**: T001-T003 complete
  - **Estimated Time**: 20 minutes

- [ ] **T011** Integration test: End-to-End HybridGraphRAG Workflows
  - **Path**: Create `/Users/intersystems-community/ws/rag-templates/tests/integration/test_hybridgraphrag_e2e.py`
  - **Contract**: `contracts/e2e_integration_contract.md`
  - **Requirements**: FR-026, FR-027, FR-028
  - **Test Cases**:
    - TC-026: test_all_query_methods_end_to_end (all 5 methods: hybrid, rrf, text, vector, kg)
    - TC-027: test_multiple_sequential_queries_consistent (10+ queries, same pipeline)
    - TC-028: test_retrieval_metadata_completeness (metadata fields validation)
  - **Assertions**: All methods work, sequential execution, metadata completeness
  - **Markers**: @pytest.mark.requires_database, @pytest.mark.integration
  - **Note**: Tests complete workflows, not mocked
  - **Dependency**: T004-T010 complete (validates contract tests pass first)
  - **Estimated Time**: 30 minutes

## Phase 3.3: Fixture Enhancements

- [ ] **T012** Add mocking fixtures to conftest.py
  - **Path**: Edit `/Users/intersystems-community/ws/rag-templates/tests/conftest.py`
  - **Add Fixtures**:
    - `mock_iris_graph_core_unavailable`: Mock IRIS_GRAPH_CORE_AVAILABLE = False
    - `mock_zero_results_retrieval`: Mock retrieval methods returning ([], method_name)
    - `mock_connection_failure`: Mock iris_graph_core connection exception
  - **Scope**: Function-scoped for test isolation
  - **Usage**: Support contract tests T004-T010
  - **Dependency**: Can run in parallel with T004-T010, but should complete before T011
  - **Estimated Time**: 15 minutes

## Phase 3.4: Validation & Coverage

- [ ] **T013** Run full contract test suite and verify all pass
  - **Path**: Execute `pytest tests/contract/test_*_contract.py -v`
  - **Verify**: All 22 contract tests pass (19 new + 6 existing from Feature 033)
  - **Verify**: Execution time <3 minutes
  - **Verify**: No skipped tests (all dependencies available)
  - **Verify**: All @pytest.mark.requires_database tests connect to live IRIS
  - **Dependency**: T004-T012 complete
  - **Estimated Time**: 5 minutes

- [ ] **T014** Run integration tests and verify all pass
  - **Path**: Execute `pytest tests/integration/test_hybridgraphrag_e2e.py -v`
  - **Verify**: All 3 E2E tests pass
  - **Verify**: All 5 query methods validated
  - **Verify**: Sequential queries maintain state consistency
  - **Verify**: Metadata includes retrieval_method, execution_time, num_retrieved
  - **Dependency**: T011, T013 complete
  - **Estimated Time**: 3 minutes

- [ ] **T015** Validate test coverage for all 28 functional requirements
  - **Path**: Review all test files and spec.md requirements
  - **Verify Coverage**:
    - FR-001 to FR-003: ✓ T004 (test_hybrid_fusion_contract.py)
    - FR-004 to FR-006: ✓ T005 (test_rrf_contract.py)
    - FR-007 to FR-009: ✓ T006 (test_text_search_contract.py)
    - FR-010 to FR-012: ✓ T007 (test_hnsw_vector_contract.py)
    - FR-013 to FR-015: ✓ T008 (test_kg_traversal_contract.py)
    - FR-016 to FR-019: ✓ T009 (test_fallback_mechanism_contract.py)
    - FR-020 to FR-022: ✓ Existing (test_dimension_validation_contract.py from Feature 033)
    - FR-023 to FR-025: ✓ T010 (test_error_handling_contract.py)
    - FR-026 to FR-028: ✓ T011 (test_hybridgraphrag_e2e.py)
  - **Success Criteria**: 100% FR coverage with passing tests
  - **Dependency**: T013, T014 complete
  - **Estimated Time**: 10 minutes

## Dependencies

```
Phase 3.1 (Setup):
T001 (IRIS connectivity) → T002 (test data)
T003 (fixture review) ← independent

Phase 3.2 (Contract Tests):
T001-T003 → T004-T010 [All Parallel] → T011 (integration)
T012 (fixtures) can run parallel with T004-T010

Phase 3.4 (Validation):
T004-T012 → T013 (contract suite) → T014 (integration suite) → T015 (coverage)
```

## Parallel Execution Examples

### Example 1: All Contract Tests (Maximum Parallelism)
```bash
# Launch T004-T010 simultaneously (7 independent test files)
pytest tests/contract/test_hybrid_fusion_contract.py \
       tests/contract/test_rrf_contract.py \
       tests/contract/test_text_search_contract.py \
       tests/contract/test_hnsw_vector_contract.py \
       tests/contract/test_kg_traversal_contract.py \
       tests/contract/test_fallback_mechanism_contract.py \
       tests/contract/test_error_handling_contract.py \
       -n auto -v
```

### Example 2: Using pytest-xdist for Parallel Test Execution
```bash
# Install pytest-xdist if not already installed
pip install pytest-xdist

# Run all contract tests in parallel
pytest tests/contract/ -n auto -v

# Run all tests (contract + integration) in parallel
pytest tests/contract/ tests/integration/test_hybridgraphrag_e2e.py -n auto -v
```

### Example 3: Sequential Execution for Debugging
```bash
# Run tests sequentially with verbose output
pytest tests/contract/test_hybrid_fusion_contract.py -vv -s
pytest tests/contract/test_rrf_contract.py -vv -s
# ... etc
```

## Notes

- **[P] tasks** indicate different files with no dependencies - can execute in parallel
- **TDD Approach**: Tests validate existing HybridGraphRAG implementation (Feature 033)
- **All tests use live IRIS database** - Constitution requirement (@pytest.mark.requires_database)
- **Mocking strategy**: Only mock iris_graph_core methods to test fallback paths
- **Execution order**: Setup → Contract Tests [P] → Integration → Validation
- **Performance target**: Full test suite <5 minutes
- **Coverage target**: 100% of 28 functional requirements

## Task Generation Rules Applied

1. **From Contracts**:
   - ✓ 8 contract files → 7 test tasks (T004-T010) + 1 reference to existing
   - ✓ All contract tests marked [P] (different files)

2. **From Data Model**:
   - N/A - Testing-only feature, no new entities

3. **From User Stories**:
   - ✓ Integration scenarios → T011 (E2E test)
   - ✓ Quickstart scenarios → T013-T015 (validation)

4. **Ordering**:
   - ✓ Setup (T001-T003) → Tests (T004-T011) → Validation (T013-T015)
   - ✓ Contract tests before integration (T004-T010 → T011)
   - ✓ All tests before validation (T004-T011 → T013-T015)

## Validation Checklist

- [x] All contracts have corresponding tests (8 contracts → 7 new tasks + 1 existing)
- [x] All entities have model tasks (N/A - testing-only feature)
- [x] All tests come before implementation (tests validate existing code)
- [x] Parallel tasks truly independent (different files, no shared state)
- [x] Each task specifies exact file path (absolute paths provided)
- [x] No task modifies same file as another [P] task (T012 is sequential, T004-T010 all different files)
- [x] All 28 FRs covered by test tasks
- [x] Constitutional compliance (live IRIS database, TDD, error handling)

## Success Criteria

✅ **Feature Complete When**:
- All 15 tasks completed
- 25 new tests pass (22 contract + 3 integration)
- All 28 functional requirements validated
- Test execution time <5 minutes
- 100% coverage of HybridGraphRAG query paths
- All tests use live IRIS database connection
- Fallback mechanisms validated for all query methods
