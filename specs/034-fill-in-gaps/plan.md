
# Implementation Plan: Fill in Testing Gaps for HybridGraphRAG Query Paths

**Branch**: `034-fill-in-gaps` | **Date**: 2025-10-07 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/Users/tdyar/ws/rag-templates/specs/034-fill-in-gaps/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from file system structure or context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
This feature addresses the testing gaps identified during Feature 033 (Fix GraphRAG Vector Retrieval Logic). The primary requirement is to create comprehensive integration tests for all HybridGraphRAG query processing paths to ensure the system works correctly in all scenarios, including fallback mechanisms when external dependencies (iris_graph_core) fail. The test suite must validate all 5 query methods (hybrid, rrf, text, vector, kg) and their fallback behaviors, achieving 100% coverage of HybridGraphRAG query processing paths with passing tests for all 28 functional requirements.

## Technical Context
**Language/Version**: Python 3.11+
**Primary Dependencies**: pytest, iris-vector-graph (optional), iris_rag framework, pytest-mock for mocking iris_graph_core
**Storage**: InterSystems IRIS database (vector store with VECTOR_DOT_PRODUCT, 384D all-MiniLM-L6-v2 embeddings)
**Testing**: pytest with contract tests (@pytest.mark.requires_database), integration tests, mocking for fallback scenarios
**Target Platform**: Linux/macOS development environment with Docker for IRIS database
**Project Type**: Single project (RAG framework testing)
**Performance Goals**: Test execution <5 minutes for full suite, validate retrieval operations complete within reasonable timeframes
**Constraints**: Tests must work with existing HybridGraphRAG implementation (Feature 033), must handle iris_graph_core optional dependency gracefully
**Scale/Scope**: 28 functional requirements across 5 query methods, comprehensive fallback coverage, integration with existing test infrastructure

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**: ✓ Tests validate framework components (HybridGraphRAG) | ✓ No application-specific logic (pure testing) | ✓ Tests executable via pytest/make targets

**II. Pipeline Validation & Requirements**: ✓ Tests validate pipeline requirement checking | ✓ Tests validate idempotent setup (FR-019 graceful degradation)

**III. Test-Driven Development**: ✓ This IS the TDD phase - writing comprehensive tests | ✓ Tests validate 10K+ scenario compatibility with existing enterprise tests | ✓ All contract tests MUST use live IRIS database (@pytest.mark.requires_database)

**IV. Performance & Enterprise Scale**: ✓ Tests validate incremental operations (FR-027 sequential queries) | ✓ Tests validate IRIS vector operations via IRISVectorStore fallback

**V. Production Readiness**: ✓ Tests validate structured logging (FR-017) | ✓ Tests validate error handling paths | ✓ Tests run in Docker environment

**VI. Explicit Error Handling**: ✓ Tests validate no silent failures (FR-023-025 error handling) | ✓ Tests validate exception messages | ✓ Tests validate fallback context (FR-018 metadata)

**VII. Standardized Database Interfaces**: ✓ Tests use existing IRISVectorStore utilities | ✓ Tests validate proper use of proven patterns | ✓ Tests ensure no ad-hoc queries in HybridGraphRAG

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
/Users/tdyar/ws/rag-templates/
├── iris_rag/                           # Framework package (testing target)
│   ├── pipelines/
│   │   ├── hybrid_graphrag.py          # HybridGraphRAG implementation (Feature 033)
│   │   └── graphrag.py                 # Base GraphRAG with fallback logic
│   ├── storage/
│   │   └── vector_store_iris.py        # IRISVectorStore fallback target
│   └── config/
│       └── manager.py                  # Configuration management
│
├── tests/                              # Test suite (implementation target)
│   ├── contract/                       # Contract tests (NEW for this feature)
│   │   ├── test_hybrid_fusion_contract.py         # FR-001 to FR-003
│   │   ├── test_rrf_contract.py                   # FR-004 to FR-006
│   │   ├── test_text_search_contract.py           # FR-007 to FR-009
│   │   ├── test_hnsw_vector_contract.py           # FR-010 to FR-012
│   │   ├── test_kg_traversal_contract.py          # FR-013 to FR-015
│   │   ├── test_fallback_mechanism_contract.py    # FR-016 to FR-019
│   │   ├── test_dimension_validation_contract.py  # FR-020 to FR-022 (already exists)
│   │   └── test_error_handling_contract.py        # FR-023 to FR-025
│   │
│   ├── integration/                    # Integration tests (NEW for this feature)
│   │   ├── test_hybridgraphrag_e2e.py             # FR-026 to FR-028
│   │   └── test_graphrag_vector_search.py         # Existing (Feature 033)
│   │
│   └── conftest.py                     # Pytest fixtures (may need updates)
│
└── specs/034-fill-in-gaps/             # Design artifacts
    ├── spec.md                         # Feature specification
    ├── plan.md                         # This file
    ├── contracts/                      # Test contracts (Phase 1 output)
    └── quickstart.md                   # Test execution guide (Phase 1 output)
```

**Structure Decision**: Single project structure with focus on comprehensive test coverage. New test files will be added to `tests/contract/` and `tests/integration/` directories. The implementation target (HybridGraphRAG) already exists from Feature 033; this feature purely adds test coverage to validate all query paths and fallback mechanisms work correctly.

## Phase 0: Outline & Research ✅ COMPLETE

**Status**: All technical context items were already resolved in specification. No NEEDS CLARIFICATION markers existed.

**Output**: `research.md` created with 6 key technical decisions:
1. Testing framework and strategy (pytest with contract tests, mocking for fallback scenarios)
2. Test organization by functional requirements (7 contract test files + 1 integration test file)
3. Mocking strategy for iris_graph_core (pytest-mock for 0 results and exceptions)
4. Fixture reuse and extension (leverage existing conftest.py fixtures)
5. Test data strategy (use existing 2,376 documents from Feature 033)
6. Test execution performance (<5 minute target, parallel execution)

## Phase 1: Design & Contracts ✅ COMPLETE

**Status**: Design artifacts generated for comprehensive test coverage.

**Outputs**:

1. **Test Contracts** (8 contract specifications in `/contracts/`):
   - `hybrid_fusion_contract.md` - TC-001 to TC-003 (FR-001 to FR-003)
   - `rrf_contract.md` - TC-004 to TC-006 (FR-004 to FR-006)
   - `text_search_contract.md` - TC-007 to TC-009 (FR-007 to FR-009)
   - `hnsw_vector_contract.md` - TC-010 to TC-012 (FR-010 to FR-012)
   - `kg_traversal_contract.md` - TC-013 to TC-015 (FR-013 to FR-015)
   - `fallback_mechanism_contract.md` - TC-016 to TC-019 (FR-016 to FR-019)
   - `error_handling_contract.md` - TC-023 to TC-025 (FR-023 to FR-025)
   - `e2e_integration_contract.md` - TC-026 to TC-028 (FR-026 to FR-028)
   - *Note: FR-020 to FR-022 covered by existing test_dimension_validation_contract.py from Feature 033*

2. **Quickstart Guide**: `quickstart.md` with:
   - Prerequisites and setup instructions
   - Step-by-step test execution commands
   - Expected results (25 new tests + 6 existing = 31 total)
   - Validation checklist
   - Troubleshooting guide

3. **Agent Context**: CLAUDE.md updated with:
   - Python 3.11+ testing context
   - pytest and mocking framework details
   - IRIS database testing requirements

**Note**: No data-model.md needed - this is a testing-only feature with no new entities. All tests validate existing HybridGraphRAG implementation from Feature 033.

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:

1. **Setup Tasks** (T001-T003):
   - Verify IRIS database connectivity
   - Verify test data loaded (2,376 documents)
   - Review existing conftest.py fixtures

2. **Contract Test Tasks [P]** (T004-T010) - All parallel, different files:
   - T004: Implement test_hybrid_fusion_contract.py (3 tests: TC-001 to TC-003)
   - T005: Implement test_rrf_contract.py (3 tests: TC-004 to TC-006)
   - T006: Implement test_text_search_contract.py (3 tests: TC-007 to TC-009)
   - T007: Implement test_hnsw_vector_contract.py (3 tests: TC-010 to TC-012)
   - T008: Implement test_kg_traversal_contract.py (3 tests: TC-013 to TC-015)
   - T009: Implement test_fallback_mechanism_contract.py (4 tests: TC-016 to TC-019)
   - T010: Implement test_error_handling_contract.py (3 tests: TC-023 to TC-025)

3. **Integration Test Task** (T011):
   - Implement test_hybridgraphrag_e2e.py (3 tests: TC-026 to TC-028)

4. **Fixture Enhancement Task** (T012):
   - Add new mocking fixtures to conftest.py (mock_iris_graph_core_unavailable, etc.)

5. **Validation Tasks** (T013-T015):
   - T013: Run full contract test suite and verify all pass
   - T014: Run integration tests and verify all pass
   - T015: Validate test coverage for all 28 FRs

**Ordering Strategy**:
- Setup first (T001-T003)
- Contract tests in parallel [P] (T004-T010) - independent files
- Fixtures can be done in parallel with contract tests (T012)
- Integration tests after contracts (T011)
- Validation after all implementation (T013-T015)

**Parallel Execution Example**:
```bash
# All contract tests can run simultaneously
pytest tests/contract/test_hybrid_fusion_contract.py \
       tests/contract/test_rrf_contract.py \
       tests/contract/test_text_search_contract.py \
       tests/contract/test_hnsw_vector_contract.py \
       tests/contract/test_kg_traversal_contract.py \
       tests/contract/test_fallback_mechanism_contract.py \
       tests/contract/test_error_handling_contract.py \
       -n auto  # pytest-xdist for parallel execution
```

**Estimated Output**: 15 numbered tasks with clear dependencies and parallel execution guidance

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command) - research.md created with 6 technical decisions
- [x] Phase 1: Design complete (/plan command) - 8 contract specs + quickstart.md + CLAUDE.md updated
- [x] Phase 2: Task planning complete (/plan command - describe approach only) - 15 tasks outlined
- [x] Phase 3: Tasks generated (/tasks command) - tasks.md created with 15 numbered tasks
- [ ] Phase 4: Implementation complete - **READY TO EXECUTE**
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS - All principles satisfied by testing feature
- [x] Post-Design Constitution Check: PASS - Test design follows all constitutional requirements
- [x] All NEEDS CLARIFICATION resolved - No clarifications needed, all context was clear
- [x] Complexity deviations documented - No deviations, straightforward testing implementation

---
*Based on Constitution v1.2.0 - See `/.specify/memory/constitution.md`*
