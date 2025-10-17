
# Implementation Plan: Retrofit GraphRAG Testing Improvements to Other Pipelines

**Branch**: `036-retrofit-graphrag-s` | **Date**: 2025-10-08 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/Users/intersystems-community/ws/rag-templates/specs/036-retrofit-graphrag-s/spec.md`

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

Retrofit comprehensive testing patterns from GraphRAG (Feature 034) to four RAG pipelines: BasicRAG, CRAG, BasicRerankRAG, and PyLateColBERT. Implement contract tests validating API behavior, error handling with diagnostic messages, fallback mechanisms for graceful degradation, dimension validation, and end-to-end integration tests. This ensures consistent testing quality and operational reliability across all RAG framework pipelines.

## Technical Context
**Language/Version**: Python 3.12
**Primary Dependencies**: pytest, pytest-mock, iris_rag framework (RAGPipeline base class), IRIS database, OpenAI/Anthropic LLM APIs
**Storage**: InterSystems IRIS database with vector storage and SQL capabilities
**Testing**: pytest with contract tests, integration tests, mocking via pytest-mock
**Target Platform**: macOS/Linux development environments, CI/CD pipelines
**Project Type**: Single project (RAG framework testing infrastructure)
**Performance Goals**: Contract tests <30 seconds for CI/CD, integration tests <2 minutes, support 10K+ document testing
**Constraints**: Must maintain consistency with GraphRAG testing patterns (Feature 034), no breaking changes to existing pipeline APIs, must support both mock and live IRIS testing
**Scale/Scope**: 4 pipelines, 6 testing patterns, 28 functional requirements, ~20-30 new test files

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**: ✓ No new components - enhancing existing RAGPipeline tests | ✓ Test infrastructure is framework-level | ✓ Tests executed via Make targets and pytest CLI

**II. Pipeline Validation & Requirements**: ✓ Tests validate pipeline initialization and requirements | ✓ Test fixtures provide idempotent setup/teardown

**III. Test-Driven Development**: ✓ This feature IS test development (contract tests, integration tests) | ✓ Tests will validate 10K+ document scenarios with RAGAS

**IV. Performance & Enterprise Scale**: N/A - Testing infrastructure feature | ✓ Tests validate IRIS operations in existing pipelines

**V. Production Readiness**: ✓ Tests validate logging and error handling | ✓ Tests validate health checks | ✓ Tests validate Docker compatibility

**VI. Explicit Error Handling**: ✓ Tests validate error messages and diagnostic logging | ✓ Tests validate exception handling | ✓ Tests validate error context

**VII. Standardized Database Interfaces**: ✓ Tests validate usage of IRISVectorStore utilities | ✓ Tests ensure no ad-hoc queries | ✓ Tests validate database interaction patterns

**Initial Assessment**: PASS - This is a testing infrastructure feature that validates constitutional compliance in existing pipelines. No violations detected.

---

**Post-Design Re-evaluation** (after Phase 1):

**I. Framework-First Architecture**: ✅ PASS - Test infrastructure validates framework patterns, all tests use create_pipeline() factory

**II. Pipeline Validation & Requirements**: ✅ PASS - Contract tests validate pipeline requirement validation (FR-007)

**III. Test-Driven Development**: ✅ PASS - Feature creates tests for TDD workflow, validates 10K+ scenarios with RAGAS (FR-025 to FR-028)

**IV. Performance & Enterprise Scale**: ✅ PASS - Tests validate IRIS vector operations, contract tests <30s (FR-005)

**V. Production Readiness**: ✅ PASS - Tests validate logging (FR-009, FR-013), error handling (FR-009 to FR-014), Docker compatibility

**VI. Explicit Error Handling**: ✅ PASS - Comprehensive error contract (FR-009 to FR-014), actionable messages (FR-010)

**VII. Standardized Database Interfaces**: ✅ PASS - Tests validate IRISVectorStore usage, no ad-hoc queries, live IRIS testing (Constitutional requirement III)

**Final Assessment**: ✅ PASS - Design maintains full constitutional compliance. No violations, no complexity deviations. Ready for Phase 2.

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
iris_rag/                          # Framework source (read-only for this feature)
├── pipelines/
│   ├── basic.py                   # BasicRAG - target for testing
│   ├── crag.py                    # CRAG - target for testing
│   ├── basic_rerank.py            # BasicRerankRAG - target for testing
│   └── pylate_colbert.py          # PyLateColBERT - target for testing
├── core/
│   └── base.py                    # RAGPipeline base class
└── storage/
    └── iris_vector_store.py       # Vector store for fallback testing

tests/                             # Primary modification area
├── contract/                      # Contract tests (NEW FILES)
│   ├── test_basic_rag_contract.py
│   ├── test_crag_contract.py
│   ├── test_basic_rerank_contract.py
│   ├── test_pylate_colbert_contract.py
│   ├── test_basic_error_handling.py
│   ├── test_crag_error_handling.py
│   ├── test_basic_rerank_error_handling.py
│   ├── test_pylate_colbert_error_handling.py
│   ├── test_basic_dimension_validation.py
│   ├── test_crag_dimension_validation.py
│   ├── test_basic_rerank_dimension_validation.py
│   └── test_pylate_colbert_dimension_validation.py
├── integration/                   # Integration tests (NEW FILES)
│   ├── test_basic_rag_e2e.py
│   ├── test_crag_e2e.py
│   ├── test_basic_rerank_e2e.py
│   └── test_pylate_colbert_e2e.py
└── conftest.py                    # Shared fixtures (MODIFY)
```

**Structure Decision**: Single project structure. All changes are isolated to the `tests/` directory, adding new contract and integration test files for the 4 target pipelines. No modifications to framework source code (iris_rag/) required - this is purely a testing infrastructure enhancement.

## Phase 0: Outline & Research

**Status**: ✅ Complete

### Research Executed

No NEEDS CLARIFICATION markers in Technical Context. Research focused on analyzing GraphRAG testing patterns (Feature 034) and documenting their application to 4 target pipelines.

### Key Research Findings

1. **Testing Pattern Categories** (6 patterns from Feature 034):
   - Contract Tests: API behavior validation with Given-When-Then structure
   - Error Handling: Graceful degradation with actionable error messages
   - Diagnostic Logging: INFO/DEBUG level standardization
   - Dimension Validation: 384D embedding compatibility for all-MiniLM-L6-v2
   - Fallback Mechanisms: Pipeline-specific recovery strategies
   - Integration Tests: E2E query path validation

2. **Target Pipeline Characteristics**:
   - **BasicRAG**: Single retrieval method, embedding fallback only
   - **CRAG**: Relevance evaluator → vector+text fusion fallback
   - **BasicRerankRAG**: Cross-encoder → vector similarity fallback
   - **PyLateColBERT**: ColBERT → dense vector fallback

3. **Test Infrastructure Decisions**:
   - Test data: Reuse Feature 034 PMC diabetes documents
   - Fixtures: Session-scoped pipeline fixtures, function-scoped utilities
   - Mocking: Mock LLM APIs, NOT IRIS database (constitutional requirement)
   - Performance: Contract tests <30s, integration tests <2m

4. **Error Message Standard**:
   - Template: Error → Context → Expected → Actual → Fix
   - All errors MUST include actionable guidance
   - Context includes pipeline type, operation, current state

**Output**: [research.md](./research.md) with all decisions documented and alternatives considered

## Phase 1: Design & Contracts

**Status**: ✅ Complete

### Artifacts Generated

1. **[data-model.md](./data-model.md)**: Test infrastructure entities
   - 8 test entities defined (Pipeline, ContractTest, IntegrationTest, DiagnosticError, FallbackMechanism, DimensionValidator, TestFixture, TestDocument)
   - Entity relationships documented with diagram
   - Mock specifications for LLM and embedding services
   - Test data schema (JSON format, expected responses)

2. **[contracts/](./contracts/)**: API behavior contracts
   - **[pipeline_api_contract.md](./contracts/pipeline_api_contract.md)**: FR-001 to FR-004
     - Query method contract (input validation, response structure, errors)
     - Load documents method contract
     - Embed method contract
   - **[error_handling_contract.md](./contracts/error_handling_contract.md)**: FR-009 to FR-014
     - Configuration error templates
     - Transient failure retry strategy
     - Contextual error messages
     - Error chain logging
   - **[fallback_mechanism_contract.md](./contracts/fallback_mechanism_contract.md)**: FR-015 to FR-020
     - Pipeline-specific fallback strategies
     - Fallback logging requirements
     - Query semantics preservation
     - Fallback configuration and chain termination
   - **[dimension_validation_contract.md](./contracts/dimension_validation_contract.md)**: FR-021 to FR-024
     - Expected dimensions per pipeline (384D for BasicRAG/CRAG/BasicRerankRAG)
     - Validation points (initialization, query, indexing)
     - Error message templates
     - Dimension transformation support

3. **Contract Test Specifications** (in contract files):
   - Test case templates for each contract
   - Pytest fixture requirements
   - Mock strategies documented
   - Acceptance criteria defined

4. **[quickstart.md](./quickstart.md)**: Validation guide
   - Quick test execution commands
   - Validation checklists for all 4 pipelines
   - Troubleshooting guide
   - Performance benchmarks
   - Success criteria

5. **Agent Context Update**: CLAUDE.md updated
   - Added Python 3.12, pytest, pytest-mock
   - Added IRIS database context
   - Updated with Feature 036 technical context

### Design Decisions

**Test File Structure**:
```
tests/contract/
├── test_basic_rag_contract.py          # FR-001 to FR-008
├── test_basic_error_handling.py        # FR-009 to FR-014
├── test_basic_dimension_validation.py  # FR-021 to FR-024
├── [Similar files for CRAG, BasicRerankRAG, PyLateColBERT]
└── test_crag_fallback.py               # FR-015 to FR-020 (CRAG only)

tests/integration/
├── test_basic_rag_e2e.py               # FR-025 to FR-028
├── test_crag_e2e.py
├── test_basic_rerank_e2e.py
└── test_pylate_colbert_e2e.py
```

**Fixture Strategy** (to be added to conftest.py):
- Session-scoped: `basic_rag_pipeline`, `crag_pipeline`, `basic_rerank_pipeline`, `pylate_colbert_pipeline`
- Function-scoped: `log_capture`, `sample_documents`, `sample_query`
- Shared: `mocker` (pytest-mock), `caplog` (pytest built-in)

**Output**: All Phase 1 artifacts complete. Ready for Phase 2 (task planning).

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
1. Load `.specify/templates/tasks-template.md` as base template
2. Generate tasks from Phase 1 artifacts:
   - **From contracts/** → Contract test implementation tasks
   - **From data-model.md** → Pytest fixture creation tasks
   - **From quickstart.md** → Integration test implementation tasks
   - **From spec.md FR requirements** → Validation tasks

3. **Task Categories** (with counts):
   - **Setup Tasks** (3-5 tasks):
     - Update conftest.py with pipeline fixtures
     - Create sample test data files
     - Configure pytest markers
   - **Contract Test Tasks** (12-16 tasks):
     - BasicRAG contract tests (3 files: API, error, dimension)
     - CRAG contract tests (4 files: API, error, dimension, fallback)
     - BasicRerankRAG contract tests (4 files)
     - PyLateColBERT contract tests (4 files)
   - **Integration Test Tasks** (4 tasks):
     - One E2E test file per pipeline
   - **Validation Tasks** (2-3 tasks):
     - Run full test suite, verify <30s contract tests
     - Validate test coverage, verify FR traceability

**Ordering Strategy**:
- **Phase A (Setup)**: Fixtures and test data [P - parallel OK]
- **Phase B (Contract Tests)**: TDD order, all failing initially
  - Pipeline-specific tests can run in parallel [P]
  - Within pipeline: API → Error → Dimension → Fallback (sequential)
- **Phase C (Integration Tests)**: After contract tests pass
  - Each pipeline's E2E test independent [P]
- **Phase D (Validation)**: After all tests implemented

**Parallelization Markers**:
- [P] = Can execute in parallel (independent files)
- Sequential within each pipeline's test suite

**Estimated Output**: 25-30 numbered tasks in tasks.md

**Task Template Example**:
```markdown
### Task 12: Create BasicRAG Contract Test Suite [P]

**File**: tests/contract/test_basic_rag_contract.py
**Dependencies**: Task 1 (fixtures), Task 2 (test data)
**Functional Requirements**: FR-001, FR-002, FR-003, FR-004

**Implementation**:
1. Create test class TestBasicRAGContract
2. Implement test_query_method (validates query API)
3. Implement test_load_documents_method (validates document loading)
4. Implement test_query_validates_inputs (input validation)
5. Implement test_query_returns_valid_structure (response structure)
6. Implement test_query_handles_errors (error conditions)

**Acceptance**:
- All tests use basic_rag_pipeline fixture
- All tests follow Given-When-Then format
- All tests include FR traceability in docstrings
- Test suite executes in <10s
```

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
- [x] Phase 0: Research complete (/plan command) ✅
- [x] Phase 1: Design complete (/plan command) ✅
- [x] Phase 2: Task planning complete (/plan command - describe approach only) ✅
- [ ] Phase 3: Tasks generated (/tasks command) - NEXT STEP
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS ✅
- [x] Post-Design Constitution Check: PASS ✅
- [x] All NEEDS CLARIFICATION resolved (none existed) ✅
- [x] Complexity deviations documented (none exist) ✅

**Artifacts Generated**:
- [x] research.md ✅
- [x] data-model.md ✅
- [x] contracts/pipeline_api_contract.md ✅
- [x] contracts/error_handling_contract.md ✅
- [x] contracts/fallback_mechanism_contract.md ✅
- [x] contracts/dimension_validation_contract.md ✅
- [x] quickstart.md ✅
- [x] CLAUDE.md updated ✅
- [ ] tasks.md (awaiting /tasks command)

**Requirements Coverage**:
- FR-001 to FR-008: Contract tests → pipeline_api_contract.md
- FR-009 to FR-014: Error handling → error_handling_contract.md
- FR-015 to FR-020: Fallback mechanisms → fallback_mechanism_contract.md
- FR-021 to FR-024: Dimension validation → dimension_validation_contract.md
- FR-025 to FR-028: Integration tests → quickstart.md + integration test design

---
*Based on Constitution v1.2.0 - See `/.specify/memory/constitution.md`*
