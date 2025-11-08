# Implementation Plan: Testing Framework Fixes for Coverage and Functional Correctness

**Branch**: `025-fixes-for-testing` | **Date**: 2025-10-03 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/025-fixes-for-testing/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path ✅
   → Spec found at /Users/intersystems-community/ws/rag-templates/specs/025-fixes-for-testing/spec.md
2. Fill Technical Context (scan for NEEDS CLARIFICATION) ✅
   → Detect Project Type: Python framework (testing infrastructure)
   → Structure Decision: Single project (tests/ directory structure)
3. Fill Constitution Check section ✅
   → Based on constitution v1.6.0
4. Evaluate Constitution Check section ✅
   → Violations: None (test infrastructure improvements)
   → Update Progress Tracking: Initial Constitution Check PASS
5. Execute Phase 0 → research.md ✅
   → Coverage tools, test patterns, IRIS test requirements
6. Execute Phase 1 → contracts, data-model.md, quickstart.md ✅
   → Test contracts, test data models, quickstart guide
7. Re-evaluate Constitution Check section ✅
   → No new violations
   → Update Progress Tracking: Post-Design Constitution Check PASS
8. Plan Phase 2 → Describe task generation approach ✅
9. STOP - Ready for /tasks command ✅
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary

**Primary Requirement**: Fix 60 failing E2E tests and 11 GraphRAG test errors, improve test coverage from 10% to target threshold, and ensure all tests run reliably against IRIS database.

**Technical Approach**:
1. Align test expectations with actual production API contracts
2. Fix IRIS vector store tests to use correct TO_VECTOR(DOUBLE) behavior
3. Resolve GraphRAG test setup/dependency issues
4. Add missing test coverage for critical modules (pipelines, storage, validation)
5. Ensure pytest-randomly compatibility or explicit disabling
6. Implement proper test isolation and cleanup

## Technical Context

**Language/Version**: Python 3.12 (framework uses Python 3.12.9)
**Primary Dependencies**: pytest 8.4.1, pytest-cov 6.1.1, IRIS Database (licensed version 2025.3.0), sentence-transformers, langchain
**Storage**: InterSystems IRIS database on ports 11972 (default), 21972 (licensed), 31972 (community)
**Testing**: pytest with markers (@pytest.mark.e2e, @pytest.mark.integration, @pytest.mark.requires_database)
**Target Platform**: macOS Darwin 24.5.0 (development), Linux (CI/CD)
**Project Type**: Single Python framework (test infrastructure improvements)
**Performance Goals**:
  - Test suite execution < 2 minutes (current: ~106 seconds for 206 tests)
  - Coverage target: 60% overall (current: 10%)
  - Critical module coverage: 80% (pipelines, storage, validation)
**Constraints**:
  - Must run with `-p no:randomly` flag (numpy/thinc seed issues)
  - Must use real IRIS database for E2E tests (no mocks per constitution)
  - Must maintain all 206 currently passing tests
  - Must fix or properly skip 60 failing + 11 error tests
**Scale/Scope**:
  - 11,470 lines of production code to test
  - 6 E2E test files (2,775 lines existing)
  - 4 unit test files (remaining after cleanup)
  - Target: 400+ total tests with high quality

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**: ✓ Tests validate RAGPipeline framework components | ✓ No application-specific logic (framework testing) | ✓ CLI interface via Make targets (`make test`, `make test-unit`, `make test-e2e`)

**II. Pipeline Validation & Requirements**: ✓ Tests validate automated requirement validation systems | ✓ Setup procedures tested for idempotency

**III. Test-Driven Development**: ✓ Fix existing failing tests (aligns with TDD) | ✓ Add performance tests for 10K+ scenarios (GraphRAG, CRAG)

**IV. Performance & Enterprise Scale**: ✓ Tests validate incremental indexing support | ✓ Tests validate IRIS vector operations optimization

**V. Production Readiness**: ✓ Tests validate structured logging | ✓ Tests validate health checks | ✓ Tests validate Docker deployment

**VI. Explicit Error Handling**: ✓ Tests validate no silent failures | ✓ Tests validate clear exception messages | ✓ Tests validate actionable error context

**VII. Standardized Database Interfaces**: ✓ Tests use common/db_vector_utils.py, common/vector_sql_utils.py | ✓ Tests validate no ad-hoc IRIS queries | ✓ Tests ensure patterns work correctly

**GATE RESULT**: ✅ PASS - Testing infrastructure improvements align with all constitutional principles

## Project Structure

### Documentation (this feature)
```
specs/025-fixes-for-testing/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
│   ├── test_execution_contract.md
│   ├── coverage_reporting_contract.md
│   └── test_isolation_contract.md
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
tests/
├── unit/                # Unit tests with minimal dependencies
│   ├── conftest.py      # Mock fixtures
│   ├── test_final_massive_coverage.py  # Existing unit tests
│   ├── test_working_units.py           # Existing unit tests
│   └── test_configuration_coverage.py  # Existing unit tests
├── e2e/                 # End-to-end tests with real IRIS
│   ├── conftest.py      # Real IRIS fixtures
│   ├── test_basic_pipeline_e2e.py      # BasicRAG tests (22 tests)
│   ├── test_basic_rerank_pipeline_e2e.py  # BasicRerank tests (26 tests)
│   ├── test_crag_pipeline_e2e.py       # CRAG tests (50+ tests, 19 failing)
│   ├── test_graphrag_pipeline_e2e.py   # GraphRAG tests (60+ tests, 39 failing + 11 errors)
│   ├── test_pylate_pipeline_e2e.py     # PyLate tests (11 tests, 7 failing)
│   ├── test_vector_store_comprehensive_e2e.py  # Vector store tests (70+ tests, 8 failing)
│   ├── test_vector_store_iris_e2e.py   # IRIS vector tests (failing)
│   ├── test_schema_manager_e2e.py      # Schema manager tests
│   ├── test_configuration_e2e.py       # Config tests (1 failing)
│   └── test_core_framework_e2e.py      # Core framework tests (4 failing)
└── integration/         # Integration tests (cross-component)
    └── test_rag_bridge.py

common/
├── db_vector_utils.py   # Vector insertion utilities (TO_VECTOR with DOUBLE)
├── vector_sql_utils.py  # Vector SQL query builders (needs fixes)
└── utils.py             # LLM and utility functions

iris_rag/
├── pipelines/           # Pipeline implementations to test
│   ├── basic.py
│   ├── basic_rerank.py
│   ├── crag.py
│   ├── graphrag.py
│   └── colbert_pylate/
├── storage/             # Storage implementations to test
│   ├── vector_store_iris.py
│   └── schema_manager.py
└── validation/          # Validation framework to test
    ├── requirements.py
    └── orchestrator.py
```

**Structure Decision**: Single Python framework project with standard tests/ directory structure. Tests are organized by type (unit, e2e, integration) per pytest conventions and constitutional requirements.

## Phase 0: Outline & Research

### Research Tasks

1. **pytest Best Practices for IRIS Database Testing**
   - Research: How to properly test database-dependent code with pytest
   - Research: Test isolation patterns for database tests (fixtures, cleanup, transactions)
   - Research: pytest-randomly issues with numpy/thinc and resolution strategies

2. **Coverage Tools and Reporting**
   - Research: pytest-cov configuration for accurate coverage reporting
   - Research: Excluding test files from coverage calculations
   - Research: Coverage thresholds and enforcement in CI/CD

3. **API Contract Testing Patterns**
   - Research: How to validate production APIs match test expectations
   - Research: Contract testing tools and patterns for Python
   - Research: Test fixtures that mirror production behavior

4. **GraphRAG Testing Requirements**
   - Research: Dependencies needed for GraphRAG tests (entity extraction, graph storage)
   - Research: Setup procedures for GraphRAG test environment
   - Research: Mock vs real LLM for entity extraction tests

5. **IRIS Vector Store Testing**
   - Research: TO_VECTOR function requirements (DOUBLE datatype, no parameter markers)
   - Research: Vector similarity search validation patterns
   - Research: IRIS-specific test data setup and cleanup

### Research Output

See [research.md](./research.md) for detailed findings, decisions, rationales, and alternatives considered for each research task.

**Key Decisions**:
- Use pytest fixtures with scope="module" for IRIS connections (minimize setup overhead)
- Disable pytest-randomly globally via pytest.ini `-p no:randomly` flag
- Set coverage targets: 60% overall, 80% for critical modules (pipelines, storage, validation)
- Use real IRIS database for all E2E tests per constitutional requirement
- Implement proper test cleanup via teardown fixtures
- Fix API mismatches by updating tests to match production, not vice versa

## Phase 1: Design & Contracts

### Data Model (test entities)

See [data-model.md](./data-model.md) for complete entity definitions.

**Key Entities**:
1. **TestCase**: name, status (pass/fail/skip/error), execution_time, coverage_lines
2. **CoverageReport**: module_name, total_lines, covered_lines, percentage, missing_lines
3. **APIContract**: endpoint_name, expected_signature, actual_signature, match_status
4. **TestFixture**: name, scope (function/class/module), setup_code, teardown_code, dependencies
5. **IrisConnection**: host, port, username, password, namespace, connection_handle

### API Contracts

See [contracts/](./contracts/) directory for detailed contract specifications.

**Contract Files**:
1. `test_execution_contract.md`: Pytest execution requirements and fixtures
2. `coverage_reporting_contract.md`: Coverage calculation and reporting requirements
3. `test_isolation_contract.md`: Database cleanup and isolation requirements
4. `api_alignment_contract.md`: Production API contracts that tests must validate
5. `graphrag_setup_contract.md`: GraphRAG test environment setup requirements

### Contract Tests

Contract tests validate that:
1. All E2E tests can connect to IRIS database (test_execution_contract)
2. Coverage reports accurately reflect line coverage (coverage_reporting_contract)
3. Tests properly clean up database state (test_isolation_contract)
4. Test expectations match production APIs (api_alignment_contract)
5. GraphRAG tests have proper dependencies and setup (graphrag_setup_contract)

**Contract Test Files** (to be created in tests/contract/):
- `test_pytest_execution_contract.py` - Validates pytest can run with IRIS
- `test_coverage_contract.py` - Validates coverage reporting accuracy
- `test_isolation_contract.py` - Validates test cleanup procedures
- `test_api_contract.py` - Validates test/production API alignment
- `test_graphrag_contract.py` - Validates GraphRAG test requirements

### Quickstart

See [quickstart.md](./quickstart.md) for developer quickstart guide.

**Quickstart validates**:
1. Developer can run all passing tests successfully
2. Developer can see which tests are failing and why
3. Developer can run tests with coverage reports
4. Developer can add new tests following established patterns
5. Developer can fix failing tests by understanding API contracts

### Agent Context Update

Execute: `.specify/scripts/bash/update-agent-context.sh claude`

Updates CLAUDE.md with:
- New testing patterns and requirements
- Coverage targets and enforcement
- IRIS database testing procedures
- API contract alignment guidelines
- Recent changes to testing infrastructure

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

### Task Generation Strategy

**Generate tasks from Phase 1 design docs**:

1. **From test_execution_contract.md**:
   - Task: Create pytest.ini with `-p no:randomly` flag
   - Task: Create conftest.py with IRIS connection fixtures
   - Task: Verify all E2E tests can import and initialize

2. **From coverage_reporting_contract.md**:
   - Task: Configure .coveragerc to exclude test files
   - Task: Add coverage targets to pytest.ini
   - Task: Create coverage report generation script

3. **From test_isolation_contract.md**:
   - Task: Implement database cleanup fixtures
   - Task: Add transaction rollback for unit tests
   - Task: Verify tests don't pollute each other

4. **From api_alignment_contract.md** (60 failing tests):
   - Task: Fix BasicRAG load_documents API mismatch (documents kwarg)
   - Task: Fix CRAG pipeline query API mismatches (19 tests)
   - Task: Fix GraphRAG pipeline API mismatches (39 tests)
   - Task: Fix PyLate pipeline API mismatches (7 tests)
   - Task: Fix vector store metadata filtering tests (4 tests)
   - Task: Fix vector store similarity threshold tests (3 tests)
   - Task: Fix IRIS vector store core tests (3 tests)
   - Task: Fix configuration reload tests (1 test)
   - Task: Fix core framework tests (4 tests)

5. **From graphrag_setup_contract.md** (11 errors):
   - Task: Investigate GraphRAG test setup failures
   - Task: Add missing GraphRAG dependencies to test environment
   - Task: Fix or skip GraphRAG tests with proper explanations

6. **From data-model.md**:
   - Task: Create test result aggregation utility
   - Task: Create coverage trend tracking utility

7. **Coverage improvement tasks**:
   - Task: Identify modules with <60% coverage
   - Task: Add unit tests for uncovered critical paths
   - Task: Add E2E tests for missing pipeline workflows

### Ordering Strategy

**TDD order** (tests before implementation):
1. Contract tests first (validate requirements)
2. Fix infrastructure (pytest.ini, conftest.py, fixtures)
3. Fix failing tests (API alignment)
4. Add coverage tests (new test files)
5. Validation (run all tests, check coverage)

**Dependency order**:
1. Infrastructure setup (pytest config, IRIS fixtures)
2. Test isolation (cleanup fixtures)
3. API contract fixes (align tests with production)
4. GraphRAG fixes (resolve setup errors)
5. Coverage improvements (add new tests)

**Parallel execution** [P]:
- API contract fixes can run in parallel (independent test files)
- Coverage improvements can run in parallel (independent modules)

### Estimated Output

**75-85 numbered, ordered tasks** in tasks.md:
- 5 infrastructure tasks
- 5 test isolation tasks
- 60 API alignment tasks (1 per failing test group)
- 11 GraphRAG error resolution tasks
- 5-10 coverage improvement tasks

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
| (none) | (no violations) | (no violations) |

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved (reasonable defaults chosen)
- [x] Complexity deviations documented (none)

---
*Based on Constitution v1.6.0 - See `.specify/memory/constitution.md`*
