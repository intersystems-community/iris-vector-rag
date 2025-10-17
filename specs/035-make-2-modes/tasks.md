# Tasks: Configurable Test Backend Modes (Enterprise & Community)

**Feature**: 035-make-2-modes
**Input**: Design documents from `/Users/intersystems-community/ws/rag-templates/specs/035-make-2-modes/`
**Prerequisites**: plan.md, research.md, data-model.md, contracts/, quickstart.md

## Execution Flow (main)
```
1. Load plan.md ✓
   → Tech stack: Python 3.12, pytest, iris-devtools, IRIS database
   → Structure: Single Python project (iris_rag/)
2. Load design documents ✓
   → data-model.md: 5 entities (BackendMode, IRISEdition, BackendConfiguration, ConnectionPool, IrisDevToolsBridge)
   → contracts/: backend_config_contract.yaml (8 test scenarios)
   → research.md: 4 technical decisions
3. Generate 30 tasks by category
4. Apply task rules (TDD, dependencies, parallel markers)
5. Number tasks T001-T030
6. Validate completeness ✓
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Paths relative to repository root: `/Users/intersystems-community/ws/rag-templates/`

## Phase 3.1: Setup & Prerequisites

- [x] **T001** Verify iris-devtools dependency at ../iris-devtools exists and is importable
  - Check: `../iris-devtools/iris_devtools/` exists
  - Check: Can import `from iris_devtools.containers import IRISContainer`
  - Error: If missing, fail with clear message per FR-007
  - File: N/A (validation task)

- [x] **T002** Create iris_rag/testing/ package structure
  - Create: `iris_rag/testing/__init__.py`
  - Create: `iris_rag/testing/backend_manager.py` (stub)
  - Create: `iris_rag/testing/iris_devtools_bridge.py` (stub)
  - Create: `iris_rag/testing/validators.py` (stub)
  - Create: `iris_rag/testing/connection_pool.py` (stub)
  - File: `iris_rag/testing/`

- [x] **T003** Create .specify/config/backend_modes.yaml configuration file
  - Create config file with schema from quickstart.md
  - Default: `backend_mode: community`
  - Optional: `iris_devtools_path: ../iris-devtools`
  - File: `.specify/config/backend_modes.yaml`

- [x] **T004** [P] Add pytest markers to pytest.ini
  - Add: `@pytest.mark.requires_backend_mode` marker
  - Update marker descriptions
  - File: `pytest.ini`

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

- [x] **T005** [P] Contract test: Backend configuration loading (FR-001, FR-002, FR-009)
  - Test file: `tests/contract/test_backend_mode_config.py`
  - Test scenarios from `contracts/backend_config_contract.yaml`:
    * `test_load_from_environment_variable()` - env var takes precedence
    * `test_load_from_config_file()` - config file when no env var
    * `test_load_default()` - defaults to COMMUNITY
    * `test_invalid_mode_value()` - raises ConfigurationError
  - Must import from `iris_rag.testing.backend_manager`
  - Tests MUST FAIL (no implementation yet)
  - File: `tests/contract/test_backend_mode_config.py`

- [x] **T006** [P] Contract test: Backend configuration validation (FR-008, FR-009)
  - Test file: `tests/contract/test_backend_mode_config.py` (same file as T005)
  - Test scenarios:
    * `test_validate_matching_edition()` - validation passes when match
    * `test_validate_mismatched_edition()` - raises EditionMismatchError
    * `test_validate_missing_iris_devtools()` - raises IrisDevtoolsMissingError
  - Tests MUST FAIL (no implementation yet)
  - File: `tests/contract/test_backend_mode_config.py`

- [x] **T007** [P] Contract test: Backend mode logging (FR-012)
  - Test file: `tests/contract/test_backend_mode_config.py` (same file as T005)
  - Test scenario:
    * `test_log_mode_at_session_start()` - logs "Backend mode: community (source: environment)"
  - Use pytest caplog fixture
  - Tests MUST FAIL (no implementation yet)
  - File: `tests/contract/test_backend_mode_config.py`

- [x] **T008** [P] Contract test: IRIS edition detection (FR-008)
  - Test file: `tests/contract/test_edition_detection.py`
  - Test scenarios:
    * `test_detect_community_edition()` - detects COMMUNITY from SQL query
    * `test_detect_enterprise_edition()` - detects ENTERPRISE from SQL query
    * `test_detection_failure()` - raises EditionDetectionError on SQL failure
  - Mock database connection for unit-style contract tests
  - Tests MUST FAIL (no implementation yet)
  - File: `tests/contract/test_edition_detection.py`

- [x] **T009** [P] Contract test: iris-devtools bridge operations (FR-006, FR-013)
  - Test file: `tests/contract/test_iris_devtools_integration.py`
  - Test scenarios:
    * `test_import_iris_devtools()` - successfully imports from ../iris-devtools
    * `test_start_community_container()` - starts Community Edition container
    * `test_start_enterprise_container()` - starts Enterprise Edition container
    * `test_reset_schema()` - resets database schema
    * `test_validate_connection()` - validates connection health
    * `test_check_health()` - returns health metrics
  - Mock iris-devtools imports for contract testing
  - Tests MUST FAIL (no implementation yet)
  - File: `tests/contract/test_iris_devtools_integration.py`

- [x] **T010** [P] Contract test: iris-devtools missing error (FR-007)
  - Test file: `tests/contract/test_iris_devtools_integration.py` (same as T009)
  - Test scenario:
    * `test_iris_devtools_unavailable()` - raises IrisDevtoolsMissingError with clear message
  - Mock path.exists() to return False
  - Tests MUST FAIL (no implementation yet)
  - File: `tests/contract/test_iris_devtools_integration.py`

- [x] **T011** [P] Contract test: Connection pool limits (FR-003, FR-011)
  - Test file: `tests/contract/test_connection_pooling.py`
  - Test scenarios:
    * `test_community_mode_single_connection()` - max_connections = 1
    * `test_enterprise_mode_unlimited_connections()` - max_connections = 999
    * `test_connection_pool_timeout()` - raises ConnectionPoolTimeout when limit exceeded
    * `test_acquire_and_release()` - connection lifecycle works
  - Mock threading.Semaphore for deterministic testing
  - Tests MUST FAIL (no implementation yet)
  - File: `tests/contract/test_connection_pooling.py`

- [x] **T012** [P] Contract test: Execution strategies (FR-004, FR-005)
  - Test file: `tests/contract/test_execution_strategies.py`
  - Test scenarios:
    * `test_community_mode_sequential_strategy()` - execution_strategy = SEQUENTIAL
    * `test_enterprise_mode_parallel_strategy()` - execution_strategy = PARALLEL
  - Verify BackendConfiguration.execution_strategy property
  - Tests MUST FAIL (no implementation yet)
  - File: `tests/contract/test_execution_strategies.py`

- [x] **T013** [P] Integration test: Community mode end-to-end (NFR-002)
  - Test file: `tests/integration/test_community_mode_execution.py`
  - Test scenario:
    * `test_community_mode_prevents_license_errors()` - run 10 sequential tests, verify >95% success
    * `test_community_mode_blocks_parallel_execution()` - verify connection limit enforced
  - Requires live IRIS Community Edition database
  - Mark: `@pytest.mark.requires_database`
  - Tests MUST FAIL (no implementation yet)
  - File: `tests/integration/test_community_mode_execution.py`

- [x] **T014** [P] Integration test: Enterprise mode end-to-end (NFR-003)
  - Test file: `tests/integration/test_enterprise_mode_execution.py`
  - Test scenario:
    * `test_enterprise_mode_allows_parallel_execution()` - run 10 parallel tests successfully
    * `test_enterprise_mode_no_performance_degradation()` - verify no artificial throttling
  - Requires live IRIS Enterprise Edition database
  - Mark: `@pytest.mark.requires_database`
  - Tests MUST FAIL (no implementation yet)
  - File: `tests/integration/test_enterprise_mode_execution.py`

- [x] **T015** [P] Integration test: Backend mode switching (NFR-001)
  - Test file: `tests/integration/test_mode_switching.py`
  - Test scenario:
    * `test_mode_switch_via_environment_variable()` - change env var, verify mode changes
    * `test_mode_immutable_during_session()` - verify mode locked after session start
  - Tests MUST FAIL (no implementation yet)
  - File: `tests/integration/test_mode_switching.py`

## Phase 3.3: Core Implementation (ONLY after tests are failing)

- [ ] **T016** [P] Implement BackendMode enum in iris_rag/config/backend_modes.py
  - Create enum with COMMUNITY and ENTERPRISE values
  - Add `from_string()` classmethod for case-insensitive parsing
  - Raise ConfigurationError for invalid values
  - Reference: data-model.md § BackendMode
  - Makes T005 partial pass (enum creation only)
  - File: `iris_rag/config/backend_modes.py`

- [ ] **T017** [P] Implement IRISEdition enum in iris_rag/testing/validators.py
  - Create enum with COMMUNITY and ENTERPRISE values
  - Add `detect()` classmethod that queries `SELECT $SYSTEM.License.LicenseType()`
  - Raise EditionDetectionError on failure
  - Reference: data-model.md § IRISEdition, research.md § 2
  - Makes T008 pass
  - File: `iris_rag/testing/validators.py`

- [ ] **T018** [P] Implement ConfigSource and ExecutionStrategy enums
  - ConfigSource: ENVIRONMENT, CONFIG_FILE, DEFAULT
  - ExecutionStrategy: SEQUENTIAL, PARALLEL
  - Reference: data-model.md § BackendConfiguration
  - File: `iris_rag/config/backend_modes.py`

- [ ] **T019** Implement BackendConfiguration dataclass
  - Frozen dataclass with: mode, source, iris_devtools_path
  - Properties: max_connections, execution_strategy
  - Methods: `load()` classmethod (env var > config > default precedence)
  - Method: `validate(detected_edition)` - check mode matches edition and iris-devtools exists
  - Method: `log_session_start()` - log mode at INFO level
  - Reference: data-model.md § BackendConfiguration, research.md § 4
  - Makes T005, T006, T007 pass
  - File: `iris_rag/config/backend_modes.py`
  - Depends on: T016, T017, T018

- [ ] **T020** [P] Implement IrisDevToolsBridge class
  - Initialize: import iris-devtools from path, store IRISContainer class
  - Method: `start_container(edition)` - start Community or Enterprise container
  - Method: `stop_container()` - stop active container
  - Method: `reset_schema(connection)` - reset database schema
  - Method: `validate_connection(connection)` - SELECT 1 health check
  - Method: `check_health()` - get resource metrics
  - Error: Raise IrisDevtoolsMissingError if import fails
  - Reference: data-model.md § IrisDevToolsBridge, research.md § 1
  - Makes T009, T010 pass
  - File: `iris_rag/testing/iris_devtools_bridge.py`

- [ ] **T021** [P] Implement ConnectionPool class
  - Initialize: create Semaphore(config.max_connections)
  - Method: `acquire_connection(iris_config, timeout=30)` - block if limit reached
  - Method: `release_connection(conn)` - close and release semaphore
  - Method: `get_active_count()` - return active connection count
  - Error: Raise ConnectionPoolTimeout if timeout exceeded
  - Reference: data-model.md § ConnectionPool, research.md § 3
  - Makes T011 pass
  - File: `iris_rag/testing/connection_pool.py`
  - Depends on: T019 (needs BackendConfiguration)

- [ ] **T022** Verify execution_strategy property derivation
  - Verify BackendConfiguration.execution_strategy returns correct value based on mode
  - Community → SEQUENTIAL, Enterprise → PARALLEL
  - Reference: data-model.md § BackendConfiguration
  - Makes T012 pass
  - File: `iris_rag/config/backend_modes.py` (part of T019)
  - Depends on: T019

- [ ] **T023** Implement pytest fixtures in tests/conftest.py
  - Fixture: `backend_mode()` - session-scoped, returns BackendMode
  - Fixture: `backend_configuration()` - session-scoped, returns BackendConfiguration
  - Fixture: `connection_pool(backend_configuration)` - session-scoped, returns ConnectionPool
  - Fixture: `iris_connection(connection_pool, iris_config)` - function-scoped, acquire/release
  - Fixture: `iris_devtools_bridge(backend_configuration)` - session-scoped
  - Add pytest_configure hook to log backend mode at session start
  - Reference: research.md § 3 (pytest integration)
  - Makes T013, T014, T015 pass
  - File: `tests/conftest.py`
  - Depends on: T019, T020, T021

## Phase 3.4: Integration & Validation

- [ ] **T024** Update common/iris_connection_manager.py for mode-aware connections
  - Integrate with ConnectionPool for connection acquisition
  - Respect backend mode connection limits
  - Add logging for connection pool status
  - Reference: plan.md § Project Structure
  - File: `common/iris_connection_manager.py`
  - Depends on: T021

- [ ] **T025** Add Make targets for backend mode selection
  - Target: `test-community` - run tests with IRIS_BACKEND_MODE=community
  - Target: `test-enterprise` - run tests with IRIS_BACKEND_MODE=enterprise
  - Target: `test-mode-switching` - run mode switching integration test
  - Reference: quickstart.md § Make Targets
  - File: `Makefile`

- [ ] **T026** Create error classes hierarchy
  - Base: BackendModeError
  - Subclasses: ConfigurationError, EditionDetectionError, EditionMismatchError
  - Subclasses: IrisDevtoolsError (with IrisDevtoolsMissingError, IrisDevtoolsImportError)
  - Subclasses: ConnectionPoolError (with ConnectionPoolTimeout, ConnectionLimitExceeded)
  - All with actionable error messages
  - Reference: data-model.md § Error Hierarchy
  - File: `iris_rag/testing/exceptions.py`

- [ ] **T027** Validate all contract tests pass
  - Run: `pytest tests/contract/test_backend_mode_config.py -v`
  - Run: `pytest tests/contract/test_edition_detection.py -v`
  - Run: `pytest tests/contract/test_iris_devtools_integration.py -v`
  - Run: `pytest tests/contract/test_connection_pooling.py -v`
  - Run: `pytest tests/contract/test_execution_strategies.py -v`
  - Verify: All tests PASS
  - File: N/A (validation task)
  - Depends on: T016-T026

- [ ] **T028** Run integration tests against Community Edition
  - Prerequisite: Start IRIS Community Edition container
  - Run: `IRIS_BACKEND_MODE=community pytest tests/integration/test_community_mode_execution.py -v`
  - Verify: >95% license error prevention (NFR-002)
  - File: N/A (validation task)
  - Depends on: T027

- [ ] **T029** Run integration tests against Enterprise Edition
  - Prerequisite: Start IRIS Enterprise Edition container
  - Run: `IRIS_BACKEND_MODE=enterprise pytest tests/integration/test_enterprise_mode_execution.py -v`
  - Verify: Parallel execution works, no performance degradation (NFR-003)
  - File: N/A (validation task)
  - Depends on: T027

## Phase 3.5: Polish & Documentation

- [ ] **T030** [P] Update CLAUDE.md with implementation patterns
  - Add: Backend mode configuration examples
  - Add: iris-devtools integration patterns
  - Add: Common troubleshooting scenarios
  - Keep under 150 lines
  - File: `CLAUDE.md`

## Dependencies

### Critical Path
```
T001 (verify iris-devtools)
  ↓
T002 (create package structure)
  ↓
T003-T004 (config files, markers)
  ↓
T005-T015 (ALL contract & integration tests - MUST FAIL)
  ↓
T016-T022 (core implementation - make tests pass)
  ↓
T023 (pytest fixtures)
  ↓
T024-T026 (integration & errors)
  ↓
T027-T029 (validation)
  ↓
T030 (documentation)
```

### Parallel Opportunities

**Phase 3.2 (Tests)**: T005-T015 can ALL run in parallel
```bash
# All are different files, no dependencies
pytest tests/contract/test_backend_mode_config.py &
pytest tests/contract/test_edition_detection.py &
pytest tests/contract/test_iris_devtools_integration.py &
pytest tests/contract/test_connection_pooling.py &
pytest tests/contract/test_execution_strategies.py &
pytest tests/integration/test_community_mode_execution.py &
pytest tests/integration/test_enterprise_mode_execution.py &
pytest tests/integration/test_mode_switching.py &
wait
```

**Phase 3.3 (Implementation)**: T016, T017, T018, T020, T026 can run in parallel
```bash
# Different files
vim iris_rag/config/backend_modes.py       # T016, T018
vim iris_rag/testing/validators.py          # T017
vim iris_rag/testing/iris_devtools_bridge.py # T020
vim iris_rag/testing/exceptions.py          # T026
```

**Sequential Dependencies**:
- T019 depends on T016, T017, T018
- T021 depends on T019
- T022 depends on T019
- T023 depends on T019, T020, T021
- T024 depends on T021
- T027 depends on T016-T026
- T028-T029 depend on T027

## Task Execution Examples

### Launch Phase 3.2 tests in parallel
```bash
# Using Task agent (pseudocode)
for task in T005 T006 T007 T008 T009 T010 T011 T012 T013 T014 T015; do
  Task: "Execute task $task from tasks.md"
done
```

### Sequential core implementation
```bash
# T016-T018 can be parallel
Task: "Execute T016: Implement BackendMode enum"
Task: "Execute T017: Implement IRISEdition enum"
Task: "Execute T018: Implement ConfigSource/ExecutionStrategy enums"

# T019 after T016-T018
Task: "Execute T019: Implement BackendConfiguration dataclass"

# T020, T021 can be parallel (both depend on T019)
Task: "Execute T020: Implement IrisDevToolsBridge"
Task: "Execute T021: Implement ConnectionPool"
```

## Notes

- **TDD Enforcement**: T005-T015 MUST be completed and failing before T016-T023
- **[P] Marker**: Indicates tasks that can run in parallel (different files)
- **Contract Tests**: Must fail initially (no implementation) then pass after core implementation
- **Integration Tests**: Require live IRIS database (Community or Enterprise)
- **Validation Tasks**: T027-T029 verify all requirements met

## Constitutional Alignment

- **III. TDD**: Tests (T005-T015) before implementation (T016-T023) ✓
- **II. Pipeline Validation**: T001 validates iris-devtools, T027-T029 validate setup ✓
- **VI. Explicit Errors**: T026 creates complete error hierarchy ✓
- **VII. Standardized Interfaces**: T020 uses iris-devtools proven patterns ✓

## Success Criteria

- ✅ All 30 tasks completed in order
- ✅ All contract tests pass (T027)
- ✅ Community mode: >95% license error prevention (T028, NFR-002)
- ✅ Enterprise mode: Parallel execution works (T029, NFR-003)
- ✅ No constitutional violations
- ✅ Documentation complete (T030)
