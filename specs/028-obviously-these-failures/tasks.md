# Tasks: Test Infrastructure Resilience and Database Schema Management

**Input**: Design documents from `/Users/tdyar/ws/rag-templates/specs/028-obviously-these-failures/`
**Prerequisites**: plan.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → ✓ Tech stack: Python 3.12, pytest 8.4.1, iris.dbapi
   → ✓ Structure: Single project (tests/ and common/)
2. Load design documents:
   → ✓ data-model.md: 6 entities extracted
   → ✓ contracts/: 3 contract test files found
   → ✓ research.md: 4 technical decisions extracted
3. Generate tasks by category:
   → Setup: 3 tasks
   → Tests: 3 contract test tasks, 5 integration tests
   → Core: 6 implementation tasks (entities + utilities)
   → Integration: 3 tasks (fixtures, plugins, configuration)
   → Polish: 3 tasks (validation, docs, cleanup)
4. Apply task rules:
   → Contract tests [P]: Different files, can run parallel
   → Model tasks [P]: Different entities, independent
   → Sequential: conftest.py (shared file), integration tasks
5. Numbered tasks: T001-T020
6. Dependencies validated
7. Parallel execution examples generated
8. Validation: All contracts have tests ✓
9. SUCCESS: 20 tasks ready for execution
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- All paths are absolute from repository root

## Phase 3.1: Setup and Prerequisites

- [ ] **T001** Create test infrastructure directory structure
  ```
  Create directories:
  - /Users/tdyar/ws/rag-templates/tests/fixtures/
  - /Users/tdyar/ws/rag-templates/tests/utils/
  - /Users/tdyar/ws/rag-templates/tests/fixtures/__init__.py
  - /Users/tdyar/ws/rag-templates/tests/utils/__init__.py

  Validate: Directory structure matches plan.md Section "Project Structure"
  ```

- [ ] **T002** Verify pytest plugins are loadable in pytest.ini
  ```
  Update /Users/tdyar/ws/rag-templates/pytest.ini:
  - Add contract marker to markers section
  - Verify plugins list includes all 3 Feature 026 plugins
  - Add placeholder for contract_test_marker plugin

  Test: Run `pytest --markers | grep contract`
  Acceptance: contract marker appears in output
  ```

- [ ] **T003** Validate IRIS database connectivity and schema
  ```
  Create pre-flight validation script:
  /Users/tdyar/ws/rag-templates/tests/utils/preflight_checks.py

  Implement PreflightChecker with methods:
  - check_iris_connectivity() → connects to localhost:11972
  - check_api_keys() → validates OPENAI_API_KEY in .env
  - check_schema_tables() → lists existing RAG.* tables

  Acceptance: Checker runs in <2 seconds (NFR-003)
  ```

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### Contract Tests (from contracts/ directory)

- [ ] **T004** [P] Contract test for schema validation in tests/contract/test_schema_manager_contract.py
  ```
  Copy contract from specs/028-obviously-these-failures/contracts/schema_manager_contract.py
  to /Users/tdyar/ws/rag-templates/tests/contract/test_schema_manager_contract.py

  Expected: 3 tests FAIL (SchemaValidator not implemented)
  - test_schema_validator_detects_missing_table
  - test_schema_validator_detects_type_mismatch
  - test_schema_validator_detects_missing_column
  ```

- [ ] **T005** [P] Contract test for schema reset in tests/contract/test_schema_reset_contract.py
  ```
  Extract reset tests from schema_manager_contract.py:
  - test_schema_reset_is_idempotent
  - test_schema_reset_completes_under_5_seconds
  - test_schema_reset_handles_nonexistent_tables

  File: /Users/tdyar/ws/rag-templates/tests/contract/test_schema_reset_contract.py
  Expected: 3 tests FAIL (SchemaResetter not implemented)
  ```

- [ ] **T006** [P] Contract test for test fixtures in tests/contract/test_fixtures_contract.py
  ```
  Copy from specs/028-obviously-these-failures/contracts/test_fixtures_contract.py
  to /Users/tdyar/ws/rag-templates/tests/contract/test_fixtures_contract.py

  Expected: Tests FAIL (fixtures not implemented)
  - test_clean_schema_fixture_provides_valid_connection
  - test_clean_schema_fixture_overhead_under_100ms
  ```

- [ ] **T007** [P] Contract test for contract test plugin in tests/contract/test_contract_plugin_contract.py
  ```
  Copy from specs/028-obviously-these-failures/contracts/contract_tests_contract.py
  to /Users/tdyar/ws/rag-templates/tests/contract/test_contract_plugin_contract.py

  Expected: Tests FAIL (plugin not implemented)
  - test_contract_test_marked_as_xfail_when_failing
  - test_contract_test_failure_message_indicates_future_feature
  ```

### Integration Tests (from quickstart.md scenarios)

- [ ] **T008** [P] Integration test for fresh database scenario
  ```
  File: /Users/tdyar/ws/rag-templates/tests/integration/test_schema_reset_integration.py

  Test: test_fresh_database_schema_creation()
  Given: IRIS running, no RAG tables exist
  When: Schema reset executed
  Then: All 4 tables created (SourceDocuments, DocumentChunks, Entities, Relationships)

  Expected: FAIL until schema reset implemented
  ```

- [ ] **T009** [P] Integration test for stale schema auto-reset
  ```
  File: /Users/tdyar/ws/rag-templates/tests/integration/test_schema_validation_integration.py

  Test: test_stale_schema_detected_and_reset()
  Given: Table exists with wrong schema (missing column)
  When: Schema validator runs
  Then: Mismatch detected, reset triggered, schema corrected

  Expected: FAIL until validator + reset implemented
  ```

- [ ] **T010** [P] Integration test for test isolation and cleanup
  ```
  File: /Users/tdyar/ws/rag-templates/tests/integration/test_cleanup_integration.py

  Test: test_cleanup_after_test_failure()
  Given: Test inserts data then fails
  When: Next test runs
  Then: Database is clean (no leftover data)

  Expected: FAIL until cleanup handlers implemented
  ```

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### Data Models (from data-model.md entities)

- [ ] **T011** [P] SchemaDefinition and ColumnDefinition models
  ```
  File: /Users/tdyar/ws/rag-templates/tests/utils/schema_models.py

  Implement dataclasses:
  - SchemaDefinition (table_name, schema_name, columns, indexes, version)
  - ColumnDefinition (name, data_type, is_nullable, is_primary_key)
  - IndexDefinition (name, columns, is_unique)

  Validation:
  - table_name is valid SQL identifier
  - version follows semver
  - primary_key columns have is_nullable=False

  Acceptance: Models pass validation tests
  ```

- [ ] **T012** [P] SchemaValidationResult and SchemaMismatch models
  ```
  File: /Users/tdyar/ws/rag-templates/tests/utils/schema_models.py (extend)

  Implement:
  - SchemaValidationResult (is_valid, table_name, mismatches, missing_tables)
  - SchemaMismatch (column_name, expected_type, actual_type, issue, severity)

  Usage pattern per data-model.md:
  result = validator.validate_schema("SourceDocuments")
  if not result.is_valid:
      for mismatch in result.mismatches:
          log.error(f"{mismatch.column_name}: {mismatch.issue}")

  Acceptance: Result objects serialize to dict for logging
  ```

- [ ] **T013** [P] TestDatabaseState model
  ```
  File: /Users/tdyar/ws/rag-templates/tests/fixtures/database_state.py

  Implement:
  - TestDatabaseState class
  - Attributes: connection, cleanup_functions, is_clean, schema_version
  - Methods: add_cleanup(func), execute_cleanup(), mark_dirty()
  - State transitions: created → used → cleaned → disposed

  Acceptance: State tracking works, cleanup functions execute in order
  ```

### Core Utilities

- [ ] **T014** SchemaValidator implementation
  ```
  File: /Users/tdyar/ws/rag-templates/tests/utils/schema_validator.py

  Implement SchemaValidator class:
  - validate_schema(table_name) → SchemaValidationResult
  - Uses INFORMATION_SCHEMA queries per research.md
  - Detects: missing tables, type mismatches, missing/extra columns
  - Includes SQLCODE in error messages (FR-021)

  Query pattern from research.md:
  SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
  FROM INFORMATION_SCHEMA.COLUMNS
  WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = ?

  Acceptance: T004 contract tests pass
  Dependencies: T011, T012 (models)
  ```

- [ ] **T015** SchemaResetter implementation
  ```
  File: /Users/tdyar/ws/rag-templates/tests/fixtures/schema_reset.py

  Implement SchemaResetter class:
  - reset_schema() → drops and recreates 4 RAG tables
  - Uses IF EXISTS for idempotency (FR-003)
  - Logs operations with timestamps (FR-019)
  - Completes in <5 seconds (NFR-001)

  DROP/CREATE sequence:
  1. DROP TABLE IF EXISTS RAG.Relationships
  2. DROP TABLE IF EXISTS RAG.Entities
  3. DROP TABLE IF EXISTS RAG.DocumentChunks
  4. DROP TABLE IF EXISTS RAG.SourceDocuments
  5. CREATE TABLE RAG.SourceDocuments (...)
  6. CREATE TABLE RAG.DocumentChunks (...)
  7. CREATE TABLE RAG.Entities (...)
  8. CREATE TABLE RAG.Relationships (...)

  Acceptance: T005 contract tests pass, completes in ~1.8s
  Dependencies: Uses common/iris_connector.py
  ```

- [ ] **T016** Database cleanup handlers
  ```
  File: /Users/tdyar/ws/rag-templates/tests/fixtures/database_cleanup.py

  Implement:
  - CleanupHandler class
  - Methods: cleanup_documents(), cleanup_entities(), cleanup_all()
  - Handles partial data (FR-009)
  - Executes in <100ms per class (NFR-002)

  Pattern:
  def cleanup_all(conn):
      cursor = conn.cursor()
      cursor.execute("DELETE FROM RAG.Relationships WHERE test_run_id = ?")
      cursor.execute("DELETE FROM RAG.Entities WHERE test_run_id = ?")
      cursor.execute("DELETE FROM RAG.DocumentChunks WHERE test_run_id = ?")
      cursor.execute("DELETE FROM RAG.SourceDocuments WHERE test_run_id = ?")
      conn.commit()

  Acceptance: Cleanup completes in <50ms (target from research.md)
  Dependencies: T013 (TestDatabaseState)
  ```

## Phase 3.4: Integration

- [ ] **T017** Pytest fixtures in conftest.py
  ```
  File: /Users/tdyar/ws/rag-templates/tests/conftest.py (modify existing)

  Add fixtures:

  @pytest.fixture(scope="session", autouse=True)
  def validate_test_environment():
      """Pre-flight checks before test session."""
      from tests.utils.preflight_checks import PreflightChecker
      checker = PreflightChecker()
      results = checker.run_all_checks()
      if not all(r.passed for r in results):
          pytest.exit("Pre-flight checks failed")

  @pytest.fixture(scope="session")
  def iris_schema_manager():
      """Provides schema validator and resetter."""
      from tests.utils.schema_validator import SchemaValidator
      from tests.fixtures.schema_reset import SchemaResetter

      validator = SchemaValidator()
      result = validator.validate_schema("SourceDocuments")
      if not result.is_valid:
          resetter = SchemaResetter()
          resetter.reset_schema()

      return validator

  @pytest.fixture(scope="class")
  def database_with_clean_schema(request, iris_schema_manager):
      """Provides clean database for test class."""
      from tests.fixtures.database_state import TestDatabaseState
      from tests.fixtures.database_cleanup import CleanupHandler

      conn = get_iris_connection()
      state = TestDatabaseState(conn)

      def cleanup():
          CleanupHandler().cleanup_all(conn)

      request.addfinalizer(cleanup)
      yield conn

  Acceptance: T006 fixture tests pass
  Dependencies: T003, T014, T015, T016
  ```

- [ ] **T018** Contract test marker plugin
  ```
  File: /Users/tdyar/ws/rag-templates/tests/plugins/contract_test_marker.py

  Implement pytest plugin:
  - Hook: pytest_runtest_makereport(item, call)
  - If test has @pytest.mark.contract and fails:
    - Check if ImportError (feature not implemented)
    - Reclassify as "xfail" (expected failure)
    - Add message: "Contract test - feature not implemented"

  Pattern from research.md:
  def pytest_runtest_makereport(item, call):
      if "contract" in item.keywords and call.excinfo:
          if isinstance(call.excinfo.value, ImportError):
              # Feature not implemented - expected failure
              item.outcome = "xfailed"
              item.wasxfail = "Contract test - feature not implemented"

  Acceptance: T007 plugin tests pass, MCP tests show as xfail
  Dependencies: None (pure pytest plugin)
  ```

- [ ] **T019** Update pytest.ini configuration
  ```
  File: /Users/tdyar/ws/rag-templates/pytest.ini (modify existing)

  Changes:
  1. Add to plugins section:
     tests.plugins.contract_test_marker

  2. Verify contract marker exists (should be there from T002)

  3. Add test session hooks configuration (if needed)

  Test: Run `pytest --trace-config | grep contract_test_marker`
  Acceptance: Plugin loads successfully
  Dependencies: T018 (plugin exists)
  ```

## Phase 3.5: Polish and Validation

- [ ] **T020** [P] End-to-end validation test
  ```
  File: /Users/tdyar/ws/rag-templates/tests/e2e/test_infrastructure_e2e.py

  Test: test_full_test_suite_medical_grade_quality()
  1. Run pre-flight checks
  2. Validate schema
  3. Run sample from each test category (contract, integration, e2e, unit)
  4. Verify:
     - 0 schema errors
     - Contract tests marked as xfail (not ERROR)
     - Cleanup executes after each class
     - Total overhead <8 seconds

  Success criteria from spec.md:
  - All 771 tests executable
  - 0 schema errors (down from 69)
  - 47 xfail (down from 47 ERROR)
  - ~650-700 passing (up from 476)

  Acceptance: Medical-grade quality metrics achieved
  Dependencies: ALL previous tasks
  ```

- [ ] **T021** Update common/database_schema_manager.py with reusable patterns
  ```
  File: /Users/tdyar/ws/rag-templates/common/database_schema_manager.py (modify)

  Add methods that can be reused beyond tests:
  - validate_table_schema(table_name, expected_schema) → ValidationResult
  - reset_table_safe(table_name) → uses IF EXISTS pattern
  - get_table_info(table_name) → query INFORMATION_SCHEMA

  Rationale: Make schema utilities available to production code (Constitutional Principle VII)

  Acceptance: Methods documented, unit tested
  Dependencies: T014, T015 (extract patterns from test utilities)
  ```

- [ ] **T022** Documentation and cleanup
  ```
  Files to update:
  1. /Users/tdyar/ws/rag-templates/README.md
     - Add section on test infrastructure
     - Document pytest fixtures and markers

  2. /Users/tdyar/ws/rag-templates/CLAUDE.md
     - Already updated by /plan command ✓

  3. /Users/tdyar/ws/rag-templates/Makefile
     - Add target: make test-reset-schema
     - Add target: make test-preflight-check

  Acceptance: Developers can find fixture documentation
  Dependencies: None (documentation only)
  ```

## Dependencies Graph

```
Setup Layer (T001-T003)
  ↓
Contract Tests (T004-T007) [P] - MUST FAIL
  ↓
Integration Tests (T008-T010) [P] - MUST FAIL
  ↓
Models (T011-T013) [P]
  ↓
Core Utilities (T014-T016)
  ├─ T014 (SchemaValidator) ← T011, T012
  ├─ T015 (SchemaResetter)
  └─ T016 (CleanupHandlers) ← T013
  ↓
Integration (T017-T019)
  ├─ T017 (conftest.py) ← T003, T014, T015, T016
  ├─ T018 (plugin) - independent
  └─ T019 (pytest.ini) ← T018
  ↓
Polish (T020-T022) [P where applicable]
  ├─ T020 (E2E validation) ← ALL
  ├─ T021 (common/ utils) ← T014, T015
  └─ T022 (docs) - independent
```

## Parallel Execution Examples

### Phase 3.2: All Contract Tests in Parallel
```bash
# Launch all 4 contract test tasks together (different files):
pytest tests/contract/test_schema_manager_contract.py & \
pytest tests/contract/test_schema_reset_contract.py & \
pytest tests/contract/test_fixtures_contract.py & \
pytest tests/contract/test_contract_plugin_contract.py &
wait

# Expected: All 4 test files FAIL (implementations don't exist yet)
```

### Phase 3.2: All Integration Tests in Parallel
```bash
# Launch all 3 integration test tasks together:
pytest tests/integration/test_schema_reset_integration.py & \
pytest tests/integration/test_schema_validation_integration.py & \
pytest tests/integration/test_cleanup_integration.py &
wait
```

### Phase 3.3: All Model Tasks in Parallel
```bash
# T011-T013 can run in parallel (different aspects of data model):
# Create all model files simultaneously
```

### Phase 3.5: Documentation Tasks in Parallel
```bash
# T022 documentation updates can be done concurrently
# (different files: README.md, Makefile)
```

## Task Validation Checklist

- [x] All contracts have corresponding tests (T004-T007)
- [x] All entities have model tasks (T011-T013)
- [x] All tests come before implementation (Phase 3.2 before 3.3)
- [x] Parallel tasks are truly independent (checked [P] markers)
- [x] Each task specifies exact file path (all paths included)
- [x] No [P] task modifies same file as another [P] task
- [x] TDD order enforced (tests MUST FAIL before implementation)
- [x] Performance requirements validated (NFR-001 to NFR-005)
- [x] Dependencies tracked (dependency graph included)

## Success Criteria

After completing all 20 tasks, verify:

1. **Functional Requirements**:
   - ✓ Schema validation detects mismatches (FR-001, FR-004)
   - ✓ Automatic schema reset works (FR-002, FR-003)
   - ✓ Test isolation prevents pollution (FR-006, FR-007, FR-010)
   - ✓ Contract tests properly marked (FR-011, FR-012, FR-014)
   - ✓ Pre-flight checks validate prerequisites (FR-015, FR-016)

2. **Non-Functional Requirements**:
   - ✓ Schema reset <5 seconds (NFR-001) - target: 1.8s
   - ✓ Test isolation overhead <100ms (NFR-002) - target: 50ms
   - ✓ Pre-flight checks <2 seconds (NFR-003)
   - ✓ Error messages include SQLCODE (NFR-004)

3. **Test Results**:
   - ✓ 0 schema errors (down from 69)
   - ✓ 0 contract test ERRORs (47 properly marked as xfail)
   - ✓ ~650-700 passing tests (up from 476)
   - ✓ Medical-grade test reliability achieved

## Notes

- **TDD Compliance**: Tests T004-T010 MUST be written and MUST FAIL before implementing T011-T019
- **[P] Markers**: Only applied to tasks with different target files and no dependencies
- **Commit Strategy**: Commit after each task completes
- **Idempotency**: All schema operations safe to run multiple times (FR-003)
- **Audit Trail**: All schema operations logged with timestamps (FR-019)
- **Constitutional Compliance**: Uses standardized database interfaces (Principle VII)

## Estimated Timeline

- Setup (T001-T003): ~1 hour
- Contract Tests (T004-T007): ~2 hours (parallel)
- Integration Tests (T008-T010): ~1.5 hours (parallel)
- Models (T011-T013): ~1 hour (parallel)
- Core Implementation (T014-T016): ~4 hours (sequential)
- Integration (T017-T019): ~2 hours (sequential)
- Polish (T020-T022): ~1.5 hours (partial parallel)

**Total**: ~13 hours of focused development

**With Parallelization**: ~9 hours wall-clock time
