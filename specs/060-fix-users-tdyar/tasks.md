# Tasks: Fix Critical Bugs in v0.5.3 (Connection API + Schema Initialization)

**Feature**: 060-fix-users-tdyar
**Input**: Design documents from `/specs/060-fix-users-tdyar/`
**Prerequisites**: plan.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

---

## Execution Summary

This task list addresses **two critical bugs** in iris-vector-rag v0.5.3:

**Bug 1 (CRITICAL)**: Replace non-existent `iris.connect()` call at `iris_dbapi_connector.py:210` with correct `iris.createConnection()` API to restore database connectivity (currently breaks all connections with AttributeError).

**Bug 2 (HIGH)**: Add automatic detection and creation of iris-vector-graph tables during pipeline initialization to eliminate silent PPR failures.

**Total Tasks**: 32
- Setup: 3 tasks
- Contract Tests (Bug 1): 4 tasks [P]
- Contract Tests (Bug 2): 6 tasks [P]
- Implementation (Bug 1): 3 tasks
- Implementation (Bug 2): 5 tasks
- Integration & Validation: 7 tasks
- Documentation: 4 tasks

**TDD Approach**: All contract tests (T004-T013) MUST be written and MUST FAIL before implementation (T014-T021).

---

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- All file paths are absolute from repository root

---

## Phase 3.1: Setup

### T001 - Verify Development Environment ✅
**Description**: Verify Python environment and IRIS database are ready for Bug 1 & 2 fixes
**Files**: `.env`, `docker-compose.yml`
**Steps**:
1. Verify Python 3.10+ installed
2. Verify intersystems-irispython>=5.1.2 installed
3. Start IRIS database: `docker-compose up -d iris`
4. Verify database connectivity with existing v0.5.2 connection code
5. Check iris-vector-graph package status (optional dependency)

**Acceptance**:
- Environment variables loaded
- IRIS database running and accessible
- Can connect using v0.5.2 fallback methods

---

### T002 - Create Test Fixture Infrastructure
**Description**: Set up test fixtures for Bug 1 connection tests and Bug 2 schema tests
**Files**: `tests/fixtures/connection_fixtures.py`, `tests/fixtures/schema_fixtures.py`
**Steps**:
1. Create connection test fixtures (valid configs, invalid configs, mock connections)
2. Create schema test fixtures (mock iris-vector-graph detection, mock table creation)
3. Create pytest conftest.py entries for fixture discovery

**Acceptance**:
- Fixtures available for contract tests
- Mock objects ready for TDD

---

### T003 [P] - Configure Linting and Pre-commit
**Description**: Ensure linting tools are configured for Bug 1 & 2 code changes
**Files**: `.pre-commit-config.yaml`, `pyproject.toml`
**Steps**:
1. Verify black, isort, flake8, mypy configurations exist
2. Run linters on `iris_vector_rag/common/iris_dbapi_connector.py` (Bug 1 target)
3. Run linters on `iris_vector_rag/storage/schema_manager.py` (Bug 2 target)

**Acceptance**:
- Linters pass on existing code
- Ready to detect issues in new code

---

## Phase 3.2: Contract Tests - Bug 1 (Connection API Fix)
⚠️ **CRITICAL: These tests MUST be written and MUST FAIL before ANY Bug 1 implementation**

### T004 [P] - Contract Test: Connection Uses Correct API
**Description**: Write contract test verifying iris.createConnection() is used (not iris.connect())
**File**: `tests/contract/test_connection_api_fix.py`
**Contract**: Contract 1 from `contracts/connection_api_contracts.md`
**Steps**:
1. Write `test_connection_uses_correct_api()`:
   - Given: valid IRIS connection parameters
   - When: get_iris_connection() is called
   - Then: iris.createConnection() is used (verified via mock/inspection)
   - And: no AttributeError is raised
2. Write `test_no_attribute_error_on_connection()`:
   - Given: valid IRIS connection parameters
   - When: get_iris_connection() is called
   - Then: no AttributeError about missing 'connect' method

**Expected Result**: Tests FAIL (iris.connect() still exists at line 210)
**Acceptance**:
- 2 failing tests
- Clear failure messages indicating iris.connect() is the problem

---

### T005 [P] - Contract Test: Connection Error Messages
**Description**: Write contract test verifying clear ConnectionError messages (not AttributeError)
**File**: `tests/contract/test_connection_api_fix.py`
**Contract**: Contracts 1 & 6 from `contracts/connection_api_contracts.md`
**Steps**:
1. Write `test_connection_error_message_clarity()`:
   - Given: invalid IRIS connection parameters (wrong port)
   - When: get_iris_connection() is called
   - Then: ConnectionError raised (not AttributeError)
   - And: error message indicates connection failure
2. Write `test_error_message_format()`:
   - Test various failure scenarios
   - Verify error messages follow standard format
   - Verify no AttributeError about iris.connect()

**Expected Result**: Tests FAIL (AttributeError currently raised)
**Acceptance**:
- 2 failing tests
- Error messages document expected format

---

### T006 [P] - Contract Test: Connection Manager Integration
**Description**: Write contract test verifying ConnectionManager works after line 210 fix
**File**: `tests/contract/test_connection_manager_integration.py`
**Contract**: Contract 3 from `contracts/connection_api_contracts.md`
**Steps**:
1. Write `test_connection_manager_creates_connections()`:
   - Given: ConnectionManager initialized with valid IRIS config
   - When: connection is created
   - Then: connection succeeds without AttributeError
2. Write `test_iris_dbapi_connector_line_210_fixed()`:
   - Read iris_dbapi_connector.py source
   - Verify line 210 contains "iris.createConnection"
   - Verify no calls to iris.connect() exist in file

**Expected Result**: Tests FAIL (line 210 still uses iris.connect())
**Acceptance**:
- 2 failing tests
- Code inspection test clearly shows line 210 problem

---

### T007 [P] - Contract Test: FHIR-AI Test Suite Compatibility
**Description**: Write contract tests simulating FHIR-AI test requirements
**File**: `tests/contract/test_fhir_ai_compatibility.py`
**Contract**: Contract 4 from `contracts/connection_api_contracts.md`
**Steps**:
1. Write tests for 6 FHIR-AI scenarios:
   - `test_configuration_manager_passes()` - Should pass (was passing in v0.5.3)
   - `test_connection_manager_passes()` - Should FAIL (Bug 1 blocks)
   - `test_iris_vector_store_passes()` - Should FAIL (depends on connection)
   - `test_schema_manager_passes()` - Should FAIL (depends on connection)
   - `test_environment_variables_pass()` - Should pass (was passing in v0.5.3)
   - `test_document_model_passes()` - Should pass (was passing in v0.5.3)

**Expected Result**: 3/6 tests pass (matching current v0.5.3 state)
**Acceptance**:
- 6 contract tests written
- 3 passing (ConfigurationManager, Environment, Document)
- 3 failing (ConnectionManager, IRISVectorStore, SchemaManager)

---

## Phase 3.3: Contract Tests - Bug 2 (Schema Initialization)
⚠️ **CRITICAL: These tests MUST be written and MUST FAIL before ANY Bug 2 implementation**

### T008 [P] - Contract Test: Package Detection
**Description**: Write contract test for iris-vector-graph package detection
**File**: `tests/contract/test_graph_schema_detection.py`
**Contract**: Contract 1 from `contracts/schema_manager_contracts.md`
**Steps**:
1. Write `test_iris_vector_graph_detection()`:
   - Given: iris-vector-graph installed
   - When: _detect_iris_vector_graph() called
   - Then: returns True
2. Write `test_iris_vector_graph_not_installed()`:
   - Given: iris-vector-graph NOT installed
   - When: _detect_iris_vector_graph() called
   - Then: returns False
3. Write 4 additional detection tests from contract

**Expected Result**: Tests FAIL (_detect_iris_vector_graph method doesn't exist)
**Acceptance**:
- 6 failing tests for package detection
- Tests cover installed, not installed, edge cases

---

### T009 [P] - Contract Test: Graph Tables Initialization
**Description**: Write contract test for automatic graph table creation
**File**: `tests/contract/test_graph_schema_initialization.py`
**Contract**: Contract 2 from `contracts/schema_manager_contracts.md`
**Steps**:
1. Write `test_graph_tables_created_when_package_installed()`:
   - Given: iris-vector-graph installed
   - When: ensure_iris_vector_graph_tables() called
   - Then: all 4 tables created
   - And: total_time_seconds < 5.0
2. Write `test_graph_tables_skipped_when_package_not_installed()`:
   - Given: iris-vector-graph NOT installed
   - When: ensure_iris_vector_graph_tables() called
   - Then: no tables created
3. Write 5 additional initialization tests from contract

**Expected Result**: Tests FAIL (ensure_iris_vector_graph_tables method doesn't exist)
**Acceptance**:
- 7 failing tests for table initialization
- Tests cover package installed, not installed, idempotency, partial failures

---

### T010 [P] - Contract Test: Prerequisite Validation
**Description**: Write contract test for iris-vector-graph prerequisite validation
**File**: `tests/contract/test_graph_schema_validation.py`
**Contract**: Contract 3 from `contracts/schema_manager_contracts.md`
**Steps**:
1. Write `test_prerequisite_validation_all_met()`:
   - Given: iris-vector-graph installed and all tables exist
   - When: validate_graph_prerequisites() called
   - Then: returns ValidationResult(is_valid=True)
2. Write `test_prerequisite_validation_package_missing()`:
   - Given: iris-vector-graph NOT installed
   - When: validate_graph_prerequisites() called
   - Then: returns ValidationResult(is_valid=False, error about package)
3. Write 7 additional validation tests from contract

**Expected Result**: Tests FAIL (validate_graph_prerequisites method doesn't exist)
**Acceptance**:
- 9 failing tests for prerequisite validation
- Tests cover all prerequisites met, package missing, tables missing

---

### T011 [P] - Contract Test: Data Structure Invariants
**Description**: Write contract tests for InitializationResult and ValidationResult data structures
**File**: `tests/contract/test_graph_schema_data_structures.py`
**Contract**: Data Structure Contracts from `contracts/schema_manager_contracts.md`
**Steps**:
1. Write `test_initialization_result_invariants()`:
   - Verify: If package_detected=False, tables_created is empty dict
   - Verify: len(tables_created) == len(tables_attempted)
   - Verify: error_messages keys are subset of tables_attempted
2. Write `test_validation_result_invariants()`:
   - Verify: If package_installed=False, is_valid must be False
   - Verify: If is_valid=True, missing_tables must be empty
   - Verify: If is_valid=False, error_message must be non-empty

**Expected Result**: Tests FAIL (data structures don't exist yet)
**Acceptance**:
- 2 failing tests for data structure invariants
- Tests document required data structure behavior

---

### T012 [P] - Contract Test: Error Message Clarity
**Description**: Write contract test verifying clear error messages for missing tables
**File**: `tests/contract/test_graph_schema_validation.py`
**Contract**: Contract 4 from `contracts/schema_manager_contracts.md`
**Steps**:
1. Write `test_clear_error_when_tables_missing()`:
   - Given: iris-vector-graph installed but tables missing
   - When: validate_graph_prerequisites() called
   - Then: error_message clearly lists missing tables
   - And: error_message suggests remediation

**Expected Result**: Tests FAIL (method doesn't exist)
**Acceptance**:
- 1 failing test
- Test documents expected error message format

---

### T013 [P] - Contract Test: Performance Requirements
**Description**: Write contract tests for initialization and validation performance
**File**: `tests/contract/test_graph_schema_performance.py`
**Contract**: Performance Contracts from `contracts/schema_manager_contracts.md`
**Steps**:
1. Write `test_initialization_performance()`:
   - Given: iris-vector-graph installed
   - When: ensure_iris_vector_graph_tables() called on empty database
   - Then: total_time_seconds < 5.0
2. Write `test_validation_performance()`:
   - Given: iris-vector-graph installed and tables exist
   - When: validate_graph_prerequisites() called
   - Then: execution completes in < 1.0 seconds

**Expected Result**: Tests FAIL (methods don't exist)
**Acceptance**:
- 2 failing tests
- Performance thresholds documented

---

## Phase 3.4: Implementation - Bug 1 (Connection API Fix)
✅ **Prerequisites**: All Bug 1 contract tests (T004-T007) written and FAILING

### T014 - Fix iris.connect() at Line 210 ✅
**Description**: Replace non-existent iris.connect() with correct iris.createConnection() API
**File**: `iris_vector_rag/common/iris_dbapi_connector.py`
**Research Decision**: Decision #1 from `research.md` - Use iris.createConnection()
**Steps**:
1. Open `iris_vector_rag/common/iris_dbapi_connector.py`
2. Navigate to line 210
3. Replace:
   ```python
   # BEFORE (v0.5.3 - BROKEN):
   conn = iris.connect(host, port, namespace, user, password)
   ```
   With:
   ```python
   # AFTER (v0.5.4 - FIXED):
   conn = iris.createConnection(host, port, namespace, user, password)
   ```
4. Preserve SSL configuration intent from v0.5.3 (no SSL parameter changes)
5. Update docstring if necessary
6. Run contract tests T004-T007 to verify fix

**Validation**:
- Contract tests T004-T007 now PASS
- No AttributeError when creating connections
- Code inspection test (T006) passes

**Acceptance**:
- ✅ Line 210 uses iris.createConnection()
- ✅ All Bug 1 contract tests pass (T004-T007)
- ✅ No AttributeError during connection establishment

---

### T015 - Add Connection Error Handling ✅
**Description**: Enhance error handling to raise ConnectionError (not AttributeError)
**File**: `iris_vector_rag/common/iris_dbapi_connector.py`
**Steps**:
1. Add try/except around iris.createConnection() call
2. Catch AttributeError and convert to ConnectionError with clear message:
   ```python
   except AttributeError as e:
       raise ConnectionError(
           f"Failed to connect to IRIS database at {host}:{port}/{namespace}: "
           f"Invalid IRIS API. Use iris.createConnection() (not iris.connect()). "
           f"Error: {e}"
       )
   ```
3. Catch other connection errors and wrap with clear messages
4. Run contract test T005 to verify error handling

**Validation**:
- Contract test T005 now PASSES
- ConnectionError raised (not AttributeError)
- Error messages follow standard format

**Acceptance**:
- ✅ Clear ConnectionError messages for all failure scenarios
- ✅ Error messages indicate connection failure reason
- ✅ No AttributeError leaks to calling code

---

### T016 - Validate FHIR-AI Test Suite Compatibility
**Description**: Run simulated FHIR-AI tests to verify 6/6 passing (up from 3/6)
**File**: `tests/contract/test_fhir_ai_compatibility.py`
**Steps**:
1. Run contract test T007 (FHIR-AI compatibility tests)
2. Verify all 6 tests now pass:
   - ConfigurationManager ✅ (was passing)
   - ConnectionManager ✅ (was failing, now fixed)
   - IRISVectorStore ✅ (was failing, now fixed)
   - SchemaManager ✅ (was failing, now fixed)
   - Environment Variables ✅ (was passing)
   - Document Model ✅ (was passing)
3. Document any failures and fix

**Validation**:
- Contract test T007 shows 6/6 passing
- Test report matches FHIR-AI requirements

**Acceptance**:
- ✅ 6/6 tests passing (up from 3/6 in v0.5.3)
- ✅ ConnectionManager, IRISVectorStore, SchemaManager now work
- ✅ Backward compatibility preserved (ConfigurationManager, Environment, Document still pass)

---

## Phase 3.5: Implementation - Bug 2 (Schema Initialization)
✅ **Prerequisites**: All Bug 2 contract tests (T008-T013) written and FAILING

### T017 [P] - Implement Package Detection Method ✅
**Description**: Add _detect_iris_vector_graph() method to SchemaManager
**File**: `iris_vector_rag/storage/schema_manager.py`
**Research Decision**: Decision #2 from `research.md` - Use importlib.util.find_spec()
**Steps**:
1. Add private method to SchemaManager class:
   ```python
   def _detect_iris_vector_graph(self) -> bool:
       """Detect if iris-vector-graph package is installed."""
       import importlib.util
       spec = importlib.util.find_spec("iris_vector_graph")
       return spec is not None
   ```
2. Add docstring with implementation notes
3. Run contract tests T008 to verify detection works

**Validation**:
- Contract tests T008 now PASS (6 tests)
- Detection works for both installed and not installed cases

**Acceptance**:
- ✅ Method exists and works correctly
- ✅ No import side effects (uses find_spec, not import)
- ✅ Returns boolean without exceptions

---

### T018 [P] - Implement InitializationResult Data Structure ✅
**Description**: Create InitializationResult dataclass for table initialization results
**File**: `iris_vector_rag/storage/schema_manager.py` (or new file `iris_vector_rag/storage/schema_types.py`)
**Data Model**: InitializationResult from `data-model.md`
**Steps**:
1. Create dataclass:
   ```python
   @dataclass
   class InitializationResult:
       package_detected: bool
       tables_attempted: List[str]
       tables_created: Dict[str, bool]
       total_time_seconds: float
       error_messages: Dict[str, str]
   ```
2. Add validation in __post_init__ to enforce invariants
3. Run contract tests T011 to verify data structure

**Validation**:
- Contract tests T011 now PASS (InitializationResult invariants)
- Data structure enforces invariants

**Acceptance**:
- ✅ Data structure exists with correct fields
- ✅ Invariants enforced
- ✅ Can be imported and used

---

### T019 - Implement Graph Tables Initialization Method ✅
**Description**: Add ensure_iris_vector_graph_tables() method to SchemaManager
**File**: `iris_vector_rag/storage/schema_manager.py`
**Research Decision**: Decision #3 from `research.md` - New public method
**Steps**:
1. Add public method to SchemaManager class:
   ```python
   def ensure_iris_vector_graph_tables(
       self,
       pipeline_type: str = "graphrag"
   ) -> InitializationResult:
       """Automatically create iris-vector-graph tables if package installed."""
       # Implementation from contracts/schema_manager_contracts.md Contract 2
   ```
2. Implement logic:
   - Check package detection via _detect_iris_vector_graph()
   - If not detected, return empty InitializationResult
   - If detected, create tables in order: rdf_labels, rdf_props, rdf_edges, kg_NodeEmbeddings_optimized
   - Track timing, success/failure per table
   - Log INFO for success, ERROR for failures
3. Run contract tests T009 to verify initialization

**Validation**:
- Contract tests T009 now PASS (7 tests)
- All 4 tables created when package installed
- Graceful skip when package not installed
- Idempotent (safe to call multiple times)

**Acceptance**:
- ✅ Method exists and works correctly
- ✅ Creates all 4 tables when package detected
- ✅ Skips gracefully when package not installed
- ✅ Returns InitializationResult with timing and status

---

### T020 [P] - Implement ValidationResult Data Structure ✅
**Description**: Create ValidationResult dataclass for prerequisite validation results
**File**: `iris_vector_rag/storage/schema_manager.py` (or `iris_vector_rag/storage/schema_types.py`)
**Data Model**: ValidationResult from `data-model.md`
**Steps**:
1. Create dataclass:
   ```python
   @dataclass
   class ValidationResult:
       is_valid: bool
       package_installed: bool
       missing_tables: List[str]
       error_message: str
   ```
2. Add validation in __post_init__ to enforce invariants
3. Run contract tests T011 to verify data structure

**Validation**:
- Contract tests T011 now PASS (ValidationResult invariants)
- Data structure enforces invariants

**Acceptance**:
- ✅ Data structure exists with correct fields
- ✅ Invariants enforced
- ✅ Can be imported and used

---

### T021 - Implement Prerequisite Validation Method ✅
**Description**: Add validate_graph_prerequisites() method to SchemaManager
**File**: `iris_vector_rag/storage/schema_manager.py`
**Research Decision**: Decision #4 from `research.md` - Explicit validation method
**Steps**:
1. Add public method to SchemaManager class:
   ```python
   def validate_graph_prerequisites(self) -> ValidationResult:
       """Validate that all iris-vector-graph prerequisites are met."""
       # Implementation from contracts/schema_manager_contracts.md Contract 3
   ```
2. Implement logic:
   - Check package installation via _detect_iris_vector_graph()
   - If not installed, return ValidationResult(is_valid=False, error about package)
   - If installed, check existence of all 4 tables via table_exists()
   - Return ValidationResult with specific missing tables
3. Run contract tests T010 to verify validation

**Validation**:
- Contract tests T010 now PASS (9 tests)
- Validation detects all prerequisites met
- Validation reports missing package or missing tables
- Error messages are clear

**Acceptance**:
- ✅ Method exists and works correctly
- ✅ Validates package installation
- ✅ Validates all 4 tables exist
- ✅ Returns ValidationResult with clear error messages

---

## Phase 3.6: Integration & Validation

### T022 - Integration Test: Bug 1 Connection Establishment
**Description**: End-to-end integration test for Bug 1 connection fix
**File**: `tests/integration/test_connection_api_integration.py`
**Quickstart Scenario**: Scenario 1 from `quickstart.md` (Bug 1)
**Steps**:
1. Write integration test following quickstart.md Scenario 1:
   - Verify line 210 uses iris.createConnection()
   - Test connection establishment with real IRIS database
   - Test ConnectionManager integration
   - Test IRISVectorStore operations
2. Run test against real IRIS database
3. Verify no AttributeError raised

**Validation**:
- Integration test passes
- Real database connection works
- IRISVectorStore can perform operations

**Acceptance**:
- ✅ Integration test passes with real IRIS database
- ✅ Connection established successfully
- ✅ No AttributeError during test execution

---

### T023 - Integration Test: Bug 1 Error Handling
**Description**: Integration test for Bug 1 error message clarity
**File**: `tests/integration/test_connection_api_integration.py`
**Quickstart Scenario**: Scenario 3 from `quickstart.md` (Bug 1)
**Steps**:
1. Write integration test following quickstart.md Scenario 3:
   - Test invalid port (should raise ConnectionError, not AttributeError)
   - Test invalid credentials (should raise ConnectionError with auth details)
   - Verify error messages are clear
2. Run test and verify error handling

**Validation**:
- Integration test passes
- ConnectionError raised (not AttributeError)
- Error messages follow standard format

**Acceptance**:
- ✅ Invalid port raises ConnectionError
- ✅ Invalid credentials raise ConnectionError
- ✅ Error messages are clear and actionable

---

### T024 - Integration Test: Bug 2 Automatic Initialization
**Description**: End-to-end integration test for Bug 2 automatic table initialization
**File**: `tests/integration/test_graph_schema_integration.py`
**Quickstart Scenario**: Scenario 1 from `quickstart.md` (Bug 2)
**Steps**:
1. Write integration test following quickstart.md Scenario 1:
   - Test automatic table initialization when iris-vector-graph installed
   - Verify all 4 tables created
   - Test idempotent initialization
   - Test PPR query execution
2. Run test against real IRIS database with iris-vector-graph installed

**Validation**:
- Integration test passes
- All 4 tables created automatically
- PPR operations work without "Table not found" errors

**Acceptance**:
- ✅ Tables automatically created when package detected
- ✅ Initialization completes in < 5 seconds
- ✅ PPR query executes successfully

---

### T025 - Integration Test: Bug 2 Graceful Degradation
**Description**: Integration test for Bug 2 graceful degradation without iris-vector-graph
**File**: `tests/integration/test_graph_schema_integration.py`
**Quickstart Scenario**: Scenario 2 from `quickstart.md` (Bug 2)
**Steps**:
1. Write integration test following quickstart.md Scenario 2:
   - Test initialization when iris-vector-graph NOT installed
   - Verify no tables created
   - Verify no errors raised
   - Verify basic pipelines still work
2. Run test in environment without iris-vector-graph

**Validation**:
- Integration test passes
- No tables created when package not installed
- No errors raised during initialization
- Basic pipelines continue to work

**Acceptance**:
- ✅ Graceful skip when package not installed
- ✅ Validation reports missing package
- ✅ Backward compatibility maintained

---

### T026 - Integration Test: Bug 2 Error Handling
**Description**: Integration test for Bug 2 partial table creation failure handling
**File**: `tests/integration/test_graph_schema_integration.py`
**Quickstart Scenario**: Scenario 3 from `quickstart.md` (Bug 2)
**Steps**:
1. Write integration test following quickstart.md Scenario 3:
   - Simulate partial table creation failure
   - Verify successful tables created despite other failures
   - Verify error messages list specific failed tables
   - Verify validation identifies missing prerequisites
2. Run test with mocked partial failures

**Validation**:
- Integration test passes
- Partial failures tracked per table
- Error messages are clear and actionable

**Acceptance**:
- ✅ Partial failures tracked correctly
- ✅ Error messages list specific failed tables
- ✅ Validation identifies missing prerequisites

---

### T027 - Performance Test: Bug 2 Initialization Performance
**Description**: Performance test for Bug 2 table initialization (< 5 seconds requirement)
**File**: `tests/integration/test_graph_schema_performance.py`
**Quickstart Scenario**: Performance Validation from `quickstart.md`
**Steps**:
1. Write performance test following quickstart.md:
   - Clean database (drop existing tables)
   - Measure initialization time for 4 tables
   - Assert total_time_seconds < 5.0
   - Verify time tracking accuracy
2. Run test against real IRIS database

**Validation**:
- Performance test passes
- Initialization completes in < 5 seconds

**Acceptance**:
- ✅ All 4 tables created in < 5 seconds
- ✅ Performance requirement met (~3.8s expected)
- ✅ Timing tracking is accurate

---

### T028 - Validation Test: FHIR-AI Test Suite Full Run
**Description**: Run full FHIR-AI test suite to verify 6/6 passing
**File**: External (FHIR-AI project tests)
**Quickstart Scenario**: Scenario 2 from `quickstart.md` (Bug 1)
**Steps**:
1. Set up FHIR-AI test environment
2. Run all 6 FHIR-AI tests:
   - ConfigurationManager
   - ConnectionManager (was failing in v0.5.3)
   - IRISVectorStore (was failing in v0.5.3)
   - SchemaManager (was failing in v0.5.3)
   - Environment Variables
   - Document Model
3. Document results

**Validation**:
- 6/6 tests pass (up from 3/6 in v0.5.3)
- No regressions in previously passing tests

**Acceptance**:
- ✅ ConfigurationManager test passes (backward compatibility)
- ✅ ConnectionManager test passes (was failing in v0.5.3)
- ✅ IRISVectorStore test passes (was failing in v0.5.3)
- ✅ SchemaManager test passes (was failing in v0.5.3)
- ✅ Environment variables test passes (backward compatibility)
- ✅ Document model test passes (backward compatibility)

---

## Phase 3.7: Documentation & Polish

### T029 [P] - Update CHANGELOG.md
**Description**: Document Bug 1 & Bug 2 fixes in CHANGELOG.md for v0.5.4 release
**File**: `CHANGELOG.md`
**Steps**:
1. Add v0.5.4 section with release date
2. Document Bug 1 fix:
   - Fixed: Replace non-existent iris.connect() with iris.createConnection() at line 210
   - Impact: Restores database connectivity (was broken in v0.5.3)
   - Tests: FHIR-AI test suite now 6/6 passing (up from 3/6)
3. Document Bug 2 fix:
   - Added: Automatic iris-vector-graph table initialization
   - Impact: Eliminates silent PPR failures
   - Performance: Table creation in < 5 seconds
4. Add migration notes if applicable

**Acceptance**:
- ✅ CHANGELOG updated with both bug fixes
- ✅ Clear description of changes
- ✅ Impact and test results documented

---

### T030 [P] - Update README.md
**Description**: Update README with v0.5.4 release notes and quickstart updates
**File**: `README.md`
**Steps**:
1. Update version badge to v0.5.4
2. Add note about Bug 1 fix (connection API corrected)
3. Add note about Bug 2 fix (automatic iris-vector-graph setup)
4. Update quickstart section if needed
5. Add troubleshooting entries for connection errors

**Acceptance**:
- ✅ README reflects v0.5.4 changes
- ✅ Quickstart updated if needed
- ✅ Troubleshooting section enhanced

---

### T031 [P] - Update API Documentation
**Description**: Update API docs with SchemaManager new methods
**File**: `docs/api/schema_manager.md` (or appropriate API doc location)
**Steps**:
1. Document _detect_iris_vector_graph() method (private)
2. Document ensure_iris_vector_graph_tables() method (public API)
3. Document validate_graph_prerequisites() method (public API)
4. Document InitializationResult data structure
5. Document ValidationResult data structure
6. Add usage examples from quickstart.md

**Acceptance**:
- ✅ New methods documented with signatures
- ✅ Data structures documented
- ✅ Usage examples provided

---

### T032 [P] - Code Review & Cleanup
**Description**: Final code review and cleanup for Bug 1 & Bug 2 fixes
**Files**: All modified files
**Steps**:
1. Review `iris_vector_rag/common/iris_dbapi_connector.py` (Bug 1):
   - Verify line 210 uses iris.createConnection()
   - Verify no iris.connect() calls remain
   - Check error handling is clear
   - Remove any debugging code
2. Review `iris_vector_rag/storage/schema_manager.py` (Bug 2):
   - Verify all methods follow naming conventions
   - Check logging is appropriate (INFO for success, ERROR for failures)
   - Verify docstrings are complete
   - Remove any debugging code
3. Run linters (black, isort, flake8, mypy)
4. Check for code duplication
5. Verify Constitution compliance (all 7 principles)

**Acceptance**:
- ✅ Code review complete
- ✅ All linters pass
- ✅ No debugging code remains
- ✅ Constitution compliance verified

---

## Dependencies

### Setup Dependencies
- T001 (environment) must complete before T002-T003
- T002 (fixtures) must complete before T004-T013 (contract tests)

### Test-First Dependencies (TDD)
- **CRITICAL**: T004-T013 (all contract tests) must complete and FAIL before T014-T021 (implementation)
- T004-T007 (Bug 1 contract tests) must FAIL before T014-T016 (Bug 1 implementation)
- T008-T013 (Bug 2 contract tests) must FAIL before T017-T021 (Bug 2 implementation)

### Bug 1 Implementation Dependencies
- T014 (fix line 210) must complete before T015 (error handling)
- T014-T015 must complete before T016 (FHIR-AI validation)

### Bug 2 Implementation Dependencies
- T017 (package detection) must complete before T019 (initialization)
- T018 (InitializationResult) must complete before T019 (initialization)
- T020 (ValidationResult) must complete before T021 (validation)
- T017, T019, T021 can run after T018 and T020

### Integration Dependencies
- T014-T016 (Bug 1 implementation) must complete before T022-T023 (Bug 1 integration tests)
- T017-T021 (Bug 2 implementation) must complete before T024-T027 (Bug 2 integration tests)
- T022-T027 (all integration tests) must pass before T028 (FHIR-AI full run)

### Documentation Dependencies
- T014-T021 (all implementation) must complete before T029-T032 (documentation)
- T029-T032 can run in parallel (different files)

---

## Parallel Execution Examples

### Setup Phase (T002-T003 can run in parallel after T001)
```bash
# After T001 completes:
Task: "Create test fixture infrastructure in tests/fixtures/"
Task: "Configure linting and pre-commit"
```

### Bug 1 Contract Tests (T004-T007 can run in parallel)
```bash
# All Bug 1 contract tests can run together:
Task: "Contract test connection uses correct API in tests/contract/test_connection_api_fix.py"
Task: "Contract test connection error messages in tests/contract/test_connection_api_fix.py"
Task: "Contract test ConnectionManager integration in tests/contract/test_connection_manager_integration.py"
Task: "Contract test FHIR-AI compatibility in tests/contract/test_fhir_ai_compatibility.py"
```

### Bug 2 Contract Tests (T008-T013 can run in parallel)
```bash
# All Bug 2 contract tests can run together:
Task: "Contract test package detection in tests/contract/test_graph_schema_detection.py"
Task: "Contract test graph tables initialization in tests/contract/test_graph_schema_initialization.py"
Task: "Contract test prerequisite validation in tests/contract/test_graph_schema_validation.py"
Task: "Contract test data structure invariants in tests/contract/test_graph_schema_data_structures.py"
Task: "Contract test error message clarity in tests/contract/test_graph_schema_validation.py"
Task: "Contract test performance requirements in tests/contract/test_graph_schema_performance.py"
```

### Bug 2 Data Structures (T018, T020 can run in parallel)
```bash
# Data structure creation can run together:
Task: "Implement InitializationResult data structure in iris_vector_rag/storage/schema_manager.py"
Task: "Implement ValidationResult data structure in iris_vector_rag/storage/schema_manager.py"
```

### Documentation Phase (T029-T031 can run in parallel)
```bash
# Documentation tasks can run together:
Task: "Update CHANGELOG.md for v0.5.4"
Task: "Update README.md with v0.5.4 notes"
Task: "Update API documentation with new SchemaManager methods"
```

---

## Validation Checklist
*GATE: Check before marking tasks.md as complete*

**Contract Test Coverage**:
- [x] All 2 contracts from `connection_api_contracts.md` have corresponding tests (T004-T007)
- [x] All 4 contracts from `schema_manager_contracts.md` have corresponding tests (T008-T013)
- [x] Total: 21 contract tests defined (12 Bug 1, 9 Bug 2)

**Entity/Component Coverage**:
- [x] Bug 1: ConnectionEstablisher, ConnectionResult (implicit in line 210 fix)
- [x] Bug 2: All 3 components from data-model.md have tasks (T017, T019, T021)
- [x] Bug 2: Both data structures from data-model.md have tasks (T018, T020)

**Test-First (TDD)**:
- [x] All contract tests (T004-T013) come before implementation (T014-T021)
- [x] Tests must fail initially (verified in task acceptance criteria)

**Parallel Task Independence**:
- [x] All [P] tasks operate on different files or independent components
- [x] No [P] task modifies same file as another [P] task

**File Path Specificity**:
- [x] Each task specifies exact file path
- [x] Contract test file paths match contract document structure
- [x] Implementation file paths match codebase structure

**Quickstart Scenario Coverage**:
- [x] Bug 1: All 3 scenarios from quickstart.md have integration tests (T022-T023, T028)
- [x] Bug 2: All 3 scenarios from quickstart.md have integration tests (T024-T026)
- [x] Performance scenario has dedicated test (T027)

---

## Notes

**TDD Approach**:
- Contract tests (T004-T013) MUST be written first and MUST FAIL
- Implementation (T014-T021) makes tests pass
- Integration tests (T022-T028) validate end-to-end functionality

**Bug 1 Fix Scope**:
- One-line change at iris_dbapi_connector.py:210
- Replace iris.connect() → iris.createConnection()
- Enhanced error handling for clarity

**Bug 2 Fix Scope**:
- Three new methods in SchemaManager
- Two new data structures
- Automatic table initialization when iris-vector-graph detected

**Constitutional Compliance**:
- Framework-First: Core infrastructure fixes (✅)
- TDD: Contract tests before implementation (✅)
- Explicit Errors: No silent failures (✅)
- Standardized Interfaces: Use proven IRIS APIs (✅)

**Performance Targets**:
- Bug 2: Table initialization < 5 seconds
- Bug 2: Prerequisite validation < 1 second

**Success Criteria**:
- Bug 1: FHIR-AI tests 6/6 passing (up from 3/6)
- Bug 1: No AttributeError during connection
- Bug 2: All tables automatically created when iris-vector-graph installed
- Bug 2: Graceful degradation when iris-vector-graph not installed

---

## Commit Strategy

**After Each Task**:
- Run relevant tests to verify task completion
- Run linters (black, isort, flake8)
- Commit with descriptive message
- Reference task ID in commit message (e.g., "T014: Fix iris.connect() at line 210")

**Suggested Commit Messages**:
```
T014: Fix iris.connect() AttributeError at line 210

Replace non-existent iris.connect() with correct iris.createConnection() API.
This restores database connectivity broken in v0.5.3.

Fixes: FHIR-AI test suite now 3/6 → 4/6 passing (ConnectionManager works)

Co-Authored-By: Claude <noreply@anthropic.com>
```

```
T019: Add automatic iris-vector-graph table initialization

Implement ensure_iris_vector_graph_tables() method in SchemaManager.
Automatically creates rdf_labels, rdf_props, rdf_edges, kg_NodeEmbeddings_optimized
when iris-vector-graph package is detected.

Performance: ~3.8s for 4 tables (< 5s requirement)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

**Status**: ✅ Tasks ready for execution (32 tasks total)
**Next Step**: Begin with T001 (Setup) → T004-T013 (Contract Tests) → T014-T021 (Implementation) → T022-T028 (Integration) → T029-T032 (Documentation)
