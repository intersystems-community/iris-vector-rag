# Data Model: Test Infrastructure Resilience

**Feature**: 028-obviously-these-failures
**Date**: 2025-10-05

## Core Entities

### SchemaDefinition
**Purpose**: Represents expected database table schema structure

**Attributes**:
- `table_name`: str - Table name (e.g., "SourceDocuments")
- `schema_name`: str - Schema/namespace (default: "RAG")
- `columns`: List[ColumnDefinition] - Expected columns
- `indexes`: List[IndexDefinition] - Expected indexes
- `version`: str - Schema version identifier

**Relationships**:
- Has many `ColumnDefinition` instances
- Has many `IndexDefinition` instances

**Validation Rules**:
- `table_name` must be valid SQL identifier (alphanumeric + underscore)
- Must have at least one column
- Primary key column(s) must exist
- `version` follows semver format (e.g., "1.0.0")

**State Transitions**: Immutable after creation

### ColumnDefinition
**Purpose**: Defines expected column structure and constraints

**Attributes**:
- `name`: str - Column name
- `data_type`: str - SQL data type (e.g., "VARCHAR", "INTEGER", "VECTOR")
- `is_nullable`: bool - Allows NULL values
- `is_primary_key`: bool - Part of primary key
- `max_length`: Optional[int] - For VARCHAR types
- `precision`: Optional[int] - For numeric types

**Validation Rules**:
- `data_type` must be valid IRIS SQL type
- If `is_primary_key`, then `is_nullable` must be False
- `max_length` required for VARCHAR types

### SchemaValidationResult
**Purpose**: Result of schema validation operation

**Attributes**:
- `is_valid`: bool - Overall validation result
- `table_name`: str - Table being validated
- `mismatches`: List[SchemaMismatch] - Specific issues found
- `missing_tables`: List[str] - Tables that should exist but don't
- `extra_columns`: List[str] - Columns that exist but shouldn't

**State**: Read-only after creation

**Usage Pattern**:
```python
result = schema_validator.validate_schema("SourceDocuments")
if not result.is_valid:
    for mismatch in result.mismatches:
        log.error(f"{mismatch.column_name}: {mismatch.issue}")
```

### SchemaMismatch
**Purpose**: Describes a specific schema discrepancy

**Attributes**:
- `column_name`: str - Column with issue
- `expected_type`: str - Expected SQL type
- `actual_type`: str - Actual SQL type in database
- `issue`: str - Description (e.g., "type mismatch", "missing column", "extra column")
- `severity`: str - "error" or "warning"

### TestDatabaseState
**Purpose**: Manages database state for test isolation

**Attributes**:
- `connection`: iris.dbapi.Connection - Active database connection
- `cleanup_functions`: List[Callable] - Cleanup operations to run
- `is_clean`: bool - Whether state is pristine
- `schema_version`: str - Schema version for this test run
- `isolation_level`: str - "session", "class", or "function"

**State Transitions**:
```
created → used → cleaned → disposed
         ↓
    (on failure) → partially_cleaned → disposed
```

**Validation Rules**:
- Must execute all `cleanup_functions` before disposal
- `is_clean` set to False when any test modifies database
- Connection must be valid when state is "used"

**Cleanup Pattern**:
```python
state = TestDatabaseState(conn)
try:
    state.add_cleanup(lambda: cursor.execute("TRUNCATE TABLE ..."))
    run_test()
finally:
    state.cleanup()  # Runs all cleanup functions
```

### ContractTestMarker
**Purpose**: Identifies and manages contract test expectations

**Attributes**:
- `test_name`: str - Full test name (e.g., "test_mcp_server_startup")
- `is_contract_test`: bool - Whether marked with @pytest.mark.contract
- `expected_to_fail`: bool - Whether failure is expected (unimplemented feature)
- `failure_reason`: Optional[str] - Why failure is expected
- `implementation_status`: str - "not_started", "in_progress", "complete"

**State Transitions**:
```
not_started → in_progress → complete
     ↓              ↓
 (expected_to_fail=True → False)
```

### PreflightCheckResult
**Purpose**: Result of pre-test validation checks

**Attributes**:
- `check_name`: str - What was checked (e.g., "IRIS Connectivity")
- `passed`: bool - Whether check passed
- `message`: str - Status message or error details
- `remediation`: Optional[str] - How to fix if failed
- `duration_ms`: int - How long check took

**Aggregation**:
Multiple `PreflightCheckResult` instances combined into overall pass/fail

## Relationships Diagram

```
SchemaDefinition (1) ──┬─> (N) ColumnDefinition
                       └─> (N) IndexDefinition

SchemaValidator ──uses──> SchemaDefinition
                ──produces──> SchemaValidationResult (1) ──> (N) SchemaMismatch

TestFixtureManager ──manages──> TestDatabaseState
                   ──uses──> SchemaValidator

ContractTestPlugin ──creates──> ContractTestMarker
                   ──references──> TestDatabaseState

PreflightChecker ──produces──> (N) PreflightCheckResult
                 ──uses──> SchemaValidator
```

## Data Flow

### Test Session Startup
```
1. PreflightChecker validates prerequisites
   → IRIS connectivity
   → Schema validity
   → API keys present

2. SchemaValidator checks database schema
   → If mismatches found → SchemaManager.reset_schema()
   → If valid → proceed

3. TestFixtureManager creates session-scoped state
   → Connection pool initialized
   → Cleanup handlers registered
```

### Per-Test-Class Execution
```
1. TestFixtureManager provides clean database state
   → Class-scoped fixture activated
   → Cleanup function registered

2. Tests execute
   → Modify database as needed
   → TestDatabaseState tracks mutations

3. Cleanup executes (always, even on failure)
   → All registered cleanup functions run
   → TestDatabaseState reset to clean
```

### Contract Test Handling
```
1. Test marked with @pytest.mark.contract executes
2. If test fails:
   → ContractTestPlugin intercepts failure
   → Checks if feature is implemented
   → If not implemented: reclassify as "xfail"
   → If implemented: report as real failure
3. Test outcome adjusted in pytest report
```

## Validation Rules Summary

1. **Schema Validation**:
   - All expected tables must exist
   - All expected columns must exist with correct types
   - No extra columns (indicates schema drift)
   - Indexes present for vector operations

2. **Test Isolation**:
   - Each test class gets clean database state
   - Cleanup must complete within 100ms
   - No data pollution between tests

3. **Contract Tests**:
   - Must be marked with @pytest.mark.contract
   - Failure reason must be documented
   - Expected failures don't contribute to failure count

4. **Performance**:
   - Schema reset <5 seconds
   - Test isolation overhead <100ms per class
   - Pre-flight checks <2 seconds

## Implementation Notes

- All entities are immutable after creation (except TestDatabaseState)
- Use dataclasses for simple entities (SchemaDefinition, ColumnDefinition)
- Use proper exception handling with SQLCODE context
- Log all schema operations for audit trail
