# Connection API Contracts: iris.connect() Fix

**Feature**: 060-fix-users-tdyar (Bug 1)
**Date**: 2025-01-14
**Status**: Design Phase

---

## Contract 1: Connection Establishment with Correct API

### Method Signature
```python
def get_iris_connection(
    host: str,
    port: int,
    namespace: str,
    user: str,
    password: str,
    max_retries: int = 3
) -> Optional[Connection]:
    """
    Establish IRIS database connection using correct iris.createConnection() API.

    Args:
        host: IRIS database host
        port: IRIS database port
        namespace: IRIS namespace
        user: Database user
        password: Database password
        max_retries: Maximum connection retry attempts (default: 3)

    Returns:
        Connection object if successful, None if all retries exhausted.

    Raises:
        ValueError: If required connection parameters are missing
        ConnectionError: If connection fails after all retries

    Implementation Notes:
        - MUST use iris.createConnection() (not the non-existent iris.connect())
        - Retry with exponential backoff on transient failures
        - Log connection attempts at INFO level
        - Log failures at ERROR level with specific error details
        - Preserve SSL configuration from v0.5.3
    """
```

### Preconditions
- intersystems-irispython package installed (>=5.1.2)
- IRIS database is running and accessible
- Connection parameters are valid and non-empty
- iris.createConnection() method exists

### Postconditions
**Success Case**:
- Returns valid Connection object
- Connection is open and usable
- No AttributeError raised
- INFO log written with connection details

**Failure Case**:
- Returns None or raises ConnectionError
- ERROR log written with specific error reason
- No AttributeError about missing iris.connect()

### Invariants
- Never calls iris.connect() (does not exist)
- Always uses iris.createConnection() or iris.dbapi.connect()
- Retry logic consistent with max_retries parameter
- Connection timeout respected

### Test Contracts

**Test**: `test_connection_uses_correct_api()`
- GIVEN valid IRIS connection parameters
- WHEN get_iris_connection() is called
- THEN iris.createConnection() is used (NOT iris.connect())
- AND no AttributeError is raised
- AND connection is successfully established

**Test**: `test_no_attribute_error_on_connection()`
- GIVEN valid IRIS connection parameters
- WHEN get_iris_connection() is called
- THEN no AttributeError about 'connect' is raised
- AND connection succeeds or raises ConnectionError (not AttributeError)

**Test**: `test_connection_retry_logic()`
- GIVEN IRIS database temporarily unavailable
- WHEN get_iris_connection() is called with max_retries=3
- THEN up to 3 connection attempts are made
- AND exponential backoff is applied between retries
- AND connection succeeds on retry or returns None after exhaustion

**Test**: `test_connection_error_message_clarity()`
- GIVEN invalid IRIS connection parameters (wrong port)
- WHEN get_iris_connection() is called
- THEN ConnectionError raised with clear message indicating connection failure
- AND error message does NOT contain "AttributeError: module 'iris' has no attribute 'connect'"

---

## Contract 2: Connection API Validation

### Method Signature
```python
def validate_iris_api_available() -> Tuple[bool, str]:
    """
    Validate that correct IRIS connection API is available.

    Returns:
        Tuple of (is_valid, error_message):
            - is_valid: True if iris.createConnection exists
            - error_message: Empty string if valid, error details if not

    Implementation Notes:
        - Check hasattr(iris, 'createConnection')
        - If not available, provide clear error with correct API name
        - Does not attempt actual connection (validation only)
    """
```

### Preconditions
- intersystems-irispython package imported

### Postconditions
**Valid API**:
- Returns (True, "")
- iris.createConnection is accessible

**Invalid API**:
- Returns (False, "iris.createConnection not available. Check intersystems-irispython version.")
- Does not raise exceptions

### Test Contracts

**Test**: `test_api_validation_detects_correct_method()`
- GIVEN intersystems-irispython package installed
- WHEN validate_iris_api_available() is called
- THEN returns (True, "") indicating iris.createConnection exists

**Test**: `test_api_validation_error_message_for_wrong_method()`
- GIVEN iris module imported
- WHEN code attempts to use iris.connect() (simulated via hasattr check)
- THEN validation indicates this method does not exist
- AND suggests using iris.createConnection() instead

---

## Contract 3: Connection Manager Integration

### Test Contract
**Test**: `test_connection_manager_creates_connections()`
- GIVEN ConnectionManager initialized with valid IRIS config
- WHEN ConnectionManager creates a connection
- THEN connection is established successfully
- AND no AttributeError is raised
- AND connection can execute queries

**Location**: `iris_vector_rag/common/iris_dbapi_connector.py:210`
**Before (v0.5.3 - BROKEN)**:
```python
conn = iris.connect(host, port, namespace, user, password)  # ❌ AttributeError
```

**After (v0.5.4 - FIXED)**:
```python
conn = iris.createConnection(host, port, namespace, user, password)  # ✅ Correct API
```

### Test Contracts

**Test**: `test_iris_dbapi_connector_line_210_fixed()`
- GIVEN iris_dbapi_connector.py source code
- WHEN reading line 210
- THEN line contains "iris.createConnection" (not "iris.connect")
- AND no calls to non-existent iris.connect() exist anywhere in file

**Test**: `test_connection_manager_integration()`
- GIVEN ConnectionManager from iris_dbapi_connector
- WHEN creating a connection via get_iris_connection()
- THEN connection succeeds without AttributeError
- AND returned connection can execute SQL queries
- AND connection can be closed cleanly

---

## Contract 4: FHIR-AI Test Suite Compatibility

### Test Contracts

**Test**: `test_configuration_manager_passes()`
- GIVEN ConfigurationManager test from FHIR-AI suite
- WHEN test executes
- THEN test passes (was passing in v0.5.2, must continue passing)

**Test**: `test_connection_manager_passes()`
- GIVEN ConnectionManager test from FHIR-AI suite
- WHEN test executes
- THEN test passes (was failing in v0.5.3 due to iris.connect() bug)

**Test**: `test_iris_vector_store_passes()`
- GIVEN IRISVectorStore test from FHIR-AI suite
- WHEN test executes
- THEN test passes (was failing in v0.5.3 due to connection dependency)

**Test**: `test_schema_manager_passes()`
- GIVEN SchemaManager test from FHIR-AI suite
- WHEN test executes
- THEN test passes (was failing in v0.5.3 due to connection dependency)

**Test**: `test_environment_variables_pass()`
- GIVEN environment variable tests from FHIR-AI suite
- WHEN tests execute
- THEN tests pass (was passing in v0.5.2, must continue passing)

**Test**: `test_document_model_passes()`
- GIVEN document model tests from FHIR-AI suite
- WHEN tests execute
- THEN tests pass (was passing in v0.5.2, must continue passing)

**Target**: 6/6 tests passing (currently 3/6 in v0.5.3)

---

## Contract 5: SSL Configuration Preservation

### Test Contract
**Test**: `test_ssl_configuration_preserved()`
- GIVEN connection code added SSL handling in v0.5.3
- WHEN iris.connect() is replaced with iris.createConnection()
- THEN SSL configuration parameters are still passed correctly
- AND SSL connections work as intended in v0.5.3

**Context**: The v0.5.3 change to iris.connect() was intended to fix SSL issues. The correct fix is to use iris.createConnection() with the same SSL parameters.

### Implementation Notes
```python
# v0.5.3 (BROKEN - wrong API but correct SSL intent):
# Line 210: conn = iris.connect(host, port, namespace, user, password)

# v0.5.4 (FIXED - correct API, preserve SSL intent):
conn = iris.createConnection(host, port, namespace, user, password)
# SSL parameters are handled by iris.createConnection() correctly
```

---

## Contract 6: Backward Compatibility

### Test Contracts

**Test**: `test_v052_functionality_preserved()`
- GIVEN tests that passed in v0.5.2 (before iris.connect() bug)
- WHEN same tests run in v0.5.4 (after fix)
- THEN all v0.5.2 passing tests continue to pass
- AND CloudConfiguration API still works (dimension fix from v0.5.3)

**Test**: `test_no_api_breaking_changes()`
- GIVEN existing code using ConnectionManager or IRISVectorStore
- WHEN code runs with v0.5.4 connection fix
- THEN no API signature changes affect calling code
- AND all existing connection patterns continue to work

---

## Data Structure Contracts

### ConnectionResult (from data-model.md)

```python
@dataclass
class ConnectionResult:
    """Result of connection establishment attempt."""

    success: bool
    """Whether connection was established."""

    connection: Optional[Connection]
    """Connection object if successful, None if failed."""

    error_message: str
    """Error details if failed, empty string if successful."""

    api_used: str
    """API method used (e.g., 'iris.createConnection')."""

    retry_count: int
    """Number of retry attempts made."""

    elapsed_seconds: float
    """Time taken to establish connection."""
```

**Invariants**:
- If success == True, connection must be non-None
- If success == False, error_message must be non-empty
- api_used must be "iris.createConnection" or "iris.dbapi.connect" (NEVER "iris.connect")
- elapsed_seconds >= 0
- retry_count >= 0

### Test Contract
**Test**: `test_connection_result_invariants()`
- GIVEN ConnectionResult from successful connection
- THEN success == True AND connection is not None AND error_message == ""
- WHEN ConnectionResult from failed connection
- THEN success == False AND connection is None AND error_message is non-empty

---

## Performance Contracts

### Connection Establishment Performance
- **Target**: Establish connection in < 2 seconds (typical case)
- **Measurement**: ConnectionResult.elapsed_seconds
- **Test**: `test_connection_performance()`
  - GIVEN valid IRIS connection parameters
  - WHEN get_iris_connection() is called
  - THEN connection established in < 2.0 seconds
  - AND elapsed_seconds is recorded accurately

### Connection Retry Performance
- **Target**: 3 retries with exponential backoff complete in < 10 seconds
- **Test**: `test_connection_retry_performance()`
  - GIVEN temporarily unavailable IRIS database
  - WHEN get_iris_connection() retries 3 times
  - THEN all retries complete in < 10 seconds
  - AND backoff delays are approximately [1s, 2s, 4s]

---

## Integration Points

### iris_dbapi_connector.py:get_iris_connection()
**Contract**: Must be updated to use iris.createConnection()
**Behavior**: Establishes connection with retry logic
**Location**: Line 210 must be changed from iris.connect() to iris.createConnection()

### ConnectionManager
**Contract**: Uses get_iris_connection() for all database connections
**Behavior**: Provides connection pooling and management
**Dependency**: Must work after line 210 fix

### IRISVectorStore
**Contract**: Creates connections via ConnectionManager
**Behavior**: Vector storage operations require database connection
**Dependency**: Must work after ConnectionManager fix

### SchemaManager
**Contract**: Creates connections for schema operations
**Behavior**: Table creation and validation require database connection
**Dependency**: Must work after connection fix

---

## Error Message Contracts

### Error Message Format

**Error Type 1: AttributeError (Bug - Must NOT Occur)**
```
❌ BEFORE (v0.5.3):
AttributeError: module 'iris' has no attribute 'connect'

✅ AFTER (v0.5.4):
ConnectionError: Failed to connect to IRIS database at localhost:1972/USER: [specific error reason]
```

**Error Type 2: Connection Refused**
```
ConnectionError: Failed to connect to IRIS database at localhost:1972/USER: Connection refused. Verify IRIS is running and port is correct.
```

**Error Type 3: Invalid Credentials**
```
ConnectionError: Failed to connect to IRIS database at localhost:1972/USER: Authentication failed. Verify username and password.
```

### Test Contract
**Test**: `test_error_message_format()`
- GIVEN various connection failure scenarios
- WHEN connection attempt fails
- THEN error message follows standard format
- AND error message provides actionable remediation guidance
- AND error message does NOT contain "AttributeError: module 'iris' has no attribute 'connect'"

---

## Constitution Compliance Contracts

### TDD Requirement (Constitution III)
**Contract**: All contract tests must be written BEFORE implementation
**Validation**: Tests initially fail with connection errors or AttributeError
**Test Files**:
- `tests/contract/test_connection_api_fix.py` (Bug 1 specific tests)
- `tests/contract/test_connection_manager_integration.py` (integration)

### Explicit Error Handling (Constitution VI)
**Contract**: No silent connection failures
**Validation**: All connection errors logged and raised
**Test**: `test_no_silent_connection_failures()`
- GIVEN connection attempt that fails
- WHEN failure occurs
- THEN ERROR logged with specific reason
- AND ConnectionError raised (not swallowed)
- AND NOT AttributeError about missing iris.connect()

### Standardized Database Interfaces (Constitution VII)
**Contract**: Use proven connection patterns
**Validation**: iris.createConnection() is standard IRIS API
**Test**: `test_uses_standard_iris_api()`
- GIVEN connection code at line 210
- WHEN code is reviewed
- THEN uses iris.createConnection() (documented standard API)
- AND NOT iris.connect() (non-existent API)

---

## Status

✅ **Contracts Defined**
- [x] Connection establishment contract
- [x] API validation contract
- [x] Connection manager integration contract
- [x] FHIR-AI test suite compatibility contract
- [x] SSL configuration preservation contract
- [x] Backward compatibility contract
- [x] Data structure contracts (ConnectionResult)
- [x] Performance contracts
- [x] Integration point contracts
- [x] Error message contracts
- [x] Constitution compliance contracts

**Next Steps**: Write failing contract tests in `tests/contract/test_connection_api_fix.py`

---

## Test Implementation Checklist

### Contract Test File: `tests/contract/test_connection_api_fix.py`

**Tests to Implement** (12 total):
1. `test_connection_uses_correct_api()` - Verify iris.createConnection() used
2. `test_no_attribute_error_on_connection()` - No AttributeError raised
3. `test_connection_retry_logic()` - Retry with exponential backoff
4. `test_connection_error_message_clarity()` - Clear error messages
5. `test_api_validation_detects_correct_method()` - API validation works
6. `test_api_validation_error_message_for_wrong_method()` - Wrong API detected
7. `test_iris_dbapi_connector_line_210_fixed()` - Code inspection test
8. `test_connection_manager_integration()` - End-to-end connection
9. `test_ssl_configuration_preserved()` - SSL still works
10. `test_v052_functionality_preserved()` - Backward compatibility
11. `test_connection_result_invariants()` - Data structure validation
12. `test_error_message_format()` - Error message standards

**Expected Initial Result**: All 12 tests FAIL (TDD - implementation not done yet)
