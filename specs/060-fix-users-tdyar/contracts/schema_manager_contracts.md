# SchemaManager API Contracts: iris-vector-graph Auto-Initialization

**Feature**: 060-fix-users-tdyar
**Date**: 2025-01-13
**Status**: Design Phase

---

## Contract 1: Package Detection

### Method Signature
```python
def _detect_iris_vector_graph(self) -> bool:
    """
    Detect if iris-vector-graph package is installed in the Python environment.

    Returns:
        bool: True if iris-vector-graph is importable, False otherwise.

    Implementation Notes:
        - Uses importlib.util.find_spec("iris_vector_graph")
        - Does NOT import the package (no side effects)
        - Returns False on any errors (ImportError, ModuleNotFoundError, etc.)
    """
```

### Preconditions
- SchemaManager instance exists
- Python environment is accessible

### Postconditions
- Returns boolean without raising exceptions
- Does not modify any state
- Does not import iris_vector_graph module

### Invariants
- Stateless operation (repeated calls return same result unless environment changes)
- No side effects on filesystem or database

### Test Contract
**Test**: `test_iris_vector_graph_detection()`
- GIVEN iris-vector-graph is installed
- WHEN _detect_iris_vector_graph() is called
- THEN returns True

**Test**: `test_iris_vector_graph_not_installed()`
- GIVEN iris-vector-graph is NOT installed
- WHEN _detect_iris_vector_graph() is called
- THEN returns False

---

## Contract 2: Graph Tables Initialization

### Method Signature
```python
def ensure_iris_vector_graph_tables(
    self,
    pipeline_type: str = "graphrag"
) -> InitializationResult:
    """
    Automatically create iris-vector-graph tables if package is installed.

    Args:
        pipeline_type: Pipeline type for table creation context (default: "graphrag")

    Returns:
        InitializationResult with:
            - package_detected: Whether iris-vector-graph was found
            - tables_attempted: List of tables initialization was attempted for
            - tables_created: Dict mapping table names to creation success
            - total_time_seconds: Time taken for initialization
            - error_messages: Dict of error messages for failed tables

    Raises:
        ValueError: If pipeline_type is invalid

    Implementation Notes:
        - If package not detected, returns empty InitializationResult immediately
        - Creates tables in dependency order: rdf_labels, rdf_props, rdf_edges,
          kg_NodeEmbeddings_optimized
        - Uses existing SchemaManager.ensure_table_schema() for each table
        - Logs INFO for successful creation, ERROR for failures
        - Continues creating other tables even if one fails (partial failure tracking)
    """
```

### Preconditions
- SchemaManager instance initialized with valid database connection
- pipeline_type is a valid pipeline identifier

### Postconditions
**Success Case (Package Installed)**:
- All 4 graph tables exist in database
- InitializationResult.package_detected == True
- InitializationResult.tables_attempted contains all 4 table names
- InitializationResult.tables_created has entries for all 4 tables
- INFO logs written for successful table creation
- Total time < 5 seconds

**Success Case (Package NOT Installed)**:
- No tables created
- InitializationResult.package_detected == False
- InitializationResult.tables_attempted is empty list
- InitializationResult.tables_created is empty dict
- DEBUG log written indicating package not installed

**Partial Failure Case**:
- Some tables exist, others failed
- InitializationResult.tables_created shows success/failure per table
- InitializationResult.error_messages contains error details for failed tables
- ERROR logs written for each failure

### Invariants
- Method is idempotent (safe to call multiple times)
- Never raises exceptions (returns error status in InitializationResult)
- Tables created in dependency order (nodes before edges)

### Test Contracts

**Test**: `test_graph_tables_created_when_package_installed()`
- GIVEN iris-vector-graph is installed
- AND database connection is valid
- WHEN ensure_iris_vector_graph_tables() is called
- THEN all 4 tables are created successfully
- AND InitializationResult.package_detected == True
- AND InitializationResult.tables_created == {"rdf_labels": True, "rdf_props": True,
  "rdf_edges": True, "kg_NodeEmbeddings_optimized": True}
- AND total_time_seconds < 5.0

**Test**: `test_graph_tables_skipped_when_package_not_installed()`
- GIVEN iris-vector-graph is NOT installed
- WHEN ensure_iris_vector_graph_tables() is called
- THEN no tables are created
- AND InitializationResult.package_detected == False
- AND InitializationResult.tables_attempted == []
- AND InitializationResult.tables_created == {}

**Test**: `test_idempotent_table_creation()`
- GIVEN iris-vector-graph is installed
- AND ensure_iris_vector_graph_tables() has been called once (tables exist)
- WHEN ensure_iris_vector_graph_tables() is called again
- THEN operation succeeds without errors
- AND InitializationResult shows all tables as successfully created
- AND no duplicate tables are created

**Test**: `test_partial_table_creation_failure()`
- GIVEN iris-vector-graph is installed
- AND database permissions allow creating 2 tables but not the other 2
- WHEN ensure_iris_vector_graph_tables() is called
- THEN successful tables are created
- AND InitializationResult.tables_created shows mixed success/failure
- AND InitializationResult.error_messages contains details for failed tables

---

## Contract 3: Prerequisite Validation

### Method Signature
```python
def validate_graph_prerequisites(self) -> ValidationResult:
    """
    Validate that all iris-vector-graph prerequisites are met.

    Returns:
        ValidationResult with:
            - is_valid: Whether all prerequisites are met
            - package_installed: Whether iris-vector-graph is installed
            - missing_tables: List of tables that don't exist
            - error_message: Human-readable error message if not valid

    Implementation Notes:
        - First checks package installation via _detect_iris_vector_graph()
        - If package not installed, returns invalid with appropriate message
        - If package installed, checks existence of all 4 required tables
        - Uses SchemaManager.table_exists() for each table check
        - Provides specific list of missing components in error_message
    """
```

### Preconditions
- SchemaManager instance initialized with valid database connection

### Postconditions
**All Prerequisites Met**:
- ValidationResult.is_valid == True
- ValidationResult.package_installed == True
- ValidationResult.missing_tables == []
- ValidationResult.error_message == ""

**Package Not Installed**:
- ValidationResult.is_valid == False
- ValidationResult.package_installed == False
- ValidationResult.missing_tables == []
- ValidationResult.error_message contains "iris-vector-graph package not installed"

**Tables Missing**:
- ValidationResult.is_valid == False
- ValidationResult.package_installed == True
- ValidationResult.missing_tables contains specific table names
- ValidationResult.error_message lists missing tables

### Invariants
- Stateless operation (no side effects)
- Validation time < 1 second
- Never raises exceptions (returns validation status)

### Test Contracts

**Test**: `test_prerequisite_validation_all_met()`
- GIVEN iris-vector-graph is installed
- AND all 4 graph tables exist in database
- WHEN validate_graph_prerequisites() is called
- THEN ValidationResult.is_valid == True
- AND ValidationResult.package_installed == True
- AND ValidationResult.missing_tables == []
- AND ValidationResult.error_message == ""

**Test**: `test_prerequisite_validation_package_missing()`
- GIVEN iris-vector-graph is NOT installed
- WHEN validate_graph_prerequisites() is called
- THEN ValidationResult.is_valid == False
- AND ValidationResult.package_installed == False
- AND ValidationResult.error_message contains "iris-vector-graph package not installed"

**Test**: `test_prerequisite_validation_tables_missing()`
- GIVEN iris-vector-graph is installed
- AND only 2 of 4 graph tables exist (rdf_labels and rdf_props present, others missing)
- WHEN validate_graph_prerequisites() is called
- THEN ValidationResult.is_valid == False
- AND ValidationResult.package_installed == True
- AND ValidationResult.missing_tables == ["rdf_edges", "kg_NodeEmbeddings_optimized"]
- AND ValidationResult.error_message lists specific missing tables

**Test**: `test_prerequisite_validation_before_ppr()`
- GIVEN iris-vector-graph is installed
- AND NOT all tables exist
- WHEN validate_graph_prerequisites() is called before PPR operation
- THEN ValidationResult.is_valid == False
- AND calling code can raise RuntimeError with specific missing components

---

## Contract 4: Error Message Clarity

### Test Contract
**Test**: `test_clear_error_when_tables_missing()`
- GIVEN iris-vector-graph is installed
- AND tables rdf_edges and kg_NodeEmbeddings_optimized are missing
- WHEN validate_graph_prerequisites() is called
- THEN error_message clearly indicates:
  - Which specific tables are missing
  - Suggested remediation (run ensure_iris_vector_graph_tables())
  - Distinguishes from "package not installed" error

**Error Message Format**:
```
Missing required iris-vector-graph tables: rdf_edges, kg_NodeEmbeddings_optimized
Run SchemaManager.ensure_iris_vector_graph_tables() to initialize tables.
```

---

## Data Structure Contracts

### InitializationResult

```python
@dataclass
class InitializationResult:
    """Result of iris-vector-graph table initialization."""

    package_detected: bool
    """Whether iris-vector-graph package was found."""

    tables_attempted: List[str]
    """Tables that initialization was attempted for."""

    tables_created: Dict[str, bool]
    """Mapping of table names to creation success status."""

    total_time_seconds: float
    """Total time taken for initialization."""

    error_messages: Dict[str, str]
    """Error messages for failed tables (empty dict if all succeeded)."""
```

**Invariants**:
- If package_detected == False, then tables_created must be empty dict
- len(tables_created) must equal len(tables_attempted)
- error_messages keys must be subset of tables_attempted
- total_time_seconds must be >= 0

### ValidationResult

```python
@dataclass
class ValidationResult:
    """Result of prerequisite validation."""

    is_valid: bool
    """Whether all prerequisites are met."""

    package_installed: bool
    """Whether iris-vector-graph is installed."""

    missing_tables: List[str]
    """Tables that don't exist (empty if all present)."""

    error_message: str
    """Human-readable error message (empty if valid)."""
```

**Invariants**:
- If package_installed == False, then is_valid must be False
- If is_valid == True, then missing_tables must be empty
- If is_valid == False, then error_message must be non-empty
- error_message should list specific missing components

---

## Performance Contracts

### Initialization Performance
- **Target**: Complete table creation in < 5 seconds
- **Measurement**: InitializationResult.total_time_seconds
- **Test**: `test_initialization_performance()`
  - GIVEN iris-vector-graph is installed
  - WHEN ensure_iris_vector_graph_tables() is called on empty database
  - THEN total_time_seconds < 5.0

### Validation Performance
- **Target**: Complete validation in < 1 second
- **Measurement**: Execution time of validate_graph_prerequisites()
- **Test**: `test_validation_performance()`
  - GIVEN iris-vector-graph is installed and tables exist
  - WHEN validate_graph_prerequisites() is called
  - THEN execution completes in < 1.0 seconds

---

## Integration Points

### SchemaManager.ensure_table_schema()
**Contract**: Must be used for individual table creation
**Behavior**: Returns bool indicating success/failure
**Usage**: Called once per table with pipeline_type parameter

### SchemaManager.table_exists()
**Contract**: Must be used for table existence checks
**Behavior**: Returns bool indicating table presence
**Usage**: Called once per table during validation

### Logging
**Contract**: Must log at appropriate levels
- INFO: Successful table creation, package detection
- DEBUG: Package not installed (expected condition)
- ERROR: Table creation failures, permission issues

---

## Backward Compatibility Contracts

### Existing Pipeline Behavior
**Contract**: Pipelines without iris-vector-graph must continue to work exactly as before

**Test**: `test_backward_compatibility_without_package()`
- GIVEN iris-vector-graph is NOT installed
- WHEN existing pipeline initialization runs
- THEN no errors are raised
- AND pipeline proceeds normally
- AND no graph tables are created

### Existing SchemaManager API
**Contract**: All existing SchemaManager methods retain their current behavior

**Test**: `test_existing_api_unchanged()`
- GIVEN SchemaManager with new methods added
- WHEN existing methods (ensure_table_schema, table_exists, etc.) are called
- THEN they behave exactly as before
- AND no regressions in existing functionality

---

## Constitution Compliance Contracts

### TDD Requirement (Constitution III)
**Contract**: All contract tests must be written BEFORE implementation
**Validation**: Tests initially fail with ImportError or AttributeError
**Test Files**:
- `tests/contract/test_graph_schema_detection.py`
- `tests/contract/test_graph_schema_initialization.py`
- `tests/contract/test_graph_schema_validation.py`

### Explicit Error Handling (Constitution VI)
**Contract**: No silent failures
**Validation**: All errors logged, validation results explicit
**Test**: `test_no_silent_failures()`
- GIVEN tables missing
- WHEN PPR operation attempts to run
- THEN explicit RuntimeError raised with specific missing components
- AND NOT silent fallback to uniform scoring

### Standardized Database Interfaces (Constitution VII)
**Contract**: Use existing SchemaManager patterns
**Validation**: No direct SQL queries, use ensure_table_schema() and table_exists()
**Review**: Code review confirms no ad-hoc database access

---

## Status

✅ **Contracts Defined**
- [x] Package detection contract
- [x] Table initialization contract
- [x] Prerequisite validation contract
- [x] Error message clarity contract
- [x] Data structure contracts
- [x] Performance contracts
- [x] Integration point contracts
- [x] Backward compatibility contracts
- [x] Constitution compliance contracts

**Next Steps**: Write failing contract tests in `tests/contract/` directory
