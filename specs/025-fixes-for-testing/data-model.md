# Data Model: Testing Framework Entities

**Feature**: 025-fixes-for-testing
**Date**: 2025-10-03
**Phase**: Phase 1 - Design

## Entity Definitions

### 1. TestCase

**Description**: Represents a single test execution instance with results and metadata.

**Attributes**:
- `name` (string): Full test path (e.g., "tests/e2e/test_basic_pipeline_e2e.py::TestBasicRAGPipelineQuerying::test_simple_query")
- `status` (enum): Test outcome - "PASSED", "FAILED", "SKIPPED", "ERROR"
- `execution_time` (float): Time in seconds to execute the test
- `coverage_lines` (list[int]): Line numbers covered by this test
- `error_message` (string, optional): Error details if status is FAILED or ERROR
- `skip_reason` (string, optional): Explanation if status is SKIPPED

**Validation Rules**:
- name must be non-empty and follow pytest node ID format
- status must be one of the four valid enum values
- execution_time must be >= 0
- if status is FAILED or ERROR, error_message must be present
- if status is SKIPPED, skip_reason should be present

**Relationships**:
- Belongs to one TestSuite (e.g., "unit", "e2e", "integration")
- Contributes to one or more CoverageReports (one per module tested)

**State Transitions**:
```
NOT_STARTED → RUNNING → (PASSED | FAILED | ERROR)
                      → SKIPPED (can skip before running)
```

### 2. CoverageReport

**Description**: Aggregates line coverage data for a single module.

**Attributes**:
- `module_name` (string): Python module path (e.g., "iris_rag.pipelines.basic")
- `total_lines` (int): Total executable lines in the module
- `covered_lines` (int): Lines covered by at least one test
- `percentage` (float): Coverage percentage (covered_lines / total_lines * 100)
- `missing_lines` (list[int]): Line numbers not covered by any test
- `partial_lines` (list[int]): Lines with branch coverage gaps

**Validation Rules**:
- module_name must match actual Python module path
- total_lines must be > 0
- covered_lines must be <= total_lines
- percentage must be 0-100
- covered_lines + missing_lines count should equal total_lines (approximately, branches complicate this)

**Relationships**:
- Aggregates coverage from multiple TestCases
- Belongs to one CoverageSummary (overall project coverage)

**Derived Values**:
- `uncovered_percentage`: 100 - percentage
- `is_critical_module`: True if module_name starts with "iris_rag.pipelines", "iris_rag.storage", or "iris_rag.validation"
- `meets_target`: percentage >= 60 for overall, >= 80 for critical modules

### 3. APIContract

**Description**: Defines expected behavior of a production API that tests must validate.

**Attributes**:
- `api_name` (string): Full API identifier (e.g., "BasicRAGPipeline.load_documents")
- `expected_signature` (string): Expected function signature from test
- `actual_signature` (string): Actual function signature from production code
- `match_status` (enum): "MATCH", "MISMATCH", "NOT_FOUND"
- `test_file` (string): Test file validating this contract (e.g., "tests/e2e/test_basic_pipeline_e2e.py")
- `production_file` (string): Production file implementing this API (e.g., "iris_rag/pipelines/basic.py")

**Validation Rules**:
- api_name must be in format "ClassName.method_name" or "function_name"
- If match_status is MISMATCH, expected_signature != actual_signature
- If match_status is NOT_FOUND, production_file does not contain api_name
- test_file and production_file must exist in repository

**Relationships**:
- Validated by one or more TestCases
- Belongs to one ProductionAPI (e.g., BasicRAGPipeline, IRISVectorStore)

**Examples**:
```python
# MATCH:
APIContract(
    api_name="BasicRAGPipeline.query",
    expected_signature="query(query: str, top_k: int = 5, generate_answer: bool = True) -> Dict",
    actual_signature="query(query: str, top_k: int = 5, generate_answer: bool = True) -> Dict",
    match_status="MATCH"
)

# MISMATCH:
APIContract(
    api_name="BasicRAGPipeline.load_documents",
    expected_signature="load_documents(documents: List[Document]) -> None",
    actual_signature="load_documents(documents_path: str, documents: List[Document] = None) -> None",
    match_status="MISMATCH"  # Test expects positional, actual uses kwarg
)
```

### 4. TestFixture

**Description**: Reusable test setup providing dependencies to test cases.

**Attributes**:
- `name` (string): Fixture function name (e.g., "iris_connection", "sample_documents")
- `scope` (enum): "function", "class", "module", "session"
- `setup_code` (string): Code executed to create fixture value
- `teardown_code` (string, optional): Code executed to clean up fixture
- `dependencies` (list[string]): Names of other fixtures this fixture depends on
- `provides_type` (string): Type of value provided (e.g., "ConnectionManager", "List[Document]")

**Validation Rules**:
- name must be valid Python identifier
- scope must be one of the four valid pytest scopes
- dependencies must reference existing fixtures
- If teardown_code is present, it must be valid Python

**Relationships**:
- Depends on zero or more other TestFixtures
- Provides dependency for one or more TestCases

**Examples**:
```python
# Module-scoped IRIS connection
TestFixture(
    name="iris_connection",
    scope="module",
    setup_code="""
        config_manager = ConfigurationManager()
        conn_manager = ConnectionManager(config_manager)
        return conn_manager.get_connection()
    """,
    teardown_code="conn.close()",
    dependencies=["config_manager"],
    provides_type="Connection"
)

# Function-scoped sample documents
TestFixture(
    name="sample_documents",
    scope="function",
    setup_code="""
        return [
            Document(id="doc1", page_content="Test content 1"),
            Document(id="doc2", page_content="Test content 2")
        ]
    """,
    teardown_code=None,  # No cleanup needed
    dependencies=[],
    provides_type="List[Document]"
)
```

### 5. IrisConnection

**Description**: Represents a connection to an IRIS database instance for testing.

**Attributes**:
- `host` (string): IRIS hostname (usually "localhost")
- `port` (int): IRIS SuperServer port (11972, 21972, or 31972)
- `username` (string): IRIS username (usually "_SYSTEM")
- `password` (string): IRIS password
- `namespace` (string): IRIS namespace (usually "IRIS")
- `connection_handle` (object): Active database connection handle
- `is_healthy` (bool): Whether connection passed health check
- `version` (string): IRIS version (e.g., "2025.3.0EHAT.127.0")

**Validation Rules**:
- host must be valid hostname or IP address
- port must be 1-65535
- connection_handle must be non-null if is_healthy is True
- Before use, must call health check validation

**Relationships**:
- Created by ConnectionManager
- Used by TestCases that require IRIS database
- May be shared across tests via TestFixture(scope="module")

**State Transitions**:
```
DISCONNECTED → CONNECTING → CONNECTED → VALIDATED (health check passed)
                         → FAILED (connection error)
CONNECTED → DISCONNECTED (after test cleanup)
```

**Health Check Requirements** (per constitution):
```python
def validate_iris_connection(conn: IrisConnection) -> bool:
    """Validate IRIS connection before test execution."""
    if not conn.is_healthy:
        return False

    # Test basic query
    cursor = conn.connection_handle.cursor()
    cursor.execute("SELECT 1")
    result = cursor.fetchone()

    # Test namespace access
    cursor.execute("SELECT $NAMESPACE")
    namespace = cursor.fetchone()[0]

    return result[0] == 1 and namespace == conn.namespace
```

## Entity Relationships Diagram

```
TestSuite (e.g., "e2e")
  ├── TestCase[1..*]
  │     ├── uses TestFixture[0..*]
  │     ├── validates APIContract[0..*]
  │     ├── contributes to CoverageReport[0..*]
  │     └── may use IrisConnection[0..1]
  │
  ├── CoverageSummary
  │     └── aggregates CoverageReport[1..*]
  │           └── tracks coverage from TestCase[1..*]
  │
  └── TestFixture[0..*]
        ├── depends on TestFixture[0..*] (transitive)
        ├── may create IrisConnection[0..1]
        └── provides dependency for TestCase[1..*]

APIContract[1..*]
  ├── validated by TestCase[1..*]
  └── implemented in ProductionAPI (e.g., BasicRAGPipeline)

IrisConnection
  ├── created by TestFixture (scope="module")
  └── used by TestCase[1..*] (E2E tests only)
```

## Usage Patterns

### Pattern 1: Test Execution Flow

```python
# 1. Setup: Fixture creates IRIS connection
iris_conn = IrisConnection(host="localhost", port=21972, ...)
assert validate_iris_connection(iris_conn)

# 2. Test: Validate API contract
test_case = TestCase(name="test_load_documents", status="RUNNING")
api_contract = APIContract(api_name="BasicRAGPipeline.load_documents", ...)

# Execute test, capture coverage
pipeline.load_documents("", documents=sample_documents)  # Match actual signature
test_case.status = "PASSED"
test_case.coverage_lines = [45, 46, 47, ...]  # Lines executed

# 3. Teardown: Close connection
iris_conn.connection_handle.close()
iris_conn.is_healthy = False
```

### Pattern 2: Coverage Reporting

```python
# Aggregate coverage from all test cases
coverage_report = CoverageReport(module_name="iris_rag.pipelines.basic")

for test_case in test_cases:
    coverage_report.covered_lines.extend(test_case.coverage_lines)

# Calculate metrics
coverage_report.covered_lines = set(coverage_report.covered_lines)  # Deduplicate
coverage_report.percentage = len(coverage_report.covered_lines) / coverage_report.total_lines * 100

# Check target
if coverage_report.is_critical_module:
    assert coverage_report.percentage >= 80, "Critical module must have 80% coverage"
else:
    assert coverage_report.percentage >= 60, "Overall coverage must be 60%"
```

### Pattern 3: API Contract Validation

```python
# Detect API mismatch
import inspect

actual_sig = inspect.signature(BasicRAGPipeline.load_documents)
expected_sig = "load_documents(documents: List[Document]) -> None"  # From test

contract = APIContract(
    api_name="BasicRAGPipeline.load_documents",
    expected_signature=expected_sig,
    actual_signature=str(actual_sig),
    match_status="MISMATCH" if str(actual_sig) != expected_sig else "MATCH"
)

if contract.match_status == "MISMATCH":
    # Fix test to match production
    print(f"Update test: {contract.test_file}")
    print(f"Expected: {contract.expected_signature}")
    print(f"Actual:   {contract.actual_signature}")
```

## Next Steps

Use these entity definitions to:
1. Create API contracts (Phase 1: contracts/ directory)
2. Design test fixtures (Phase 1: update conftest.py files)
3. Generate tasks (Phase 2: /tasks command)
4. Implement test fixes (Phase 4: execution)
