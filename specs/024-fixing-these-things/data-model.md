# Data Model: Quality Improvement Initiative

**Generated**: 2025-10-02
**Feature**: RAG-Templates Quality Improvement

## Overview

This data model defines the entities and relationships for tracking test quality, coverage metrics, and CI/CD pipeline status in the RAG-Templates framework.

## Entities

### TestResult
Represents the execution result of a single test.

**Fields**:
- `test_id`: string (unique identifier, e.g., "test_module::TestClass::test_method")
- `test_name`: string (human-readable test name)
- `module_path`: string (e.g., "tests/unit/test_config_unit.py")
- `status`: enum ["PASS", "FAIL", "SKIP", "ERROR"]
- `failure_type`: string? (e.g., "AttributeError", "AssertionError", "MockError")
- `failure_message`: string? (detailed error message)
- `execution_time`: float (seconds)
- `timestamp`: datetime
- `commit_hash`: string (git commit hash)

**Validation Rules**:
- test_id must follow pytest naming convention
- status must be one of the defined enum values
- execution_time must be non-negative
- timestamp must be UTC

### CoverageMetric
Represents code coverage data for a module or file.

**Fields**:
- `module_name`: string (e.g., "iris_rag.config.manager")
- `file_path`: string (relative path from repo root)
- `total_lines`: integer (>= 0)
- `covered_lines`: integer (>= 0, <= total_lines)
- `coverage_percentage`: float (0.0-100.0)
- `is_critical`: boolean (true for config/validation/pipelines/services/storage)
- `target_coverage`: float (80.0 for critical, 60.0 for others)
- `branch_coverage`: float? (0.0-100.0)
- `timestamp`: datetime
- `commit_hash`: string

**Validation Rules**:
- coverage_percentage must equal (covered_lines / total_lines * 100)
- is_critical determined by module_name prefix
- target_coverage based on is_critical flag

### QualityGate
Represents a CI/CD quality check result.

**Fields**:
- `gate_id`: string (unique identifier)
- `gate_type`: enum ["TEST_PASS", "COVERAGE_OVERALL", "COVERAGE_CRITICAL", "PERFORMANCE", "LINT"]
- `status`: enum ["PASS", "FAIL", "WARNING"]
- `actual_value`: float (e.g., 95.5 for percentage, 150.3 for seconds)
- `expected_value`: float (threshold)
- `comparison_operator`: enum [">=", "<=", "==", "<", ">"]
- `message`: string (human-readable result)
- `timestamp`: datetime
- `build_id`: string (CI/CD build identifier)

**Validation Rules**:
- actual_value and expected_value must be same unit/type
- comparison_operator must be valid for gate_type
- message must explain pass/fail reason

### TestSession
Aggregates test results for a complete test run.

**Fields**:
- `session_id`: string (unique identifier)
- `start_time`: datetime
- `end_time`: datetime
- `total_tests`: integer (>= 0)
- `passed_tests`: integer (>= 0)
- `failed_tests`: integer (>= 0)
- `skipped_tests`: integer (>= 0)
- `error_tests`: integer (>= 0)
- `overall_coverage`: float (0.0-100.0)
- `critical_coverage`: object (module -> percentage mapping)
- `quality_gates`: array[QualityGate]
- `environment`: object (Python version, OS, dependencies)

**Validation Rules**:
- total_tests must equal sum of passed/failed/skipped/error
- end_time must be after start_time
- all coverage values must be valid percentages

## Relationships

```
TestSession
    |
    +-- has many --> TestResult
    |
    +-- has many --> CoverageMetric
    |
    +-- has many --> QualityGate
```

## State Transitions

### TestResult States
```
PENDING -> RUNNING -> [PASS|FAIL|ERROR]
                 \-> SKIP
```

### QualityGate States
```
PENDING -> EVALUATING -> [PASS|FAIL|WARNING]
```

## Usage Examples

### Creating a Test Session
```python
session = TestSession(
    session_id=f"session_{timestamp}",
    start_time=datetime.utcnow(),
    total_tests=349,
    environment={
        "python_version": "3.11.5",
        "os": "Darwin",
        "pytest_version": "7.4.0"
    }
)
```

### Recording Test Result
```python
result = TestResult(
    test_id="tests.unit.test_config_unit::TestConfigurationManager::test_get_method",
    test_name="test_get_method",
    module_path="tests/unit/test_config_unit.py",
    status="FAIL",
    failure_type="AttributeError",
    failure_message="'ConfigurationManager' object has no attribute 'get_value'",
    execution_time=0.023,
    timestamp=datetime.utcnow(),
    commit_hash="abc123"
)
```

### Evaluating Quality Gate
```python
gate = QualityGate(
    gate_id="coverage_overall_check",
    gate_type="COVERAGE_OVERALL",
    status="FAIL",
    actual_value=9.0,
    expected_value=60.0,
    comparison_operator=">=",
    message="Overall coverage 9.0% is below required 60.0%",
    timestamp=datetime.utcnow(),
    build_id="github-actions-123"
)
```

## Migration Notes

Since this is a quality improvement initiative for testing infrastructure, these entities will be used for:
1. Tracking progress during test repair phase
2. CI/CD pipeline integration
3. Coverage monitoring dashboard
4. Historical quality trend analysis

The data will be stored in:
- JSON files for local development tracking
- CI/CD artifacts for pipeline results
- Potentially IRIS database for historical analysis