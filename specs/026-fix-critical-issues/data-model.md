# Data Model: Fix Critical Issues from Analyze Command

**Feature**: 026-fix-critical-issues
**Version**: 1.0.0

## Entity Definitions

### 1. CoverageWarning

Represents a coverage threshold violation detected during test execution.

**Attributes**:
- `module_path` (str): Full Python module path (e.g., "iris_rag.pipelines.basic")
- `current_coverage` (float): Actual coverage percentage (0.0-100.0)
- `threshold` (float): Expected minimum coverage (60.0 or 80.0)
- `severity` (enum): WARNING (below 60%) or CRITICAL (critical module below 80%)
- `missing_lines` (list[int]): Line numbers not covered by tests
- `timestamp` (datetime): When the warning was generated

**Relationships**:
- References CriticalModuleConfig if module is critical
- Part of CoverageReport aggregate

**Validation Rules**:
- current_coverage must be between 0.0 and 100.0
- threshold must be 60.0 or 80.0
- severity must match threshold violation type
- module_path must be valid Python module

### 2. ErrorMessageValidation

Validates test failure messages meet quality standards.

**Attributes**:
- `test_name` (str): Full test identifier (e.g., "test_module.py::TestClass::test_method")
- `error_message` (str): The actual error message produced
- `has_what` (bool): Message describes what failed
- `has_why` (bool): Message explains why it failed
- `has_action` (bool): Message suggests corrective action
- `validation_status` (enum): VALID, MISSING_WHAT, MISSING_WHY, MISSING_ACTION, INVALID
- `suggestions` (list[str]): Improvements for invalid messages

**Relationships**:
- Associated with test execution results
- Aggregated in validation reports

**Validation Rules**:
- test_name must match pytest node ID format
- validation_status must reflect component analysis
- At least one suggestion required if status != VALID

### 3. ContractTestState

Tracks TDD compliance for contract tests.

**Attributes**:
- `test_file` (str): Path to contract test file (e.g., "tests/contract/test_foo_contract.py")
- `initial_state` (enum): FAIL, PASS, ERROR, SKIP, NOT_FOUND
- `initial_commit` (str): Git commit SHA where test was introduced
- `implementation_commit` (str): Git commit SHA where implementation was added
- `compliance_status` (enum): COMPLIANT (failed first), VIOLATION (never failed), UNKNOWN
- `violation_reason` (str): Explanation if non-compliant

**Relationships**:
- Links to git commit history
- Part of TDD compliance report

**Validation Rules**:
- test_file must match contract test pattern
- initial_commit must exist in git history
- compliance_status must match state analysis
- violation_reason required if status == VIOLATION

### 4. RequirementTaskMapping

Links requirements to implementation tasks.

**Attributes**:
- `requirement_id` (str): Requirement identifier (e.g., "FR-001")
- `requirement_text` (str): Brief requirement description
- `task_ids` (list[str]): Task identifiers that implement this requirement
- `coverage_status` (enum): COVERED, PARTIAL, MISSING
- `validation_timestamp` (datetime): When mapping was validated
- `source_file` (str): Path to spec.md containing requirement

**Relationships**:
- Links spec.md requirements to tasks.md tasks
- Aggregated in coverage gap reports

**Validation Rules**:
- requirement_id must match FR-XXX pattern
- coverage_status must reflect task_ids presence
- task_ids must exist in tasks.md if COVERED

### 5. CriticalModuleConfig

Configuration for modules requiring higher coverage.

**Attributes**:
- `module_pattern` (str): Glob pattern or module path (e.g., "iris_rag/pipelines/*")
- `threshold` (float): Required coverage percentage (default: 80.0)
- `rationale` (str): Why this module is critical
- `active` (bool): Whether this rule is currently enforced

**Relationships**:
- Used by CoverageWarning to determine severity
- Configured in .coveragerc

**Validation Rules**:
- module_pattern must be valid glob/module syntax
- threshold must be between 0.0 and 100.0
- rationale required for documentation

## Aggregate Entities

### CoverageReport

Aggregates all coverage warnings for a test run.

**Attributes**:
- `total_modules` (int): Number of modules analyzed
- `modules_below_threshold` (int): Count of modules with warnings
- `critical_violations` (int): Count of critical modules below 80%
- `overall_coverage` (float): Repository-wide coverage percentage
- `warnings` (list[CoverageWarning]): All warnings generated
- `report_timestamp` (datetime): When report was generated

### ValidationReport

Aggregates all validation results for a test run.

**Attributes**:
- `total_tests` (int): Number of tests with errors
- `valid_messages` (int): Count of well-formed error messages
- `invalid_messages` (int): Count of messages needing improvement
- `validations` (list[ErrorMessageValidation]): All validations performed
- `common_issues` (dict[str, int]): Frequency of validation failures

### TDDComplianceReport

Aggregates TDD compliance across all contract tests.

**Attributes**:
- `total_contract_tests` (int): Number of contract tests found
- `compliant_tests` (int): Tests that failed before implementation
- `violation_tests` (int): Tests that never failed
- `unknown_tests` (int): Tests with unclear history
- `test_states` (list[ContractTestState]): All contract test analyses
- `overall_compliance` (float): Percentage of compliant tests

### RequirementCoverageReport

Aggregates requirement-to-task mapping analysis.

**Attributes**:
- `total_requirements` (int): Number of requirements in spec
- `covered_requirements` (int): Requirements with tasks
- `missing_requirements` (int): Requirements without tasks
- `mappings` (list[RequirementTaskMapping]): All requirement mappings
- `coverage_percentage` (float): Percentage of requirements with tasks
- `edge_case_coverage` (dict[str, bool]): Edge case test coverage

## State Transitions

### CoverageWarning States
```
NONE → WARNING (coverage < 60%) → RESOLVED (coverage >= 60%)
NONE → CRITICAL (critical module < 80%) → RESOLVED (coverage >= 80%)
```

### ErrorMessageValidation States
```
UNCHECKED → VALID (all components present)
UNCHECKED → INVALID (missing components) → IMPROVED (suggestions applied)
```

### ContractTestState States
```
NOT_FOUND → FAIL (test added) → PASS (implementation added) = COMPLIANT
NOT_FOUND → PASS (test and impl added together) = VIOLATION
```

### RequirementTaskMapping States
```
MISSING (no tasks) → PARTIAL (some tasks) → COVERED (sufficient tasks)
```

## Example Usage

```python
# Coverage warning example
warning = CoverageWarning(
    module_path="iris_rag.pipelines.basic",
    current_coverage=45.5,
    threshold=80.0,
    severity="CRITICAL",
    missing_lines=[23, 45, 67, 89],
    timestamp=datetime.now()
)

# Error validation example
validation = ErrorMessageValidation(
    test_name="test_basic_pipeline.py::test_load_documents",
    error_message="AssertionError: assert False",
    has_what=True,  # "AssertionError"
    has_why=False,   # No explanation
    has_action=False,  # No suggestion
    validation_status="MISSING_WHY",
    suggestions=["Add assertion message explaining expected vs actual"]
)

# TDD compliance example
compliance = ContractTestState(
    test_file="tests/contract/test_coverage_contract.py",
    initial_state="FAIL",
    initial_commit="abc123",
    implementation_commit="def456",
    compliance_status="COMPLIANT",
    violation_reason=None
)

# Requirement mapping example
mapping = RequirementTaskMapping(
    requirement_id="FR-007",
    requirement_text="System MUST warn on low coverage",
    task_ids=[],
    coverage_status="MISSING",
    validation_timestamp=datetime.now(),
    source_file="specs/025-fixes-for-testing/spec.md"
)
```

## Storage Considerations

These entities are primarily used for:
1. **Runtime validation**: Temporary objects during test execution
2. **Reporting**: Markdown/JSON output files
3. **Historical tracking**: Optional database storage for trends

No persistent storage required for MVP - all data is transient during test runs.