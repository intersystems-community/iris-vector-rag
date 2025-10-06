# Data Model: Test Coverage Enhancement

## Core Entities

### CoverageReport
**Purpose**: Represents a complete coverage analysis result for the codebase
**Fields**:
- `report_id: str` - Unique identifier for the coverage run
- `timestamp: datetime` - When the coverage analysis was performed
- `overall_coverage_percentage: float` - Overall coverage across entire codebase (target: 60%+)
- `total_lines: int` - Total lines of code analyzed
- `covered_lines: int` - Lines covered by tests
- `branch_coverage_percentage: float` - Branch coverage analysis
- `function_coverage_percentage: float` - Function coverage analysis
- `module_coverage: List[ModuleCoverage]` - Per-module coverage breakdown
- `analysis_duration_seconds: float` - Time taken for coverage analysis (target: <300s)
- `test_execution_duration_seconds: float` - Time taken for test execution with coverage
- `git_commit_hash: str` - Git commit associated with this coverage run
- `ci_build_id: str` - CI build identifier (optional)

**Validation Rules**:
- `overall_coverage_percentage >= 60.0` for production builds
- `analysis_duration_seconds <= 300.0` (5 minutes maximum)
- `test_execution_duration_seconds <= baseline_duration * 2.0`

### ModuleCoverage
**Purpose**: Coverage metrics for individual code modules
**Fields**:
- `module_name: str` - Name of the module (e.g., "iris_rag.config")
- `file_path: str` - Relative path to the module file
- `coverage_percentage: float` - Coverage percentage for this module
- `total_lines: int` - Total lines in module
- `covered_lines: int` - Covered lines in module
- `uncovered_lines: List[int]` - Line numbers not covered by tests
- `is_critical_module: bool` - Whether module requires 80% coverage
- `is_legacy_module: bool` - Whether module has reduced coverage requirements
- `target_coverage_percentage: float` - Target coverage for this module
- `priority_level: str` - "HIGH" | "MEDIUM" | "LOW" based on clarifications

**Validation Rules**:
- Critical modules: `coverage_percentage >= 80.0`
- Regular modules: `coverage_percentage >= 60.0`
- Legacy modules: `coverage_percentage >= legacy_target` (configurable)
- Priority HIGH modules must be tested first (config, validation)

### TestSuite
**Purpose**: Represents the complete test suite configuration and results
**Fields**:
- `suite_id: str` - Unique identifier for test suite configuration
- `total_tests: int` - Total number of tests in suite
- `passing_tests: int` - Number of passing tests
- `failing_tests: int` - Number of failing tests
- `skipped_tests: int` - Number of skipped tests
- `test_categories: List[TestCategory]` - Breakdown by test type
- `async_tests: int` - Number of async tests (special handling required)
- `database_tests: int` - Number of tests requiring IRIS database
- `mocked_tests: int` - Number of tests using mocks
- `execution_time_seconds: float` - Total test execution time

**State Transitions**:
- PENDING → RUNNING → COMPLETED
- COMPLETED → FAILED (if coverage targets not met)
- FAILED → RETRYING (for transient failures)

### TestCategory
**Purpose**: Categorizes tests by type and execution requirements
**Fields**:
- `category_name: str` - "unit" | "integration" | "e2e"
- `test_count: int` - Number of tests in this category
- `coverage_contribution: float` - Coverage percentage contributed by this category
- `execution_time_seconds: float` - Time taken for this category
- `requires_database: bool` - Whether category needs IRIS database
- `requires_mocking: bool` - Whether category uses mocked dependencies

### CoverageTrend
**Purpose**: Tracks coverage changes over time for monthly reporting
**Fields**:
- `trend_id: str` - Unique identifier for trend record
- `month_year: str` - Month and year (e.g., "2025-10")
- `baseline_coverage: float` - Coverage at start of month
- `current_coverage: float` - Current coverage percentage
- `coverage_delta: float` - Change in coverage during month
- `milestone_achieved: bool` - Whether monthly milestone was met
- `critical_modules_at_target: int` - Number of critical modules at 80%
- `total_critical_modules: int` - Total number of critical modules

## Relationships

### CoverageReport → ModuleCoverage
- One-to-many: Each coverage report contains multiple module coverage records
- Cascade delete: Removing coverage report removes all associated module coverage

### TestSuite → TestCategory
- One-to-many: Each test suite contains multiple test categories
- Aggregate relationship: Test suite metrics calculated from category metrics

### CoverageReport → CoverageTrend
- Many-to-one: Multiple coverage reports contribute to monthly trend data
- Calculated relationship: Trend data derived from coverage report time series

## Configuration Entities

### CoverageConfiguration
**Purpose**: Stores coverage targets and configuration settings
**Fields**:
- `config_id: str` - Configuration identifier
- `overall_target_percentage: float` - Overall coverage target (60%)
- `critical_module_target_percentage: float` - Critical module target (80%)
- `legacy_module_target_percentage: float` - Legacy module target (configurable)
- `max_analysis_duration_seconds: int` - Maximum analysis time (300)
- `max_test_overhead_multiplier: float` - Maximum test overhead (2.0)
- `critical_modules: List[str]` - List of critical module names
- `legacy_modules: List[str]` - List of legacy module names with exemptions
- `excluded_patterns: List[str]` - File patterns to exclude from coverage
- `reporting_frequency: str` - "MONTHLY" based on clarifications

### LegacyModuleExemption
**Purpose**: Documents exemptions for difficult-to-test legacy modules
**Fields**:
- `module_name: str` - Name of exempted module
- `reduced_target_percentage: float` - Reduced coverage target for this module
- `justification: str` - Written justification for exemption
- `review_date: date` - When exemption should be reviewed
- `improvement_plan: str` - Optional plan for future improvement
- `approved_by: str` - Who approved the exemption

## Data Flow

```
CI/CD Pipeline → TestSuite Execution → CoverageReport Generation → ModuleCoverage Analysis
       ↓                    ↓                    ↓                        ↓
  Test Results    Coverage Metrics    Report Storage    Monthly Trend Update
       ↓                    ↓                    ↓                        ↓
  Pass/Fail       Target Validation   Artifact Storage   Milestone Reporting
```

## Validation Rules Summary

1. **Coverage Targets**:
   - Overall: ≥60%
   - Critical modules: ≥80%
   - Legacy modules: ≥exemption_target with justification

2. **Performance Constraints**:
   - Coverage analysis: ≤5 minutes
   - Test execution overhead: ≤2x baseline

3. **Quality Requirements**:
   - No flaky tests introduced
   - Deterministic test results
   - Meaningful test coverage (not just line coverage)