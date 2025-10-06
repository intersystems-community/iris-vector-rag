# Tasks: Fix Critical Issues from Analyze Command

**Feature**: 026-fix-critical-issues
**Branch**: `026-fix-critical-issues`
**Input**: Design documents from `/specs/026-fix-critical-issues/`
**Prerequisites**: plan.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

## Execution Flow (main)
```
1. Load plan.md from feature directory ✅
   → Tech stack: Python 3.12, pytest 8.4.1, coverage 7.6.1, GitPython
   → Structure: Single project (tests/plugins/, scripts/)
2. Load optional design documents ✅
   → data-model.md: 5 entities (CoverageWarning, ErrorMessageValidation, etc.)
   → contracts/: 4 contract files (coverage, error, TDD, mapping)
   → research.md: 5 research areas (hooks, validation, git analysis)
3. Generate tasks by category ✅
   → Setup: Plugin directory, dependencies
   → Tests: 4 contract tests [P]
   → Core: 4 plugin implementations
   → Integration: 2 scripts, configuration
   → Polish: Documentation, examples
4. Apply task rules ✅
   → Contract tests marked [P] (different files)
   → Plugin implementations sequential (shared infrastructure)
   → Scripts marked [P] (independent)
5. Number tasks sequentially (T001-T021) ✅
6. Generate dependency graph ✅
7. Create parallel execution examples ✅
8. Validate task completeness ✅
   → All contracts have tests: 4/4 ✅
   → All plugins have implementations: 4/4 ✅
   → Configuration tasks included: 3/3 ✅
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions
- Plugin implementations: `tests/plugins/`
- Contract tests: `tests/contract/`
- Standalone scripts: `scripts/`
- Configuration files: Repository root

---

## Phase 3.1: Setup & Infrastructure (4 tasks)

### T001: Create pytest plugins directory structure
**File**: `tests/plugins/`
**Description**: Create the plugins directory with __init__.py to make it a Python package. Add basic plugin registration infrastructure.
**Dependencies**: None
**Acceptance Criteria**:
- Directory `tests/plugins/` exists
- `tests/plugins/__init__.py` contains plugin list
- Plugins are discoverable by pytest

```python
# tests/plugins/__init__.py
"""Pytest plugins for testing framework compliance."""

# Plugin modules to be discovered
pytest_plugins = [
    "tests.plugins.coverage_warnings",
    "tests.plugins.error_message_validator",
    "tests.plugins.tdd_compliance",
]
```

### T002: Install required dependencies
**Command**: Update requirements-dev.txt and install
**Description**: Add GitPython 3.1.43 and ensure coverage dependencies are present for TDD validation and coverage analysis.
**Dependencies**: None
**Acceptance Criteria**:
- GitPython added to requirements-dev.txt
- Dependencies installed successfully
- Import tests pass

### T003: Create test fixtures directory
**File**: `tests/fixtures/`
**Description**: Create fixtures directory for test helpers including low_coverage_module.py for validation testing.
**Dependencies**: T001
**Acceptance Criteria**:
- Directory `tests/fixtures/` exists
- `tests/fixtures/low_coverage_module.py` with intentional low coverage
- Module importable in tests

### T004: Update pytest.ini with plugin registrations
**File**: `pytest.ini`
**Description**: Register the new plugins in pytest.ini so they're automatically loaded during test runs.
**Dependencies**: T001
**Acceptance Criteria**:
- pytest.ini updated with plugin list
- Plugins load without errors
- `pytest --trace-config` shows plugins

```ini
# Add to pytest.ini
plugins =
    tests.plugins.coverage_warnings
    tests.plugins.error_message_validator
```

---

## Phase 3.2: Contract Tests (TDD) ⚠️ MUST COMPLETE BEFORE 3.3 (4 tasks, all [P])

### T005 [P]: Create test_coverage_warning_contract.py
**File**: `tests/contract/test_coverage_warning_contract.py`
**Description**: Implement contract tests for coverage warning system per COV-001. Tests must fail initially.
**Dependencies**: T001, T003
**Contract**: `specs/026-fix-critical-issues/contracts/coverage_warning_contract.md`
**Acceptance Criteria**:
- 5 failing tests for coverage warning behavior
- Tests validate hook registration, threshold detection, critical modules, format, non-failing
- Uses mock coverage data for testing

```python
def test_COV001_plugin_registration():
    """Verify plugin registers correctly."""
    # This should fail until plugin exists
    assert "coverage_warnings" in pytest_plugins

def test_COV001_threshold_detection():
    """Verify modules below threshold are detected."""
    # Mock coverage at 50%, expect warning
    assert False  # Fails until implemented
```

### T006 [P]: Create test_error_message_contract.py
**File**: `tests/contract/test_error_message_contract.py`
**Description**: Implement contract tests for error message validation per ERR-001. Tests must fail initially.
**Dependencies**: T001
**Contract**: `specs/026-fix-critical-issues/contracts/error_message_contract.md`
**Acceptance Criteria**:
- 4 failing tests for error validation
- Tests validate three-part detection, missing components, context, suggestions
- Uses sample error messages

```python
def test_ERR001_three_part_detection():
    """Verify three-part structure is detected correctly."""
    good_message = """
    Test failed: Database connection error.
    Expected connection to localhost:5432 but got timeout.
    Check PostgreSQL is running and accepting connections.
    """
    # Should pass validation once implemented
    assert False  # Fails until implemented
```

### T007 [P]: Create test_tdd_validation_contract.py
**File**: `tests/contract/test_tdd_validation_contract.py`
**Description**: Implement contract tests for TDD compliance validation per TDD-001. Tests must fail initially.
**Dependencies**: T001, T002
**Contract**: `specs/026-fix-critical-issues/contracts/tdd_validation_contract.md`
**Acceptance Criteria**:
- 5 failing tests for TDD validation
- Tests validate discovery, violation detection, compliant workflow, reporting
- Uses mock git repository

### T008 [P]: Create test_task_mapping_contract.py
**File**: `tests/contract/test_task_mapping_contract.py`
**Description**: Implement contract tests for requirement-task mapping per MAP-001. Tests must fail initially.
**Dependencies**: T001
**Contract**: `specs/026-fix-critical-issues/contracts/task_mapping_contract.md`
**Acceptance Criteria**:
- 5 failing tests for mapping validation
- Tests validate requirement extraction, task detection, gap finding, reporting
- Uses sample spec/tasks content

---

## Phase 3.3: Plugin Implementations (4 tasks)

### T009: Implement coverage_warnings.py plugin
**File**: `tests/plugins/coverage_warnings.py`
**Description**: Create pytest plugin with pytest_terminal_summary hook that displays coverage warnings without failing tests.
**Dependencies**: T005 (contract test defines behavior)
**Acceptance Criteria**:
- Hook reads .coverage file
- Calculates module coverage percentages
- Displays warnings for modules below thresholds
- Critical modules (80%) vs normal (60%)
- Contract test T005 passes

```python
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Display coverage warnings after test execution."""
    # Implementation based on research.md decisions
```

### T010: Implement error_message_validator.py plugin
**File**: `tests/plugins/error_message_validator.py`
**Description**: Create pytest plugin with pytest_exception_interact hook that validates error message structure. Implements FR-006 (three-part structure), FR-007 (error context), and FR-008 (consistent format).
**Dependencies**: T006 (contract test defines behavior)
**Acceptance Criteria**:
- Hook intercepts test failures (FR-006)
- Validates three-part message structure: what/why/action (FR-006)
- Includes relevant context (test name, values, assertions) (FR-007)
- Enforces consistent format across test types (FR-008)
- Provides improvement suggestions
- Configurable validation rules with regex patterns (FR-009)
- Contract test T006 passes

### T011: Implement TDD compliance plugin base
**File**: `tests/plugins/tdd_compliance.py`
**Description**: Create pytest plugin that provides TDD validation helpers and marks for contract tests.
**Dependencies**: T007 (contract test defines behavior)
**Acceptance Criteria**:
- Plugin provides @pytest.mark.contract decorator
- Integrates with standalone script
- Helps identify contract tests
- Contract test T007 partially passes

### T012: Update .coveragerc with critical module patterns
**File**: `.coveragerc`
**Description**: Add [coverage:critical_modules] section with patterns for modules requiring 80% coverage.
**Dependencies**: T009
**Acceptance Criteria**:
- Section added with patterns from spec
- Coverage plugin reads configuration
- Critical modules properly identified

```ini
[coverage:critical_modules]
patterns =
    iris_rag/pipelines/
    iris_rag/storage/
    iris_rag/validation/
```

---

## Phase 3.4: Standalone Scripts (2 tasks, all [P])

### T013 [P]: Create validate_tdd_compliance.py script
**File**: `scripts/validate_tdd_compliance.py`
**Description**: Implement standalone script that analyzes git history to verify contract tests failed before implementation.
**Dependencies**: T002 (GitPython), T007
**Acceptance Criteria**:
- Finds all contract test files
- Analyzes git history for each test
- Detects if test failed initially
- Generates compliance report
- Supports CI mode with --fail-on-violations
- Contract test T007 fully passes

### T014 [P]: Create validate_task_mapping.py script
**File**: `scripts/validate_task_mapping.py`
**Description**: Implement script that validates all requirements have corresponding tasks.
**Dependencies**: T008
**Acceptance Criteria**:
- Extracts requirements from spec.md
- Extracts tasks from tasks.md
- Maps requirements to tasks
- Reports gaps and coverage
- Contract test T008 passes

---

## Phase 3.5: Integration & Configuration (3 tasks)

### T015: Create example validation test file
**File**: `tests/validation/test_compliance_examples.py`
**Description**: Create example test file demonstrating all compliance features with intentional violations.
**Dependencies**: T009, T010
**Acceptance Criteria**:
- Shows good and bad error messages
- Includes low coverage module
- Demonstrates all validators
- Used in quickstart guide

### T016: Add CI/CD workflow for TDD validation
**File**: `.github/workflows/tdd-check.yml`
**Description**: Create GitHub Actions workflow that runs TDD compliance check on PRs with contract tests.
**Dependencies**: T013
**Acceptance Criteria**:
- Workflow triggers on PR
- Runs TDD validation script
- Comments on PR if violations found
- Non-blocking initially

### T017: Create pre-commit hook configuration
**File**: `.pre-commit-config.yaml`
**Description**: Add pre-commit hooks for TDD validation and task mapping checks.
**Dependencies**: T013, T014
**Acceptance Criteria**:
- TDD check for contract test files
- Task mapping for spec/task updates
- Hooks are optional/warnings only

---

## Phase 3.6: Documentation & Polish (4 tasks, all [P])

### T018 [P]: Document coverage warning usage
**File**: `docs/testing/coverage-warnings.md`
**Description**: Create detailed documentation for the coverage warning system with examples and configuration.
**Dependencies**: T009, T012
**Acceptance Criteria**:
- Explains thresholds and critical modules
- Shows configuration examples
- Includes troubleshooting

### T019 [P]: Document error message best practices
**File**: `docs/testing/error-messages.md`
**Description**: Create guide for writing good test error messages with three-part structure examples.
**Dependencies**: T010
**Acceptance Criteria**:
- Explains what/why/action structure
- Good and bad examples
- Integration with existing tests

### T020 [P]: Document TDD compliance workflow
**File**: `docs/testing/tdd-compliance.md`
**Description**: Create guide for TDD workflow with contract tests and compliance validation.
**Dependencies**: T011, T013
**Acceptance Criteria**:
- Step-by-step TDD process
- How to write compliant tests
- CI/CD integration

### T021 [P]: Update main testing documentation
**File**: `README.md` (Testing section)
**Description**: Update main README with information about new compliance tools and links to detailed docs.
**Dependencies**: T018, T019, T020
**Acceptance Criteria**:
- Brief overview of each tool
- Links to detailed documentation
- Quick start commands

---

## Dependency Graph

```
Setup (T001-T004)
  ↓
Contract Tests [P] (T005-T008) ← Must fail before implementation
  ↓
Plugin Implementations (T009-T012)
  ├── Coverage warnings (T009)
  ├── Error validation (T010)
  ├── TDD helpers (T011)
  └── Configuration (T012)
  ↓
Standalone Scripts [P] (T013-T014)
  ↓
Integration (T015-T017)
  ↓
Documentation [P] (T018-T021)
```

## Parallel Execution Examples

### Example 1: Run all contract tests in parallel
```bash
# After setup (T001-T004) is complete
pytest tests/contract/test_*_contract.py -n 4

# Or as Task agents:
Task "Run coverage warning contract test T005"
Task "Run error message contract test T006"
Task "Run TDD validation contract test T007"
Task "Run task mapping contract test T008"
```

### Example 2: Implement standalone scripts in parallel
```bash
# After contract tests are written
Task "Create TDD compliance validator script T013"
Task "Create task mapping validator script T014"
```

### Example 3: Write all documentation in parallel
```bash
# After implementations complete
Task "Document coverage warnings T018"
Task "Document error messages T019"
Task "Document TDD compliance T020"
Task "Update main README T021"
```

## Task Summary

**Total Tasks**: 21
- **Setup & Infrastructure**: 4 tasks (T001-T004)
- **Contract Tests [P]**: 4 tasks (T005-T008)
- **Plugin Implementations**: 4 tasks (T009-T012)
- **Standalone Scripts [P]**: 2 tasks (T013-T014)
- **Integration & Config**: 3 tasks (T015-T017)
- **Documentation [P]**: 4 tasks (T018-T021)

**Parallel Tasks**: 10 tasks can run in parallel (marked [P])
**Sequential Tasks**: 11 tasks must run sequentially

**Estimated Completion**:
- Sequential tasks: ~4-5 hours
- Parallel tasks (with 4 workers): ~1-2 hours
- **Total**: ~6-7 hours of focused work

## Validation Checklist

After completing all tasks:

- [ ] All 4 contract tests pass (T005-T008)
- [ ] Coverage warnings display correctly (T009)
- [ ] Error messages are validated (T010)
- [ ] TDD compliance can be checked (T013)
- [ ] Task mapping validates correctly (T014)
- [ ] CI/CD integration works (T016)
- [ ] Documentation is complete (T018-T021)
- [ ] Example tests demonstrate features (T015)

## Success Criteria

✅ **Definition of Done**:
1. All 21 tasks completed
2. All contract tests pass
3. Plugins load without errors
4. Scripts execute successfully
5. Coverage warnings appear in test output
6. Error validation provides helpful feedback
7. TDD compliance detects violations
8. Task mapping finds gaps
9. Documentation explains usage
10. Quickstart examples work

---

*Generated from specs/026-fix-critical-issues/ design documents*
*Based on Constitution v1.6.0 - Test-Driven Development principles*