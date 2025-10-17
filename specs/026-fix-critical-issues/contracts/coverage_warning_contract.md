# Contract: Coverage Warning System

**Contract ID**: COV-001
**Version**: 1.0.0
**Component**: tests/plugins/coverage_warnings.py

## Purpose

Define the behavior of the pytest plugin that warns developers when code coverage falls below configured thresholds without failing the test suite.

## Requirements

This contract implements:
- **FR-001**: Warn when any module falls below 60% coverage
- **FR-002**: Warn when critical modules fall below 80% coverage
- **FR-003**: Include module name, current coverage, and threshold
- **FR-004**: Display warnings without failing tests
- **FR-005**: Identify critical modules via configuration

## Interface Definition

### Configuration

```ini
# .coveragerc
[coverage:run]
source = iris_rag,common

[coverage:report]
fail_under = 0  # Don't fail, just warn

[coverage:critical_modules]
# Patterns for critical modules requiring 80% coverage
patterns =
    iris_rag/pipelines/
    iris_rag/storage/
    iris_rag/validation/
    common/db_vector_utils.py
    common/vector_sql_utils.py
```

### Plugin Registration

```python
# pytest_plugins in conftest.py or pytest.ini
pytest_plugins = ["tests.plugins.coverage_warnings"]
```

### Hook Implementation

```python
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Display coverage warnings after test execution."""
    # 1. Check if coverage is enabled
    # 2. Load coverage data
    # 3. Analyze thresholds
    # 4. Display warnings
```

## Behavior Specifications

### REQ-1: Coverage Data Collection

**Given** tests are run with `--cov` option
**When** pytest_terminal_summary hook is called
**Then** the plugin loads coverage data from .coverage file

**Validation**:
```python
assert os.path.exists('.coverage')
cov = coverage.Coverage()
cov.load()
assert cov.get_data() is not None
```

### REQ-2: Module Analysis

**Given** coverage data is available
**When** analyzing each module
**Then** calculate percentage and check against thresholds

**Validation**:
```python
for module in covered_modules:
    percentage = calculate_coverage(module)
    threshold = 80.0 if is_critical(module) else 60.0
    if percentage < threshold:
        warnings.append(CoverageWarning(module, percentage, threshold))
```

### REQ-3: Critical Module Detection

**Given** a module path
**When** checking if critical
**Then** match against configured patterns

**Validation**:
```python
def is_critical(module_path):
    patterns = config.get('coverage:critical_modules', 'patterns')
    for pattern in patterns:
        if fnmatch.fnmatch(module_path, pattern):
            return True
    return False
```

### REQ-4: Warning Display Format

**Given** coverage warnings exist
**When** displaying in terminal
**Then** show formatted warnings with color coding

**Expected Output**:
```
======================== Coverage Warnings ========================
⚠️  iris_rag.pipelines.basic: 45.5% < 80.0% (CRITICAL)
    Missing lines: 23, 45, 67, 89-95
⚠️  common.utils: 55.2% < 60.0%
    Missing lines: 12, 34, 56

3 modules below threshold (1 critical)
==================================================================
```

### REQ-5: Non-Failing Behavior

**Given** coverage warnings exist
**When** test suite completes
**Then** exit status remains unchanged (warnings don't fail tests)

**Validation**:
```python
original_status = exitstatus
# Display warnings
assert exitstatus == original_status  # Not modified
```

## Error Handling

### ERR-1: Missing Coverage Data

**Given** --cov not used or .coverage missing
**When** plugin runs
**Then** skip silently without warnings

### ERR-2: Corrupted Coverage File

**Given** .coverage file is corrupted
**When** loading coverage data
**Then** display single error message and continue

```
⚠️  Coverage data corrupted - cannot analyze thresholds
```

### ERR-3: Invalid Configuration

**Given** .coveragerc has invalid patterns
**When** parsing configuration
**Then** use defaults (60% all, 80% for */pipelines/*, */storage/*, */validation/*)

## Integration Points

### With pytest-cov

- Must not interfere with normal coverage reporting
- Runs after pytest-cov completes its work
- Uses same .coverage data file

### With CI/CD

- Warnings appear in CI logs
- Can be parsed for metrics/trends
- Compatible with coverage badge generation

## Contract Tests

```python
# tests/contract/test_coverage_warning_contract.py

def test_COV001_plugin_registration():
    """Verify plugin registers correctly."""
    assert "coverage_warnings" in pytest_plugins

def test_COV001_threshold_detection():
    """Verify modules below threshold are detected."""
    # Mock coverage data with module at 50%
    # Run analysis
    # Assert warning generated

def test_COV001_critical_module_identification():
    """Verify critical modules use 80% threshold."""
    # Test with iris_rag/pipelines/basic.py
    # Assert 80% threshold applied

def test_COV001_warning_format():
    """Verify warning output format."""
    # Generate warnings
    # Capture output
    # Assert format matches specification

def test_COV001_non_failing_behavior():
    """Verify warnings don't change exit status."""
    # Run with warnings present
    # Assert exit status unchanged
```

## Performance Requirements

- Coverage analysis must complete within 5 seconds
- Memory usage must not exceed coverage.py baseline + 10MB
- Must handle repositories with 1000+ modules

## Security Considerations

- No external data access
- Read-only operations on .coverage file
- No code execution or evaluation

## Future Extensions

1. **Trend Tracking**: Store historical coverage data
2. **Baseline Comparison**: Warn only on coverage decreases
3. **Team Goals**: Configure team-specific thresholds
4. **Exclusions**: Ignore specific modules/patterns