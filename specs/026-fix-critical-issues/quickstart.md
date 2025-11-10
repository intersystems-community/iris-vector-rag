# Quickstart: Testing Framework Compliance Tools

**Feature**: Fix Critical Issues from Analyze Command
**Version**: 1.0.0

## Overview

This guide helps you use the new testing framework compliance tools that ensure your tests follow constitutional principles: proper coverage, actionable error messages, and TDD compliance.

## Prerequisites

- Python 3.12+ with pytest installed
- Git repository (for TDD validation)
- Existing test suite using pytest

## Installation

1. **Install the pytest plugins**:
```bash
# Ensure plugins are in Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Verify plugins are available
pytest --trace-config | grep plugins
```

2. **Update configuration files**:

**.coveragerc**:
```ini
[coverage:critical_modules]
patterns =
    iris_rag/pipelines/
    iris_rag/storage/
    iris_rag/validation/
```

**pytest.ini**:
```ini
[pytest]
plugins =
    tests.plugins.coverage_warnings
    tests.plugins.error_message_validator
```

## Feature 1: Coverage Warnings

See warnings when modules fall below coverage thresholds without failing tests.

### Basic Usage

```bash
# Run tests with coverage warnings
pytest --cov=iris_rag --cov=common

# Output includes:
# ======================== Coverage Warnings ========================
# ⚠️  iris_rag.pipelines.basic: 45.5% < 80.0% (CRITICAL)
#     Missing lines: 23, 45, 67, 89-95
# ⚠️  common.utils: 55.2% < 60.0%
#     Missing lines: 12, 34, 56
```

### Configuration

Modify `.coveragerc` to set critical modules:
```ini
[coverage:critical_modules]
patterns =
    your_package/core/
    your_package/critical/
    specific_module.py
```

### Validation

Run this test to verify coverage warnings work:

```python
# tests/validation/test_coverage_warnings.py
def test_coverage_warnings_display():
    """Verify coverage warnings appear in output."""
    # This module intentionally has low coverage
    from tests.fixtures.low_coverage_module import barely_tested_function

    assert barely_tested_function(1) == 1
    # Function has other branches not tested
    # Should see warning: "low_coverage_module: 25% < 60%"
```

## Feature 2: Error Message Validation

Get feedback when test error messages lack required components.

### Basic Usage

Write tests with descriptive assertion messages:

```python
# ❌ Bad: No context
def test_pipeline_config():
    config = load_config()
    assert config.get('llm_model')  # AssertionError with no context

# ✅ Good: Three-part message
def test_pipeline_config():
    config = load_config()
    assert config.get('llm_model'), (
        "Pipeline configuration validation failed. "
        "Expected 'llm_model' in config but got None. "
        "Check config.yaml includes required 'llm_model' field."
    )
```

### Validation Output

When a test fails with a poor message:
```
⚠️  Error Message Validation Failed: test_pipeline.py::test_config
    ❌ Missing: Why it failed
    ❌ Missing: Suggested action

    Current: "AssertionError"

    Better example:
    "Configuration validation failed.
     Expected 'api_key' in environment but not found.
     Set OPENAI_API_KEY environment variable."
```

### Configuration

Control validation strictness in `pytest.ini`:
```ini
[tool.pytest.error_validation]
enabled = true
strict = false  # true = fail tests with bad messages
show_suggestions = true
```

## Feature 3: TDD Compliance Validation

Verify contract tests were written before implementation.

### Basic Usage

```bash
# Check all contract tests
python scripts/validate_tdd_compliance.py

# Output:
# ✅ tests/contract/test_api_contract.py - COMPLIANT
# ❌ tests/contract/test_cache_contract.py - VIOLATION: Never failed
#
# Summary: 12/15 compliant (80%)
```

### Writing Compliant Contract Tests

1. **Write failing test first**:
```python
# tests/contract/test_feature_contract.py
def test_FEAT001_new_capability():
    """Contract test for new feature."""
    # This should fail initially
    feature = NewFeature()
    assert feature.do_something() == "expected"
```

2. **Commit the failing test**:
```bash
git add tests/contract/test_feature_contract.py
git commit -m "Add failing contract test for new feature"
```

3. **Implement feature** (in separate commit):
```bash
git add src/new_feature.py
git commit -m "Implement new feature"
```

4. **Validate compliance**:
```bash
python scripts/validate_tdd_compliance.py tests/contract/test_feature_contract.py
# ✅ COMPLIANT
```

### CI Integration

Add to `.github/workflows/tests.yml`:
```yaml
- name: Check TDD Compliance
  run: python scripts/validate_tdd_compliance.py --ci
  continue-on-error: true  # Warning for now
```

## Feature 4: Task Mapping Validation

Ensure all requirements have implementation tasks.

### Basic Usage

```bash
# Validate current feature
python scripts/validate_task_mapping.py

# Output:
# Coverage: 17/20 (85%)
#
# Missing:
# - FR-007: Coverage threshold warnings
# - FR-013: Error message validation
```

### Writing Mappable Tasks

In `tasks.md`, reference requirements explicitly:

```markdown
### T001: Implement coverage warnings (FR-007)
**File**: tests/plugins/coverage_warnings.py
**Description**: Create pytest plugin for coverage threshold warnings per FR-007
```

### Validation Report

Generate detailed gap analysis:
```bash
python scripts/validate_task_mapping.py --report gaps.md
```

## Complete Example

Here's a complete test file demonstrating all features:

```python
# tests/example/test_compliant_example.py
"""Example showing all compliance features."""

def test_vector_search_with_good_message():
    """Demonstrate proper error messages."""
    from iris_rag.storage import VectorStore

    store = VectorStore()
    results = store.search("test query", k=3)

    # Good assertion message with three parts
    assert len(results) == 3, (
        f"Vector search returned wrong number of results. "
        f"Expected 3 documents but got {len(results)}. "
        f"Check if documents are properly indexed and k parameter is correct."
    )

# This module should have >60% coverage
# Contract tests should fail first
# All requirements should have tasks
```

## Troubleshooting

### Coverage warnings not appearing?

1. Check pytest-cov is installed: `pip install pytest-cov`
2. Verify plugin loaded: `pytest --trace-config | grep coverage_warnings`
3. Ensure `--cov` option is used

### Error validation too strict?

Set `strict = false` in configuration to get warnings without failures.

### TDD validation taking too long?

Use `--include` to check specific test files:
```bash
python scripts/validate_tdd_compliance.py --include "tests/contract/test_new_*.py"
```

### Task mapping missing requirements?

Check requirement ID format matches pattern: `**FR-001**:`

## Next Steps

1. Run coverage warnings on your test suite
2. Fix error messages that lack components
3. Validate TDD compliance for contract tests
4. Ensure all requirements have tasks

For more details, see the contract specifications in `specs/026-fix-critical-issues/contracts/`.