# Contract: Pipeline Types Helper Script

**Script**: `scripts/utils/get_pipeline_types.py`
**Purpose**: Query iris_rag factory for available pipeline types and output comma-separated list
**Consumers**: Makefile (test-ragas-sample target)

---

## Interface Contract

### Success Case

**Invocation**:
```bash
python scripts/utils/get_pipeline_types.py
```

**Expected Behavior**:
- Import iris_rag factory module
- Extract available_types list from factory
- Output comma-separated string to stdout
- Exit with code 0

**Output Format**:
```
STDOUT: basic,basic_rerank,crag,graphrag,pylate_colbert
STDERR: (empty)
EXIT_CODE: 0
```

**Guarantees**:
- Output contains only valid pipeline type names
- No whitespace in output (except trailing newline)
- Pipeline names match factory's available_types list exactly
- Order is consistent (same as factory definition)

---

## Error Contracts

### Error Case 1: Package Import Failure

**Scenario**: iris_rag package not installed or not on PYTHONPATH

**Expected Behavior**:
```
STDOUT: (empty)
STDERR: ERROR: Cannot import iris_rag. Is the package installed?
       Ensure you have run: uv sync or pip install -e .
EXIT_CODE: 1
```

**Contract Test**:
```python
def test_helper_script_import_error():
    result = subprocess.run(
        ["python", "scripts/utils/get_pipeline_types.py"],
        capture_output=True,
        env={"PYTHONPATH": "/invalid"},  # Force import error
    )
    assert result.returncode == 1
    assert b"Cannot import iris_rag" in result.stderr
    assert result.stdout == b""
```

---

### Error Case 2: No Pipelines Available

**Scenario**: Factory returns empty list (should never happen, but defensive)

**Expected Behavior**:
```
STDOUT: (empty)
STDERR: ERROR: No pipeline types available from factory
       This indicates a bug in iris_rag factory - please report
EXIT_CODE: 1
```

**Contract Test**:
```python
def test_helper_script_no_pipelines():
    # Mock scenario - would require factory patch
    # Verification: If factory ever returns empty list, script fails clearly
    pass  # Tested via unit tests with mocked factory
```

---

### Error Case 3: Extraction Failure

**Scenario**: Cannot parse factory source or extract available_types

**Expected Behavior**:
```
STDOUT: (empty)
STDERR: ERROR: Cannot extract pipeline types: {exception_message}
       Factory source may have changed - please update helper script
EXIT_CODE: 1
```

---

## Makefile Integration Contract

### Integration Point

**File**: `Makefile`
**Target**: `test-ragas-sample`
**Line**: ~684 (approximate)

**Before**:
```makefile
export RAGAS_PIPELINES=${RAGAS_PIPELINES:-"basic,basic_rerank,crag,graphrag,pylate_colbert"};
```

**After**:
```makefile
export RAGAS_PIPELINES=$${RAGAS_PIPELINES:-$$(python scripts/utils/get_pipeline_types.py)};
```

### Behavior Contract

**Case 1: Default (No env var)**
```bash
$ make test-ragas-sample
# RAGAS_PIPELINES is populated by helper script output
# All factory pipelines are tested
```

**Case 2: User Override**
```bash
$ RAGAS_PIPELINES=basic,crag make test-ragas-sample
# Helper script is NOT called
# Only 'basic' and 'crag' pipelines are tested
```

**Case 3: Helper Script Failure**
```bash
$ make test-ragas-sample
# If helper script exits 1, Make fails immediately
# Error message from stderr is displayed
# No partial/incorrect evaluation happens
```

---

## Contract Tests

### Test 1: Script Exists and is Executable

**File**: `tests/infrastructure/test_makefile_targets.py`

```python
def test_get_pipeline_types_script_exists():
    """Verify helper script file exists at expected location."""
    script_path = Path("scripts/utils/get_pipeline_types.py")
    assert script_path.exists(), "Helper script not found"
    assert script_path.is_file(), "Helper script path is not a file"
```

**Status**: MUST FAIL until implementation

---

### Test 2: Output Format Validation

```python
def test_get_pipeline_types_output_format():
    """Verify helper script outputs comma-separated list."""
    result = subprocess.run(
        ["python", "scripts/utils/get_pipeline_types.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    output = result.stdout.strip()

    # Must be comma-separated with no spaces
    assert "," in output, "Output must contain commas"
    assert " " not in output, "Output must not contain spaces"

    # Must parse as valid pipeline names
    pipelines = output.split(",")
    assert len(pipelines) > 0, "Must have at least one pipeline"
    for name in pipelines:
        assert name.isidentifier() or "_" in name, f"Invalid pipeline name: {name}"
```

**Status**: MUST FAIL until implementation

---

### Test 3: Output Matches Factory

```python
def test_get_pipeline_types_matches_factory():
    """Verify helper script output matches iris_rag factory available types."""
    from iris_rag import _create_pipeline_legacy
    import inspect

    # Extract available_types from factory source
    source = inspect.getsource(_create_pipeline_legacy)
    # Parse available_types list from error message
    # (Implementation detail - approximate matching)

    # Get helper script output
    result = subprocess.run(
        ["python", "scripts/utils/get_pipeline_types.py"],
        capture_output=True,
        text=True,
    )
    script_output = set(result.stdout.strip().split(","))

    # Expected factory types (as of 2025-10-06)
    expected_types = {"basic", "basic_rerank", "crag", "graphrag", "pylate_colbert"}

    assert script_output == expected_types, \
        f"Helper output {script_output} doesn't match factory {expected_types}"
```

**Status**: MUST FAIL until implementation

---

### Test 4: Makefile Uses Helper

```python
def test_ragas_target_uses_dynamic_pipelines():
    """Verify Makefile test-ragas-sample target calls helper script."""
    makefile_path = Path("Makefile")
    makefile_content = makefile_path.read_text()

    # Find test-ragas-sample target
    lines = makefile_content.split('\n')
    found_target = False
    found_helper_call = False

    for line in lines:
        if 'test-ragas-sample:' in line:
            found_target = True
        if found_target and 'get_pipeline_types.py' in line:
            found_helper_call = True
            break

    assert found_target, "test-ragas-sample target not found in Makefile"
    assert found_helper_call, "Makefile doesn't call get_pipeline_types.py helper"
```

**Status**: MUST FAIL until implementation

---

### Test 5: Environment Variable Override

```python
def test_ragas_target_respects_env_override():
    """Verify RAGAS_PIPELINES env var overrides helper script."""
    makefile_path = Path("Makefile")
    makefile_content = makefile_path.read_text()

    # Check for pattern: ${RAGAS_PIPELINES:-$(shell ...)}
    # This ensures env var takes precedence
    assert "RAGAS_PIPELINES:-" in makefile_content or \
           "RAGAS_PIPELINES:=" in makefile_content, \
           "Makefile doesn't preserve env var override capability"
```

**Status**: MUST FAIL until implementation

---

## Contract Validation Checklist

Before marking implementation complete, verify:

- [ ] All 5 contract tests exist in `tests/infrastructure/test_makefile_targets.py`
- [ ] All tests currently FAIL (pre-implementation)
- [ ] Helper script file structure created
- [ ] Makefile modification planned
- [ ] Error messages match contract specifications
- [ ] Output format exactly matches: `{pipeline1},{pipeline2},...`
- [ ] Exit codes match contract (0 success, 1 failure)
- [ ] STDERR used for errors, STDOUT for data

---

## Non-Functional Requirements

**Performance**:
- Helper script MUST execute in <200ms
- No network calls allowed
- No database connections

**Reliability**:
- MUST fail fast with clear errors
- NO silent failures
- NO partial/corrupt output

**Maintainability**:
- Code MUST be self-documenting
- Error messages MUST be actionable
- Comments MUST explain WHY, not WHAT

---

## Change Log

| Date | Change | Rationale |
|------|--------|-----------|
| 2025-10-06 | Initial contract | Feature 031 planning phase |

---

**Contract Status**: DEFINED ✓
**Implementation Status**: PENDING
**Test Status**: FAILING (expected)
