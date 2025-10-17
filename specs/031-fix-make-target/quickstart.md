# Quickstart: Dynamic Pipeline Discovery Testing

**Feature**: Fix RAGAS Make Target Pipeline List
**Purpose**: Verify the helper script and Makefile integration work correctly
**Time to Complete**: ~5 minutes

---

## Prerequisites

```bash
# 1. Repository setup
cd /path/to/rag-templates
git checkout 031-fix-make-target

# 2. Python environment (use uv per constitution)
uv sync

# 3. Activate virtual environment
source .venv/bin/activate

# 4. Verify iris_rag is installed
python -c "import iris_rag; print('✓ iris_rag available')"
```

---

## Step 1: Verify Helper Script Exists

```bash
# Check script was created
ls -l scripts/utils/get_pipeline_types.py

# Expected output:
# -rw-r--r--  1 user  staff  XXX Oct  6 HH:MM scripts/utils/get_pipeline_types.py
```

**If missing**: Implementation not complete yet (expected during TDD phase)

---

## Step 2: Test Helper Script Directly

```bash
# Run helper script
python scripts/utils/get_pipeline_types.py

# Expected output (to stdout):
# basic,basic_rerank,crag,graphrag,pylate_colbert

# Verify exit code
echo $?

# Expected: 0
```

### Success Criteria
- ✓ Output is comma-separated list with no spaces
- ✓ Output contains 5 pipeline names (as of 2025-10-06)
- ✓ Exit code is 0
- ✓ No error messages to stderr

### Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'iris_rag'`
```bash
# Solution: Install package in editable mode
uv pip install -e .
# or
pip install -e .
```

**Problem**: Script outputs nothing
```bash
# Debug: Check for errors
python scripts/utils/get_pipeline_types.py 2>&1

# If you see "ERROR: Cannot import iris_rag":
# → Verify virtual environment is activated
# → Verify package is installed (pip list | grep rag-templates)
```

---

## Step 3: Run Contract Tests

```bash
# Run all infrastructure tests
pytest tests/infrastructure/test_makefile_targets.py -v

# Run only dynamic pipeline tests
pytest tests/infrastructure/test_makefile_targets.py -v -k "pipeline_types"
```

### Expected Output (All Passing)

```
tests/infrastructure/test_makefile_targets.py::test_get_pipeline_types_script_exists PASSED
tests/infrastructure/test_makefile_targets.py::test_get_pipeline_types_output_format PASSED
tests/infrastructure/test_makefile_targets.py::test_get_pipeline_types_matches_factory PASSED
tests/infrastructure/test_makefile_targets.py::test_ragas_target_uses_dynamic_pipelines PASSED
tests/infrastructure/test_makefile_targets.py::test_ragas_target_respects_env_override PASSED

===================== 5 passed in 0.5s =====================
```

### If Tests Fail

**During TDD Phase** (before implementation):
- ✓ Expected: All 5 tests FAIL
- ✓ This confirms tests are properly asserting requirements
- ✓ Proceed to implementation phase

**After Implementation**:
- ❌ Any failing test indicates incomplete implementation
- ❌ Review test output for specific assertion failures
- ❌ Fix implementation before proceeding

---

## Step 4: Test Makefile Integration (Default Behavior)

```bash
# Unset any existing RAGAS_PIPELINES override
unset RAGAS_PIPELINES

# Run RAGAS evaluation (this will take several minutes)
make test-ragas-sample
```

### Expected Behavior

1. **Helper Script Execution**:
   - Make calls `python scripts/utils/get_pipeline_types.py`
   - Script outputs: `basic,basic_rerank,crag,graphrag,pylate_colbert`
   - Make sets `RAGAS_PIPELINES` to this value

2. **RAGAS Evaluation**:
   - All 5 pipelines are tested
   - Results appear in `outputs/reports/ragas_evaluations/`

3. **Success Output**:
```
📊 RAGAS Evaluation Complete
Pipelines tested: basic, basic_rerank, crag, graphrag, pylate_colbert
Report: outputs/reports/ragas_evaluations/simple_ragas_report_YYYYMMDD_HHMMSS.html
```

### Verify All Pipelines Were Tested

```bash
# Check latest RAGAS report
ls -t outputs/reports/ragas_evaluations/*.html | head -1

# Open report in browser (macOS)
open outputs/reports/ragas_evaluations/simple_ragas_report_*.html

# Or view JSON results
cat outputs/reports/ragas_evaluations/simple_ragas_report_*.json | jq 'keys'

# Expected: ["basic", "basic_rerank", "crag", "graphrag", "pylate_colbert"]
```

---

## Step 5: Test Environment Variable Override

```bash
# Override to test only 2 pipelines
RAGAS_PIPELINES=basic,crag make test-ragas-sample
```

### Expected Behavior

1. **Helper Script NOT Called**:
   - Make sees `RAGAS_PIPELINES` is set
   - Uses env var value instead of calling helper
   - Helper script is NOT executed

2. **RAGAS Evaluation**:
   - Only `basic` and `crag` pipelines tested
   - Other 3 pipelines skipped

3. **Success Output**:
```
📊 RAGAS Evaluation Complete
Pipelines tested: basic, crag
Report: outputs/reports/ragas_evaluations/simple_ragas_report_YYYYMMDD_HHMMSS.html
```

### Verify Override Worked

```bash
# Check latest report shows only 2 pipelines
cat outputs/reports/ragas_evaluations/simple_ragas_report_*.json | jq 'keys'

# Expected: ["basic", "crag"]
```

---

## Step 6: Test Error Handling

### Test 6a: Invalid PYTHONPATH (Force Import Error)

```bash
# Create a shell with broken PYTHONPATH
PYTHONPATH=/invalid python scripts/utils/get_pipeline_types.py

# Expected output to STDERR:
# ERROR: Cannot import iris_rag. Is the package installed?
#        Ensure you have run: uv sync or pip install -e .

# Expected exit code:
echo $?  # Should be 1
```

### Test 6b: Makefile Fails Fast on Helper Error

```bash
# Simulate helper script failure by making it non-executable
chmod -x scripts/utils/get_pipeline_types.py

# Try running make target
make test-ragas-sample

# Expected: Make fails immediately with error
# Expected: No partial RAGAS evaluation
# Expected: Clear error message

# Restore permissions
chmod +x scripts/utils/get_pipeline_types.py
```

---

## Step 7: Regression Test (Future-Proofing)

**Scenario**: A new pipeline is added to the factory

```bash
# Simulate adding a new pipeline (don't actually edit factory)
# Just verify current behavior is correct

# 1. Check current factory list
python -c "
from iris_rag import _create_pipeline_legacy
import inspect
source = inspect.getsource(_create_pipeline_legacy)
if 'available_types' in source:
    print('Factory contains available_types list')
else:
    print('WARNING: Factory structure may have changed')
"

# 2. Verify helper matches factory
pytest tests/infrastructure/test_makefile_targets.py::test_get_pipeline_types_matches_factory -v

# 3. When factory DOES change:
# → Helper script should auto-detect new pipeline
# → No manual Makefile update required
# → This is the whole point of the feature!
```

---

## Success Checklist

After completing all steps, verify:

- [ ] Helper script exists and is executable
- [ ] Helper script outputs comma-separated pipeline list
- [ ] Helper script exits with code 0 on success
- [ ] All 5 contract tests pass
- [ ] `make test-ragas-sample` tests all factory pipelines
- [ ] `RAGAS_PIPELINES=basic,crag make test-ragas-sample` tests only those 2
- [ ] Helper script fails clearly on import errors
- [ ] Makefile fails fast when helper script fails
- [ ] Helper output matches factory available_types exactly

---

## Common Issues & Solutions

### Issue 1: Helper Script Not Found

**Symptom**:
```bash
python scripts/utils/get_pipeline_types.py
# python: can't open file 'scripts/utils/get_pipeline_types.py': [Errno 2] No such file or directory
```

**Solution**:
- Ensure you're in the repository root directory
- Check branch: `git branch` (should be on `031-fix-make-target`)
- Verify implementation is complete

### Issue 2: Wrong Number of Pipelines

**Symptom**: Helper outputs 4 pipelines instead of 5

**Solution**:
- Check factory source: `grep -A 10 "available_types" iris_rag/__init__.py`
- Verify helper script parsing logic
- Ensure all pipelines are included in factory list

### Issue 3: RAGAS Tests Wrong Pipelines

**Symptom**: Make target tests pipelines not in helper output

**Solution**:
- Check Makefile line ~684 for hardcoded list
- Ensure Makefile uses `$(shell python scripts/utils/get_pipeline_types.py)`
- Run `make -n test-ragas-sample` to see dry-run (what commands will execute)

### Issue 4: Env Var Override Not Working

**Symptom**: Setting `RAGAS_PIPELINES=basic` still tests all 5 pipelines

**Solution**:
- Check Makefile pattern: Should be `${RAGAS_PIPELINES:-$(shell ...)}`
- The `:-` is critical for fallback behavior
- Try: `RAGAS_PIPELINES=basic make test-ragas-sample -e` (force env vars)

---

## Performance Validation

```bash
# Benchmark helper script execution time
time python scripts/utils/get_pipeline_types.py

# Expected: <100ms (usually 20-50ms)
# Acceptable: <200ms
# If slower: Check for unnecessary imports or file I/O
```

---

## Next Steps

### If All Tests Pass ✅
- Feature is working correctly
- Ready for code review
- Can proceed to documentation updates (README, CLAUDE.md)

### If Tests Fail ❌
- Review test output carefully
- Check implementation against contracts
- Verify factory source hasn't changed unexpectedly
- Run tests with `-vv` for more detail: `pytest -vv tests/infrastructure/test_makefile_targets.py`

---

## Rollback Plan

If you need to revert to the old hardcoded behavior:

```bash
# Option 1: Revert git commits
git log --oneline  # Find commits to revert
git revert <commit-hash>

# Option 2: Quick hotfix (edit Makefile directly)
# Change line ~684 back to:
export RAGAS_PIPELINES=${RAGAS_PIPELINES:-"basic,basic_rerank,crag,graphrag,pylate_colbert"};

# Verify old behavior works
make test-ragas-sample
```

---

**Quickstart Status**: ✓ Complete
**Estimated Time**: 5-10 minutes (excluding RAGAS evaluation time)
**Dependencies**: iris_rag installed, pytest available
