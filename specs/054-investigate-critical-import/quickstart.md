# Fix Verification Quickstart

**Feature**: 054-investigate-critical-import
**Fix**: Move common module inside iris_vector_rag package to resolve namespace conflicts
**Version**: 0.5.0 (or 0.4.2 if decided as patch)

## Prerequisites

- Python 3.10+
- Clean virtual environment (recommended for testing)
- Access to HippoRAG2 pipeline repository (for integration testing)

## Quick Verification (5 minutes)

### Step 1: Install Patched Version

```bash
# Create clean test environment
python3 -m venv test-venv
source test-venv/bin/activate  # On Windows: test-venv\Scripts\activate

# Install from PyPI (after release)
pip install iris-vector-rag==0.5.0

# OR install from local build (before release)
pip install /path/to/iris-vector-rag/dist/iris_vector_rag-0.5.0-py3-none-any.whl
```

### Step 2: Verify Imports Work

```python
# Test in Python REPL
python3 << 'EOF'
# This is what was failing before the fix
from iris_vector_rag.core.connection import ConnectionManager

print("✅ SUCCESS: ConnectionManager imported without ImportError")

# Verify common module is in correct location
from iris_vector_rag.common.iris_dbapi_connector import get_iris_dbapi_connection
from iris_vector_rag.common.iris_connection_manager import get_iris_connection

print("✅ SUCCESS: All common module imports work")
print(f"   - get_iris_dbapi_connection: {callable(get_iris_dbapi_connection)}")
print(f"   - get_iris_connection: {callable(get_iris_connection)}")
EOF
```

**Expected Output**:
```
✅ SUCCESS: ConnectionManager imported without ImportError
✅ SUCCESS: All common module imports work
   - get_iris_dbapi_connection: True
   - get_iris_connection: True
```

**If you see ImportError**: The fix is not applied or package not installed correctly.

### Step 3: Run Contract Tests

```bash
# From iris-vector-rag-private repository
pytest specs/054-investigate-critical-import/contracts/test_common_imports.py -v

# Expected: All 7 tests PASS
```

**Expected Output**:
```
test_common_imports.py::test_import_iris_dbapi_connector PASSED                 [ 14%]
test_common_imports.py::test_import_iris_connection_manager PASSED              [ 28%]
test_common_imports.py::test_connection_manager_imports PASSED                  [ 42%]
test_common_imports.py::test_common_module_location PASSED                      [ 57%]
test_common_imports.py::test_old_top_level_common_removed PASSED                [ 71%]
test_common_imports.py::test_connection_manager_can_create_connection_mock PASSED [ 85%]

================================ 7 passed in 0.23s =================================
```

## Full Validation (15 minutes)

### Step 4: Test in HippoRAG2 Pipeline

This is the **critical test** - verify the fix resolves the original bug report.

```bash
# Navigate to HippoRAG2 pipeline
cd ../hipporag2-pipeline

# Activate HippoRAG2 environment
source .venv/bin/activate  # Or your venv activation command

# Upgrade to patched version
pip install --upgrade iris-vector-rag==0.5.0

# Verify installation
python -c "import iris_vector_rag; print(f'iris-vector-rag version: {iris_vector_rag.__version__}')"
# Expected: iris-vector-rag version: 0.5.0

# Run entity extraction test (this was hanging before)
python tests/test_e2e_simple.py

# OR run with timing
time python tests/test_e2e_simple.py
```

### Step 5: Verify Entity Extraction Logging

**Before fix**: No logging, appeared to hang
**After fix**: Should see INFO-level logging during entity extraction

**Expected console output** (example):
```
INFO:iris_vector_rag.services.entity_extraction:=======================================================================
INFO:iris_vector_rag.services.entity_extraction:🤖 Entity Extraction Service - LLM Configuration
INFO:iris_vector_rag.services.entity_extraction:=======================================================================
INFO:iris_vector_rag.services.entity_extraction:  Provider:    openai
INFO:iris_vector_rag.services.entity_extraction:  API Type:    openai
INFO:iris_vector_rag.services.entity_extraction:  Model:       gpt-4o-mini
INFO:iris_vector_rag.services.entity_extraction:  API Base:    https://api.openai.com/v1
INFO:iris_vector_rag.services.entity_extraction:  Method:      llm_basic
INFO:iris_vector_rag.services.entity_extraction:=======================================================================
INFO:iris_vector_rag.services.entity_extraction:Processing document doc_001...
INFO:iris_vector_rag.services.entity_extraction:Extracted 15 entities from document doc_001
```

## Success Criteria

✅ **All must pass**:
1. `from iris_vector_rag.core.connection import ConnectionManager` works without ImportError
2. All 7 contract tests pass
3. HippoRAG2 entity extraction completes (doesn't hang)
4. Entity extraction logging appears in console
5. No namespace conflict errors in any environment

## Troubleshooting

### Issue: Still getting ImportError

**Symptom**:
```python
ModuleNotFoundError: No module named 'iris_vector_rag.common'
```

**Solution**:
1. Verify you installed the correct version:
   ```bash
   pip show iris-vector-rag
   # Check Version: 0.5.0 or later
   ```

2. Uninstall and reinstall:
   ```bash
   pip uninstall iris-vector-rag -y
   pip install iris-vector-rag==0.5.0
   ```

3. Check for editable installs:
   ```bash
   pip list | grep iris
   # Should show: iris-vector-rag 0.5.0
   # NOT: iris-vector-rag 0.4.1 (editable)
   ```

### Issue: Old top-level common module still exists

**Symptom**:
```python
import common
print(common.__file__)
# Shows: /path/to/venv/lib/python3.12/site-packages/common/__init__.py
```

**Diagnosis**: Another package provides `common` module.

**Solution**: This is OK! The fix ensures iris-vector-rag uses its own `iris_vector_rag.common`, not the top-level one.

To verify the fix worked:
```python
from iris_vector_rag import common
print(common.__file__)
# Should show: /path/to/venv/lib/python3.12/site-packages/iris_vector_rag/common/__init__.py
```

### Issue: HippoRAG2 tests still fail

**Symptom**: Tests still hang or fail with import errors

**Checklist**:
1. [ ] Verified iris-vector-rag 0.5.0 is installed in HippoRAG2 venv
2. [ ] Ran `pip list | grep iris-vector-rag` to confirm version
3. [ ] Restarted Python interpreter (old imports may be cached)
4. [ ] Checked for conflicting `iris-vector-rag` installations (editable vs site-packages)

**Still failing?** Check if HippoRAG2 has any hardcoded imports:
```bash
cd ../hipporag2-pipeline
grep -r "from common\." . --include="*.py"
# Should return: (no results)
```

If HippoRAG2 code imports `from common.X import Y`, update to:
```python
from iris_vector_rag.common.X import Y
```

## Migration Notes for Users

### If you were importing common directly (rare):

**Before (0.4.1)**:
```python
from common.iris_dbapi_connector import get_iris_dbapi_connection
from common.iris_connection_manager import get_iris_connection
```

**After (0.5.0)**:
```python
from iris_vector_rag.common.iris_dbapi_connector import get_iris_dbapi_connection
from iris_vector_rag.common.iris_connection_manager import get_iris_connection
```

### If you were only using ConnectionManager (normal usage):

**No changes required** - ConnectionManager imports are handled internally.

```python
# This works in both 0.4.1 and 0.5.0
from iris_vector_rag.core.connection import ConnectionManager
```

## Performance Impact

**None** - This is a packaging fix with zero performance impact:
- Import paths changed, but import overhead is negligible (<1ms)
- No runtime behavior changes
- No database query changes
- No LLM call changes

## Rollback Plan

If issues arise, rollback to 0.4.0 (not 0.4.1, which had the bug):

```bash
pip install iris-vector-rag==0.4.0
```

**Note**: 0.4.0 and earlier used `iris_rag` module name, not `iris_vector_rag`.

---
**Verification Status**: ⏳ Pending fix implementation
**Last Updated**: 2025-11-09
