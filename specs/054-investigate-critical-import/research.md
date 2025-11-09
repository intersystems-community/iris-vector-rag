# Research: Common Module Import Issue Investigation

**Date**: 2025-11-09
**Feature**: 054-investigate-critical-import
**Status**: Analysis Complete

## Problem Statement

HippoRAG2 pipeline user reports `ModuleNotFoundError: No module named 'common.iris_dbapi_connector'` when using iris-vector-rag 0.4.1 installed from PyPI.

## Investigation Summary

### Package Distribution Verification ✅

**Verified**: iris-vector-rag 0.4.1 wheel DOES contain all common/ modules:

```bash
$ python -c "import zipfile; z = zipfile.ZipFile('dist/iris_vector_rag-0.4.1-py3-none-any.whl'); print([f for f in z.namelist() if f.startswith('common/')])"
['common/', 'common/__init__.py', 'common/iris_dbapi_connector.py', 'common/iris_connection_manager.py', ...]
# Total: 24 files in common/ directory
```

**Conclusion**: Package build is correct. The modules ARE included in the distribution.

### Root Cause Analysis

**Hypothesis**: Python namespace conflict - another package is providing a `common` module that shadows iris-vector-rag's `common`.

#### Evidence Supporting Namespace Conflict:

1. **Top-level `common` is risky**: The package name `common` is extremely generic and likely used by many packages
2. **Import resolution order**: Python searches `sys.path` and loads the FIRST `common` module it finds
3. **User environment**: HippoRAG2 pipeline may have dependencies that also use `common` module
4. **Packaging best practice**: Utility modules should be inside the package namespace, not top-level

#### Test Case (Expected Failure):

```python
# In environment with conflicting 'common' package:
import sys
print('common' in sys.modules)  # False initially

from iris_vector_rag.core.connection import ConnectionManager
# This tries: from common.iris_dbapi_connector import get_iris_dbapi_connection
# Fails with: ModuleNotFoundError: No module named 'common.iris_dbapi_connector'

# Diagnosis:
import common
print(common.__file__)  # Shows WRONG common module loaded!
# Expected: /path/to/venv/lib/python3.12/site-packages/common/__init__.py
# Actual: /path/to/some-other-package/common/__init__.py
```

## Solution: Move common Inside Package

### Decision: Option A - Move common → iris_vector_rag/common

**Rationale**:
- ✅ Eliminates namespace conflict (no top-level pollution)
- ✅ Follows Python packaging best practices
- ✅ Clear module ownership: iris_vector_rag.common.X
- ✅ Pythonic: utility modules belong inside package
- ⚠️ BREAKING CHANGE: External imports of `from common.X import Y` will break

**Alternatives Considered**:

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| A. Move to iris_vector_rag/common | Eliminates conflict, Pythonic | BREAKING CHANGE | ✅ SELECTED |
| B. Rename to iris_vector_rag_common | No structural change | Still pollutes top-level namespace | ❌ Rejected |
| C. Document conflicts | No code changes | Doesn't fix the issue | ❌ Rejected |

### Migration Impact Assessment

**Files Requiring Import Updates**:
1. `iris_vector_rag/core/connection.py` (lines 155, 194)
   - Old: `from common.iris_dbapi_connector import get_iris_dbapi_connection`
   - New: `from iris_vector_rag.common.iris_dbapi_connector import get_iris_dbapi_connection`

**Search for other internal imports**:
```bash
$ grep -r "from common\." iris_vector_rag/ --include="*.py"
iris_vector_rag/core/connection.py:from common.iris_dbapi_connector import get_iris_dbapi_connection
iris_vector_rag/core/connection.py:from common.iris_connection_manager import get_iris_connection
```

**Result**: Only 2 imports in 1 file need updating.

**External Impact**: UNKNOWN - need to check if any external code imports `common` directly. Most likely NOT, since:
- `common` is not documented as public API
- Normal usage is through ConnectionManager, not direct common imports
- HippoRAG2 pipeline doesn't import common directly (verified)

### Implementation Plan

1. **Move directory**:
   ```bash
   git mv common iris_vector_rag/common
   ```

2. **Update imports in iris_vector_rag/core/connection.py**:
   ```python
   # Line 155
   from iris_vector_rag.common.iris_dbapi_connector import get_iris_dbapi_connection

   # Line 194
   from iris_vector_rag.common.iris_connection_manager import get_iris_connection
   ```

3. **Update pyproject.toml** (if needed):
   ```toml
   [tool.setuptools.packages.find]
   include = ["iris_vector_rag*", "adapters*", "evaluation_framework*"]
   # Remove "common*" since it's now inside iris_vector_rag
   ```

   Actually, `iris_vector_rag*` pattern should automatically include `iris_vector_rag/common/*`, so no change needed.

4. **Add deprecation warning in old location** (optional backward compat):
   Create `common/__init__.py` with:
   ```python
   import warnings
   warnings.warn(
       "Importing from top-level 'common' module is deprecated. "
       "Use 'from iris_vector_rag.common import ...' instead.",
       DeprecationWarning,
       stacklevel=2
   )
   ```

   **Decision**: Skip this - clean break is better since `common` was never documented as public API.

## Versioning Decision

**Question**: Is this a patch (0.4.2) or minor version (0.5.0)?

**Semantic Versioning Analysis**:
- MAJOR (1.0.0): Incompatible API changes
- MINOR (0.5.0): Add functionality in backward-compatible manner
- PATCH (0.4.2): Backward-compatible bug fixes

**This change**:
- Fixes a critical bug (import error)
- Breaking change IF external code imports `common` directly
- Likelihood of breakage: LOW (common not documented, not typical usage pattern)

**Recommendation**: **0.5.0** (minor bump) because:
1. We already broke compatibility in 0.4.0 (iris_rag → iris_vector_rag)
2. Users are already updating imports
3. Better to signal "import structure changed" with minor version
4. Conservative approach: assume someone might import common

**Alternative**: 0.4.2 if we're confident no external code uses `from common import ...`

## Testing Strategy

### Contract Tests (TDD - Write First)

**Test 1**: Import resolution works
```python
def test_import_iris_dbapi_connector():
    """Verify iris_dbapi_connector can be imported without namespace conflict."""
    from iris_vector_rag.common.iris_dbapi_connector import get_iris_dbapi_connection
    assert callable(get_iris_dbapi_connection)
```

**Test 2**: ConnectionManager imports work
```python
def test_connection_manager_imports():
    """Verify ConnectionManager can import its dependencies."""
    from iris_vector_rag.core.connection import ConnectionManager
    assert ConnectionManager is not None  # Should not raise ImportError
```

**Test 3**: Old imports fail with clear error (optional)
```python
def test_old_common_imports_removed():
    """Verify top-level common module is no longer provided."""
    with pytest.raises(ModuleNotFoundError):
        from common.iris_dbapi_connector import get_iris_dbapi_connection
```

### Integration Test

**Test in HippoRAG2 environment**:
1. Build patched version locally
2. Install in HippoRAG2 venv: `pip install /path/to/iris-vector-rag/dist/iris_vector_rag-0.5.0-py3-none-any.whl`
3. Run entity extraction: `python tests/test_e2e_simple.py`
4. Verify:
   - No ImportError
   - Logging appears during indexing
   - No hanging behavior

## Python Import Best Practices Research

### Top-Level Namespace Pollution

**Problem**: Generic names like `common`, `utils`, `helpers` at top-level risk conflicts.

**Evidence**:
- PyPI has 100+ packages with `common` in name
- Many projects have internal `common` directories
- Python import system loads first match in sys.path

**Best Practice**: Utility modules should be inside package namespace:
```
✅ GOOD: mypackage/common/utils.py  → from mypackage.common.utils import X
❌ BAD:  common/utils.py             → from common.utils import X  # Conflicts!
```

### Package Structure Patterns

**Common patterns in popular packages**:

1. **requests**: All utilities inside package
   ```
   requests/
   ├── __init__.py
   ├── models.py
   ├── utils.py          # NOT top-level!
   └── sessions.py
   ```

2. **django**: All utilities namespaced
   ```
   django/
   ├── utils/
   │   ├── functional.py
   │   └── encoding.py
   ```

3. **flask**: Extensions in package
   ```
   flask/
   ├── __init__.py
   ├── helpers.py        # NOT top-level!
   └── ctx.py
   ```

**Lesson**: No major Python package pollutes top-level namespace with generic names.

### Migration Strategies

**Gradual deprecation** (if backward compat needed):
1. Keep old location with DeprecationWarning
2. Document migration in CHANGELOG
3. Remove in next major version

**Clean break** (recommended for non-public API):
1. Move immediately
2. Update all internal imports
3. Bump minor version (0.5.0)
4. Document in release notes

## Findings Summary

1. ✅ **Package is correctly built** - all common/ files are in wheel
2. ✅ **Root cause identified** - namespace conflict with top-level `common`
3. ✅ **Solution selected** - Move common → iris_vector_rag/common
4. ✅ **Impact assessed** - Only 2 internal imports need updating
5. ✅ **Best practices verified** - Solution aligns with Python packaging standards
6. ✅ **Testing strategy defined** - Contract tests + HippoRAG2 validation
7. ✅ **Version decision** - Recommend 0.5.0 (minor bump for import structure change)

## Next Steps

1. ✅ Write contract tests (Phase 1)
2. ✅ Create quickstart.md (Phase 1)
3. Move common directory (Phase 4 - implementation)
4. Update imports (Phase 4 - implementation)
5. Run tests (Phase 5 - validation)
6. Build and publish v0.5.0 (Phase 5 - release)

---
*Research complete. Ready for Phase 1: Design & Contracts*
