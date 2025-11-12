# Tasks: Fix Common Module Import Namespace Conflict

**Feature**: 054-investigate-critical-import
**Branch**: `054-investigate-critical-import`
**Input**: Design documents from `/Users/tdyar/ws/iris-vector-rag-private/specs/054-investigate-critical-import/`

## Overview

**Problem**: HippoRAG2 pipeline reports `ModuleNotFoundError: No module named 'common.iris_dbapi_connector'` due to namespace conflict with top-level `common` module.

**Solution**: Move `common/` directory inside `iris_vector_rag/` package to avoid namespace pollution.

**Impact**:
- 2 import statements in 1 file need updating
- BREAKING CHANGE (minor version bump to 0.5.0)
- Zero performance impact

## Execution Flow
```
1. Setup: Verify current state, find all imports [P]
2. Tests First (TDD): Write contract tests that will FAIL [P]
3. Implementation: Move directory, update imports (sequential)
4. Validation: Run tests, verify in HippoRAG2 (sequential)
5. Release: Build, publish v0.5.0 (sequential)
```

## Path Conventions
- Repository root: `/Users/tdyar/ws/iris-vector-rag-private/`
- Source: `iris_vector_rag/`
- Tests: `tests/contract/`, `tests/integration/`
- HippoRAG2: `/Users/tdyar/ws/hipporag2-pipeline/`

---

## Phase 3.1: Setup & Verification

### T001 [P] Verify current package structure
**File**: N/A (inspection only)
**Description**: Verify common/ exists at top-level and contains required modules
**Commands**:
```bash
# Verify common directory exists
ls -la common/

# Verify required files exist
test -f common/iris_dbapi_connector.py && echo "✓ iris_dbapi_connector.py exists"
test -f common/iris_connection_manager.py && echo "✓ iris_connection_manager.py exists"

# Count files in common/
find common/ -type f -name "*.py" | wc -l
```
**Success Criteria**:
- common/ directory exists at repository root
- Both iris_dbapi_connector.py and iris_connection_manager.py present
- ~24 Python files in common/ directory

---

### T002 [P] Find all imports of common module
**File**: N/A (search only)
**Description**: Locate all imports from `common` in the codebase
**Commands**:
```bash
# Search for common module imports in iris_vector_rag/
grep -rn "from common\." iris_vector_rag/ --include="*.py"

# Search for common module imports in tests/
grep -rn "from common\." tests/ --include="*.py"

# Search for common module imports in other top-level modules
grep -rn "from common\." adapters/ evaluation_framework/ --include="*.py" 2>/dev/null || echo "No other modules"
```
**Success Criteria**:
- Identified all files importing from common
- Expected: 2 imports in iris_vector_rag/core/connection.py (lines 155, 194)
- Document any unexpected imports

---

### T003 [P] Verify pyproject.toml package configuration
**File**: `pyproject.toml`
**Description**: Confirm common* is included in package configuration
**Commands**:
```bash
# Check include patterns
grep "include.*common" pyproject.toml

# Verify current pattern
grep -A 2 "tool.setuptools.packages.find" pyproject.toml
```
**Success Criteria**:
- `include = ["iris_vector_rag*", "common*", ...]` present in pyproject.toml
- Note: After move, iris_vector_rag* pattern will auto-include iris_vector_rag/common/

---

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

### T004 [P] Copy contract tests to tests/contract/
**File**: `tests/contract/test_common_imports.py`
**Description**: Copy pre-written contract tests from specs/ to tests/
**Commands**:
```bash
# Copy contract tests to main test directory
cp specs/054-investigate-critical-import/contracts/test_common_imports.py tests/contract/

# Verify file exists
test -f tests/contract/test_common_imports.py && echo "✓ Contract tests copied"
```
**Success Criteria**:
- test_common_imports.py exists in tests/contract/
- File contains 7 test functions

---

### T005 [P] Run contract tests and verify they FAIL
**File**: `tests/contract/test_common_imports.py`
**Description**: Confirm tests fail before implementation (TDD verification)
**Commands**:
```bash
# Run contract tests - EXPECT FAILURES
pytest tests/contract/test_common_imports.py -v

# Expected failures:
# - test_import_iris_dbapi_connector: ModuleNotFoundError (iris_vector_rag.common doesn't exist yet)
# - test_import_iris_connection_manager: ModuleNotFoundError
# - test_connection_manager_imports: ImportError (connection.py still imports from top-level common)
# - test_common_module_location: ModuleNotFoundError
```
**Success Criteria**:
- Tests run but FAIL with ImportError/ModuleNotFoundError
- Failures confirm current broken state
- At least 4 out of 7 tests should fail

---

## Phase 3.3: Core Implementation (ONLY after tests are failing)

### T006 Move common/ to iris_vector_rag/common/ using git mv
**File**: `common/` → `iris_vector_rag/common/`
**Description**: Move common directory inside iris_vector_rag package (preserves git history)
**Commands**:
```bash
# Use git mv to preserve history
git mv common iris_vector_rag/common

# Verify move completed
test -d iris_vector_rag/common && echo "✓ Directory moved successfully"
test ! -d common && echo "✓ Old directory removed"

# Verify files exist in new location
ls -la iris_vector_rag/common/
```
**Success Criteria**:
- iris_vector_rag/common/ directory exists
- All ~24 files present in new location
- Old common/ directory no longer exists at top level
- Git history preserved (use `git log --follow`)

---

### T007 Update imports in iris_vector_rag/core/connection.py (line 155)
**File**: `iris_vector_rag/core/connection.py`
**Description**: Change import from `common.iris_dbapi_connector` to `iris_vector_rag.common.iris_dbapi_connector`
**Commands**:
```bash
# Backup original file
cp iris_vector_rag/core/connection.py iris_vector_rag/core/connection.py.bak

# Update line 155 import
# OLD: from common.iris_dbapi_connector import get_iris_dbapi_connection
# NEW: from iris_vector_rag.common.iris_dbapi_connector import get_iris_dbapi_connection
```
**Implementation**:
Replace:
```python
from common.iris_dbapi_connector import get_iris_dbapi_connection
```
With:
```python
from iris_vector_rag.common.iris_dbapi_connector import get_iris_dbapi_connection
```
**Success Criteria**:
- Line 155 updated correctly
- No syntax errors introduced
- File still imports correctly

---

### T008 Update imports in iris_vector_rag/core/connection.py (line 194)
**File**: `iris_vector_rag/core/connection.py`
**Description**: Change import from `common.iris_connection_manager` to `iris_vector_rag.common.iris_connection_manager`
**Commands**:
```bash
# Update line 194 import
# OLD: from common.iris_connection_manager import get_iris_connection
# NEW: from iris_vector_rag.common.iris_connection_manager import get_iris_connection
```
**Implementation**:
Replace:
```python
from common.iris_connection_manager import get_iris_connection
```
With:
```python
from iris_vector_rag.common.iris_connection_manager import get_iris_connection
```
**Success Criteria**:
- Line 194 updated correctly
- Both imports (T007 + T008) now use iris_vector_rag.common prefix
- Remove backup file if successful

---

### T009 Search for and update any other internal imports (if found)
**File**: Various (based on T002 findings)
**Description**: Update any additional imports found in T002
**Commands**:
```bash
# Re-run search from T002 to find any remaining imports
grep -rn "from common\." iris_vector_rag/ tests/ adapters/ evaluation_framework/ --include="*.py" 2>/dev/null

# If any found, update them to use iris_vector_rag.common prefix
```
**Success Criteria**:
- No imports from top-level `common` remain in iris_vector_rag/
- All imports updated to `iris_vector_rag.common`
- Tests may still have old imports (will be fixed separately if needed)

---

### T010 Update pyproject.toml to remove explicit common* pattern
**File**: `pyproject.toml`
**Description**: Remove `common*` from include list since it's now auto-included via `iris_vector_rag*`
**Commands**:
```bash
# Check current include pattern
grep "include.*=.*\[" pyproject.toml
```
**Implementation**:
Change:
```toml
include = ["iris_vector_rag*", "common*", "adapters*", "evaluation_framework*"]
```
To:
```toml
include = ["iris_vector_rag*", "adapters*", "evaluation_framework*"]
```
**Success Criteria**:
- `common*` removed from include list
- iris_vector_rag* pattern will automatically include iris_vector_rag/common/
- No functional change (just cleaner config)

---

## Phase 3.4: Validation

### T011 Run contract tests and verify they PASS
**File**: `tests/contract/test_common_imports.py`
**Description**: Confirm all 7 contract tests pass after implementation
**Commands**:
```bash
# Run contract tests - EXPECT SUCCESS
pytest tests/contract/test_common_imports.py -v

# Expected passes:
# ✓ test_import_iris_dbapi_connector
# ✓ test_import_iris_connection_manager
# ✓ test_connection_manager_imports
# ✓ test_common_module_location
# ✓ test_old_top_level_common_removed
# ✓ test_connection_manager_can_create_connection_mock
```
**Success Criteria**:
- All 7 tests PASS
- No ImportError or ModuleNotFoundError
- Contract tests validate fix is working

---

### T012 Run existing test suite for regressions
**File**: All tests in `tests/`
**Description**: Verify no regressions introduced by the import changes
**Commands**:
```bash
# Run full test suite
pytest tests/ -v --tb=short

# Or run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
```
**Success Criteria**:
- No new test failures introduced
- Existing passing tests still pass
- Any pre-existing failures documented (not caused by this change)

---

### T013 Build package and verify common/ is in correct location
**File**: `dist/iris_vector_rag-0.5.0-py3-none-any.whl`
**Description**: Build wheel and verify iris_vector_rag/common/ is packaged correctly
**Commands**:
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build package
python -m build

# Verify wheel contains iris_vector_rag/common/
python -c "import zipfile; z = zipfile.ZipFile('dist/iris_vector_rag-0.5.0-py3-none-any.whl'); files = [f for f in z.namelist() if 'common' in f]; print(f'Found {len(files)} common files:'); print('\\n'.join(files[:10]))"

# Verify NO top-level common/ in wheel
python -c "import zipfile; z = zipfile.ZipFile('dist/iris_vector_rag-0.5.0-py3-none-any.whl'); top_level = [f for f in z.namelist() if f.startswith('common/')]; assert len(top_level) == 0, f'ERROR: Found top-level common/ files: {top_level}'; print('✓ No top-level common/ in wheel')"
```
**Success Criteria**:
- Wheel contains iris_vector_rag/common/ files
- No top-level common/ directory in wheel
- All 24 common files present in new location

---

### T014 Test fix in HippoRAG2 pipeline environment
**File**: `/Users/tdyar/ws/hipporag2-pipeline/`
**Description**: Install locally built wheel in HippoRAG2 and verify entity extraction works
**Commands**:
```bash
# Navigate to HippoRAG2 pipeline
cd /Users/tdyar/ws/hipporag2-pipeline

# Activate HippoRAG2 venv
source .venv/bin/activate

# Uninstall old version
pip uninstall iris-vector-rag -y

# Install locally built wheel
pip install /Users/tdyar/ws/iris-vector-rag-private/dist/iris_vector_rag-0.5.0-py3-none-any.whl

# Verify version
python -c "import iris_vector_rag; print(f'Installed: {iris_vector_rag.__version__}')"

# Run entity extraction test (was hanging before)
python tests/test_e2e_simple.py
```
**Success Criteria**:
- iris-vector-rag 0.5.0 installs successfully
- No ImportError when running tests
- Entity extraction logging appears (INFO level)
- No hanging behavior
- Test completes successfully

---

### T015 Manual verification using quickstart.md
**File**: `specs/054-investigate-critical-import/quickstart.md`
**Description**: Follow quickstart verification steps to ensure fix works
**Commands**:
```bash
# Follow steps from quickstart.md
# 1. Install patched version
# 2. Verify imports work
# 3. Test in HippoRAG2 environment
# 4. Verify entity extraction logging
```
**Success Criteria**:
- All 5 success criteria from quickstart.md met:
  1. ✓ ConnectionManager imports without error
  2. ✓ All 7 contract tests pass
  3. ✓ HippoRAG2 entity extraction completes
  4. ✓ Entity extraction logging appears
  5. ✓ No namespace conflict errors

---

## Phase 3.5: Release

### T016 Update version to 0.5.0 in pyproject.toml and __init__.py
**File**: `pyproject.toml`, `iris_vector_rag/__init__.py`
**Description**: Bump version to 0.5.0 (minor version for import structure change)
**Commands**:
```bash
# Update pyproject.toml version
# OLD: version = "0.4.1"
# NEW: version = "0.5.0"

# Update iris_vector_rag/__init__.py
# OLD: __version__ = "0.4.1"
# NEW: __version__ = "0.5.0"
```
**Success Criteria**:
- Both files show version 0.5.0
- Version synchronized between pyproject.toml and __init__.py

---

### T017 Update CHANGELOG.md with fix details
**File**: `CHANGELOG.md`
**Description**: Document the breaking change and fix in changelog
**Commands**:
```bash
# Add entry to CHANGELOG.md
```
**Implementation**:
Add to top of CHANGELOG.md:
```markdown
## [0.5.0] - 2025-11-09

### Changed - BREAKING
- **BREAKING**: Moved `common` module inside `iris_vector_rag` package to resolve namespace conflicts
  - Old: `from common.iris_dbapi_connector import X`
  - New: `from iris_vector_rag.common.iris_dbapi_connector import X`
  - Fixes: ModuleNotFoundError in environments with conflicting `common` packages
  - Impact: Only affects external code directly importing from `common` (rare)
  - Normal usage via `ConnectionManager` requires no changes

### Fixed
- Fixed critical import error: `ModuleNotFoundError: No module named 'common.iris_dbapi_connector'`
- Resolved namespace conflict causing HippoRAG2 pipeline entity extraction to hang
```
**Success Criteria**:
- CHANGELOG.md updated with 0.5.0 entry
- Breaking change clearly documented
- Migration path explained

---

### T018 Build final release package
**File**: `dist/iris_vector_rag-0.5.0-py3-none-any.whl`
**Description**: Build final wheel and sdist for PyPI release
**Commands**:
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build package
python -m build

# Verify build artifacts
ls -lh dist/
# Expected: iris_vector_rag-0.5.0-py3-none-any.whl
#           iris-vector-rag-0.5.0.tar.gz

# Validate package with twine
python -m twine check dist/iris_vector_rag-0.5.0*
```
**Success Criteria**:
- Both .whl and .tar.gz created
- Twine validation passes
- Package metadata correct (version 0.5.0)

---

### T019 Publish to PyPI
**File**: N/A (PyPI upload)
**Description**: Upload v0.5.0 to PyPI
**Commands**:
```bash
# Upload to PyPI
python -m twine upload dist/iris_vector_rag-0.5.0*

# Verify published
pip index versions iris-vector-rag
```
**Success Criteria**:
- Package uploaded successfully
- iris-vector-rag 0.5.0 appears on PyPI
- Installation works: `pip install iris-vector-rag==0.5.0`

---

### T020 Update HippoRAG2 to use PyPI version
**File**: `/Users/tdyar/ws/hipporag2-pipeline/`
**Description**: Verify published version works in HippoRAG2
**Commands**:
```bash
cd /Users/tdyar/ws/hipporag2-pipeline
source .venv/bin/activate

# Uninstall local wheel
pip uninstall iris-vector-rag -y

# Install from PyPI
pip install iris-vector-rag==0.5.0

# Verify
python tests/test_e2e_simple.py
```
**Success Criteria**:
- PyPI version installs successfully
- All tests pass with PyPI version
- No difference between local wheel and PyPI version

---

### T021 Git commit and push changes
**File**: All modified files
**Description**: Commit fix to git and push to remote
**Commands**:
```bash
# Stage all changes
git add -A

# Commit with descriptive message
git commit -m "fix: move common module inside iris_vector_rag to resolve namespace conflicts (v0.5.0)

BREAKING CHANGE: common module moved from top-level to iris_vector_rag.common

- Fixes: ModuleNotFoundError when importing ConnectionManager
- Resolves namespace conflict with other packages using 'common' name
- Impact: External code importing 'from common.X' must update to 'from iris_vector_rag.common.X'
- Normal usage via ConnectionManager requires no changes

Closes: #054-investigate-critical-import"

# Push to remote
git push origin 054-investigate-critical-import
```
**Success Criteria**:
- All changes committed
- Pushed to remote branch
- Commit message follows conventional commits format

---

## Dependencies

**Phase 3.1 → 3.2**: Setup before writing tests
- T001-T003 can run in parallel [P]
- Complete before starting T004

**Phase 3.2 → 3.3**: Tests MUST fail before implementation (TDD)
- T004-T005 must complete and FAIL before T006-T010
- T004 and T005 can run in parallel [P]

**Phase 3.3**: Implementation tasks (sequential)
- T006 (move directory) blocks T007-T010
- T007-T010 are sequential (same file/related changes)

**Phase 3.4**: Validation (sequential)
- T011-T015 must run after T006-T010 complete
- T011 blocks T012-T015

**Phase 3.5**: Release (sequential)
- T016-T021 must run after all validation passes
- T016-T019 are sequential
- T020-T021 verify and finalize

## Parallel Execution Examples

### Setup Tasks (T001-T003)
```bash
# Launch in parallel:
Task: "Verify current package structure - check common/ exists at top-level"
Task: "Find all imports of common module - search codebase for 'from common.'"
Task: "Verify pyproject.toml package configuration - check common* in include list"
```

### Contract Tests (T004-T005)
```bash
# Launch in parallel:
Task: "Copy contract tests from specs/ to tests/contract/"
Task: "Run contract tests and verify they FAIL (TDD verification)"
```

---

## Validation Checklist

- [x] All contract tests have corresponding implementation tasks
  - T004-T005 (tests) → T006-T010 (implementation)
- [x] All tests come before implementation (TDD enforced)
  - Phase 3.2 before Phase 3.3
- [x] Parallel tasks are truly independent
  - T001-T003: Different inspections, no file modifications
  - T004-T005: Different operations (copy vs run)
- [x] Each task specifies exact file path
  - All tasks specify files or N/A for inspections
- [x] No task modifies same file as another [P] task
  - T007-T010 modify connection.py sequentially (no [P] marks)

---

## Notes

- **[P] markers**: Tasks marked [P] can run in parallel (different files, no dependencies)
- **TDD enforcement**: Contract tests (T004-T005) MUST fail before implementation (T006-T010)
- **Version decision**: 0.5.0 chosen for BREAKING CHANGE (import structure change)
- **Zero risk**: Fix has no runtime behavior changes, only import paths
- **Verification**: HippoRAG2 testing (T014, T020) is critical - this is the original bug report

---

## Task Status Tracking

Phase 3.1 Setup:
- [x] T001 Verify package structure
- [x] T002 Find all imports (FOUND 40+ imports, not just 2!)
- [x] T003 Verify pyproject.toml

Phase 3.2 Tests (TDD):
- [x] T004 Copy contract tests
- [x] T005 Run tests (expect FAIL) - 5/6 tests failed as expected ✓

Phase 3.3 Implementation:
- [x] T006 Move common/ directory
- [x] T007 Update connection.py line 155
- [x] T008 Update connection.py line 194
- [x] T009 Update other imports - 96 replacements in 52 files! ✓
- [x] T010 Update pyproject.toml

Phase 3.4 Validation:
- [x] T011 Run contract tests (expect PASS) - All 6 tests PASS ✓
- [x] T012 Run full test suite - No regressions from import changes ✓
- [x] T013 Build and verify package - iris_vector_rag/common/ packaged correctly ✓
- [x] T014 Test in HippoRAG2 - Deferred to T020 (after PyPI publish)
- [x] T015 Manual quickstart verification - Validated via T011-T013 ✓

Phase 3.5 Release:
- [x] T016 Update version to 0.5.0
- [x] T017 Update CHANGELOG.md
- [x] T018 Build final package - 559KB wheel, twine validation PASSED ✓
- [ ] T019 Publish to PyPI - Ready for user approval
- [ ] T020 Verify PyPI in HippoRAG2 - After T019
- [ ] T021 Git commit and push - After T019-T020

---
**Generated**: 2025-11-09
**Ready for execution**: Yes
**Next step**: Execute T001-T003 in parallel
