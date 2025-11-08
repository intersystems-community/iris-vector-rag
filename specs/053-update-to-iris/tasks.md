# Tasks: Update iris-vector-rag to use iris-vector-graph 1.1.1

**Input**: Design documents from `/specs/053-update-to-iris/`
**Prerequisites**: plan.md, research.md, data-model.md, contracts/, quickstart.md
**Branch**: `053-update-to-iris`
**Estimated Time**: 15-20 minutes total

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → ✅ Loaded - pure import path migration, 4 files to update
   → Extract: Python 3.10+, iris-vector-graph >= 1.1.1, pytest for testing
2. Load optional design documents:
   → ✅ data-model.md: Import path mapping (old → new)
   → ✅ contracts/: test_import_iris_vector_graph.py (11 contract tests)
   → ✅ research.md: Migration strategy and version constraint
3. Generate tasks by category:
   → Setup: Dependency update only (pyproject.toml)
   → Tests: 11 contract tests for import validation
   → Core: Update import statements in 1 primary file
   → Integration: Update test files referencing old imports
   → Polish: Validation and documentation
4. Apply task rules:
   → Contract tests can run in parallel [P] (independent test methods)
   → Import updates sequential (same file, hybrid_graphrag_discovery.py)
   → Dependency update before implementation
5. Number tasks sequentially (T001, T002...)
6. Generate dependency graph
7. Create parallel execution examples
8. Validate task completeness:
   → ✅ All contracts have tests (11 tests in 1 file)
   → ✅ All import paths mapped (4 imports documented)
   → ✅ All affected files identified (4 total)
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different test methods, no file conflicts)
- All file paths are absolute and verified

## Path Conventions
**Project Type**: Single Python package (iris_rag/)
- Source code: `iris_rag/` at repository root
- Tests: `tests/` at repository root
- Config: `pyproject.toml` at repository root
- Docs: `docs/` at repository root

---

## Phase 3.1: Setup

### T001: ✅ Update pyproject.toml dependency constraint
**File**: `/Users/tdyar/ws/rag-templates/pyproject.toml`
**Lines**: 76, 79
**Description**: Update iris-vector-graph version constraint from `>=2.0.0` to `>=1.1.1`

**Changes Required**:
```toml
# Line 76 (embedding optional dependencies)
OLD: "iris-vector-graph>=2.0.0"
NEW: "iris-vector-graph>=1.1.1"

# Line 79 (hybrid-graphrag optional dependencies)
OLD: "iris-vector-graph>=2.0.0"
NEW: "iris-vector-graph>=1.1.1"
```

**Validation**:
```bash
grep "iris-vector-graph>=1.1.1" pyproject.toml
# Should show 2 matches
```

**Dependencies**: None
**Blocks**: T008, T009, T010 (implementation tasks require correct dependency)

---

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL**: These contract tests validate the new import structure. They should PASS for direct imports (iris-vector-graph 1.1.1 is correctly structured) but MAY FAIL for GraphCoreDiscovery until implementation is complete.

### T002 [P]: ✅ Contract test - Import IRISGraphEngine
**File**: `/Users/tdyar/ws/rag-templates/specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py`
**Test Method**: `test_import_iris_graph_engine()`
**Description**: Validate `from iris_vector_graph import IRISGraphEngine` succeeds

**Expected Behavior**:
- ✅ PASS (iris-vector-graph 1.1.1 correctly exports IRISGraphEngine)

**Validates**: FR-001 from feature specification

### T003 [P]: ✅ Contract test - Import HybridSearchFusion
**File**: `/Users/tdyar/ws/rag-templates/specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py`
**Test Method**: `test_import_hybrid_search_fusion()`
**Description**: Validate `from iris_vector_graph import HybridSearchFusion` succeeds

**Expected Behavior**:
- ✅ PASS (iris-vector-graph 1.1.1 correctly exports HybridSearchFusion)

**Validates**: FR-002 from feature specification

### T004 [P]: ✅ Contract test - Import TextSearchEngine
**File**: `/Users/tdyar/ws/rag-templates/specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py`
**Test Method**: `test_import_text_search_engine()`
**Description**: Validate `from iris_vector_graph import TextSearchEngine` succeeds

**Expected Behavior**:
- ✅ PASS (iris-vector-graph 1.1.1 correctly exports TextSearchEngine)

**Validates**: FR-003 from feature specification

### T005 [P]: ✅ Contract test - Import VectorOptimizer
**File**: `/Users/tdyar/ws/rag-templates/specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py`
**Test Method**: `test_import_vector_optimizer()`
**Description**: Validate `from iris_vector_graph import VectorOptimizer` succeeds

**Expected Behavior**:
- ✅ PASS (iris-vector-graph 1.1.1 correctly exports VectorOptimizer)

**Validates**: FR-004 from feature specification

### T006 [P]: ✅ Contract test - Old import path fails
**File**: `/Users/tdyar/ws/rag-templates/specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py`
**Test Method**: `test_old_iris_vector_graph_core_import_fails()`
**Description**: Validate `from iris_vector_graph_core.* import` raises ImportError

**Expected Behavior**:
- ✅ PASS (iris-vector-graph 1.1.1 removed iris_vector_graph_core module)

**Validates**: FR-006 from feature specification

### T007 [P]: ✅ Contract test - Version requirement
**File**: `/Users/tdyar/ws/rag-templates/specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py`
**Test Method**: `test_iris_vector_graph_version_requirement()`
**Description**: Validate iris-vector-graph version >= 1.1.1

**Expected Behavior**:
- ✅ PASS (dependency constraint enforced by T001)

**Validates**: FR-005, FR-009 from feature specification

### T008 [P]: Integration test - GraphCoreDiscovery imports
**File**: `/Users/tdyar/ws/rag-templates/specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py`
**Test Method**: `test_graph_core_discovery_imports_successfully()`
**Description**: Validate GraphCoreDiscovery.import_graph_core_modules() returns correct modules

**Expected Behavior BEFORE Implementation**:
- ❌ FAIL (hybrid_graphrag_discovery.py still uses old imports)

**Expected Behavior AFTER Implementation**:
- ✅ PASS (hybrid_graphrag_discovery.py updated to use new imports)

**Validates**: FR-008 from feature specification (existing functionality continues to work)

**Dependencies**: Blocks T012 (must validate imports work before marking complete)

### T009 [P]: Integration test - Import caching
**File**: `/Users/tdyar/ws/rag-templates/specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py`
**Test Method**: `test_graph_core_discovery_caches_imports()`
**Description**: Validate imports are cached after first load (performance check)

**Expected Behavior**: PASS after T012 (implementation complete)

**Validates**: Performance expectation (not a functional requirement)

### T010 [P]: ✅ Integration test - HybridGraphRAGPipeline imports
**File**: `/Users/tdyar/ws/rag-templates/specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py`
**Test Method**: `test_hybrid_graphrag_pipeline_imports()`
**Description**: Validate HybridGraphRAGPipeline can be imported without errors

**Expected Behavior**: PASS after T012 (implementation complete)

**Validates**: FR-008 (backward compatibility)

### T011 [P]: ✅ Integration test - Pipeline factory
**File**: `/Users/tdyar/ws/rag-templates/specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py`
**Test Method**: `test_create_pipeline_graphrag_type()`
**Description**: Validate create_pipeline("graphrag") import works

**Expected Behavior**: PASS (smoke test, doesn't create pipeline, just validates import)

**Validates**: FR-008 (backward compatibility)

---

## Phase 3.3: Core Implementation (ONLY after tests T002-T007 pass)

### T012: Update hybrid_graphrag_discovery.py import statements
**File**: `/Users/tdyar/ws/rag-templates/iris_rag/pipelines/hybrid_graphrag_discovery.py`
**Description**: Update all iris_vector_graph_core imports to iris_vector_graph

**Changes Required**:

**Change #1: Installed package imports (lines 127-130)**
```python
# OLD:
from iris_vector_graph_core.engine import IRISGraphEngine
from iris_vector_graph_core.fusion import HybridSearchFusion
from iris_vector_graph_core.text_search import TextSearchEngine
from iris_vector_graph_core.vector_utils import VectorOptimizer

# NEW:
from iris_vector_graph import IRISGraphEngine
from iris_vector_graph import HybridSearchFusion
from iris_vector_graph import TextSearchEngine
from iris_vector_graph import VectorOptimizer
```

**Change #2: Local path fallback imports (lines 166-169)**
```python
# OLD:
from iris_vector_graph_core.engine import IRISGraphEngine
from iris_vector_graph_core.fusion import HybridSearchFusion
from iris_vector_graph_core.text_search import TextSearchEngine
from iris_vector_graph_core.vector_utils import VectorOptimizer

# NEW:
from iris_vector_graph import IRISGraphEngine
from iris_vector_graph import HybridSearchFusion
from iris_vector_graph import TextSearchEngine
from iris_vector_graph import VectorOptimizer
```

**Change #3: Path validation logic (lines 82-86)**
```python
# OLD:
# Check for either new or legacy package structure
new_package = path / "iris_vector_graph_core"
legacy_package = path / "iris_graph_core"

return (new_package.exists() and new_package.is_dir()) or (
    legacy_package.exists() and legacy_package.is_dir()
)

# NEW:
# Check for iris_vector_graph package directory (top-level module)
package_dir = path / "iris_vector_graph"
return package_dir.exists() and package_dir.is_dir()
```

**Validation After Change**:
```bash
# Verify old imports removed
grep "iris_vector_graph_core" iris_rag/pipelines/hybrid_graphrag_discovery.py
# Should return: 0 matches

# Verify new imports added
grep "from iris_vector_graph import" iris_rag/pipelines/hybrid_graphrag_discovery.py
# Should return: 8 matches (4 imports × 2 locations)
```

**Dependencies**:
- Requires: T001 (dependency constraint updated)
- Blocks: T013, T014 (test file updates depend on this)

**Validates**: FR-001, FR-002, FR-003, FR-004, FR-006, FR-007, FR-008 from feature specification

---

## Phase 3.4: Integration

### T013: Update test_import_requirements.py
**File**: `/Users/tdyar/ws/rag-templates/tests/contract/test_import_requirements.py`
**Description**: Update any references to iris_vector_graph_core in contract tests

**Changes Required**:
```bash
# Search for old import references
grep -n "iris_vector_graph_core" tests/contract/test_import_requirements.py

# If found, replace with iris_vector_graph
sed -i '' 's/iris_vector_graph_core/iris_vector_graph/g' tests/contract/test_import_requirements.py
```

**Validation**:
```bash
# Should return 0 matches:
grep "iris_vector_graph_core" tests/contract/test_import_requirements.py
```

**Dependencies**: Requires T012 (main implementation complete)

### T014: Update test_hybrid_graphrag.py
**File**: `/Users/tdyar/ws/rag-templates/tests/unit/test_hybrid_graphrag.py`
**Description**: Update any references to iris_vector_graph_core in unit tests

**Changes Required**:
```bash
# Search for old import references
grep -n "iris_vector_graph_core" tests/unit/test_hybrid_graphrag.py

# If found, replace with iris_vector_graph
sed -i '' 's/iris_vector_graph_core/iris_vector_graph/g' tests/unit/test_hybrid_graphrag.py
```

**Validation**:
```bash
# Should return 0 matches:
grep "iris_vector_graph_core" tests/unit/test_hybrid_graphrag.py
```

**Dependencies**: Requires T012 (main implementation complete)

---

## Phase 3.5: Polish

### T015: Run all contract tests
**File**: `/Users/tdyar/ws/rag-templates/specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py`
**Description**: Validate all 11 contract tests pass after implementation

**Command**:
```bash
pytest specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py -v
```

**Expected Output**:
```
test_import_iris_graph_engine PASSED
test_import_hybrid_search_fusion PASSED
test_import_text_search_engine PASSED
test_import_vector_optimizer PASSED
test_old_iris_vector_graph_core_import_fails PASSED
test_iris_vector_graph_version_requirement PASSED
test_graph_core_discovery_imports_successfully PASSED
test_graph_core_discovery_caches_imports PASSED
test_hybrid_graphrag_pipeline_imports PASSED
test_create_pipeline_graphrag_type PASSED
test_helpful_error_when_package_missing PASSED

============== 11 passed in 0.5s ==============
```

**Dependencies**:
- Requires: T012 (implementation), T013, T014 (test updates)
- Blocks: T016 (local installation validation)

**Validates**: FR-010 from feature specification (all tests pass)

### T016: Test local installation
**Description**: Install updated package and verify imports work

**Command**:
```bash
# Reinstall package with updated dependencies
uv pip install -e .[hybrid-graphrag]

# Verify direct imports work
python -c "
from iris_vector_graph import IRISGraphEngine, HybridSearchFusion, TextSearchEngine, VectorOptimizer
print('✅ All imports successful')
print(f'  - IRISGraphEngine: {IRISGraphEngine.__name__}')
print(f'  - HybridSearchFusion: {HybridSearchFusion.__name__}')
print(f'  - TextSearchEngine: {TextSearchEngine.__name__}')
print(f'  - VectorOptimizer: {VectorOptimizer.__name__}')
"

# Verify GraphCoreDiscovery works
python -c "
from iris_rag.pipelines.hybrid_graphrag_discovery import GraphCoreDiscovery
discovery = GraphCoreDiscovery()
success, modules = discovery.import_graph_core_modules()
assert success, 'Import should succeed'
print('✅ GraphCoreDiscovery successfully imports modules')
print(f'  - Modules: {list(modules.keys())}')
"
```

**Expected Output**:
```
✅ All imports successful
  - IRISGraphEngine: IRISGraphEngine
  - HybridSearchFusion: HybridSearchFusion
  - TextSearchEngine: TextSearchEngine
  - VectorOptimizer: VectorOptimizer
✅ GraphCoreDiscovery successfully imports modules
  - Modules: ['IRISGraphEngine', 'HybridSearchFusion', 'TextSearchEngine', 'VectorOptimizer']
```

**Dependencies**: Requires T015 (all tests pass)
**Blocks**: T017 (documentation update)

### T017: Update documentation
**File**: `/Users/tdyar/ws/rag-templates/docs/CLAUDE.md`
**Description**: Verify HybridGraphRAG requirements section reflects iris-vector-graph >= 1.1.1

**Validation**:
```bash
grep "iris-vector-graph>=1.1.1" docs/CLAUDE.md
# Should show at least 1 match with context about Feature 053
```

**Note**: Documentation was already updated in Phase 1 (planning). This task just validates it.

**Dependencies**: Requires T016 (installation validated)

### T018: Validate quickstart guide
**File**: `/Users/tdyar/ws/rag-templates/specs/053-update-to-iris/quickstart.md`
**Description**: Walk through quickstart.md steps 1-9 to ensure they work

**Command**:
```bash
# Step 1: Verify current state
source .venv/bin/activate
python -c "from importlib.metadata import version; print(f'iris-vector-graph: {version(\"iris-vector-graph\")}')"

# Step 6: Run contract tests (should all PASS)
pytest specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py -v

# Step 8: Test local installation (should succeed)
python -c "from iris_vector_graph import IRISGraphEngine; print('✅ Import successful')"
```

**Expected**: All commands succeed with expected output from quickstart.md

**Dependencies**: Requires T017 (documentation verified)
**Blocks**: T019 (commit changes)

### T019: Commit implementation changes
**Description**: Commit all changes with clear commit message

**Command**:
```bash
# Stage all changes
git add pyproject.toml
git add iris_rag/pipelines/hybrid_graphrag_discovery.py
git add tests/unit/test_hybrid_graphrag.py
git add tests/contract/test_import_requirements.py
git add specs/053-update-to-iris/
git add docs/CLAUDE.md

# Commit with comprehensive message
git commit -m "feat: update to iris-vector-graph 1.1.1 top-level imports

Update import statements from iris_vector_graph_core to iris_vector_graph
to support iris-vector-graph 1.1.1+ which removed the _core submodule.

Changes:
- Update pyproject.toml dependency: iris-vector-graph>=1.1.1 (from >=2.0.0)
- Update hybrid_graphrag_discovery.py imports (lines 127-130, 166-169)
- Update path validation logic to check for iris_vector_graph package
- Update test files to use new import paths
- Add contract tests for import validation (11 tests)
- Update docs/CLAUDE.md with new requirements

All contract tests pass (11/11).
Existing HybridGraphRAG functionality preserved.

Validates: FR-001 through FR-010 from specs/053-update-to-iris/spec.md"
```

**Validation**:
```bash
git status
# Should show clean working directory
```

**Dependencies**: Requires T018 (quickstart validated)

---

## Dependencies Graph

```
T001 (pyproject.toml)
  ↓
T002-T011 [P] (contract tests - can run in parallel)
  ↓
T012 (update hybrid_graphrag_discovery.py)
  ↓
T013 (update test_import_requirements.py)
  ↓
T014 (update test_hybrid_graphrag.py)
  ↓
T015 (run all contract tests)
  ↓
T016 (test local installation)
  ↓
T017 (validate documentation)
  ↓
T018 (validate quickstart guide)
  ↓
T019 (commit changes)
```

**Critical Path**: T001 → T012 → T013 → T014 → T015 → T016 → T017 → T018 → T019

**Parallel Opportunities**: T002-T011 (10 contract tests can run simultaneously)

---

## Parallel Execution Examples

### Run Contract Tests in Parallel (T002-T011)
```bash
# All contract tests are in the same file but test different methods
# pytest automatically runs test methods in parallel with pytest-xdist plugin
pytest specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py -v -n auto
```

**Without pytest-xdist**: Tests run sequentially but complete in < 1 second total

---

## Task Summary

**Total Tasks**: 19
- **Setup**: 1 task (T001)
- **Contract Tests**: 10 tasks (T002-T011) [P]
- **Implementation**: 1 task (T012)
- **Integration**: 2 tasks (T013-T014)
- **Validation**: 5 tasks (T015-T019)

**Estimated Time**:
- Setup: 1 minute (T001)
- Tests: 2 minutes (T002-T011 in parallel)
- Implementation: 5 minutes (T012 - careful editing)
- Integration: 2 minutes (T013-T014)
- Validation: 5 minutes (T015-T019)
- **Total**: ~15 minutes

**Parallelizable Tasks**: 10 (T002-T011)
**Sequential Tasks**: 9 (T001, T012-T019)

---

## Validation Checklist

**GATE: All items must be checked before considering feature complete**

- [x] All contracts have corresponding tests (11 tests for 10 functional requirements)
- [x] All import paths mapped (4 imports × 2 locations = 8 total changes documented)
- [x] All tests come before implementation (T002-T011 before T012)
- [x] Parallel tasks truly independent (T002-T011 test different methods)
- [x] Each task specifies exact file path (all tasks include absolute paths)
- [x] No task modifies same file as another [P] task (T012 is sequential, T002-T011 are read-only tests)
- [x] Setup tasks before implementation (T001 before T012)
- [x] Integration tasks after core implementation (T013-T014 after T012)
- [x] Polish tasks at end (T015-T019 after implementation)

---

## Notes

- **TDD Approach**: Contract tests (T002-T011) written first, should mostly PASS because iris-vector-graph 1.1.1 is correctly structured. T008 will FAIL until T012 is complete.
- **Version Constraint**: T001 ensures pip won't install incompatible versions
- **No Database Required**: This is a pure import path migration, no IRIS database needed for testing
- **Quick Turnaround**: Estimated 15 minutes from start to commit
- **Constitutional Compliance**: Follows TDD (tests first), no silent failures (explicit ImportError), standardized approach
- **Backward Compatible for Users**: Transparent upgrade via pip dependency resolution

---

## Success Criteria

Feature 053 is **COMPLETE** when:

1. ✅ All 19 tasks completed (T001-T019)
2. ✅ All 11 contract tests PASS
3. ✅ `pyproject.toml` shows `iris-vector-graph>=1.1.1`
4. ✅ No `iris_vector_graph_core` references in source code
5. ✅ `GraphCoreDiscovery.import_graph_core_modules()` returns `success=True`
6. ✅ Local installation succeeds with `uv pip install -e .[hybrid-graphrag]`
7. ✅ Changes committed to branch `053-update-to-iris`
8. ✅ Quickstart guide validated (all steps work)

**Next Step After Tasks Complete**: Merge to main, sync to public GitHub, and optionally publish new version to PyPI (if bundled with other changes).
