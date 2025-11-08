# Quickstart: Update iris-vector-rag to use iris-vector-graph 1.1.1

**Feature**: 053-update-to-iris
**Date**: 2025-11-08
**Estimated Time**: 5-10 minutes

## Prerequisites

- ✅ Git repository: `rag-templates` on branch `053-update-to-iris`
- ✅ Python 3.10+ with uv package manager
- ✅ Virtual environment: `.venv` created
- ✅ iris-vector-graph 1.1.1 available (will be installed via dependency update)

## Step 1: Verify Current State (30 seconds)

**Validate that old imports currently fail with iris-vector-graph 1.1.1:**

```bash
# Activate virtual environment
source .venv/bin/activate

# Check current iris-vector-graph version (if installed)
python -c "from importlib.metadata import version; print(f'iris-vector-graph: {version(\"iris-vector-graph\")}')" 2>/dev/null || echo "Not installed"

# Try old import path (should fail if iris-vector-graph >= 1.1.1)
python -c "from iris_vector_graph_core.engine import IRISGraphEngine" 2>&1 | grep -q "ModuleNotFoundError" && echo "✅ Old import fails (expected)" || echo "⚠️ Old import works (unexpected)"

# Try new import path (should succeed)
python -c "from iris_vector_graph import IRISGraphEngine; print('✅ New import works')" 2>/dev/null || echo "❌ New import fails (iris-vector-graph not installed or < 1.1.1)"
```

**Expected Output**:
```
iris-vector-graph: 1.1.1 (or higher)
✅ Old import fails (expected)
✅ New import works
```

## Step 2: Run Contract Tests (BEFORE Implementation) (1 minute)

**TDD Principle**: Run tests FIRST - they should fail before implementation.

```bash
# Run contract tests (should FAIL initially)
pytest specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py -v

# Expected: FAILURES because hybrid_graphrag_discovery.py still uses old imports
```

**Expected Failures**:
- ❌ `test_graph_core_discovery_imports_successfully` - hybrid_graphrag_discovery uses old imports
- ✅ `test_import_iris_graph_engine` - direct import works (iris-vector-graph 1.1.1 installed)
- ✅ `test_import_hybrid_search_fusion` - direct import works
- ✅ `test_import_text_search_engine` - direct import works
- ✅ `test_import_vector_optimizer` - direct import works
- ✅ `test_old_iris_vector_graph_core_import_fails` - old module doesn't exist
- ✅ `test_iris_vector_graph_version_requirement` - version constraint met

## Step 3: Update Dependency Constraint (30 seconds)

**Edit `pyproject.toml`:**

```bash
# Open in editor
code pyproject.toml  # or vim, nano, etc.

# Find lines 76 and 79, update from:
#   iris-vector-graph>=2.0.0
# To:
#   iris-vector-graph>=1.1.1

# Or use sed:
sed -i '' 's/iris-vector-graph>=2\.0\.0/iris-vector-graph>=1.1.1/g' pyproject.toml
```

**Verify change:**
```bash
grep "iris-vector-graph" pyproject.toml
# Should show: iris-vector-graph>=1.1.1 (twice)
```

## Step 4: Update Import Statements in hybrid_graphrag_discovery.py (2 minutes)

**Primary changes needed:**

```bash
# Open the file
code iris_rag/pipelines/hybrid_graphrag_discovery.py
```

**Change #1: Installed package imports (lines 127-130)**

Replace:
```python
from iris_vector_graph_core.engine import IRISGraphEngine
from iris_vector_graph_core.fusion import HybridSearchFusion
from iris_vector_graph_core.text_search import TextSearchEngine
from iris_vector_graph_core.vector_utils import VectorOptimizer
```

With:
```python
from iris_vector_graph import IRISGraphEngine
from iris_vector_graph import HybridSearchFusion
from iris_vector_graph import TextSearchEngine
from iris_vector_graph import VectorOptimizer
```

**Change #2: Local path fallback imports (lines 166-169)**

Replace:
```python
from iris_vector_graph_core.engine import IRISGraphEngine
from iris_vector_graph_core.fusion import HybridSearchFusion
from iris_vector_graph_core.text_search import TextSearchEngine
from iris_vector_graph_core.vector_utils import VectorOptimizer
```

With:
```python
from iris_vector_graph import IRISGraphEngine
from iris_vector_graph import HybridSearchFusion
from iris_vector_graph import TextSearchEngine
from iris_vector_graph import VectorOptimizer
```

**Change #3: Path validation logic (lines 82-86)**

Replace:
```python
# Check for either new or legacy package structure
new_package = path / "iris_vector_graph_core"
legacy_package = path / "iris_graph_core"

return (new_package.exists() and new_package.is_dir()) or (
    legacy_package.exists() and legacy_package.is_dir()
)
```

With:
```python
# Check for iris_vector_graph package directory (top-level module)
package_dir = path / "iris_vector_graph"
return package_dir.exists() and package_dir.is_dir()
```

**Quick edit with sed (alternative to manual editing):**
```bash
# Backup original file
cp iris_rag/pipelines/hybrid_graphrag_discovery.py iris_rag/pipelines/hybrid_graphrag_discovery.py.bak

# Replace old imports with new imports
sed -i '' 's/from iris_vector_graph_core\.engine import/from iris_vector_graph import/g' iris_rag/pipelines/hybrid_graphrag_discovery.py
sed -i '' 's/from iris_vector_graph_core\.fusion import/from iris_vector_graph import/g' iris_rag/pipelines/hybrid_graphrag_discovery.py
sed -i '' 's/from iris_vector_graph_core\.text_search import/from iris_vector_graph import/g' iris_rag/pipelines/hybrid_graphrag_discovery.py
sed -i '' 's/from iris_vector_graph_core\.vector_utils import/from iris_vector_graph import/g' iris_rag/pipelines/hybrid_graphrag_discovery.py
```

## Step 5: Update Test Files (1 minute)

**Update unit tests that reference old imports:**

```bash
# Check for old import references in tests
grep -n "iris_vector_graph_core" tests/unit/test_hybrid_graphrag.py tests/contract/test_import_requirements.py

# If found, update them to use iris_vector_graph instead
# Example:
sed -i '' 's/iris_vector_graph_core/iris_vector_graph/g' tests/unit/test_hybrid_graphrag.py
sed -i '' 's/iris_vector_graph_core/iris_vector_graph/g' tests/contract/test_import_requirements.py
```

## Step 6: Run Contract Tests (AFTER Implementation) (1 minute)

**Validate that tests now PASS:**

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

## Step 7: Run Existing Integration Tests (2 minutes)

**Validate that existing HybridGraphRAG tests still pass:**

```bash
# Run HybridGraphRAG integration tests
pytest tests/integration/test_hybridgraphrag_*.py -v -k "not e2e"

# If skipped with "requires iris-vector-graph setup", that's expected
# The contract tests already validated the import paths work
```

## Step 8: Test Local Installation (1 minute)

**Install updated package locally and verify imports:**

```bash
# Reinstall package in editable mode with updated dependencies
uv pip install -e .[hybrid-graphrag]

# Verify new imports work
python -c "
from iris_vector_graph import IRISGraphEngine, HybridSearchFusion, TextSearchEngine, VectorOptimizer
print('✅ All imports successful')
print(f'  - IRISGraphEngine: {IRISGraphEngine.__name__}')
print(f'  - HybridSearchFusion: {HybridSearchFusion.__name__}')
print(f'  - TextSearchEngine: {TextSearchEngine.__name__}')
print(f'  - VectorOptimizer: {VectorOptimizer.__name__}')
"

# Verify HybridGraphRAG pipeline creation works
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

## Step 9: Commit Changes (1 minute)

```bash
# Stage changes
git add pyproject.toml
git add iris_rag/pipelines/hybrid_graphrag_discovery.py
git add tests/unit/test_hybrid_graphrag.py
git add tests/contract/test_import_requirements.py
git add specs/053-update-to-iris/

# Commit with clear message
git commit -m "feat: update to iris-vector-graph 1.1.1 top-level imports

Update import statements from iris_vector_graph_core to iris_vector_graph
to support iris-vector-graph 1.1.1+ which removed the _core submodule.

Changes:
- Update pyproject.toml dependency: iris-vector-graph>=1.1.1
- Update hybrid_graphrag_discovery.py imports (lines 127-130, 166-169)
- Update path validation logic to check for iris_vector_graph package
- Update test files to use new import paths
- Add contract tests for import validation

All contract tests pass (11/11).
Existing HybridGraphRAG functionality preserved."
```

## Validation Checklist

After completing the quickstart, verify:

- [x] **Dependency Updated**: `grep "iris-vector-graph>=1.1.1" pyproject.toml` shows 2 matches
- [x] **Imports Updated**: `grep "from iris_vector_graph import" iris_rag/pipelines/hybrid_graphrag_discovery.py` shows 8 matches
- [x] **Old Imports Removed**: `grep "iris_vector_graph_core" iris_rag/pipelines/hybrid_graphrag_discovery.py` shows 0 matches
- [x] **Contract Tests Pass**: All 11 tests in `test_import_iris_vector_graph.py` pass
- [x] **Direct Imports Work**: `python -c "from iris_vector_graph import IRISGraphEngine"` succeeds
- [x] **Discovery Works**: `GraphCoreDiscovery.import_graph_core_modules()` returns success=True
- [x] **Changes Committed**: Git status shows clean working directory

## Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'iris_vector_graph'"

**Solution**: Install iris-vector-graph 1.1.1+
```bash
uv pip install 'iris-vector-graph>=1.1.1'
```

### Problem: "ModuleNotFoundError: No module named 'iris_vector_graph_core'"

**Solution**: This is expected! iris-vector-graph 1.1.1 removed iris_vector_graph_core.
Ensure you've completed Step 4 (update import statements).

### Problem: Contract tests still failing after implementation

**Solution**: Verify all import statements were updated:
```bash
# Should show 0 results:
grep -r "from iris_vector_graph_core" iris_rag/pipelines/

# Should show new imports:
grep -r "from iris_vector_graph import" iris_rag/pipelines/
```

### Problem: Integration tests skip with "requires iris-vector-graph setup"

**Solution**: This is expected behavior. Contract tests validate the imports work.
Full integration tests require database + LLM setup which is beyond scope of this feature.

## Success Criteria

✅ Feature 053 is complete when:

1. All 11 contract tests pass
2. `pyproject.toml` shows `iris-vector-graph>=1.1.1`
3. No `iris_vector_graph_core` references in `hybrid_graphrag_discovery.py`
4. Direct imports work: `from iris_vector_graph import IRISGraphEngine`
5. `GraphCoreDiscovery.import_graph_core_modules()` returns `success=True`
6. Changes committed to branch `053-update-to-iris`

**Estimated Total Time**: 8 minutes actual + 2 minutes for validation = **10 minutes**

**Next Steps**: After quickstart validation passes, proceed to `/tasks` command to generate the complete task list for implementation tracking.
