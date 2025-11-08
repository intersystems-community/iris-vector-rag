# Research: Update iris-vector-rag to use iris-vector-graph 1.1.1

**Feature**: 053-update-to-iris
**Date**: 2025-11-08
**Status**: Complete

## Overview
This research documents the import path migration from `iris_vector_graph_core` to `iris_vector_graph` required to support iris-vector-graph version 1.1.1.

## Decision: Use iris_vector_graph Top-Level Imports

**What was chosen**: Update all imports to use `from iris_vector_graph import ...` instead of `from iris_vector_graph_core.* import ...`

**Why chosen**:
1. **Package Restructuring**: iris-vector-graph 1.1.1 eliminated the `iris_vector_graph_core` submodule in favor of direct top-level exports
2. **Simplified API**: Users can now import directly from `iris_vector_graph` without navigating submodules
3. **Version Compatibility**: Older versions (< 1.1.1) used `iris_vector_graph_core`, newer versions use `iris_vector_graph`
4. **Breaking Change**: iris-vector-graph 1.1.1 no longer provides `iris_vector_graph_core` module at all

**Alternatives considered**:
1. **Dual import path support** (try new, fallback to old):
   - Rejected: Adds complexity and masks version incompatibility
   - Rejected: Would hide the actual package version being used
   - Rejected: Creates confusion about which module structure is canonical

2. **Keep old imports and pin to iris-vector-graph < 1.1.1**:
   - Rejected: Prevents users from benefiting from iris-vector-graph 1.1.1+ improvements
   - Rejected: Creates version conflict if users want newer iris-vector-graph

3. **Create compatibility shim module**:
   - Rejected: Unnecessary complexity for a straightforward import path change
   - Rejected: Would require maintenance and testing of compatibility layer

## Module Structure Analysis

### iris-vector-graph 1.1.1 Structure (New)
```python
# Top-level exports from iris_vector_graph
from iris_vector_graph import IRISGraphEngine
from iris_vector_graph import HybridSearchFusion
from iris_vector_graph import TextSearchEngine
from iris_vector_graph import VectorOptimizer
```

**Confirmation**: User diagnosis confirmed this structure:
> "Perfect! iris_vector_graph 1.1.1 is correctly structured:
> ✅ IRISGraphEngine is directly available from iris_vector_graph
> ❌ No iris_vector_graph_core module"

### iris-vector-graph < 1.1.1 Structure (Old)
```python
# Submodule exports from iris_vector_graph_core
from iris_vector_graph_core.engine import IRISGraphEngine
from iris_vector_graph_core.fusion import HybridSearchFusion
from iris_vector_graph_core.text_search import TextSearchEngine
from iris_vector_graph_core.vector_utils import VectorOptimizer
```

## Implementation Impact Analysis

### Files Requiring Updates

1. **iris_rag/pipelines/hybrid_graphrag_discovery.py** (PRIMARY):
   - Lines 127-130: Installed package imports (try block)
   - Lines 166-169: Local path imports (fallback, also in try block)
   - Lines 82-86: Path validation logic referencing `iris_vector_graph_core`
   - **Impact**: Core HybridGraphRAG discovery mechanism

2. **pyproject.toml**:
   - Line 76: `embedding` optional dependencies
   - Line 79: `hybrid-graphrag` optional dependencies
   - **Current**: `iris-vector-graph>=2.0.0`
   - **Updated**: `iris-vector-graph>=1.1.1`
   - **Impact**: Dependency resolution for package installation

3. **tests/contract/test_import_requirements.py**:
   - Contract tests validating import paths
   - **Impact**: Test validation of new import structure

4. **tests/unit/test_hybrid_graphrag.py**:
   - Unit tests potentially referencing old import paths
   - **Impact**: Test compatibility with new imports

### Migration Strategy

**Phase 1: Update Imports**
1. Replace `from iris_vector_graph_core.engine import IRISGraphEngine`
   - With: `from iris_vector_graph import IRISGraphEngine`
2. Replace `from iris_vector_graph_core.fusion import HybridSearchFusion`
   - With: `from iris_vector_graph import HybridSearchFusion`
3. Replace `from iris_vector_graph_core.text_search import TextSearchEngine`
   - With: `from iris_vector_graph import TextSearchEngine`
4. Replace `from iris_vector_graph_core.vector_utils import VectorOptimizer`
   - With: `from iris_vector_graph import VectorOptimizer`

**Phase 2: Update Path Validation**
Remove references to `iris_vector_graph_core` in path validation:
```python
# OLD (lines 82-86):
new_package = path / "iris_vector_graph_core"
legacy_package = path / "iris_graph_core"

# NEW:
# Only check for iris_vector_graph (top-level package marker)
package_dir = path / "iris_vector_graph"
```

**Phase 3: Update Dependency Constraint**
```toml
# OLD:
iris-vector-graph>=2.0.0

# NEW:
iris-vector-graph>=1.1.1
```

**Rationale**: Version 1.1.1 is the first version with the new top-level import structure

## Error Handling Strategy

### Import Failure Scenarios

1. **iris-vector-graph not installed**:
   - Current behavior: Falls back to local path discovery
   - New behavior: Same (no change)
   - Error message: Already clear ("install with: pip install rag-templates[hybrid-graphrag]")

2. **iris-vector-graph < 1.1.1 installed**:
   - Current behavior: Works (uses iris_vector_graph_core)
   - New behavior: ImportError (iris_vector_graph_core doesn't exist)
   - **Solution**: Dependency constraint `>= 1.1.1` prevents this scenario

3. **iris-vector-graph >= 1.1.1 with old iris-vector-rag**:
   - Current behavior: ImportError (iris_vector_graph_core doesn't exist)
   - New behavior: N/A (this update fixes the issue)
   - **This is the bug we're fixing**

### Version Validation

**Recommended approach**: Rely on pip dependency resolution
- `pyproject.toml` specifies `iris-vector-graph>=1.1.1`
- pip enforces version constraint at install time
- No runtime version checking needed (pip already validated)

**Alternative rejected**: Runtime version checking
- Would add complexity
- pip dependency resolution already handles this
- Would create redundant validation logic

## Testing Strategy

### Contract Tests (TDD Approach)

Create `specs/053-update-to-iris/contracts/test_import_iris_vector_graph.py`:

```python
"""Contract tests for iris-vector-graph 1.1.1 import compatibility."""

def test_import_iris_graph_engine():
    """FR-001: IRISGraphEngine must import from iris_vector_graph."""
    from iris_vector_graph import IRISGraphEngine
    assert IRISGraphEngine is not None

def test_import_hybrid_search_fusion():
    """FR-002: HybridSearchFusion must import from iris_vector_graph."""
    from iris_vector_graph import HybridSearchFusion
    assert HybridSearchFusion is not None

def test_import_text_search_engine():
    """FR-003: TextSearchEngine must import from iris_vector_graph."""
    from iris_vector_graph import TextSearchEngine
    assert TextSearchEngine is not None

def test_import_vector_optimizer():
    """FR-004: VectorOptimizer must import from iris_vector_graph."""
    from iris_vector_graph import VectorOptimizer
    assert VectorOptimizer is not None

def test_old_import_path_fails():
    """FR-006: Old iris_vector_graph_core imports must fail."""
    import pytest
    with pytest.raises(ImportError):
        from iris_vector_graph_core.engine import IRISGraphEngine
```

### Integration Tests

Validate HybridGraphRAG pipeline works with new imports:
1. Install iris-vector-graph 1.1.1
2. Import HybridGraphRAGPipeline
3. Execute query
4. Verify uses new iris_vector_graph modules

## Risk Analysis

### Low Risk
- ✅ Pure import path change (no logic changes)
- ✅ Dependency constraint enforces compatible version
- ✅ Existing error handling covers import failures
- ✅ No database schema changes
- ✅ No API changes

### Medium Risk
- ⚠️ Users with pinned iris-vector-graph < 1.1.1 will need to update
  - Mitigation: Clear error message, documentation update
- ⚠️ Local development setups may need iris-vector-graph reinstall
  - Mitigation: Documented in quickstart.md

### No Risk
- N/A Performance (imports are cached after first load)
- N/A Data integrity (no data changes)
- N/A Security (same package, different import path)

## Performance Considerations

**Import Performance**: No measurable impact
- Python caches imports after first load
- Import path length difference negligible
- No additional import overhead

**Runtime Performance**: Zero impact
- Same underlying classes
- Same method calls
- No compatibility shims or wrappers

## Backward Compatibility

**Breaking Change**: Yes, but managed via dependency constraint

**Migration Path for Users**:
```bash
# Option 1: Upgrade iris-vector-rag (recommended)
pip install --upgrade iris-vector-rag

# Option 2: Explicitly upgrade iris-vector-graph
pip install --upgrade iris-vector-graph>=1.1.1

# Option 3: Fresh install with extras
pip install iris-vector-rag[hybrid-graphrag]
```

**Version Compatibility Matrix**:
| iris-vector-rag | iris-vector-graph | Compatible? |
|----------------|-------------------|-------------|
| 0.2.4 (old)    | < 1.1.1           | ✅ Yes      |
| 0.2.4 (old)    | >= 1.1.1          | ❌ No (this bug) |
| 0.2.5 (new)    | < 1.1.1           | ❌ No (constraint blocks) |
| 0.2.5 (new)    | >= 1.1.1          | ✅ Yes      |

## Documentation Updates Required

1. **CLAUDE.md**: Update HybridGraphRAG dependency section
   - Change iris-vector-graph version requirement to >= 1.1.1
   - Note about top-level import structure

2. **README.md**: Update installation instructions
   - Verify extras installation still works
   - Update any import examples

3. **Quickstart**: Add validation steps
   - Verify iris-vector-graph version
   - Test HybridGraphRAG import

## Conclusion

This is a straightforward import path migration with:
- ✅ Clear technical direction (use top-level iris_vector_graph imports)
- ✅ Low risk (pure import change, no logic changes)
- ✅ Strong compatibility guarantees (dependency constraint)
- ✅ Comprehensive test coverage plan (contract + integration tests)
- ✅ Clear error handling (existing + dependency resolution)

**Ready for Phase 1: Design & Contracts**
