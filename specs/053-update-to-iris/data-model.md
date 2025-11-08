# Data Model: Update iris-vector-rag to use iris-vector-graph 1.1.1

**Feature**: 053-update-to-iris
**Date**: 2025-11-08

## Overview
This feature does not introduce new data entities or modify database schemas. It is a pure import path refactoring that changes Python module references from `iris_vector_graph_core.*` to `iris_vector_graph`.

## Import Path Mapping

### Entity: ImportPath
**Purpose**: Conceptual model showing old vs. new import paths

| Old Import Path | New Import Path | Module Purpose |
|----------------|-----------------|----------------|
| `iris_vector_graph_core.engine` | `iris_vector_graph` | IRISGraphEngine - Core graph database engine |
| `iris_vector_graph_core.fusion` | `iris_vector_graph` | HybridSearchFusion - Search result fusion algorithm |
| `iris_vector_graph_core.text_search` | `iris_vector_graph` | TextSearchEngine - Full-text search capabilities |
| `iris_vector_graph_core.vector_utils` | `iris_vector_graph` | VectorOptimizer - Vector operation optimizations |

### Fields
- **old_module_path**: String - Deprecated module path using `iris_vector_graph_core` submodule
- **new_module_path**: String - Current module path using `iris_vector_graph` top-level exports
- **class_name**: String - Class being imported (e.g., `IRISGraphEngine`)
- **import_location**: String - File path where import statement exists

### Relationships
None - This is a mapping table, not a persistent data entity.

## File-Level Changes

### Primary Source File
**File**: `iris_rag/pipelines/hybrid_graphrag_discovery.py`

**Import Locations**:

1. **Installed Package Imports** (lines 127-130):
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

2. **Local Path Fallback Imports** (lines 166-169):
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

### Path Validation Logic
**File**: `iris_rag/pipelines/hybrid_graphrag_discovery.py` (lines 82-86)

```python
# OLD:
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

**Rationale**:
- Remove reference to `iris_vector_graph_core` submodule (no longer exists in 1.1.1)
- Validate based on `iris_vector_graph` package presence
- Simplify logic (no legacy fallback needed)

### Dependency Configuration
**File**: `pyproject.toml`

```toml
# OLD (lines 76, 79):
embedding = [
    "torch>=2.0.0",
    "sentence-transformers>=2.2.0",
    "iris-vector-graph>=2.0.0"
]
hybrid-graphrag = [
    "iris-vector-graph>=2.0.0"
]

# NEW:
embedding = [
    "torch>=2.0.0",
    "sentence-transformers>=2.2.0",
    "iris-vector-graph>=1.1.1"
]
hybrid-graphrag = [
    "iris-vector-graph>=1.1.1"
]
```

**Rationale**:
- Version 1.1.1 is first version with top-level `iris_vector_graph` exports
- Constraint prevents incompatible versions from being installed
- Both optional dependency groups updated for consistency

## Validation Rules

### Contract Test Validations

| Rule ID | Validation | Error Condition | Expected Behavior |
|---------|-----------|-----------------|-------------------|
| FR-001 | `from iris_vector_graph import IRISGraphEngine` | ImportError | Test fails, implementation incomplete |
| FR-002 | `from iris_vector_graph import HybridSearchFusion` | ImportError | Test fails, implementation incomplete |
| FR-003 | `from iris_vector_graph import TextSearchEngine` | ImportError | Test fails, implementation incomplete |
| FR-004 | `from iris_vector_graph import VectorOptimizer` | ImportError | Test fails, implementation incomplete |
| FR-006 | `from iris_vector_graph_core.*` raises ImportError | Import succeeds | Test fails, old import still works |
| FR-009 | iris-vector-graph version >= 1.1.1 | Version < 1.1.1 | Test fails, dependency not enforced |

### Runtime Behavior

**Import Success Path**:
```python
try:
    from iris_vector_graph import IRISGraphEngine, HybridSearchFusion, TextSearchEngine, VectorOptimizer
    # SUCCESS: All modules imported, HybridGraphRAG available
except ImportError as e:
    # FAILURE: iris-vector-graph not installed or incompatible version
    logger.warning("HybridGraphRAG requires iris-vector-graph >= 1.1.1")
    logger.info("Install with: pip install rag-templates[hybrid-graphrag]")
```

**Import Failure Scenarios**:
1. **Package not installed**: ImportError with clear installation instructions
2. **Version < 1.1.1**: Blocked by pip dependency resolution (won't install)
3. **Corrupted installation**: ImportError with troubleshooting guidance

## State Transitions

### Build-Time (pip install)
```
1. User runs: pip install rag-templates[hybrid-graphrag]
2. pip reads: pyproject.toml dependencies
3. pip resolves: iris-vector-graph>=1.1.1 constraint
4. pip installs: iris-vector-graph 1.1.1 (or later)
5. Package ready: iris_vector_graph module available
```

### Runtime (import)
```
1. Code executes: from iris_vector_graph import IRISGraphEngine
2. Python checks: sys.modules cache
3. If not cached: Load iris_vector_graph/__init__.py
4. Module returns: IRISGraphEngine class
5. Import complete: Class available for instantiation
```

### Error State
```
1. Code executes: from iris_vector_graph import IRISGraphEngine
2. Python checks: sys.modules cache (not found)
3. Python searches: sys.path for iris_vector_graph
4. Not found: Raises ImportError
5. Exception caught: GraphCoreDiscovery logs helpful message
6. Fallback: HybridGraphRAG features disabled, basic RAG still works
```

## No Database Schema Changes

This feature does NOT modify:
- ✅ IRIS database tables
- ✅ Vector index structures
- ✅ Entity/relationship schemas
- ✅ SQL stored procedures
- ✅ Database connections
- ✅ Transaction handling

**Rationale**: This is purely a Python import path change with no database interaction.

## No API Changes

This feature does NOT modify:
- ✅ Public API methods
- ✅ Function signatures
- ✅ Response formats
- ✅ REST endpoints
- ✅ MCP tools
- ✅ Pipeline factory interface

**Rationale**: Import paths are internal implementation details, not exposed in public APIs.

## Migration Path

### For End Users
**No migration required** - transparent upgrade:
```bash
# Standard upgrade path
pip install --upgrade iris-vector-rag

# Or fresh install
pip install iris-vector-rag[hybrid-graphrag]
```

### For Developers
**If using local iris-vector-graph checkout**:
```bash
# 1. Update iris-vector-graph to 1.1.1+
cd /path/to/iris-vector-graph
git pull
pip install -e .

# 2. Update iris-vector-rag
cd /path/to/rag-templates
pip install -e .[hybrid-graphrag]

# 3. Verify imports work
python -c "from iris_vector_graph import IRISGraphEngine; print('✅ Import successful')"
```

## Testing Data Requirements

**No test data required** - contract tests validate imports only:
- ✅ No database fixtures needed
- ✅ No sample documents needed
- ✅ No entity/relationship data needed
- ✅ Import-only validation sufficient

**Test Execution**:
```bash
# Contract tests (no database)
pytest specs/053-update-to-iris/contracts/ -v

# Integration tests (require iris-vector-graph 1.1.1 installed)
pytest tests/integration/test_hybridgraphrag_*.py -v
```

## Summary

This data model describes the structural changes required for Feature 053:
- **Import Path Mapping**: 4 imports updated from submodule to top-level
- **File Changes**: 1 primary file + 1 config file + test files
- **Validation Rules**: 6 contract test rules enforcing new imports
- **No Data Changes**: Zero database schema modifications
- **No API Changes**: Zero public interface modifications
- **Migration**: Transparent for end users, documented for developers
