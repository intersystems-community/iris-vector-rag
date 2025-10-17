# Data Model: Dynamic Pipeline Discovery

**Feature**: Fix RAGAS Make Target Pipeline List
**Phase**: 1 (Design)
**Date**: 2025-10-06

---

## Overview

This feature has minimal data modeling requirements as it's primarily a build system enhancement. The "data" involved is the list of available pipeline types, which is read-only metadata extracted from the iris_rag factory at build time.

---

## Entities

### PipelineType (Read-Only Metadata)

**Description**: A string identifier representing an available RAG pipeline implementation in the iris_rag factory.

**Lifecycle**: Compile-time constant, queried during make execution

**Source of Truth**: `iris_rag/__init__.py` - `_create_pipeline_legacy()` function

#### Attributes

| Attribute | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `name` | string | ✓ | Pipeline type identifier | `"basic"`, `"graphrag"` |
| `is_available` | boolean | ✓ | Always true (if in list) | `true` |

#### Validation Rules

- **name**: Must be a valid Python identifier (alphanumeric + underscore)
- **name**: Must match one of the keys in factory's if/elif chain
- **name**: Must be unique within the list

#### Examples

**Valid Pipeline Types** (as of 2025-10-06):
```python
[
    "basic",           # Basic RAG with vector search
    "basic_rerank",    # Basic RAG + reranking
    "crag",            # Corrective RAG
    "graphrag",        # Graph-based RAG
    "pylate_colbert",  # ColBERT dense retrieval
]
```

**Invalid Examples**:
```python
"Basic"           # ❌ Capitalized (doesn't match factory)
"basic-rag"       # ❌ Hyphenated (not in factory)
"experimental"    # ❌ Not in factory list
""                # ❌ Empty string
```

---

## Data Flow

### Query Path (Build Time)

```
┌─────────────────┐
│   Make Target   │
│ test-ragas-     │
│    sample       │
└────────┬────────┘
         │
         │ $(shell python scripts/utils/get_pipeline_types.py)
         ▼
┌─────────────────┐
│  Helper Script  │
│ get_pipeline_   │
│   types.py      │
└────────┬────────┘
         │
         │ import iris_rag
         ▼
┌─────────────────┐
│  Factory Module │
│ iris_rag/       │
│  __init__.py    │
└────────┬────────┘
         │
         │ Extract available_types from source
         ▼
┌─────────────────┐
│ Pipeline Types  │
│ List[str]       │
└────────┬────────┘
         │
         │ ','.join(types)
         ▼
┌─────────────────┐
│     STDOUT      │
│ "basic,crag,    │
│  graphrag,..."  │
└────────┬────────┘
         │
         │ Captured by Make
         ▼
┌─────────────────┐
│ RAGAS_PIPELINES │
│   env var       │
└─────────────────┘
```

### Data Transformations

1. **Factory Source → AST/Regex Parse**
   - Input: Python source code
   - Output: List of strings
   - Error: ImportError, SyntaxError

2. **List[str] → Comma-Separated String**
   - Input: `["basic", "crag", "graphrag"]`
   - Output: `"basic,crag,graphrag"`
   - Error: EmptyListError (no pipelines)

3. **String → Environment Variable**
   - Input: `"basic,crag,graphrag"`
   - Output: `RAGAS_PIPELINES=basic,crag,graphrag`
   - Error: Shell expansion errors

---

## No Persistent Storage

**Key Point**: This feature does NOT involve:
- Database tables
- File persistence
- State management
- Caching (future enhancement possible)

All data is ephemeral and recalculated on every make invocation.

---

## Factory Interface (Existing)

**Location**: `iris_rag/__init__.py` lines 152-158

**Current Implementation**:
```python
else:
    available_types = [
        "basic",
        "basic_rerank",
        "crag",
        "graphrag",
        "pylate_colbert",
    ]
    raise ValueError(
        f"Unknown pipeline type: {pipeline_type}. Available: {available_types}"
    )
```

**Extraction Strategy**:
The helper script must:
1. Import or read source of `iris_rag.__init__`
2. Locate the `available_types` list definition
3. Parse the list (AST or regex)
4. Return the values

**Alternative (Future Refactor)**:
```python
# Module-level constant (easier to import)
AVAILABLE_PIPELINE_TYPES = [
    "basic",
    "basic_rerank",
    "crag",
    "graphrag",
    "pylate_colbert",
]

def _create_pipeline_legacy(...):
    ...
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}. Available: {AVAILABLE_PIPELINE_TYPES}")
```

Then helper script can just:
```python
from iris_rag import AVAILABLE_PIPELINE_TYPES
print(','.join(AVAILABLE_PIPELINE_TYPES))
```

---

## Error States

### State 1: Factory Not Found
**Condition**: iris_rag package not installed
**Detection**: ImportError during `import iris_rag`
**Handling**: Exit 1 with "Cannot import iris_rag" message

### State 2: Empty List
**Condition**: available_types is empty or missing
**Detection**: Parsed list has length 0
**Handling**: Exit 1 with "No pipeline types available" message

### State 3: Parse Failure
**Condition**: Source code structure changed, can't extract list
**Detection**: AST parse exception or regex no match
**Handling**: Exit 1 with "Cannot extract pipeline types" message

---

## Dependencies

### Compile-Time Dependencies
- **iris_rag**: Must be installed and importable
- **Python stdlib**: `ast`, `inspect`, or `re` for parsing
- **sys**: For exit codes and stderr

### Runtime Dependencies
- None (helper only runs during development)

---

## Constraints

### Immutability
- Pipeline types list is immutable at runtime
- Only changes when factory source is modified
- No user input modifies this data

### Single Source of Truth
- Factory function is the ONLY source of pipeline types
- Helper script is a read-only consumer
- No duplication of the list allowed

### Performance
- Extraction must complete in <100ms
- No caching required (overhead negligible)
- No network or disk I/O (except reading factory source)

---

## Future Enhancements

### Enhancement 1: Caching
**Problem**: Repeated make invocations re-parse factory source
**Solution**: Cache result in `/tmp/pipeline_types.cache` with timestamp
**Benefit**: Faster make execution (50-100ms saved)
**Trade-off**: Added complexity, stale cache risk

### Enhancement 2: Factory Refactor
**Problem**: Parsing source is fragile
**Solution**: Expose `AVAILABLE_PIPELINE_TYPES` as module constant
**Benefit**: Simpler, more reliable extraction
**Timeline**: Separate feature after this one ships

### Enhancement 3: Metadata Enrichment
**Problem**: Just names, no descriptions
**Solution**: Return JSON with `{name, description, status}` per pipeline
**Benefit**: Could generate pipeline selection UI
**Trade-off**: Make integration becomes more complex

---

## Testing Data

### Test Fixture: Expected Pipeline Types
```python
EXPECTED_PIPELINES_2025_10_06 = [
    "basic",
    "basic_rerank",
    "crag",
    "graphrag",
    "pylate_colbert",
]
```

**Usage in Tests**:
```python
def test_helper_output_matches_expected():
    result = run_helper_script()
    actual = result.stdout.strip().split(',')
    assert set(actual) == set(EXPECTED_PIPELINES_2025_10_06)
```

**Maintenance**: Update this list when factory adds/removes pipelines

---

## Data Validation

### Input Validation (None)
- Helper script takes no input arguments
- All data comes from factory source

### Output Validation
1. **Format**: Must match regex `^[a-z_,]+$` (lowercase letters, underscores, commas)
2. **Count**: Must have at least 1 pipeline
3. **Uniqueness**: No duplicate pipeline names
4. **Correctness**: Must match factory source exactly

### Example Validation Code
```python
def validate_output(output: str) -> bool:
    """Validate helper script output format."""
    if not output or ',' not in output:
        return False

    pipelines = output.split(',')
    if len(pipelines) == 0:
        return False

    # Check for duplicates
    if len(pipelines) != len(set(pipelines)):
        return False

    # Check format (lowercase + underscores only)
    for name in pipelines:
        if not re.match(r'^[a-z_]+$', name):
            return False

    return True
```

---

## Summary

This feature's data model is minimal by design:
- **Read-only** list of string identifiers
- **No persistence** (runtime query only)
- **Single source of truth** (factory function)
- **Simple transformations** (list → comma-separated string)

The simplicity is intentional - this is a build tool enhancement, not a data management feature.

---

**Data Model Status**: ✓ Complete
**Complexity**: Minimal (1 entity, 2 attributes, no persistence)
**Dependencies**: iris_rag factory source only
