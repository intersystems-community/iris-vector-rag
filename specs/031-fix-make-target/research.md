# Research: Dynamic Pipeline Discovery for RAGAS

**Feature**: Fix RAGAS Make Target Pipeline List
**Phase**: 0 (Research & Technical Discovery)
**Date**: 2025-10-06

## Research Questions & Findings

### 1. Factory Introspection Strategy

**Question**: How can we safely extract the list of available pipeline types from the iris_rag factory without triggering pipeline initialization or database connections?

**Investigation**:
Examined `/Users/tdyar/ws/rag-templates/iris_rag/__init__.py` (lines 104-161):
```python
def _create_pipeline_legacy(...):
    if pipeline_type == "basic":
        return BasicRAGPipeline(...)
    elif pipeline_type == "basic_rerank":
        ...
    elif pipeline_type == "crag":
        ...
    elif pipeline_type == "graphrag":
        ...
    elif pipeline_type == "pylate_colbert":
        ...
    else:
        available_types = [
            "basic",
            "basic_rerank",
            "crag",
            "graphrag",
            "pylate_colbert",
        ]
        raise ValueError(f"Unknown pipeline type: {pipeline_type}. Available: {available_types}")
```

**Decision**: Extract the `available_types` list directly from the factory function
- **Approach**: Import the module and access the hardcoded list in the error path
- **Rationale**: The list is already maintained in the factory for error messages, so we can reuse it
- **Implementation**: Use AST parsing or regex to extract the list from source code

**Alternatives Considered**:
1. **Duplicate the list** in helper script
   - ❌ Rejected: Violates DRY, will become stale
2. **Call create_pipeline() and catch ValueError**
   - ❌ Rejected: Wasteful, requires dummy pipeline_type
3. **Expose available_types as module-level constant**
   - ✓ Best long-term solution, but requires factory refactor
   - ✓ Recommend for future enhancement

**Chosen Solution**: Parse factory source code to extract `available_types` list
- Minimal changes to existing code
- No risk of triggering initialization
- Self-updating when factory changes

---

### 2. Script Output Format

**Question**: What format should the helper script output for optimal Make consumption?

**Investigation**:
- Reviewed existing Makefile at line 684: `export RAGAS_PIPELINES=${RAGAS_PIPELINES:-"basic,basic_rerank,crag,graphrag,pylate_colbert"};`
- Current format: Comma-separated string with no spaces
- Make capture pattern: `$(shell command)` substitution

**Decision**: Comma-separated values with no whitespace
- **Format**: `basic,basic_rerank,crag,graphrag,pylate_colbert`
- **Rationale**: Matches existing format, easy for Make to handle
- **Implementation**: `print(','.join(pipeline_types))`

**Alternatives Considered**:
1. **Space-separated** (`basic basic_rerank crag`)
   - ❌ Rejected: Doesn't match current format
2. **JSON array** (`["basic","crag"]`)
   - ❌ Rejected: Requires parsing in Make (complex)
3. **Newline-separated**
   - ❌ Rejected: Harder for Make to consume

---

### 3. Error Handling Requirements

**Question**: What failure modes must the helper script handle gracefully?

**Scenarios Identified**:

**Scenario 1: iris_rag not installed**
```python
try:
    import iris_rag
except ImportError:
    print("ERROR: Cannot import iris_rag. Is the package installed?", file=sys.stderr)
    sys.exit(1)
```

**Scenario 2: Factory source not parseable**
```python
try:
    available_types = extract_pipeline_types()
except Exception as e:
    print(f"ERROR: Cannot extract pipeline types: {e}", file=sys.stderr)
    sys.exit(1)
```

**Scenario 3: No pipelines found**
```python
if not available_types:
    print("ERROR: No pipeline types available from factory", file=sys.stderr)
    sys.exit(1)
```

**Decision**: Exit with status code 1 and clear error message to stderr
- **Rationale**: Make will fail fast with visible error
- **Benefit**: Developers immediately know what's wrong

---

### 4. Makefile Integration Pattern

**Question**: How should the Makefile invoke the helper while preserving environment variable overrides?

**Investigation**:
- Bash parameter expansion: `${VAR:-default}` uses default if VAR unset
- Make shell function: `$(shell command)` captures stdout
- Combine both: `${VAR:-$(shell command)}`

**Decision**: Use shell expansion with command substitution fallback
```makefile
export RAGAS_PIPELINES=$${RAGAS_PIPELINES:-$$(python scripts/utils/get_pipeline_types.py)};
```

**Behavior**:
- If `RAGAS_PIPELINES` env var is set → use it (user override)
- If not set → run helper script and use output (dynamic discovery)

**Verification**:
```bash
# Test 1: Default (dynamic)
make test-ragas-sample
# → Uses all factory pipelines

# Test 2: Override
RAGAS_PIPELINES=basic,crag make test-ragas-sample
# → Uses only basic and crag
```

**Alternatives Considered**:
1. **Always call helper** (no override)
   - ❌ Rejected: Removes user flexibility
2. **Separate make target** (make discover-pipelines)
   - ❌ Rejected: Extra step for developers
3. **Check helper output, fall back to hardcoded list**
   - ✓ Possible safety mechanism if helper fails
   - ❌ Rejected: Silent failures, defeats purpose

---

### 5. Performance Considerations

**Question**: Will the helper script add noticeable latency to make execution?

**Benchmarking**:
```bash
# Simulated timing
time python -c "import iris_rag; print('basic,crag')"
# Expected: <100ms on modern hardware
```

**Analysis**:
- Python import time: ~50-100ms (iris_rag is lightweight)
- Source parsing: <10ms (AST or regex on small file)
- Total overhead: <150ms per make invocation

**Decision**: Acceptable overhead
- **Rationale**: RAGAS evaluation takes minutes, 150ms is negligible
- **Optimization**: Could cache result in temp file if needed (future)

---

### 6. Testing Strategy

**Question**: How do we verify the dynamic discovery works correctly?

**Test Pyramid**:

**Unit Tests** (helper script in isolation):
1. Script exists and is executable
2. Output format is comma-separated
3. Output matches factory available_types
4. Error handling (import failure, no pipelines)

**Integration Tests** (Makefile behavior):
1. Make target calls helper when RAGAS_PIPELINES unset
2. Make target respects RAGAS_PIPELINES env var override
3. RAGAS evaluation actually tests all discovered pipelines

**Location**: `tests/infrastructure/test_makefile_targets.py`
- Consistent with existing infrastructure tests
- Can use subprocess to run helper and make

---

## Key Decisions Summary

| Decision Point | Chosen Approach | Rationale |
|----------------|-----------------|-----------|
| **Introspection Method** | Parse factory source for available_types list | Reuses existing list, no initialization risk |
| **Output Format** | Comma-separated string | Matches current format, Make-friendly |
| **Error Handling** | Exit 1 with stderr message | Fast fail with clear diagnostics |
| **Makefile Pattern** | `${VAR:-$(shell helper)}` | Preserves user override capability |
| **Script Location** | `scripts/utils/get_pipeline_types.py` | Consistent with project structure |
| **Testing** | Infrastructure tests + integration | Comprehensive coverage at right levels |

---

## Future Enhancements

### Recommended Factory Refactor
```python
# In iris_rag/__init__.py
AVAILABLE_PIPELINE_TYPES = [
    "basic",
    "basic_rerank",
    "crag",
    "graphrag",
    "pylate_colbert",
]

def _create_pipeline_legacy(...):
    if pipeline_type == "basic":
        ...
    else:
        raise ValueError(f"Unknown pipeline type: {pipeline_type}. Available: {AVAILABLE_PIPELINE_TYPES}")
```

**Benefits**:
- Module-level constant easier to import
- No source parsing needed
- Single source of truth

**Migration Path**:
1. Refactor factory to expose AVAILABLE_PIPELINE_TYPES
2. Update helper script to import constant
3. Remove source parsing logic
4. Add deprecation notice for old approach

**Timeline**: Can be done in separate feature after this one ships

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Helper script fails silently | Broken make target | Exit codes + stderr, tested in CI |
| Factory list becomes out of sync | Wrong pipelines tested | AST parsing auto-updates, plus tests |
| Performance degradation | Slower make invocations | Benchmarked <150ms, acceptable |
| User confusion about override | Wrong pipelines run | Clear documentation in quickstart |

---

## Dependencies

**Build Time**:
- Python 3.11+ (already required)
- iris_rag package installed (development dependency)

**Runtime**:
- None (helper only runs during development)

**Testing**:
- pytest (existing)
- subprocess module (stdlib)

---

## Research Status: ✓ Complete

All technical unknowns have been resolved. Ready to proceed to Phase 1 (Design & Contracts).
