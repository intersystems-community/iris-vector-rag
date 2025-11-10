# Implementation Plan: Pipeline Contract Validation

**Feature**: 047-pipeline-contract-validation
**Status**: Ready to Implement
**Estimated Effort**: 4-6 hours

---

## Implementation Order

### Step 1: Core Validator (1-2 hours)
**File**: `iris_rag/core/validators.py` (new)

**Tasks**:
- [ ] Create `PipelineContractViolation` class
- [ ] Implement `PipelineValidator.validate_pipeline_class()`
- [ ] Implement `PipelineValidator._validate_query_signature()`
- [ ] Implement `PipelineValidator.validate_response()`
- [ ] Add comprehensive docstrings

**Testing**:
- Create `tests/contract/test_pipeline_validator.py`
- Test compliant pipeline validation
- Test detection of missing query parameter
- Test deprecation warnings for query_text

### Step 2: Update HybridGraphRAG (Already Done!) ✅
**File**: `iris_rag/pipelines/hybrid_graphrag.py`

**Status**: COMPLETE - Already updated to support both `query` and `query_text`

```python
def query(
    self,
    query: str = None,
    query_text: str = None,  # Backward compatibility
    ...
):
    # Support both standard and deprecated names
    if query is None and query_text is None:
        raise ValueError("Either 'query' or 'query_text' parameter is required")
    query_text = query if query is not None else query_text
```

### Step 3: Registry Integration (1 hour)
**File**: `iris_rag/mcp/technique_handlers.py`

**Tasks**:
- [ ] Add `strict_mode` parameter to `TechniqueHandlerRegistry.__init__()`
- [ ] Implement `_register_with_validation()` method
- [ ] Update `_register_default_handlers()` to use validation
- [ ] Add logging for validation results

### Step 4: Configuration (30 minutes)
**File**: `iris_rag/config/default_config.yaml`

**Tasks**:
- [ ] Add `validation:` section
- [ ] Add `enabled`, `strict_mode`, `validate_responses`, `log_level` settings
- [ ] Update ConfigurationManager to load validation settings

### Step 5: Testing (1-2 hours)
**Files**:
- `tests/contract/test_pipeline_validator.py` (new)
- `tests/integration/test_pipeline_registration.py` (new)

**Tasks**:
- [ ] Unit tests for PipelineValidator
- [ ] Integration tests for registry validation
- [ ] Test strict mode enforcement
- [ ] Test all 6 default pipelines pass validation

### Step 6: Documentation (1 hour)
**Files**:
- `CLAUDE.md`
- `specs/047-pipeline-contract-validation/README.md`

**Tasks**:
- [ ] Add "Pipeline API Contract" section to CLAUDE.md
- [ ] Document query() method signature requirements
- [ ] Document response format requirements
- [ ] Add troubleshooting guide for common violations
- [ ] Create migration guide for custom pipelines

---

## Quick Start Commands

```bash
# Create feature branch
git checkout -b 047-pipeline-contract-validation

# Create new files
touch iris_rag/core/validators.py
touch tests/contract/test_pipeline_validator.py
touch tests/integration/test_pipeline_registration.py

# Run tests after implementation
pytest tests/contract/test_pipeline_validator.py -v
pytest tests/integration/test_pipeline_registration.py -v

# Verify all pipelines pass validation
python -c "from iris_rag.mcp.technique_handlers import TechniqueHandlerRegistry; r = TechniqueHandlerRegistry(); print('✅ All pipelines registered successfully')"
```

---

## Validation Checkpoints

After each step, verify:

1. **After Step 1**: PipelineValidator unit tests pass (100%)
2. **After Step 2**: GraphRAG supports both `query` and `query_text` (already done)
3. **After Step 3**: Registry logs validation results for all 6 pipelines
4. **After Step 4**: Configuration loads correctly
5. **After Step 5**: All 58 MCP integration tests still pass (regression check)
6. **After Step 6**: Documentation is clear and complete

---

## Success Criteria

- [ ] All 6 default pipelines pass validation without errors
- [ ] GraphRAG shows warnings for `query_text` but still registers
- [ ] Strict mode successfully blocks non-compliant pipelines
- [ ] All existing tests still pass (58/58 MCP tests)
- [ ] Documentation includes complete API contract specification
- [ ] New validator tests achieve >90% code coverage

---

## Risk Mitigation

**Risk**: Breaking existing code that uses `query_text`
**Mitigation**: GraphRAG already updated to support both parameters

**Risk**: Performance impact from validation
**Mitigation**: Validation only runs at registration time (startup), not per-query

**Risk**: False positives in validation
**Mitigation**: Warnings (not errors) for minor issues, strict mode is optional

---

## Next Actions

1. Review spec.md and plan.md
2. Create feature branch
3. Implement Step 1 (Core Validator)
4. Run initial tests
5. Proceed to Steps 2-6
