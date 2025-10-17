# Contract: API Alignment Requirements

**Feature**: 025-fixes-for-testing
**Date**: 2025-10-03

## Purpose

Ensure test expectations match actual production API signatures.

## Known API Mismatches (60 failing tests)

### BasicRAG Pipeline
- **load_documents**: Test expects `load_documents(documents)`, actual is `load_documents(documents_path, documents=None)`
- **Fix**: Update tests to pass `documents` as kwarg

### CRAG Pipeline (19 failing)
- Query metadata expectations don't match actual response structure
- Fix: Update tests to validate actual metadata fields

### GraphRAG Pipeline (39 failing)
- Entity extraction API mismatches
- Graph traversal result structure mismatches
- Fix: Align tests with actual entity/graph APIs

### PyLate Pipeline (7 failing)
- Document loading and stats tracking API mismatches
- Fix: Update to match PyLate fallback mode behavior

### Vector Store (8 failing)
- Metadata filtering expects different filter syntax
- Similarity thresholds use different parameter names
- Fix: Align with IRIS vector store actual API

## Contract Test Pattern

```python
import inspect

def test_api_signature_match():
    """Validate test expectations match production."""
    actual_sig = inspect.signature(BasicRAGPipeline.load_documents)
    expected_params = ["documents_path", "documents"]

    actual_params = list(actual_sig.parameters.keys())
    assert all(p in actual_params for p in expected_params)
```

## Success Criteria
- All 60 failing tests updated to match production APIs
- Contract tests validate signature alignment
- No test-driven API changes (tests follow production)
