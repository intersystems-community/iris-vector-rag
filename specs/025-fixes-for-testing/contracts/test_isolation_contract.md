# Contract: Test Isolation Requirements

**Feature**: 025-fixes-for-testing
**Date**: 2025-10-03

## Requirements

### REQ-1: Database Cleanup
- E2E tests must clean up IRIS data after execution
- Use teardown fixtures to delete test documents
- Prevent state pollution between tests

### REQ-2: Fixture Scope
- IRIS connections: module scope (amortize setup cost)
- Test documents: function scope (isolated per test)
- Pipelines: function scope (fresh state)

### REQ-3: Transaction Rollback
- Unit tests use transactions when possible
- Rollback after test to restore clean state

## Contract Test

```python
def test_database_cleanup(iris_connection):
    """Validate test data is cleaned up."""
    # Insert test document
    doc_id = "test_isolation_doc"
    # ... insert logic ...

    # Verify exists
    # ... check logic ...

    # Cleanup (should happen in teardown)
    # After test, verify document deleted
```

## Success Criteria
- No test data persists after test suite completes
- Tests can run in any order without failures
- Database state is consistent between test runs
