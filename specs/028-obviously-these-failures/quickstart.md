# Quickstart: Test Infrastructure Resilience

**Feature**: 028-obviously-these-failures
**Date**: 2025-10-05

## Overview

This guide demonstrates how to use the test infrastructure resilience features to run the test suite with automatic schema management and contract test handling.

## Prerequisites

- IRIS database running (docker-compose up -d)
- Python 3.12+ environment
- pytest 8.4.1+ installed

## Quick Start

### 1. Run Tests with Auto-Schema-Management

```bash
# Automatic schema validation and reset if needed
pytest tests/

# Or with explicit cleanup
pytest tests/ --clean-schema
```

**What Happens**:
1. Pre-flight checks validate IRIS connectivity
2. Schema validator detects any table mismatches
3. If mismatch found: automatic schema reset (<5s)
4. Tests run with clean, valid schema
5. Cleanup handlers remove test data after each class

### 2. Manual Schema Reset

```bash
# Reset database schema manually
make test-reset-schema

# Or via Python
python -c "from tests.fixtures.schema_reset import SchemaResetter; SchemaResetter().reset_schema()"
```

### 3. Run Contract Tests

```bash
# Contract tests (expected to fail for unimplemented features)
pytest tests/ -m contract

# These tests will show as "xfail" (expected failure) rather than ERROR
```

## Usage Scenarios

### Scenario 1: Developer Runs Full Test Suite

**Given**: Developer has IRIS database running
**When**: Developer runs `pytest tests/`
**Then**:
- Pre-flight checks validate database connectivity (< 2s)
- Schema validated against expected structure
- All 771 tests execute without schema errors
- Contract tests for MCP properly marked as "xfail"
- Test results show actual code issues, not infrastructure problems

**Example Output**:
```
============================= test session starts ==============================
Pre-flight checks: IRIS connectivity ✓ | Schema valid ✓ | API keys ✓
collecting ... collected 771 items

tests/contract/test_coverage_contract.py::test_COV001 PASSED
tests/contract/test_mcp_contract.py::test_mcp_server XFAIL (expected - unimplemented)
tests/e2e/test_basic_pipeline_e2e.py::test_simple_query PASSED
...

========== 650 passed, 47 xfailed (expected), 8 skipped in 380.00s ===========
```

### Scenario 2: Stale Schema Auto-Reset

**Given**: Tables exist from previous test run with old schema
**When**: Developer runs `pytest tests/`
**Then**:
- Schema validator detects mismatches
- Automatic reset executed (drops and recreates tables)
- Tests proceed with correct schema
- Developer sees warning about schema reset

**Example Output**:
```
WARNING: Schema mismatch detected for table 'SourceDocuments'
  - Column 'metadata': expected JSON, found VARCHAR
Resetting schema... done (1.8s)
Schema validation: PASS
```

### Scenario 3: Test Fails Leaving Partial Data

**Given**: Test inserts documents but fails during entity extraction
**When**: Next test in same class runs
**Then**:
- Cleanup handler executes (registered via finalizer)
- Partial documents removed
- Next test starts with clean state
- No data pollution

**Implementation**:
```python
@pytest.fixture(scope="class")
def database_with_clean_schema(request):
    """Provides clean database state for test class."""
    conn = get_iris_connection()

    # Cleanup handler - ALWAYS runs
    def cleanup():
        cursor = conn.cursor()
        cursor.execute("DELETE FROM RAG.SourceDocuments WHERE test_run_id = ?", [test_run_id])
        conn.commit()

    request.addfinalizer(cleanup)

    yield conn

    # Cleanup runs here even if test failed
```

### Scenario 4: Contract Test for Unimplemented Feature

**Given**: MCP server modules not yet implemented
**When**: Contract test runs: `test_mcp_server_startup()`
**Then**:
- Test attempts to import `iris_rag.mcp.server_manager`
- Import fails (module doesn't exist)
- Plugin intercepts failure
- Test marked as "xfail" (expected failure)
- Does NOT contribute to overall failure count

**Example**:
```python
@pytest.mark.contract
def test_mcp_server_startup(self):
    """Test MCP server can start (contract - not implemented yet)."""
    from iris_rag.mcp.server_manager import MCPServerManager  # Will fail

    server = MCPServerManager()
    result = server.start()
    assert result['success'] is True
```

**Output**:
```
tests/test_mcp/test_mcp_server.py::test_mcp_server_startup XFAIL
  Reason: Contract test - feature not implemented (iris_rag.mcp.server_manager)
```

## Configuration

### pytest.ini

```ini
[tool:pytest]
# Contract test marker
markers =
    contract: Contract tests that define expected behavior (may fail if unimplemented)

# Enable contract test plugin
plugins =
    tests.plugins.contract_test_marker
```

### Environment Variables

```bash
# Database connection
IRIS_HOST=localhost
IRIS_PORT=11972
IRIS_USERNAME=_SYSTEM
IRIS_PASSWORD=SYS

# API keys (for LLM tests)
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here  # optional
```

## Troubleshooting

### Issue: Schema validation fails

**Symptom**: `SchemaValidationError: Table structure mismatch`

**Solution**:
```bash
# Manual reset
make test-reset-schema

# Or
pytest tests/ --reset-schema
```

### Issue: Cleanup not running

**Symptom**: Test data persists between runs

**Solution**:
- Verify cleanup handlers use `request.addfinalizer()`
- Check pytest finalizers are enabled
- Manual cleanup: `make test-cleanup`

### Issue: Contract tests showing as ERROR

**Symptom**: Contract tests fail with ERROR instead of XFAIL

**Solution**:
- Verify `@pytest.mark.contract` decorator is present
- Check plugin is loaded: `pytest --trace-config | grep contract_test_marker`
- Update pytest.ini to enable plugin

## Performance Validation

### Expected Overhead

- Schema validation: ~200ms (one-time, test session startup)
- Per-class cleanup: ~50ms per test class
- Total for 771 tests: ~7.7s additional (2% increase)

### Measuring Actual Performance

```bash
# Run with timing
pytest tests/ --durations=10

# Check schema reset time
pytest tests/contract/schema_manager_contract.py::test_schema_reset_completes_under_5_seconds -v
```

## Next Steps

1. **Run Full Test Suite**: `pytest tests/`
2. **Verify Results**: Check for 0 schema errors
3. **Review Contract Tests**: See which features are still TDD contracts
4. **Continuous Integration**: Add to CI/CD pipeline

## Medical-Grade Quality Checklist

- [ ] All 771 tests execute without schema errors
- [ ] Contract tests properly marked (47 xfail, not ERROR)
- [ ] Schema reset completes in <5 seconds
- [ ] Test isolation overhead <100ms per class
- [ ] Pre-flight checks complete in <2 seconds
- [ ] Audit trail logged for all schema operations
- [ ] Cleanup handlers execute even on test failures

**Success Criteria**: Test suite provides reliable, reproducible results that accurately reflect code quality, not infrastructure issues.
