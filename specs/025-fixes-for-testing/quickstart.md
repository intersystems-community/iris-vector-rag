# Quickstart: Testing Framework

**Feature**: 025-fixes-for-testing
**Date**: 2025-10-03
**For**: Developers working on RAG-Templates test suite

## Prerequisites

1. **IRIS Database Running**:
   ```bash
   docker-compose up -d
   # Wait for IRIS to be healthy
   docker logs iris-pgwire-db --tail 50
   ```

2. **Python Environment**:
   ```bash
   python --version  # Requires Python 3.12+
   source .venv/bin/activate  # Or your virtual environment
   ```

3. **Dependencies Installed**:
   ```bash
   pip install pytest pytest-cov sentence-transformers langchain
   ```

## Quick Test Runs

### Run All Tests
```bash
export PYTHONPATH=/Users/intersystems-community/ws/rag-templates
pytest tests/unit/ tests/e2e/ -p no:randomly -q
```

**Expected Output**:
```
206 passed, 5 skipped in 106.03s
```

### Run With Coverage
```bash
pytest tests/unit/ tests/e2e/ --cov=iris_rag --cov=common --cov-report=term -p no:randomly
```

**Expected**: Coverage report showing 60%+ overall, 80%+ for critical modules

### Run Specific Test File
```bash
pytest tests/e2e/test_basic_pipeline_e2e.py -v
```

### Run Single Test
```bash
pytest tests/e2e/test_basic_pipeline_e2e.py::TestBasicRAGPipelineQuerying::test_simple_query -v
```

## Common Issues

### Issue 1: IRIS Not Running
**Symptom**: `Connection refused` or `IRIS database not running`
**Fix**:
```bash
docker-compose up -d
# Verify IRIS is running
docker ps | grep iris
```

### Issue 2: Import Errors
**Symptom**: `ModuleNotFoundError: No module named 'iris_rag'`
**Fix**:
```bash
export PYTHONPATH=/Users/intersystems-community/ws/rag-templates
# Or add to your shell profile
echo 'export PYTHONPATH=/Users/intersystems-community/ws/rag-templates' >> ~/.zshrc
```

### Issue 3: pytest-randomly Errors
**Symptom**: `ValueError: Seed must be between 0 and 2**32 - 1`
**Fix**: Always use `-p no:randomly` flag (already in pytest.ini)

### Issue 4: Failing Tests
**Current**: 60 failing tests (API mismatches), 11 errors (GraphRAG setup)
**Fix**: Part of this feature - see [api_alignment_contract.md](./contracts/api_alignment_contract.md)

## Adding New Tests

### 1. Unit Test (No IRIS)
```python
# tests/unit/test_my_feature.py
import pytest

def test_my_function():
    """Unit test with mocks."""
    result = my_function(input_data)
    assert result == expected
```

### 2. E2E Test (With IRIS)
```python
# tests/e2e/test_my_pipeline_e2e.py
import pytest
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager

@pytest.fixture(scope="module")
def pipeline_dependencies():
    """Real IRIS dependencies."""
    config = ConfigurationManager()
    conn = ConnectionManager(config)
    return {"config": config, "conn": conn}

@pytest.mark.e2e
@pytest.mark.requires_database
def test_my_pipeline_feature(pipeline_dependencies):
    """E2E test with real IRIS."""
    # Test code here
    pass
```

### 3. Contract Test
```python
# tests/contract/test_my_contract.py
def test_my_api_contract():
    """Validate API signature."""
    import inspect
    sig = inspect.signature(MyClass.my_method)
    # Assert signature matches expectations
```

## Test Organization

```
tests/
├── unit/           # Fast, isolated, mocked
│   └── conftest.py # Mock fixtures
├── e2e/            # Slow, integrated, real IRIS
│   └── conftest.py # Real IRIS fixtures
├── integration/    # Cross-component, real deps
└── contract/       # API validation tests
```

## Verification Steps

After making changes, verify:

1. **All Passing Tests Still Pass**:
   ```bash
   pytest tests/ -p no:randomly -q --tb=no
   # Should see 206+ passed
   ```

2. **Coverage Maintained or Improved**:
   ```bash
   pytest tests/ --cov=iris_rag --cov=common --cov-report=term -p no:randomly -q
   # Should see 60%+ overall
   ```

3. **No New Errors**:
   ```bash
   pytest tests/ -p no:randomly 2>&1 | grep ERROR
   # Should see 11 or fewer GraphRAG errors
   ```

4. **Test Execution Time Acceptable**:
   ```bash
   time pytest tests/ -p no:randomly -q --tb=no
   # Should complete in < 2 minutes
   ```

## Next Steps

1. **Fix Failing Tests**: See [api_alignment_contract.md](./contracts/api_alignment_contract.md)
2. **Resolve GraphRAG Errors**: See [graphrag_setup_contract.md](./contracts/graphrag_setup_contract.md)
3. **Improve Coverage**: Add tests for modules <60% coverage
4. **Run Full Validation**: Execute all contract tests

## Resources

- [Test Execution Contract](./contracts/test_execution_contract.md)
- [Coverage Reporting Contract](./contracts/coverage_reporting_contract.md)
- [Test Isolation Contract](./contracts/test_isolation_contract.md)
- [API Alignment Contract](./contracts/api_alignment_contract.md)
- [GraphRAG Setup Contract](./contracts/graphrag_setup_contract.md)
