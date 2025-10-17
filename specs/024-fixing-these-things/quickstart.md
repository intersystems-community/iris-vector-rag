# Quickstart: RAG-Templates Quality Improvement

This guide helps you quickly validate the quality improvements made to the RAG-Templates framework.

## Prerequisites

- Python 3.11+
- Docker Desktop running
- Git repository cloned
- uv package manager installed (`pip install uv`)

## Quick Validation Steps

### 1. Setup Test Environment

```bash
# Install dependencies using uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync

# Start IRIS database for tests
docker-compose -f docker-compose.test.yml up -d

# Verify IRIS is running
docker ps | grep iris-test
```

### 2. Run Fixed Tests

```bash
# Run all tests to verify 100% pass rate
pytest tests/ -v

# Expected output:
# =================== 349 passed in XXs ===================
```

### 3. Check Coverage Metrics

```bash
# Run tests with coverage
pytest tests/ --cov=iris_rag --cov=common --cov-report=term --cov-report=html

# View coverage report
open htmlcov/index.html  # On macOS
# xdg-open htmlcov/index.html  # On Linux
# start htmlcov/index.html  # On Windows

# Verify metrics:
# - Overall coverage: ≥60%
# - Critical modules: ≥80% each
```

### 4. Validate CI/CD Pipeline

```bash
# Run local CI checks
make ci-local

# This runs:
# - Linting (black, isort, flake8)
# - Type checking (mypy)
# - Tests with coverage
# - Quality gate validation
```

### 5. Test Specific Critical Modules

```bash
# Test individual critical modules
pytest tests/unit/test_config_unit.py -v
pytest tests/unit/test_validation_unit.py -v
pytest tests/unit/test_pipelines_unit.py -v
pytest tests/unit/test_services_unit.py -v
pytest tests/unit/test_storage_unit.py -v

# All should pass with no failures
```

### 6. Performance Validation

```bash
# Run performance benchmark
pytest tests/performance/ --benchmark-only

# Verify test execution time is within 2x baseline
# Baseline time should be recorded in benchmark_results.json
```

## Validation Checklist

- [ ] All 349 tests pass (0 failures)
- [ ] Overall coverage ≥60%
- [ ] Config module coverage ≥80%
- [ ] Validation module coverage ≥80%
- [ ] Pipelines module coverage ≥80%
- [ ] Services module coverage ≥80%
- [ ] Storage module coverage ≥80%
- [ ] CI/CD pipeline configured in .github/workflows/
- [ ] Quality gates blocking merge on failures
- [ ] Test execution time <2x original baseline
- [ ] Docker test environment working

## Troubleshooting

### Docker Issues
```bash
# If IRIS container fails to start
docker-compose -f docker-compose.test.yml down
docker-compose -f docker-compose.test.yml up -d --force-recreate

# Check logs
docker-compose -f docker-compose.test.yml logs iris-test
```

### Test Failures
```bash
# Run with detailed output
pytest tests/unit/test_failing.py -vvs

# Check for missing dependencies
uv sync --dev
```

### Coverage Issues
```bash
# Generate detailed coverage report
pytest --cov=iris_rag --cov-report=term-missing

# This shows which lines are not covered
```

## Next Steps

1. **Monitor CI/CD**: Check GitHub Actions for automated quality checks
2. **Maintain Coverage**: Ensure new code includes tests
3. **Update Documentation**: Keep test documentation current
4. **Track Metrics**: Use coverage badges in README

## Success Criteria Met

When all validation checks pass, the RAG-Templates framework has achieved:
- ✅ Production-ready quality standards
- ✅ Reliable test suite
- ✅ Comprehensive coverage
- ✅ Automated quality enforcement
- ✅ Maintainable codebase

Congratulations! The framework is now ready for production deployment with confidence.