# Test Environment Setup Guide

This guide provides instructions for setting up the test environment for the RAG-Templates framework.

## Prerequisites

- Python 3.11+
- Docker Desktop running
- Git repository cloned
- uv package manager installed (`pip install uv`)

## Quick Setup

### 1. Install Python Dependencies

```bash
# Create virtual environment using uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies (dev + runtime)
uv sync
```

### 2. Start IRIS Database

The test suite requires a running IRIS database instance. We provide Docker Compose configurations for different scenarios:

```bash
# Start test database (isolated from production)
docker-compose -f docker-compose.test.yml up -d

# Verify IRIS is running
docker ps | grep rag_iris_test

# Check health
docker-compose -f docker-compose.test.yml ps
```

**Test Database Ports:**
- SuperServer: 31972 (vs 11972 for production)
- Management Portal: 35273 (vs 15273 for production)

### 3. Environment Variables

Create a `.env` file in the project root with the following variables:

```bash
# IRIS Database Configuration (Test)
IRIS_HOST=localhost
IRIS_PORT=31972
IRIS_USERNAME=_SYSTEM
IRIS_PASSWORD=SYS
IRIS_NAMESPACE=USER

# LLM API Keys (for integration tests)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Test Configuration
PYTEST_TIMEOUT=300
TEST_DATABASE_VOLUME=test-iris-data
```

**Note:** Unit tests use mocks and don't require real API keys. Integration/E2E tests may require them.

### 4. Verify Setup

Run the test suite health check:

```bash
# Run quick smoke tests
pytest tests/ -m smoke -v

# Run unit tests (no external dependencies)
pytest tests/unit/ -v

# Run all tests
pytest tests/ -v
```

## Test Categories

The test suite is organized into categories using pytest markers:

### Unit Tests
```bash
pytest tests/unit/ -v
# or
pytest -m unit
```

**Characteristics:**
- Fast execution (< 1s per test)
- No external dependencies
- Use mocks for database/API calls
- Test individual components in isolation

### Integration Tests
```bash
pytest tests/integration/ -v
# or
pytest -m integration
```

**Characteristics:**
- Medium execution time (1-5s per test)
- Require IRIS database
- Test component interactions
- May use real database queries

### E2E Tests
```bash
pytest tests/e2e/ -v
# or
pytest -m e2e
```

**Characteristics:**
- Slower execution (5-30s per test)
- Require full stack (database + external services)
- Test complete workflows
- Validate end-to-end functionality

### Contract Tests
```bash
pytest tests/contract/ -v
```

**Characteristics:**
- API contract validation
- Ensure backward compatibility
- Test request/response schemas
- Validate public interfaces

## Running Tests with Coverage

### Basic Coverage

```bash
# Run all tests with coverage
pytest tests/ --cov=iris_rag --cov=common --cov-report=term

# Generate HTML coverage report
pytest tests/ --cov=iris_rag --cov=common --cov-report=html

# View HTML report
open htmlcov/index.html  # macOS
# xdg-open htmlcov/index.html  # Linux
# start htmlcov/index.html  # Windows
```

### Coverage Targets

The project enforces the following coverage requirements:

- **Overall Coverage**: ≥60%
- **Critical Modules**: ≥80%
  - `iris_rag/config/`
  - `iris_rag/validation/`
  - `iris_rag/pipelines/`
  - `iris_rag/services/`
  - `iris_rag/storage/`

### Check Coverage Compliance

```bash
# Run coverage validation
python iris_rag/testing/example_usage.py

# Or use pytest with coverage threshold
pytest tests/ --cov=iris_rag --cov=common --cov-fail-under=60
```

## Test Configuration

The test suite is configured via `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
    requires_docker: Requires Docker
    requires_database: Requires database connection
```

**Key Configuration:**
- `pytest-randomly` is disabled to avoid seed conflicts
- Coverage is configured for `iris_rag/` and `common/` modules
- Test timeout is set to 300 seconds (5 minutes)

## Docker Test Environment

### Architecture

The Docker test environment provides:

1. **Isolated IRIS Database** - Separate from production data
2. **Test Data Volumes** - Persistent or ephemeral test data
3. **Health Checks** - Automated service readiness verification
4. **Network Isolation** - Dedicated `rag-test-network`

### Managing Test Database

```bash
# Start test database
docker-compose -f docker-compose.test.yml up -d

# Stop test database
docker-compose -f docker-compose.test.yml down

# Reset test database (clean slate)
docker-compose -f docker-compose.test.yml down -v
docker-compose -f docker-compose.test.yml up -d

# View logs
docker-compose -f docker-compose.test.yml logs -f iris-test

# Access IRIS shell
docker exec -it rag_iris_test iris session iris -U USER
```

### Test Database Volumes

The test database supports multiple volume strategies:

```bash
# Use named volume (persistent between runs)
TEST_DATABASE_VOLUME=test-iris-data docker-compose -f docker-compose.test.yml up -d

# Use ephemeral volume (clean on restart)
TEST_DATABASE_VOLUME=$(mktemp -d) docker-compose -f docker-compose.test.yml up -d
```

## Troubleshooting

### IRIS Database Connection Issues

**Symptom:** Tests fail with "Connection refused" or "Unable to connect to IRIS"

**Solution:**
```bash
# Check if IRIS is running
docker ps | grep rag_iris_test

# If not running, start it
docker-compose -f docker-compose.test.yml up -d

# Wait for health check to pass
docker-compose -f docker-compose.test.yml ps

# Check logs for errors
docker-compose -f docker-compose.test.yml logs iris-test
```

### Port Conflicts

**Symptom:** Docker fails to start with "port already in use"

**Solution:**
```bash
# Check what's using the ports
lsof -i :31972  # Test SuperServer port
lsof -i :35273  # Test Management Portal port

# Kill conflicting process or change ports in docker-compose.test.yml
```

### Test Discovery Issues

**Symptom:** pytest can't find tests or modules

**Solution:**
```bash
# Ensure you're in the project root
cd /path/to/rag-templates

# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall package in editable mode
pip install -e .

# Verify Python path
python -c "import iris_rag; print(iris_rag.__file__)"
```

### Mock Configuration Errors

**Symptom:** Tests fail with "AttributeError: Mock object has no attribute..."

**Solution:**
- Check that mock fixtures in `tests/conftest.py` are properly configured
- Ensure test is using the correct fixture (e.g., `mock_vector_store` vs `iris_test_session`)
- Review test markers - unit tests should use mocks, integration tests use real connections

### Coverage Report Errors

**Symptom:** Coverage report shows 0% or missing modules

**Solution:**
```bash
# Ensure coverage is tracking the right modules
pytest --cov=iris_rag --cov=common --cov-report=term-missing

# Check .coveragerc configuration
cat .coveragerc

# Verify source paths are correct
pytest --cov-config=.coveragerc --cov-report=term
```

## Performance Considerations

### Parallel Test Execution

For faster test runs, use pytest-xdist:

```bash
# Run tests in parallel (auto-detect CPU cores)
pytest tests/ -n auto

# Run with specific number of workers
pytest tests/ -n 4

# Parallel with coverage (slower due to coverage overhead)
pytest tests/ -n auto --cov=iris_rag --cov=common
```

**Note:** Parallel execution may cause issues with database tests. Use carefully.

### Baseline Performance

Expected test execution times:
- Unit tests: ~10-30 seconds (full suite)
- Integration tests: ~1-2 minutes (full suite)
- E2E tests: ~2-5 minutes (full suite)
- Full test suite: ~5-10 minutes

### Optimization Tips

1. **Run unit tests first** - Fast feedback on core logic
2. **Use test markers** - Run only relevant tests during development
3. **Skip slow tests** - Use `-m "not slow"` for rapid iteration
4. **Parallelize when safe** - Unit tests are safe to parallelize
5. **Cache coverage data** - Reuse coverage between runs when possible

## CI/CD Integration

The test suite integrates with GitHub Actions via `.github/workflows/ci.yml`:

```yaml
# Key CI steps:
1. Setup Python 3.11+
2. Install dependencies via uv
3. Start IRIS test database
4. Run tests with coverage
5. Upload coverage reports
6. Enforce quality gates
```

**Quality Gates:**
- All tests must pass (0 failures)
- Overall coverage ≥60%
- Critical modules ≥80% coverage
- Test execution time <2x baseline

## Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [InterSystems IRIS Documentation](https://docs.intersystems.com/)

## Support

For test setup issues:
1. Check this document's troubleshooting section
2. Review test logs in `test-results/logs/`
3. Check GitHub Actions workflow runs
4. Open an issue in the repository
