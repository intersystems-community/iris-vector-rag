# Quickstart: Configurable Test Backend Modes

**Feature**: 035-make-2-modes
**Date**: 2025-10-08

## Overview

This guide shows how to configure and use the configurable test backend modes feature to run tests against IRIS Community or Enterprise editions with appropriate connection limits and execution strategies.

## Prerequisites

1. **iris-devtools dependency** (REQUIRED):
   ```bash
   # Clone iris-devtools as sibling directory
   cd /path/to/parent-directory
   git clone https://github.com/your-org/iris-devtools
   cd rag-templates

   # Verify iris-devtools is available
   ls ../iris-devtools  # Should show iris_devtools package
   ```

2. **Docker running**: Required for IRIS container management
   ```bash
   docker ps  # Should connect without errors
   ```

3. **IRIS database image**:
   - Community: `docker pull intersystemsdc/iris-community:latest`
   - Enterprise: Available as `intersystemsdc/iris-community:2025.3.0EHAT.127.0-linux-arm64v8`

## Configuration

### Option 1: Config File (Persistent)

Create or edit `.specify/config/backend_modes.yaml`:

```yaml
# Backend mode configuration for IRIS test execution
# Options: community | enterprise
backend_mode: community

# Optional: Custom iris-devtools path (default: ../iris-devtools)
# iris_devtools_path: /custom/path/to/iris-devtools
```

**When to use**: Set project-wide default that all developers share

### Option 2: Environment Variable (Override)

Set `IRIS_BACKEND_MODE` environment variable:

```bash
# Temporary (single session)
export IRIS_BACKEND_MODE=enterprise

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export IRIS_BACKEND_MODE=community' >> ~/.zshrc
```

**When to use**: Override config file for local development or CI/CD

### Configuration Precedence

1. **Environment variable** (highest priority)
2. **Config file**
3. **Default** (community mode)

Example:
```bash
# Config file says: community
# Env var says: enterprise
# Result: ENTERPRISE (env var wins)
IRIS_BACKEND_MODE=enterprise pytest tests/
```

## Running Tests

### Community Mode (Default)

```bash
# Using default (community mode)
pytest tests/

# Explicit community mode
IRIS_BACKEND_MODE=community pytest tests/

# Community mode with specific tests
IRIS_BACKEND_MODE=community pytest tests/integration/
```

**Behavior**:
- Maximum 1 concurrent database connection
- Sequential test execution
- Prevents >95% of license exhaustion errors
- Tests may take longer due to sequential execution

**When to use**:
- Local development with Community Edition IRIS
- CI/CD with Community Edition containers
- Limited license pool scenarios

### Enterprise Mode

```bash
# Set enterprise mode
IRIS_BACKEND_MODE=enterprise pytest tests/

# Enterprise mode with parallel execution
IRIS_BACKEND_MODE=enterprise pytest tests/ -n 4

# Enterprise mode for specific test files
IRIS_BACKEND_MODE=enterprise pytest tests/integration/test_graphrag*.py
```

**Behavior**:
- Unlimited concurrent database connections
- Parallel test execution supported
- No artificial throttling
- Faster test execution

**When to use**:
- Enterprise IRIS license available
- CI/CD with licensed IRIS containers
- Performance testing scenarios
- Parallel test execution needed

## Verifying Configuration

### Check Current Mode

```bash
# Run any test with logging to see active mode
pytest tests/unit/test_config.py -v --log-cli-level=INFO

# Look for log message:
# INFO Backend mode: community (source: environment)
```

### Validate Configuration

```python
# In Python REPL or script
from iris_rag.testing.backend_manager import BackendConfiguration

config = BackendConfiguration.load()
print(f"Mode: {config.mode.value}")
print(f"Source: {config.source.value}")
print(f"Max connections: {config.max_connections}")
print(f"Execution strategy: {config.execution_strategy.value}")
```

## Common Scenarios

### Scenario 1: Local Development (Community)

```bash
# Setup once
git clone https://github.com/your-org/iris-devtools ../iris-devtools
echo "backend_mode: community" > .specify/config/backend_modes.yaml

# Run tests
docker-compose up -d  # Start IRIS Community container
pytest tests/         # Uses community mode by default
```

### Scenario 2: CI/CD (Enterprise)

```yaml
# .github/workflows/test.yml
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      IRIS_BACKEND_MODE: enterprise  # Override to enterprise
    steps:
      - uses: actions/checkout@v3
      - name: Clone iris-devtools
        run: git clone https://github.com/your-org/iris-devtools ../iris-devtools
      - name: Start IRIS Enterprise
        run: docker-compose -f docker-compose.enterprise.yml up -d
      - name: Run tests
        run: pytest tests/ -n 4  # Parallel execution OK with enterprise
```

### Scenario 3: Switch Between Modes

```bash
# Start with community
IRIS_BACKEND_MODE=community pytest tests/unit/

# Switch to enterprise for integration tests
IRIS_BACKEND_MODE=enterprise pytest tests/integration/

# Back to community
unset IRIS_BACKEND_MODE  # Use config file default
pytest tests/
```

## Troubleshooting

### Error: "iris-devtools not found at ../iris-devtools"

**Problem**: iris-devtools dependency missing

**Solution**:
```bash
cd /path/to/parent-directory
git clone https://github.com/your-org/iris-devtools
cd rag-templates
pytest tests/  # Should work now
```

### Error: "Backend mode 'enterprise' does not match detected IRIS edition 'community'"

**Problem**: Configuration specifies enterprise mode but Community Edition IRIS is running

**Solutions**:
```bash
# Option 1: Change mode to community
IRIS_BACKEND_MODE=community pytest tests/

# Option 2: Start Enterprise IRIS container
docker-compose -f docker-compose.enterprise.yml up -d
pytest tests/
```

### Error: "Invalid backend mode: foobar"

**Problem**: Invalid mode value in config or environment variable

**Solution**:
```bash
# Check current value
echo $IRIS_BACKEND_MODE

# Set to valid value
export IRIS_BACKEND_MODE=community  # or enterprise

# Or fix config file
vim .specify/config/backend_modes.yaml
# Set: backend_mode: community
```

### Error: "Connection pool timeout after 30s"

**Problem**: Community mode connection limit (1) exceeded by parallel test execution

**Solutions**:
```bash
# Option 1: Run tests sequentially (no -n flag)
IRIS_BACKEND_MODE=community pytest tests/

# Option 2: Switch to enterprise mode
IRIS_BACKEND_MODE=enterprise pytest tests/ -n 4

# Option 3: Reduce parallelism
IRIS_BACKEND_MODE=community pytest tests/ --maxfail=1  # Stop after first failure
```

## Make Targets (Coming Soon)

Convenient make targets will be added:

```bash
# Run tests in community mode
make test-community

# Run tests in enterprise mode
make test-enterprise

# Run tests with auto-detected mode
make test
```

## Advanced Usage

### Custom iris-devtools Path

```yaml
# .specify/config/backend_modes.yaml
backend_mode: community
iris_devtools_path: /custom/path/to/iris-devtools
```

### Programmatic Configuration

```python
from iris_rag.testing.backend_manager import (
    BackendConfiguration,
    BackendMode,
    ConfigSource
)

# Load configuration
config = BackendConfiguration.load()

# Create custom configuration
custom_config = BackendConfiguration(
    mode=BackendMode.ENTERPRISE,
    source=ConfigSource.ENVIRONMENT
)

# Validate against detected edition
from iris_rag.testing.validators import detect_iris_edition

connection = get_iris_connection()
edition = detect_iris_edition(connection)
config.validate(edition)  # Raises EditionMismatchError if mismatch
```

### Connection Pool Monitoring

```python
from iris_rag.testing.connection_pool import get_connection_pool

pool = get_connection_pool()
print(f"Active connections: {pool.get_active_count()}")
print(f"Max connections: {pool.config.max_connections}")
```

## Next Steps

1. **Review contracts**: See `contracts/` directory for API specifications
2. **Run contract tests**: `pytest tests/contract/test_backend_*.py`
3. **Integration tests**: `pytest tests/integration/ -v`
4. **Read data model**: See `data-model.md` for entity definitions

## Support

- **Documentation**: See `plan.md` for implementation details
- **Issues**: Report at https://github.com/your-org/rag-templates/issues
- **Examples**: See `tests/contract/` for usage examples
